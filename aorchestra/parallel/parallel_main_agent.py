"""Parallel-capable MainAgent implementation."""
from __future__ import annotations

import json
from typing import Any, Dict, List

from pydantic import Field

from base.engine.logs import LogLevel, logger
from aorchestra.main_agent import MainAgent
from aorchestra.parallel.parallel_prompt import append_parallel_delegate_instructions
from aorchestra.parallel.parallel_utils import (
    build_delegate_error_result,
    execute_parallel_delegates,
    parse_multi_action_response,
    summarize_parallel_delegate_results,
)


class ParallelMainAgent(MainAgent):
    """MainAgent extension that can execute multiple delegate_task actions in parallel."""

    parallel_delegate: bool = Field(default=False)
    max_parallel_tasks: int = Field(default=5)

    async def step(self, observation, history, **kwargs) -> tuple:
        """Execute one orchestration decision with optional parallel delegation."""
        self.attempt += 1
        logger.info(f"[ParallelMainAgent] Step {self.attempt}/{self.max_attempts}")

        subtask_history = self._format_subtask_history()
        logger.info(f"[ParallelMainAgent] Subtask history:\n{subtask_history}")

        prompt = self._build_prompt(subtask_history)
        prompt_msg = (
            f"\n{'='*80}\n"
            f"[ParallelMainAgent Attempt {self.attempt}] PROMPT:\n"
            f"{'='*80}\n{prompt}\n{'='*80}\n"
        )
        logger.warning(prompt_msg)
        logger.log_to_file(LogLevel.INFO, prompt_msg)

        logger.info("[ParallelMainAgent] Calling LLM...")
        resp = await self.llm(prompt)

        response_msg = (
            f"\n{'='*80}\n"
            f"[ParallelMainAgent Attempt {self.attempt}] RAW RESPONSE:\n"
            f"{'='*80}\n{resp}\n{'='*80}\n"
        )
        logger.warning(response_msg)
        logger.log_to_file(LogLevel.INFO, response_msg)

        try:
            actions = parse_multi_action_response(resp)
        except Exception as exc:
            logger.error(f"[ParallelMainAgent] Failed to parse decision: {exc}")
            return {
                "action": "error",
                "error": f"Failed to parse LLM response: {exc}",
                "result": {"done": False},
                "subtask_history": subtask_history,
            }, resp

        decision_msg = (
            f"\n{'='*80}\n"
            f"[ParallelMainAgent Attempt {self.attempt}] PARSED DECISION:\n"
            f"{'='*80}\n{json.dumps(actions, indent=2, ensure_ascii=False)}\n{'='*80}\n"
        )
        logger.warning(decision_msg)
        logger.log_to_file(LogLevel.INFO, decision_msg)

        action_payload = await self._execute_actions(actions)
        action_payload["subtask_history"] = subtask_history

        context_msg = (
            f"\n{'='*80}\n"
            f"[ParallelMainAgent Attempt {self.attempt}] UPDATED CONTEXT:\n"
            f"{'='*80}\n{self.context}\n{'='*80}\n"
        )
        logger.warning(context_msg)
        logger.log_to_file(LogLevel.INFO, context_msg)

        return action_payload, resp

    def _build_prompt(self, subtask_history: str) -> str:
        if self.prompt_builder:
            prompt = self.prompt_builder.build_prompt(
                instruction=self.instruction,
                meta=self.meta,
                tools_description=self._get_tools_description(),
                prior_context=self.context,
                attempt_index=self.attempt,
                max_attempts=self.max_attempts,
                sub_models=self.masked_sub_models,
                subtask_history=subtask_history,
                model_to_alias=self.model_to_alias if self.mask_model_names else None,
            )
        else:
            prompt = self._default_prompt()

        if self.parallel_delegate:
            prompt = append_parallel_delegate_instructions(prompt, self.masked_sub_models)
        return prompt

    async def _execute_actions(self, actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not actions:
            return {"action": "error", "error": "No actions provided", "result": {"done": False}}

        if not self.parallel_delegate or len(actions) == 1:
            if len(actions) > 1 and not self.parallel_delegate:
                logger.warning(
                    "[ParallelMainAgent] parallel_delegate=False but multiple actions returned; executing first action only."
                )
            return await self._execute_single_action(actions[0])

        delegate_actions = [action for action in actions if action.get("action") == "delegate_task"]
        if len(delegate_actions) <= 1:
            logger.warning(
                "[ParallelMainAgent] Multiple actions received, but not enough delegate_task actions for parallel run."
            )
            return await self._execute_single_action(actions[0])

        non_delegate_actions = [action for action in actions if action.get("action") != "delegate_task"]
        if non_delegate_actions:
            logger.warning(
                "[ParallelMainAgent] Mixed action list detected; only delegate_task actions will run in parallel."
            )

        results = await execute_parallel_delegates(
            tools=self.tools,
            actions=delegate_actions,
            max_concurrent=self.max_parallel_tasks,
        )
        self._update_context_batch(delegate_actions, results)

        aggregated_result = summarize_parallel_delegate_results(results)
        if non_delegate_actions:
            aggregated_result["ignored_actions"] = [a.get("action") for a in non_delegate_actions]

        return {
            "action": "delegate_task",
            "params": {"actions": [action.get("params", {}) for action in delegate_actions]},
            "parallel_actions": delegate_actions,
            "result": aggregated_result,
        }

    async def _execute_single_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        action_name = action.get("action")
        params = action.get("params", {}) or {}

        tool = next((t for t in self.tools if t.name == action_name), None)
        if not tool:
            return {
                "action": "error",
                "error": f"Unknown action: {action_name}",
                "params": params,
                "result": {"done": False},
            }

        try:
            result = await tool(**params)
        except Exception as exc:
            logger.error(f"[ParallelMainAgent] Action `{action_name}` failed: {exc}")
            if action_name == "delegate_task":
                result = build_delegate_error_result(str(exc), model=params.get("model", "unknown"))
            else:
                result = {"error": str(exc), "done": False}

        self._update_context(action_name, params, result)

        return {
            "action": action_name,
            "params": params,
            "result": result,
        }

    def _update_context_batch(self, actions: List[Dict[str, Any]], results: List[Dict[str, Any]]) -> None:
        """Update context for each parallel delegate result."""
        for action, result in zip(actions, results):
            action_name = action.get("action", "delegate_task")
            params = action.get("params", {}) or {}

            if not isinstance(result, dict):
                error_result = build_delegate_error_result(
                    f"Invalid result type: {type(result).__name__}",
                    model=params.get("model", "unknown"),
                )
                self._record_error(action, error_result["error"])
                self._update_context(action_name, params, error_result)
                continue

            if result.get("error"):
                self._record_error(action, str(result.get("error")))
                if not isinstance(result.get("finish_result"), dict):
                    normalized = build_delegate_error_result(
                        str(result["error"]),
                        model=params.get("model", "unknown"),
                    )
                    normalized.update(result)
                    result = normalized

            self._update_context(action_name, params, result)

    @staticmethod
    def _record_error(action: Dict[str, Any], error: str) -> None:
        logger.error(f"[ParallelMainAgent] Action failed: {action} | error={error}")
