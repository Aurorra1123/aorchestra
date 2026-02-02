"""Open GAIA-like research entry with autonomous MainAgent/SubAgent orchestration.

Usage:
  python aorchestra/parallel/example_gaia_like_parallel_gold.py \
    --question "Analyze gold price trends since 2026 and provide investment advice, helping me avoid major events and risks."

This script is benchmark-free:
- Input: one research question
- Process: MainAgent autonomously plans/delegates, SubAgents use GAIA tools
- Output: final answer + trajectory files
- No answer scoring
"""
from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Allow direct run: `python aorchestra/parallel/example_gaia_like_parallel_gold.py`
if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from base.agent.base_action import BaseAction
from base.engine.async_llm import LLMsConfig, create_llm_instance
from base.engine.logs import logger
from benchmark.common.env import Action, BasicInfo, Environment, Observation
from benchmark.common.runner import LevelResult, Runner, StepRecord
from benchmark.gaia.tools import ExecuteCodeAction, ExtractUrlContentAction, GoogleSearchAction
from aorchestra.parallel.parallel_main_agent import ParallelMainAgent
from aorchestra.prompts.gaia import GAIAMainAgentPrompt
from aorchestra.tools.complete import CompleteTool
from aorchestra.tools.delegate import DelegateTaskTool

DEFAULT_QUESTION = "Analyze gold price trends since 2026 and provide investment advice, helping me avoid major events and risks."


class OpenResearchGaiaEnv(Environment):
    """Single-question GAIA-like environment for SubAgents (finish-based)."""

    def __init__(
        self,
        question: str,
        tools: List[BaseAction],
        max_steps: int = 30,
        task_id: str = "open_research",
        attachment_path: Optional[str] = None,
    ):
        self.task_id = task_id
        self.question = question.strip()
        self.max_steps = max(1, int(max_steps))
        self.tools: Dict[str, BaseAction] = {tool.name: tool for tool in tools}
        self.attachment_path = Path(attachment_path).expanduser().resolve() if attachment_path else None

        self.instruction = self._build_instruction()
        self._steps = 0
        self._done = False

    def _build_instruction(self) -> str:
        instruction = f"Question: {self.question}"
        if self.attachment_path:
            instruction += (
                "\n\n[ATTACHMENT]\n"
                f"File path: {self.attachment_path}\n"
                "Use available tools to inspect or process this file if relevant."
            )
        return instruction

    def _build_action_space(self) -> str:
        blocks: List[str] = []
        for name, tool in self.tools.items():
            block = f"### {name}\nDescription: {tool.description}"
            if getattr(tool, "parameters", None):
                block += f"\nParameters: {json.dumps(tool.parameters, ensure_ascii=False, indent=2)}"
            blocks.append(block)

        finish_block = (
            "### finish\n"
            "Description: Report progress back to MainAgent and stop this subtask.\n"
            "Parameters: {\"result\": \"<answer or finding>\", \"status\": \"done|partial|blocked\", \"summary\": \"<brief summary>\"}"
        )
        blocks.append(finish_block)

        return "Available actions:\n\n" + "\n\n".join(blocks)

    def get_basic_info(self) -> BasicInfo:
        return BasicInfo(
            env_id=self.task_id,
            instruction=self.instruction,
            action_space=self._build_action_space(),
            max_steps=self.max_steps,
            meta_data={
                "mode": "open-gaia-like",
                "question": self.question,
                "attachment_path": str(self.attachment_path) if self.attachment_path else None,
            },
        )

    async def reset(self, seed: int | None = None) -> Observation:
        self._steps = 0
        self._done = False
        return {
            "message": "Environment ready. Use tools and call finish when done.",
            "question": self.question,
            "current_step": 0,
            "max_steps": self.max_steps,
        }

    async def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Environment already finished. Call reset() first.")

        self._steps += 1
        action_name = action.get("action", "")
        params = action.get("params", {}) or {}

        if action_name == "finish":
            return self._handle_finish(params)

        tool = self.tools.get(action_name)
        if not tool:
            observation = {
                "error": f"Unknown action: {action_name}",
                "current_step": self._steps,
                "max_steps": self.max_steps,
            }
            return self._finalize_step(observation, 0.0, {"error": "unknown_action"})

        try:
            result = await tool(**params)
        except Exception as exc:
            result = {"success": False, "output": None, "error": str(exc), "metrics": {}}

        if not isinstance(result, dict):
            result = {
                "success": False,
                "output": None,
                "error": f"Tool returned non-dict: {type(result).__name__}",
                "metrics": {},
            }

        observation = {
            "action": action_name,
            "success": bool(result.get("success", False)),
            "output": result.get("output") if result.get("success") else None,
            "error": result.get("error") if not result.get("success") else None,
            "current_step": self._steps,
            "max_steps": self.max_steps,
        }
        step_info = {"last_action_result": result}
        return self._finalize_step(observation, 0.0, step_info)

    def _handle_finish(self, params: Dict[str, Any]) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        status = str(params.get("status", "done"))
        result = str(params.get("result", ""))
        summary = str(params.get("summary", ""))

        finish_result = {
            "result": result,
            "status": status,
            "summary": summary,
        }
        observation = {
            "message": "SubAgent reported finish to MainAgent.",
            "finish_result": finish_result,
            "current_step": self._steps,
            "max_steps": self.max_steps,
        }
        self._done = True
        return observation, 0.0, True, {
            "finished": True,
            "finish_result": finish_result,
            "last_action_result": finish_result,
        }

    def _finalize_step(
        self,
        observation: Dict[str, Any],
        reward: float,
        step_info: Dict[str, Any],
    ) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        if self._steps >= self.max_steps:
            self._done = True
            finish_result = {
                "result": "",
                "status": "timeout",
                "summary": f"Reached max steps ({self.max_steps}) without finish.",
            }
            observation["message"] = "Max steps reached"
            observation["finish_result"] = finish_result
            return observation, reward, True, {
                **step_info,
                "max_steps_reached": True,
                "finished": True,
                "finish_result": finish_result,
            }

        return observation, reward, False, step_info

    async def close(self):
        return None


class GaiaLikeSubAgentRunner(Runner):
    """SubAgent runner that passes current_step/max_steps into ReActAgent.step()."""

    async def run(self, agent, env: Environment) -> LevelResult:
        info = env.get_basic_info()
        agent.reset(info)

        reset_result = env.reset()
        obs = await reset_result if inspect.isawaitable(reset_result) else reset_result

        history: List[StepRecord] = []
        total_reward = 0.0
        max_steps = info.max_steps

        for t in range(max_steps):
            current_step = t + 1
            if self.step_timeout:
                step_result = await asyncio.wait_for(
                    agent.step(
                        observation=obs,
                        history=history,
                        current_step=current_step,
                        max_steps=max_steps,
                    ),
                    timeout=self.step_timeout,
                )
            else:
                step_result = await agent.step(
                    observation=obs,
                    history=history,
                    current_step=current_step,
                    max_steps=max_steps,
                )

            if not isinstance(step_result, (list, tuple)):
                raise TypeError(f"agent.step returned unsupported type: {type(step_result)}")
            if len(step_result) == 3:
                action, raw_response, raw_input = step_result
            elif len(step_result) == 2:
                action, raw_response = step_result
                raw_input = None
            else:
                raise ValueError(f"agent.step returned {len(step_result)} values, expected 2 or 3")

            obs_next, reward, done, step_info = await env.step(action)

            history.append(
                StepRecord(
                    observation=obs,
                    action=action,
                    reward=float(reward or 0.0),
                    raw_response=raw_response,
                    done=done,
                    info=step_info,
                    raw_input=raw_input,
                )
            )
            total_reward += float(reward or 0.0)
            obs = obs_next

            if done:
                break

        usage = agent.llm.get_usage_summary() if agent.llm else {}
        return LevelResult(
            model=usage.get("model", ""),
            total_reward=total_reward,
            steps=len(history),
            done=history[-1].done if history else False,
            trace=history,
            cost=float(usage.get("total_cost", 0.0) or 0.0),
            input_tokens=int(usage.get("total_input_tokens", 0) or 0),
            output_tokens=int(usage.get("total_output_tokens", 0) or 0),
        )


def _build_gaia_tools(enable_multimodal: bool) -> List[BaseAction]:
    tools: List[BaseAction] = [
        GoogleSearchAction(),
        ExecuteCodeAction(),
        ExtractUrlContentAction(),
    ]
    if enable_multimodal:
        from benchmark.gaia.tools.multimodal_toolkit import ImageAnalysisAction, ParseAudioAction

        tools.extend([ImageAnalysisAction(), ParseAudioAction()])
    return tools


def _resolve_models(main_model: Optional[str], sub_models_csv: Optional[str]) -> Tuple[str, List[str]]:
    cfg = LLMsConfig.default()
    all_models = cfg.get_all_names()
    if not all_models:
        raise ValueError("No models found in config. Please set config/global_config.yaml or env model config.")

    selected_main = main_model.strip() if main_model else all_models[0]
    cfg.get(selected_main)  # validate

    if sub_models_csv:
        selected_sub = [x.strip() for x in sub_models_csv.split(",") if x.strip()]
    else:
        selected_sub = [m for m in all_models if m != selected_main][:3]
        if not selected_sub:
            selected_sub = [selected_main]

    for model in selected_sub:
        cfg.get(model)  # validate

    return selected_main, selected_sub


def _fallback_answer(attempts_detail: List[Dict[str, Any]]) -> str:
    for attempt in reversed(attempts_detail):
        result = attempt.get("result", {})
        if not isinstance(result, dict):
            continue

        # Parallel delegate aggregation
        sub_results = result.get("results")
        if isinstance(sub_results, list):
            summaries: List[str] = []
            for item in sub_results:
                if not isinstance(item, dict):
                    continue
                finish = item.get("finish_result", {}) if isinstance(item.get("finish_result"), dict) else {}
                summary = str(finish.get("summary", "")).strip()
                status = str(finish.get("status", "")).strip()
                if summary:
                    summaries.append(f"[{status}] {summary}")
            if summaries:
                return "\n".join(summaries)

        finish = result.get("finish_result", {}) if isinstance(result.get("finish_result"), dict) else {}
        summary = str(finish.get("summary", "")).strip()
        if summary:
            return summary

    return "MainAgent did not output `complete` within max attempts."


def _save_outputs(
    output_dir: Path,
    question: str,
    final_answer: str,
    attempts_detail: List[Dict[str, Any]],
    main_model: str,
    sub_models: List[str],
    parallel_enabled: bool,
) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    md_path = output_dir / f"open_research_answer_{ts}.md"
    json_path = output_dir / f"open_research_trajectory_{ts}.json"

    md_text = "\n".join(
        [
            "# Open Research Answer",
            "",
            f"- Timestamp: {datetime.now().isoformat()}",
            f"- Main model: {main_model}",
            f"- Sub models: {', '.join(sub_models)}",
            f"- Parallel delegate: {parallel_enabled}",
            "",
            "## Question",
            question,
            "",
            "## Final Answer",
            final_answer,
        ]
    )
    md_path.write_text(md_text, encoding="utf-8")

    json_payload = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "final_answer": final_answer,
        "main_model": main_model,
        "sub_models": sub_models,
        "parallel_delegate": parallel_enabled,
        "attempts": attempts_detail,
    }
    json_path.write_text(json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return md_path, json_path


async def run_open_research(args: argparse.Namespace) -> int:
    main_model, sub_models = _resolve_models(args.main_model, args.sub_models)

    gaia_tools = _build_gaia_tools(enable_multimodal=args.enable_multimodal)
    env = OpenResearchGaiaEnv(
        question=args.question,
        tools=gaia_tools,
        max_steps=args.sub_max_steps,
        task_id="open_research_single_question",
        attachment_path=args.attachment_path,
    )

    sub_runner = GaiaLikeSubAgentRunner()
    delegate_tool = DelegateTaskTool(
        env=env,
        runner=sub_runner,
        models=sub_models,
        benchmark_type="gaia",
    )
    complete_tool = CompleteTool()

    main_llm = create_llm_instance(LLMsConfig.default().get(main_model))

    main_agent = ParallelMainAgent(
        llm=main_llm,
        sub_models=sub_models,
        tools=[delegate_tool, complete_tool],
        prompt_builder=GAIAMainAgentPrompt,
        max_attempts=args.max_attempts,
        benchmark_type="gaia",
        parallel_delegate=not args.serial,
        max_parallel_tasks=args.max_parallel_tasks,
        mask_model_names=False,
    )

    main_info = BasicInfo(
        env_id="open_research",
        instruction=env.instruction,
        action_space="",
        max_steps=args.max_attempts,
        meta_data={"mode": "open-gaia-like", "question": args.question},
    )
    main_agent.reset(main_info)

    attempts_detail: List[Dict[str, Any]] = []
    history: List[Dict[str, Any]] = []
    final_answer: Optional[str] = None

    logger.info(
        f"[OpenResearch] Start | main_model={main_model} | sub_models={sub_models} | "
        f"parallel={not args.serial}"
    )

    try:
        for idx in range(args.max_attempts):
            action, raw_response = await main_agent.step(observation=None, history=history)
            action_name = action.get("action")
            params = action.get("params", {})
            result = action.get("result", {})

            attempts_detail.append(
                {
                    "attempt": idx + 1,
                    "action": action_name,
                    "params": params,
                    "result": result,
                    "raw_response": raw_response,
                }
            )
            history.append({"attempt": idx + 1, "action": action_name, "result": result})

            if action_name == "complete":
                final_answer = str(params.get("answer", "")).strip() or str(result.get("answer", "")).strip()
                break

        if not final_answer:
            final_answer = _fallback_answer(attempts_detail)

        md_path, json_path = _save_outputs(
            output_dir=Path(args.output_dir),
            question=args.question,
            final_answer=final_answer,
            attempts_detail=attempts_detail,
            main_model=main_model,
            sub_models=sub_models,
            parallel_enabled=not args.serial,
        )

        print("\n=== Question ===")
        print(args.question)
        print("\n=== Final Answer ===")
        print(final_answer)
        print(f"\nAttempts: {len(attempts_detail)}")
        print(f"Saved answer: {md_path}")
        print(f"Saved trajectory: {json_path}")

        return 0
    finally:
        await env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open GAIA-like autonomous research entry (single question, no scoring)."
    )
    parser.add_argument("--question", type=str, default=DEFAULT_QUESTION, help="Research question")
    parser.add_argument("--main-model", type=str, default=None, help="MainAgent model name")
    parser.add_argument(
        "--sub-models",
        type=str,
        default=None,
        help="Comma-separated sub-models (default: auto-pick up to 3 from config)",
    )
    parser.add_argument("--max-attempts", type=int, default=6, help="MainAgent max attempts")
    parser.add_argument("--sub-max-steps", type=int, default=30, help="SubAgent max steps")
    parser.add_argument("--max-parallel-tasks", type=int, default=3, help="Max parallel delegate tasks")
    parser.add_argument("--serial", action="store_true", help="Disable parallel delegate mode")
    parser.add_argument(
        "--enable-multimodal",
        action="store_true",
        help="Enable ImageAnalysisAction and ParseAudioAction tools",
    )
    parser.add_argument(
        "--attachment-path",
        type=str,
        default=None,
        help="Optional local file path for the research task",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="workspace/open_research",
        help="Directory for answer + trajectory outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(run_open_research(args))


if __name__ == "__main__":
    main()
