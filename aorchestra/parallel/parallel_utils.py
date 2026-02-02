"""Utilities for parsing and executing parallel delegate actions."""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from base.engine.logs import logger
from aorchestra.common.utils import parse_json_response


def _normalize_action(action: Dict[str, Any], index: int) -> Dict[str, Any]:
    if not isinstance(action, dict):
        raise ValueError(f"actions[{index}] must be an object")

    action_name = action.get("action")
    if not isinstance(action_name, str) or not action_name.strip():
        raise ValueError(f"actions[{index}].action is required")

    params = action.get("params", {})
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise ValueError(f"actions[{index}].params must be an object")

    normalized = dict(action)
    normalized["action"] = action_name.strip()
    normalized["params"] = params
    return normalized


def parse_multi_action_response(resp: str) -> List[Dict[str, Any]]:
    """Parse both single-action and multi-action JSON responses."""
    decision = parse_json_response(resp)
    if not isinstance(decision, dict):
        raise ValueError("response JSON must be an object")

    if "actions" in decision:
        actions = decision.get("actions")
        if not isinstance(actions, list) or not actions:
            raise ValueError("`actions` must be a non-empty array")
        return [_normalize_action(action, idx) for idx, action in enumerate(actions)]

    return [_normalize_action(decision, 0)]


def build_delegate_error_result(error_message: str, model: str = "unknown") -> Dict[str, Any]:
    """Create a delegate-like error payload that matches existing result schema."""
    message = str(error_message or "Unknown delegate error")
    return {
        "model": model,
        "steps_taken": 0,
        "done": False,
        "cost": 0.0,
        "error": message,
        "finish_result": {
            "status": "error",
            "message": message,
            "completed": [],
            "issues": [message],
            "result": "-",
            "summary": "",
        },
        "statistics": {
            "total_steps": 0,
            "max_steps": 30,
            "completed": False,
        },
    }


async def execute_parallel_delegates(
    tools: List[Any],
    actions: List[Dict[str, Any]],
    max_concurrent: int = 5,
) -> List[Dict[str, Any]]:
    """Execute delegate actions concurrently with bounded concurrency."""
    if not actions:
        return []

    tool_by_name = {tool.name: tool for tool in tools if getattr(tool, "name", None)}
    semaphore = asyncio.Semaphore(max(1, int(max_concurrent)))

    async def _run_one(action: Dict[str, Any]) -> Dict[str, Any]:
        action_name = action.get("action")
        params = action.get("params", {})
        tool = tool_by_name.get(action_name)
        if not tool:
            return build_delegate_error_result(f"Unknown action: {action_name}")

        try:
            async with semaphore:
                result = await tool(**params)
        except Exception as exc:
            logger.error(f"[ParallelDelegate] {action_name} failed: {exc}")
            return build_delegate_error_result(str(exc), model=params.get("model", "unknown"))

        if isinstance(result, dict):
            return result
        return {"raw_result": result, "done": False}

    tasks = [asyncio.create_task(_run_one(action)) for action in actions]
    return await asyncio.gather(*tasks)


def summarize_parallel_delegate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate parallel delegate outputs into one result dict."""
    num_tasks = len(results)
    total_cost = 0.0
    total_steps = 0
    success_count = 0
    error_count = 0
    completed_items: List[Any] = []
    issues: List[str] = []
    models: List[str] = []

    for result in results:
        if not isinstance(result, dict):
            error_count += 1
            issues.append(f"Invalid result type: {type(result).__name__}")
            continue

        models.append(str(result.get("model", "unknown")))

        try:
            total_cost += float(result.get("cost", 0.0) or 0.0)
        except Exception:
            pass
        try:
            total_steps += int(result.get("steps_taken", 0) or 0)
        except Exception:
            pass

        finish = result.get("finish_result") if isinstance(result.get("finish_result"), dict) else {}
        if result.get("error"):
            error_count += 1
            issues.append(str(result["error"]))
        elif finish.get("status") == "done" or result.get("done"):
            success_count += 1
        else:
            issues.extend([str(x) for x in finish.get("issues", []) if x])

        completed_items.extend(finish.get("completed", []) or [])

    summary_status = (
        "done"
        if error_count == 0 and num_tasks > 0 and success_count == num_tasks
        else "partial"
    )
    if error_count > 0 and success_count == 0:
        summary_status = "error"

    summary_message = (
        f"Executed {num_tasks} delegate task(s) in parallel: "
        f"{success_count} success, {error_count} error."
    )

    return {
        "parallel": True,
        "num_tasks": num_tasks,
        "success_count": success_count,
        "error_count": error_count,
        "cost": total_cost,
        "steps_taken": total_steps,
        "done": summary_status == "done",
        "models": models,
        "results": results,
        "finish_result": {
            "status": summary_status,
            "message": summary_message,
            "completed": completed_items,
            "issues": issues,
            "result": "-",
            "summary": summary_message,
        },
    }
