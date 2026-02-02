"""Simple usage example for ParallelMainAgent."""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict

from pydantic import Field

from base.agent.base_action import BaseAction
from benchmark.common.env import BasicInfo
from aorchestra.parallel.parallel_main_agent import ParallelMainAgent


class MockLLM:
    """Returns two delegate_task actions to demonstrate parallel mode."""

    async def __call__(self, prompt: str) -> str:
        return json.dumps(
            {
                "actions": [
                    {
                        "action": "delegate_task",
                        "reasoning": "split task branch A",
                        "params": {
                            "task_instruction": "Collect branch A facts",
                            "context": "priority-high",
                            "model": "mock-sub-model",
                        },
                    },
                    {
                        "action": "delegate_task",
                        "reasoning": "split task branch B",
                        "params": {
                            "task_instruction": "Collect branch B facts",
                            "context": "priority-high",
                            "model": "mock-sub-model",
                        },
                    },
                ]
            }
        )

    def get_usage_summary(self) -> Dict[str, Any]:
        return {"total_cost": 0.0, "model": "mock-main-model"}


class MockDelegateTool(BaseAction):
    name: str = "delegate_task"
    description: str = "mock delegate"
    parameters: Dict[str, Any] = Field(default_factory=dict)

    async def __call__(self, task_instruction: str, model: str, context: str = "") -> Dict[str, Any]:
        await asyncio.sleep(0.05)
        return {
            "model": model,
            "steps_taken": 1,
            "done": True,
            "cost": 0.01,
            "finish_result": {
                "status": "done",
                "message": f"Finished: {task_instruction}",
                "completed": [task_instruction],
                "issues": [],
                "result": "-",
                "summary": "",
            },
            "trace_summary": context,
            "statistics": {"max_steps": 30},
        }


async def run_demo() -> None:
    agent = ParallelMainAgent(
        llm=MockLLM(),
        sub_models=["mock-sub-model"],
        tools=[MockDelegateTool()],
        parallel_delegate=True,
        max_parallel_tasks=2,
    )
    agent.reset(
        BasicInfo(
            env_id="demo",
            instruction="Demonstrate parallel delegate_task handling.",
            action_space="",
            max_steps=2,
            meta_data={},
        )
    )

    action, _ = await agent.step(observation=None, history=[])
    print(json.dumps(action, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(run_demo())
