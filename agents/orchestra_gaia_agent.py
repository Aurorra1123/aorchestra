"""Orchestra GAIA Agent - SubAgent for Orchestra mode with MainAgent delegation."""
from pydantic import Field
from typing import Dict, Any, List

from base.agent.base_agent import BaseAgent
from base.agent.memory import Memory
from base.engine.utils import parse_llm_action_response, parse_llm_output
from base.engine.logs import logger, LogLevel
from benchmark.common.env import BasicInfo, Observation, Action


ORCHESTRA_GAIA_PROMPT = """
==== Progress ====
[Step {current_step}/{max_steps}] Remaining {remaining_steps} steps
{budget_warning}

==== Your Task (from MainAgent) ====
{task_instruction}

==== Context ====
{context}

==== Original Question (for reference) ====
{original_question}

==== Available Tools ====
{action_space}

==== Guidelines ====
1. Focus on completing YOUR TASK above
2. Think step by step before outputting an action
3. Write key observations to the "memory" field
4. Use print() in ExecuteCodeAction to see computation results
5. Once done, use 'finish' IMMEDIATELY

⚠️ BUDGET: When remaining_steps <= 5, use 'finish' NOW!

==== Output Format ====
```json
{{
    "action": "<tool_name>",
    "params": {{}},
    "memory": "<observations>"
}}
```

==== Memory ====
{memory}

==== Current Observation ====
{obs}
"""


class OrchestraGAIAAgent(BaseAgent):
    """GAIA Agent for Orchestra mode."""
    name: str = Field(default="OrchestraGAIAAgent")
    description: str = Field(default="A GAIA Agent for Orchestra mode.")
    current_env_instruction: str = Field(default="")
    current_action_space: str = Field(default="")
    trajectory_folder_path: str = Field(default="")
    task_instruction: str = Field(default="")
    context: str = Field(default="")
    completion_action: str = Field(default="finish")
    memory: Memory = Field(default=None)
    allowed_tools: List[str] | None = Field(default=None)  # None = 全部可用

    def reset(self, env_info: "BasicInfo") -> None:
        if self.memory is None:
            self.memory = Memory(llm=self.llm, max_memory=10)
        else:
            self.memory.clear()
        self.current_env_instruction = env_info.instruction
        
        # 如果指定了 allowed_tools，过滤 action_space
        if self.allowed_tools:
            self.current_action_space = self._filter_action_space(env_info.action_space, self.allowed_tools)
            logger.info(f"[OrchestraGAIAAgent] Filtered to tools: {self.allowed_tools}")
        else:
            self.current_action_space = env_info.action_space

    def _filter_action_space(self, action_space: str, allowed_tools: List[str]) -> str:
        """简单过滤：在 action_space 末尾加上工具限制提示"""
        return f"{action_space}\n\n[TOOL RESTRICTION] You can ONLY use: {allowed_tools}"

    def parse_action(self, resp: str) -> Dict[str, Any]:
        return parse_llm_action_response(resp)

    def _get_memory(self) -> str:
        return self.memory.as_text()

    async def step(self, observation: Observation, history: Any, current_step: int = 1, max_steps: int = 30) -> tuple[Action, str]:
        remaining_steps = max_steps - current_step
        
        if remaining_steps <= 5:
            budget_warning = f"🚨 CRITICAL: Only {remaining_steps} steps left! Use 'finish' NOW!"
        elif remaining_steps <= 10:
            budget_warning = f"⚠️ Warning: {remaining_steps} steps remaining."
        else:
            budget_warning = ""
        
        act_prompt = ORCHESTRA_GAIA_PROMPT.format(
            task_instruction=self.task_instruction,
            context=self.context or "None",
            original_question=self.current_env_instruction,
            action_space=self.current_action_space,
            obs=observation,
            memory=self._get_memory(),
            current_step=current_step,
            max_steps=max_steps,
            remaining_steps=remaining_steps,
            budget_warning=budget_warning,
            completion_action=self.completion_action
        )
        
        logger.log_to_file(LogLevel.INFO, f"Orchestra GAIA Agent Input:\n{act_prompt}\n")
        
        try:
            resp = await self.llm(act_prompt)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            resp = ""

        memory_content = parse_llm_output(resp, "memory")
        thinking = memory_content.get("memory") if isinstance(memory_content, dict) else None
        action = self.parse_action(resp)
        logger.agent_action(f"Orchestra GAIA Agent Action: {action}")

        agent_obs = history[-1].info.get("last_action_result") if history else None
        await self.memory.add_memory(obs=agent_obs, action=action, thinking=thinking, raw_response=resp)
        
        return action, resp

    async def run(self, request: str = None) -> str:
        return ""
