"""
GAIA-specific ReAcT Agent with memory support.

This agent is optimized for GAIA benchmark tasks (question answering with tools).
Used for baseline (single-layer) mode without MainAgent orchestration.
"""
from pydantic import Field
from typing import Dict, Any

from base.agent.base_agent import BaseAgent
from base.agent.memory import Memory
from base.engine.utils import parse_llm_action_response, parse_llm_output
from base.engine.logs import logger, LogLevel
from benchmark.common.env import BasicInfo, Observation, Action


GAIA_REACT_PROMPT = """
==== Progress ====
[Step {current_step}/{max_steps}] Remaining {remaining_steps} steps

==== Question ====
{instruction}

==== Available Tools ====
{action_space}

==== Guidelines ====
1. Think step by step before outputting an action
2. Write key observations and findings to the "memory" field
3. Use print() in ExecuteCodeAction to see computation results
4. Use '{completion_action}' to report your result when done

==== Output Format ====
Output exactly ONE action in JSON format:
```json
{{
    "action": "<tool_name>",
    "params": {{
        "<param_name>": "<param_value>"
    }},
    "memory": "<write key observations, failures, findings, and next steps here>"
}}
```

==== Memory (Previous Steps) ====
{memory}

==== Current Observation ====
{obs}
"""


class GAIAReAcTAgent(BaseAgent):
    """
    GAIA-specific ReAcT Agent for baseline (single-layer) mode.
    Used when running GAIA benchmark without Orchestra (no MainAgent).
    """
    name: str = Field(default="GAIAReAcTAgent")
    description: str = Field(default="A ReAcT Agent optimized for GAIA benchmark tasks.")
    current_env_instruction: str = Field(default="")
    current_action_space: str = Field(default="")
    trajectory_folder_path: str = Field(default="")
    completion_action: str = Field(default="complete")
    memory: Memory = Field(default=None)

    def reset(self, env_info: "BasicInfo") -> None:
        """Reset agent state for a new task."""
        if self.memory is None:
            self.memory = Memory(llm=self.llm, max_memory=10)
        else:
            self.memory.clear()
        self.current_env_instruction = env_info.instruction
        self.current_action_space = env_info.action_space

    def parse_action(self, resp: str) -> Dict[str, Any]:
        """Parse LLM response to extract action data."""
        return parse_llm_action_response(resp)

    def _get_memory(self) -> str:
        """Get formatted memory text for prompt."""
        return self.memory.as_text()

    async def step(
        self, 
        observation: Observation, 
        history: Any, 
        current_step: int = 1, 
        max_steps: int = 30
    ) -> tuple[Action, str]:
        """Execute one step: build prompt, call LLM, parse action, update memory."""
        remaining_steps = max_steps - current_step
        
        act_prompt = GAIA_REACT_PROMPT.format(
            instruction=self.current_env_instruction,
            action_space=self.current_action_space,
            obs=observation,
            memory=self._get_memory(),
            current_step=current_step,
            max_steps=max_steps,
            remaining_steps=remaining_steps,
            completion_action=self.completion_action
        )
        
        logger.log_to_file(LogLevel.INFO, f"GAIA Agent Input:\n{act_prompt}\n")
        
        try:
            resp = await self.llm(act_prompt)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            resp = ""

        # Parse memory field from LLM output
        memory_content = parse_llm_output(resp, "memory")
        thinking = memory_content.get("memory") if isinstance(memory_content, dict) else None
        
        # Parse action
        action = self.parse_action(resp)
        logger.agent_action(f"GAIA Agent Action: {action}")

        # Get observation from previous step for memory
        agent_obs = history[-1].info.get("last_action_result") if history else None

        # Update memory with this step
        await self.memory.add_memory(
            obs=agent_obs, 
            action=action, 
            thinking=thinking, 
            raw_response=resp
        )
        
        return action, resp

    async def run(self, request: str = None) -> str:
        """Not used in orchestra pattern - execution is handled by Runner."""
        return ""
