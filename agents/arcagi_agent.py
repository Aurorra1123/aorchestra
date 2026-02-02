"""
ARC-AGI Agent for FoundationAgent.

Actions:
- test: Validate code on all training examples (reward=0)
- submit: Submit final code for test evaluation (reward=0/1)
"""
from typing import Any, Dict

from pydantic import Field

from base.agent.base_agent import BaseAgent
from base.agent.memory import Memory
from base.engine.logs import logger, LogLevel
from base.engine.utils import parse_llm_action_response
from benchmark.common.env import BasicInfo, Observation, Action


ARCAGI_PROMPT = """
==== Instruction ====
{instruction}

==== Action Space ====
{action_space}

==== Memory ====
Recent memory:
{memory}

==== Current Observation ====
{obs}

==== Thinking ====
You should think step by step before you output an action.
And you should write any necessary reasoning (such as environment rules or information relevant for future actions) inside memory.

==== Output Format ====
Reply with exactly:
1) A single JSON object with only "action" and "memory" keys.
2) Immediately after, one ```python``` code block containing the runnable solution.

Example:
```json
{{
    "action": "validate",
    "memory": "why this code should work"
}}
```
```python
# your python solution here
```

Action names allowed: "validate" or "submit". No other text is allowed outside the JSON + python block.
"""



class ArcAGIAgent(BaseAgent):
    """Agent for solving ARC-AGI tasks."""
    
    name: str = Field(default="ArcAGIAgent")
    description: str = Field(default="Agent for solving ARC-AGI tasks")
    memory: Memory = Field(default=None)
    current_instruction: str = Field(default="")
    current_action_space: str = Field(default="")

    def reset(self, env_info: BasicInfo) -> None:
        self.memory = Memory(llm=self.llm, max_memory=10)
        self.current_instruction = env_info.instruction
        self.current_action_space = env_info.action_space
        self.memory.clear()

    def _get_memory(self) -> str:
        return self.memory.as_text() if self.memory else "None"

    def parse_action(self, resp: str):
        """Parse LLM response to extract action data."""
        return parse_llm_action_response(resp)

    async def step(self, observation: Observation, history: Any) -> tuple[Action, str]:
        prompt = ARCAGI_PROMPT.format(
            instruction=self.current_instruction,
            action_space=self.current_action_space,
            memory=self._get_memory(),
            obs=observation,
        )
        
        logger.log_to_file(LogLevel.INFO, f"Agent Input:\n{prompt}\n")
        
        try:
            response = await self.llm(prompt)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            response = ""

        logger.log_to_file(LogLevel.INFO, f"Agent Output:\n{response}\n")
        
        action = self.parse_action(response)
        if isinstance(action, dict) and response:
            # Pass raw response for downstream fallbacks in the environment.
            action["_raw_response"] = response
        action_type = action.get("action", "unknown")
        
        logger.agent_action(f"Agent Action: {action_type}")
        
        # Store in memory
        memory_text = action.get("memory", "")
        
        await self.memory.add_memory(
            obs=str(observation),  # Full observation (no truncation for learning)
            action=action, 
            thinking=memory_text, 
            raw_response=response
        )
        
        return action, response, prompt

    async def run(self):
        pass
