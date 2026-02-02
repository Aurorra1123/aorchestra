from pydantic import Field
from typing import Dict, Any

from base.agent.base_agent import BaseAgent
from base.agent.memory import Memory
from base.engine.utils import parse_llm_action_response, parse_llm_output
from base.engine.logs import logger, LogLevel
from benchmark.common.env import BasicInfo, Observation, Action


REACT_PROMPT = """
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
2) Immediately after, one ```bash``` code block containing the runnable shell commands.

Example:
```json
{{
    "action": "",
    "memory": ""
}}
```
```bash
# your bash commands here
```
"""

class ReAcTAgent(BaseAgent):
    """
    A Basic ReAcT Agent.
    """
    name: str = Field(default="ReAcTAgent")
    description: str = Field(default="A Basic ReAcT Agent.")
    current_env_instruction: str = Field(default="")
    current_action_space: str = Field(default="")
    trajectory_folder_path: str = Field(default="")
    memory: Memory = Field(default_factory=None)

    def reset(self, env_info: "BasicInfo") -> None:
        self.memory = Memory(llm=self.llm, max_memory=10)
        self.current_env_instruction = env_info.instruction
        self.current_action_space = env_info.action_space
        self.memory.clear()

    def parse_action(self, resp: str):
        """Parse LLM response to extract action data."""
        return parse_llm_action_response(resp)

    def _get_memory(self) -> str:
        return self.memory.as_text()

    def _extract_bash_command(self, resp: str) -> str | None:
        """Pull the first ```bash ... ``` block to use as the execute command."""
        if not resp:
            return None
        marker = "```bash"
        start = resp.find(marker)
        if start == -1:
            return None
        start += len(marker)
        end = resp.find("```", start)
        block = resp[start:end if end != -1 else None]
        return block.strip() if block else None

    def _get_max_steps(self, env, env_info: Dict) -> int:
        explicit = env_info.get("max_step")
        if explicit is not None:
            try:
                return int(explicit)
            except Exception:
                pass
        configs = getattr(env, "configs", {}) or {}
        term_steps = configs.get("termination", {}).get("max_steps")
        try:
            if term_steps is not None:
                return int(term_steps)
        except Exception:
            pass
        return 20

    async def step(self, observation: Observation, history: Any) -> tuple[Action, str, str]:
        act_prompt = REACT_PROMPT.format(
            instruction = self.current_env_instruction,
            action_space = self.current_action_space,
            obs = observation,
            memory = self._get_memory()
        )
        logger.log_to_file(LogLevel.INFO, f"Agent Input:\n{act_prompt}\n")
        try:
            resp = await self.llm(act_prompt)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            resp = ""

        memory = parse_llm_output(resp, "memory")
        action = self.parse_action(resp)

        # If action is execute but no command parsed from JSON, backfill from ```bash``` block.
        if isinstance(action, dict) and action.get("action") == "execute":
            params = action.setdefault("params", {})
            if not params.get("command"):
                bash_cmd = self._extract_bash_command(resp)
                if bash_cmd:
                    params["command"] = bash_cmd
                else:
                    logger.warning("Execute action missing command and no ```bash``` block found.")

        logger.agent_action(f"Agent Action: {action}")

        agent_obs = history[-1].info.get("last_action_result") if history else None

        await self.memory.add_memory(obs=agent_obs, action=action, thinking=memory, raw_response=resp)
        return action, resp, act_prompt

    async def run(self):
        pass