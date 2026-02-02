from pydantic import Field
from typing import Dict, Any

from base.agent.base_agent import BaseAgent
from base.agent.memory import Memory
from base.engine.utils import parse_llm_action_response, parse_llm_output
from base.engine.logs import logger, LogLevel
from benchmark.common.env import BasicInfo, Observation, Action


REACT_PROMPT = """
==== Progress ====
[Step {current_step}/{max_steps}] Remaining: {remaining_steps} step(s)
{budget_warning}
If you run out of steps without "finish", your work is lost and marked as timeout.

==== Your Task (from MainAgent) ====
{task_instruction}

==== Context (from previous attempts) ====
{context}
Use this info: repeat what WORKED, avoid what FAILED.

==== Original Question (for reference) ====
{original_question}

==== Action Space ====
{action_space}

==== Memory ====
Recent memory:
{memory}

==== Current Observation ====
{obs}

==== Thinking ====
Think step by step before outputting an action. Write key reasoning in memory for future steps.

==== Action Guidelines ====
You have TWO actions available:

1. **execute** - Run shell commands and observe results
   - Use this to install packages, configure services, verify status, etc.
   - Example: "apt update && apt install -y nginx"

2. **finish** - Report your progress to MainAgent
   - Use when task is COMPLETE (status="done")
   - Use when you made PROGRESS but need more work (status="partial")
   - ⚠️ MUST use before running out of steps! Your work is LOST if you timeout.

**What to report in finish:**
- completed: List SUCCESSFUL steps that WORKED (e.g., ["apt update succeeded", "nginx installed"])
- issues: List FAILED attempts with WHY (e.g., ["nginx -v failed: command not found"])
- message: Brief summary of current state

This info helps the NEXT SubAgent know what to repeat and what to avoid.

==== Output Format ====
⚠️ CRITICAL: You MUST reply with ONLY a JSON object. No explanations, no markdown, no other text.

For execute:
{{"action": "execute", "params": {{"command": "your shell command"}}, "memory": "key findings"}}

For finish:
{{"action": "finish", "params": {{"status": "done|partial", "completed": [...], "issues": [...], "message": "..."}}, "memory": "final notes"}}


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

    async def step(self, observation: Observation, history: Any, current_step: int = 1, max_steps: int = 30) -> tuple[Action, str]:
        remaining_steps = max_steps - current_step
        
        # Dynamically generate budget warning
        if remaining_steps <= 3:
            budget_warning = f"🚨 CRITICAL: Only {remaining_steps} steps left! Use 'finish' NOW to report your progress!"
        elif remaining_steps <= 5:
            budget_warning = f"⚠️ Warning: {remaining_steps} steps remaining. Plan to finish soon."
        else:
            budget_warning = ""
        
        # Get SubAgent fields (if exist), otherwise use default values
        task_instruction = getattr(self, 'task_instruction', '') or self.current_env_instruction
        context = getattr(self, 'context', '') or "No additional context provided."
        original_question = getattr(self, 'original_question', '') or self.current_env_instruction
        
        act_prompt = REACT_PROMPT.format(
            task_instruction=task_instruction,
            context=context,
            original_question=original_question,
            action_space=self.current_action_space,
            obs=observation,
            memory=self._get_memory(),
            current_step=current_step,
            max_steps=max_steps,
            remaining_steps=remaining_steps,
            budget_warning=budget_warning,
        )
        logger.log_to_file(LogLevel.INFO, f"Agent Input:\n{act_prompt}\n")
        try:
            resp = await self.llm(act_prompt)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            resp = ""

        memory = parse_llm_output(resp, "memory")
        action = self.parse_action(resp)
        logger.agent_action(f"Agent Action: {action}")

        agent_obs = history[-1].info.get("last_action_result") if history else None

        await self.memory.add_memory(obs=agent_obs, action=action, thinking=memory, raw_response=resp)
        return action, resp, act_prompt

    async def run(self):
        pass