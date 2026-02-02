"""ReActAgent - ReAct 模式的 SubAgent 实现"""
from __future__ import annotations

import re
from typing import Any, Dict, List

from pydantic import Field

from base.agent.base_agent import BaseAgent
from base.agent.memory import Memory
from base.engine.utils import parse_llm_action_response, parse_llm_output
from base.engine.logs import logger, LogLevel
from benchmark.common.env import BasicInfo, Observation, Action


# GAIA SubAgent Prompt 模板
GAIA_PROMPT = """You are a specialized SubAgent. Complete the assigned task efficiently.

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


# TerminalBench SubAgent Prompt 模板
TERMINALBENCH_PROMPT = """
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


class ReActAgent(BaseAgent):
    """ReAct 模式的 SubAgent，支持 GAIA 和 TerminalBench"""
    
    name: str = Field(default="ReActAgent")
    description: str = Field(default="ReAct-style SubAgent for Orchestra framework")
    
    # 核心字段
    benchmark_type: str = Field(default="terminalbench")  # "gaia" | "terminalbench"
    task_instruction: str = Field(default="")             # MainAgent 分配的子任务
    context: str = Field(default="")                      # 上下文/hints
    original_question: str = Field(default="")            # 原始完整问题
    allowed_tools: List[str] | None = Field(default=None) # 工具限制
    
    # 内部状态
    current_env_instruction: str = Field(default="")
    current_action_space: str = Field(default="")
    memory: Memory = Field(default=None)
    
    class Config:
        arbitrary_types_allowed = True
    
    def reset(self, env_info: BasicInfo) -> None:
        """初始化 Agent"""
        if self.memory is None:
            self.memory = Memory(llm=self.llm, max_memory=10)
        else:
            self.memory.clear()
        
        # 保存原始问题
        if not self.original_question:
            self.original_question = env_info.instruction
        
        self.current_env_instruction = env_info.instruction
        
        # 工具过滤（如果指定了 allowed_tools）
        if self.allowed_tools:
            self.current_action_space = self._filter_action_space(
                env_info.action_space, 
                self.allowed_tools
            )
            logger.info(f"[ReActAgent] Filtered to tools: {self.allowed_tools}")
        else:
            self.current_action_space = env_info.action_space
    
    def _normalize_tool_name(self, name: str) -> str:
        """标准化工具名称，用于模糊匹配"""
        normalized = name.lower().replace("_", "")
        if normalized.endswith("action"):
            normalized = normalized[:-6]
        return normalized
    
    def _tool_matches(self, tool_name: str, allowed_tools: List[str]) -> bool:
        """检查工具名是否匹配（支持模糊匹配）"""
        if tool_name in allowed_tools:
            return True
        
        normalized_tool = self._normalize_tool_name(tool_name)
        for allowed in allowed_tools:
            if self._normalize_tool_name(allowed) == normalized_tool:
                return True
        
        return False
    
    def _filter_action_space(self, action_space: str, allowed_tools: List[str]) -> str:
        """过滤 action_space，只保留允许的工具描述"""
        blocks = re.split(r'\n(?=### )', action_space)
        
        filtered_blocks = []
        for block in blocks:
            if block.startswith("Available actions"):
                filtered_blocks.append(block.rstrip())
                continue
            
            match = re.match(r'### (\w+)', block)
            if match:
                tool_name = match.group(1)
                if self._tool_matches(tool_name, allowed_tools):
                    filtered_blocks.append(block.rstrip())
        
        return "\n\n".join(filtered_blocks)
    
    def parse_action(self, resp: str) -> Dict[str, Any]:
        """解析 LLM 响应为 action"""
        return parse_llm_action_response(resp)
    
    def _get_memory(self) -> str:
        """获取 memory 文本"""
        return self.memory.as_text()
    
    def _get_budget_warning(self, remaining_steps: int) -> str:
        """生成预算警告"""
        if remaining_steps <= 3:
            return f"🚨 CRITICAL: Only {remaining_steps} steps left! Use 'finish' NOW!"
        elif remaining_steps <= 5:
            return f"⚠️ Warning: {remaining_steps} steps remaining. Plan to finish soon."
        return ""
    
    def _build_prompt(
        self,
        observation: Any,
        current_step: int,
        max_steps: int,
        remaining_steps: int,
        budget_warning: str,
    ) -> str:
        """根据 benchmark_type 构建 prompt"""
        if self.benchmark_type == "gaia":
            return GAIA_PROMPT.format(
                task_instruction=self.task_instruction,
                context=self.context or "None",
                original_question=self.original_question,
                action_space=self.current_action_space,
                memory=self._get_memory(),
                obs=observation,
                current_step=current_step,
                max_steps=max_steps,
                remaining_steps=remaining_steps,
                budget_warning=budget_warning,
            )
        else:  # terminalbench
            return TERMINALBENCH_PROMPT.format(
                task_instruction=self.task_instruction,
                context=self.context or "No additional context provided.",
                original_question=self.original_question,
                action_space=self.current_action_space,
                memory=self._get_memory(),
                obs=observation,
                current_step=current_step,
                max_steps=max_steps,
                remaining_steps=remaining_steps,
                budget_warning=budget_warning,
            )
    
    async def step(
        self, 
        observation: Observation, 
        history: Any, 
        current_step: int = 1, 
        max_steps: int = 30
    ) -> tuple[Action, str, str]:
        """执行一步
        
        Returns:
            tuple: (action, raw_response, raw_input_prompt)
        """
        remaining_steps = max_steps - current_step
        budget_warning = self._get_budget_warning(remaining_steps)
        
        # 构建 prompt
        prompt = self._build_prompt(
            observation=observation,
            current_step=current_step,
            max_steps=max_steps,
            remaining_steps=remaining_steps,
            budget_warning=budget_warning,
        )
        
        logger.log_to_file(LogLevel.INFO, f"ReActAgent Input:\n{prompt}\n")
        
        try:
            resp = await self.llm(prompt)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            resp = ""
        
        # 解析响应
        memory_content = parse_llm_output(resp, "memory")
        thinking = memory_content.get("memory") if isinstance(memory_content, dict) else None
        action = self.parse_action(resp)
        
        logger.agent_action(f"ReActAgent Action: {action}")
        
        # 更新 memory
        agent_obs = history[-1].info.get("last_action_result") if history else None
        await self.memory.add_memory(obs=agent_obs, action=action, thinking=thinking, raw_response=resp)
        
        return action, resp, prompt
    
    async def run(self, request: str = None) -> str:
        """Standalone run - not used in Orchestra mode"""
        return ""
