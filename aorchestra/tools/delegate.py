"""DelegateTaskTool - 统一的任务委派工具"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List

from pydantic import Field

from base.agent.base_action import BaseAction
from base.agent.memory import Memory
from base.engine.async_llm import LLMsConfig, create_llm_instance
from base.engine.logs import logger
from aorchestra.subagents import ReActAgent
from aorchestra.tools.trace_formatter import (
    create_gaia_formatter,
    create_terminalbench_formatter,
    create_swebench_formatter,
)


def _make_serializable(obj: Any) -> Any:
    """Recursively convert an object to JSON-serializable format."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _make_serializable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    # Fallback: convert to string
    return str(obj)


class DelegateTaskTool(BaseAction):
    """统一的任务委派工具，支持 GAIA、TerminalBench 和 SWE-bench"""
    
    name: str = "delegate_task"
    description: str = "Delegate task to SubAgent that executes commands"
    parameters: Dict[str, Any] = Field(default_factory=lambda: {
        "type": "object",
        "properties": {
            "task_instruction": {"type": "string", "description": "Task for SubAgent"},
            "context": {"type": "string", "description": "Additional context/hints"},
            "model": {"type": "string", "description": "Model to use"},
            "tools": {"type": "array", "items": {"type": "string"}, "description": "Tools for SubAgent (optional)"},
        },
        "required": ["task_instruction", "model"]
    })
    
    # 核心依赖
    env: Any = Field(default=None, exclude=True)
    runner: Any = Field(default=None, exclude=True)
    models: list = Field(default_factory=list)
    
    # 配置
    benchmark_type: str = Field(default="terminalbench")  # "gaia" | "terminalbench" | "swebench"
    alias_to_model: Dict[str, str] = Field(default_factory=dict)  # 模型名遮蔽（可选）
    
    # 内部状态
    _trace_formatter: Any = None
    
    class Config:
        arbitrary_types_allowed = True
        underscore_attrs_are_private = True
    
    def __init__(
        self,
        env,
        runner,
        models: list,
        benchmark_type: str = "terminalbench",
        alias_to_model: Dict[str, str] = None,
    ):
        super().__init__()
        self.env = env
        self.runner = runner
        self.models = models
        self.benchmark_type = benchmark_type
        self.alias_to_model = alias_to_model or {}
        
        # 创建对应的 trace formatter
        if benchmark_type == "gaia":
            self._trace_formatter = create_gaia_formatter()
        elif benchmark_type == "swebench":
            self._trace_formatter = create_swebench_formatter()
        else:
            self._trace_formatter = create_terminalbench_formatter()
        
        # 设置 model enum（使用别名或真实名）
        display_models = list(self.alias_to_model.keys()) if self.alias_to_model else models
        self.parameters = {
            "type": "object",
            "properties": {
                "task_instruction": {"type": "string", "description": "Task for SubAgent"},
                "context": {"type": "string", "description": "Additional context/hints"},
                "model": {
                    "type": "string", 
                    "description": f"Model to use. MUST be one of: {display_models}",
                    "enum": display_models
                },
                "tools": {"type": "array", "items": {"type": "string"}, "description": "Tools for SubAgent (optional)"},
            },
            "required": ["task_instruction", "model"]
        }
    
    async def __call__(
        self, 
        task_instruction: str, 
        model: str, 
        context: str = "", 
        tools: List[str] = None
    ) -> Dict:
        """执行委派任务
        
        Args:
            task_instruction: SubAgent 要执行的任务描述
            model: 使用的模型（可以是别名）
            context: 额外的上下文信息
            tools: 允许 SubAgent 使用的工具列表
            
        Returns:
            包含执行结果的字典
        """
        # 1. 解析模型名（如果有别名映射）
        real_model = self.alias_to_model.get(model, model)
        if real_model not in self.models:
            return {"error": f"Invalid model: {model}", "steps_taken": 0, "done": False}
        
        logger.info(f"[DelegateTool] Creating SubAgent with model={real_model}, tools={tools}")
        
        # 2. 获取原始问题
        original_question = getattr(self.env, 'instruction', '') or ''
        
        # 3. 创建 SubAgent（根据 benchmark_type 选择不同的 Agent）
        llm = create_llm_instance(LLMsConfig.default().get(real_model))
        
        if self.benchmark_type == "swebench":
            # SWE-bench 使用专门的 SWEBenchSubAgent（DISCUSSION + COMMAND 格式）
            from aorchestra.subagents import SWEBenchSubAgent
            sub_agent = SWEBenchSubAgent(
                llm=llm,
                task_instruction=task_instruction,
                context=context,
                original_question=original_question,
                memory=Memory(llm=llm, max_memory=20),
            )
        else:
            # GAIA 和 TerminalBench 使用 ReActAgent（JSON 格式）
            sub_agent = ReActAgent(
                llm=llm,
                benchmark_type=self.benchmark_type,
                task_instruction=task_instruction,
                context=context,
                original_question=original_question,
                allowed_tools=tools,
                memory=Memory(llm=llm, max_memory=10),
            )
        
        # 4. 临时替换 env.instruction
        original_instruction = getattr(self.env, 'instruction', None)
        if hasattr(self.env, 'instruction'):
            self.env.instruction = task_instruction
        
        try:
            # 5. 执行
            result = await self.runner.run(sub_agent, self.env)
            
            # 6. 提取 finish_result
            finish_result = None
            if result.trace:
                last = result.trace[-1]
                if last.info.get("finished") and last.info.get("finish_result"):
                    finish_result = last.info["finish_result"]
            
            # 7. 总结轨迹（固定使用 gemini-3-flash-preview）
            trace_summary = await self._summarize_trace(result.trace, task_instruction)
            
            # Convert StepRecord objects to dicts for JSON serialization
            trace_serializable = [_make_serializable(step) for step in result.trace] if result.trace else []
            
            return {
                "model": real_model,
                "tools_assigned": tools,
                "steps_taken": result.steps,
                "done": result.done,
                "cost": result.cost,
                "finish_result": finish_result,
                "trace": trace_serializable,
                "trace_summary": trace_summary,
                "statistics": {
                    "total_steps": result.steps, 
                    "max_steps": 30, 
                    "completed": result.done
                },
            }
            
        except Exception as e:
            logger.error(f"[DelegateTool] Error: {e}")
            return {"error": str(e), "steps_taken": 0, "done": False, "cost": 0.0}
            
        finally:
            # 8. 恢复 env.instruction
            if original_instruction is not None and hasattr(self.env, 'instruction'):
                self.env.instruction = original_instruction
    
    async def _summarize_trace(self, trace, task_instruction: str) -> str:
        """总结执行轨迹（固定使用 gemini-3-flash-preview）"""
        if not trace:
            return "No steps executed"
        
        trace_text = self._trace_formatter.format_trace(trace)
        
        # 根据 benchmark_type 选择不同的总结 prompt
        if self.benchmark_type == "gaia":
            prompt = f"""You are a trajectory summarizer. Review the SubAgent's execution trace.

Task: {task_instruction[:200]}
Steps: {len(trace)}

=== Trace ===
{trace_text}
===

Summarize in 5-10 bullets: key progress, problems, remaining issues.
Output ONLY bullets."""
        elif self.benchmark_type == "swebench":
            original_question = getattr(self.env, 'instruction', '') or task_instruction
            prompt = f"""You are a trajectory summarizer for a SWE-bench task (GitHub issue fixing).
Review the SubAgent's execution trace and compare against the original issue.

== ORIGINAL ISSUE ==
{original_question[:1000]}

== EXECUTION TRACE ==
{trace_text}

== OUTPUT ==
Based on the trace, answer:
1. ✅ CODE CHANGES: What code changes were made? Which files were modified?
2. ✅ TESTS: Were tests run? Did they pass?
3. ❌ REMAINING: What is still needed to fully fix the issue?

Summarize in 5-10 bullets: key progress, problems, remaining issues.
Be specific and concise. Output ONLY the sections above."""
        else:  # terminalbench
            original_question = getattr(self.env, 'instruction', '') or task_instruction
            prompt = f"""You are a trajectory summarizer. Review the SubAgent's execution trace.
Compare the execution trace against the original task requirements.

== ORIGINAL TASK ==
{original_question}

== EXECUTION TRACE ==
{trace_text}

== OUTPUT ==
Based on the trace, answer:
1. ✅ COMPLETED: What requirements from the original task were actually done?
2. ❌ REMAINING: What requirements are still missing or not properly tested?

Summarize in 5-10 bullets: key progress, problems, remaining issues.
Be specific and concise. Output ONLY the two sections above."""
        
        try:
            review_llm = create_llm_instance(
                LLMsConfig.default().get("gemini-3-flash-preview")
            )
            return (await review_llm(prompt)).strip()
        except Exception as e:
            logger.warning(f"[DelegateTool] Trace summarization failed: {e}")
            return f"Steps: {len(trace)}"
