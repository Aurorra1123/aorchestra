# aorchestra 框架设计文档

## 概述

`aorchestra` 是一个统一的多 Agent 协作框架，支持 **GAIA**、**TerminalBench** 和 **SWE-bench** 三种 benchmark。框架采用 **MainAgent-SubAgent** 的分层架构，实现任务的规划、委派和执行。

## 目录结构

```
aorchestra/
├── __init__.py              # 模块导出
├── config.py                # 配置类（GAIAOrchestraConfig, TerminalBenchOrchestraConfig, SWEBenchOrchestraConfig）
├── main_agent.py            # MainAgent 实现
├── sub_agent.py             # 向后兼容层
├── common/
│   └── utils.py             # 公共工具函数
├── prompts/
│   ├── gaia.py              # GAIA MainAgent prompt
│   ├── terminalbench.py     # TerminalBench MainAgent prompt
│   └── swebench.py          # SWE-bench MainAgent prompt
├── runners/
│   ├── gaia_runner.py       # GAIA benchmark runner
│   ├── terminalbench_runner.py  # TerminalBench runner
│   └── swebench_runner.py   # SWE-bench runner
├── subagents/               # SubAgent 实现
│   ├── react_agent.py       # ReAct 模式 Agent（GAIA/TerminalBench）
│   ├── swebench_agent.py    # SWE-bench SubAgent（DISCUSSION + COMMAND 格式）
└── tools/
    ├── delegate.py          # DelegateTaskTool
    ├── submit.py            # SubmitTool
    ├── complete.py          # CompleteTool
    └── trace_formatter.py   # Trace 格式化
```

## 核心抽象

### 1. 分层 Agent 架构

```
┌─────────────────────────────────────────────────────────┐
│                      MainAgent                          │
│  - 理解原始问题                                           │
│  - 制定执行计划                                           │
│  - 委派子任务给 SubAgent                                   │
│  - 汇总结果并提交最终答案                                   │
└─────────────────────────────────────────────────────────┘
                           │
                           │ DelegateTaskTool
                           ▼
┌─────────────────────────────────────────────────────────┐
│                      SubAgent                           │
│  ┌──────────────┐  ┌──────────────────┐               │
│  │  ReActAgent  │  │ SWEBenchSubAgent │               │
│  │ (GAIA/TB)    │  │   (SWE-bench)    │               │
│  └──────────────┘  └──────────────────┘               │
│  - 执行具体任务                                           │
│  - 与环境交互（执行代码、搜索等）                            │
│  - 返回执行结果给 MainAgent                               │
└─────────────────────────────────────────────────────────┘
```

### 2. SubAgent 统一接口

所有 SubAgent 继承自 `BaseAgent`，共享统一接口：

```python
class SubAgentInterface:
    # 核心字段
    benchmark_type: str       # "gaia" | "terminalbench"
    task_instruction: str     # MainAgent 分配的子任务
    context: str              # 上下文信息
    original_question: str    # 原始问题（参考）
    allowed_tools: List[str]  # 工具限制（可选）
    
    # 核心方法
    def reset(self, env_info: BasicInfo) -> None: ...
    async def step(self, observation, history, current_step, max_steps) -> tuple[Action, str, str]: ...
```

### 3. benchmark_type 区分机制

通过 `benchmark_type` 参数在运行时选择不同的行为：

| 特性 | GAIA | TerminalBench | SWE-bench |
|------|------|---------------|-----------|
| Prompt 模板 | GAIA_PROMPT | TERMINALBENCH_PROMPT | SWEBENCH_PROMPT |
| SubAgent | ReActAgent (JSON) | ReActAgent (JSON) | SWEBenchSubAgent (DISCUSSION+COMMAND) |
| 工具集 | 搜索、代码执行、文件操作 | execute, finish | ACI 命令、bash、finish |
| 完成动作 | `complete` | `submit` | `submit` |
| 执行环境 | 本地 | Docker 容器 | Docker 容器 (SWE-bench 镜像) |

## 关键代码

### ReActAgent (subagents/react_agent.py)

核心的 SubAgent 实现，采用 ReAct (Reasoning + Acting) 模式：

```python
class ReActAgent(BaseAgent):
    """ReAct 模式的 SubAgent，支持 GAIA 和 TerminalBench"""
    
    benchmark_type: str = Field(default="terminalbench")
    task_instruction: str = Field(default="")
    context: str = Field(default="")
    original_question: str = Field(default="")
    allowed_tools: List[str] | None = Field(default=None)
    
    def _build_prompt(self, observation, current_step, max_steps, ...) -> str:
        """根据 benchmark_type 选择 prompt 模板"""
        if self.benchmark_type == "gaia":
            return GAIA_PROMPT.format(...)
        else:
            return TERMINALBENCH_PROMPT.format(...)
    
    async def step(self, observation, history, current_step, max_steps):
        """执行一步推理和行动"""
        prompt = self._build_prompt(...)
        resp = await self.llm(prompt)
        action = self.parse_action(resp)
        return action, resp, prompt
```

**设计要点**：
- 使用 Pydantic 进行配置管理和类型验证
- Memory 组件记录执行历史，支持上下文学习
- 预算警告机制，防止 SubAgent 超时

### DelegateTaskTool (tools/delegate.py)

MainAgent 委派任务的核心工具：

```python
class DelegateTaskTool:
    """统一的任务委派工具"""
    
    benchmark_type: str           # "gaia" | "terminalbench"
    alias_to_model: Dict[str, str]  # 模型别名映射
    
    async def __call__(
        self,
        task_instruction: str,
        model: str = "default",
        context: str = "",
        original_question: str = "",
        tools: List[str] = None,
        max_steps: int = 30,
    ) -> str:
        # 1. 解析模型别名
        real_model = self.alias_to_model.get(model, model)
        
        # 2. 创建 ReActAgent
        sub_agent = ReActAgent(
            llm=create_llm_instance(...),
            benchmark_type=self.benchmark_type,
            task_instruction=task_instruction,
            context=context,
            ...
        )
        
        # 3. 执行 SubAgent 循环
        trace = await self._run_sub_agent(sub_agent, max_steps)
        
        # 4. 总结执行轨迹
        summary = await self._summarize_trace(trace)
        return summary
    
    async def _summarize_trace(self, trace: str) -> str:
        """使用 gemini-3-flash-preview 固定模型总结轨迹"""
        llm = create_llm_instance(LLMsConfig.default().get("gemini-3-flash-preview"))
        return await llm(SUMMARY_PROMPT.format(trace=trace))
```

**设计要点**：
- 模型别名机制，MainAgent 可以指定不同的 SubAgent 模型
- 轨迹总结使用固定模型（gemini-3-flash-preview），保证一致性
- 支持工具过滤，限制 SubAgent 可用的工具集

### TraceFormatter (tools/trace_formatter.py)

执行轨迹的格式化工具：

```python
class TraceFormatter:
    """格式化 Agent 执行轨迹"""
    
    def format_step(self, step: int, action: dict, observation: str) -> str:
        """格式化单步"""
        return f"[Step {step}]\nAction: {action}\nObservation: {observation}\n"
    
    def format_trace(self, steps: List[dict]) -> str:
        """格式化完整轨迹"""
        return "\n".join(self.format_step(i, s["action"], s["obs"]) for i, s in enumerate(steps))
```

## 扩展 SubAgent

添加新的 SubAgent 类型只需：

1. 在 `subagents/` 创建新文件
2. 继承 `BaseAgent` 并实现接口
3. 在 `subagents/__init__.py` 导出

示例框架：

```python
# subagents/my_agent.py
class MyAgent(BaseAgent):
    name: str = Field(default="MyAgent")
    benchmark_type: str = Field(default="terminalbench")
    task_instruction: str = Field(default="")
    # ... 其他字段
    
    def reset(self, env_info: BasicInfo) -> None:
        # 初始化逻辑
        pass
    
    async def step(self, observation, history, current_step, max_steps):
        # 推理和行动逻辑
        pass
```

## 配置

### GAIAOrchestraConfig

```python
@dataclass
class GAIAOrchestraConfig:
    main_model: str = "gpt-4o"      # MainAgent 模型
    sub_model: str = "gpt-4o-mini"  # SubAgent 默认模型
    max_rounds: int = 10            # MainAgent 最大轮数
    sub_max_steps: int = 30         # SubAgent 最大步数
```

### TerminalBenchOrchestraConfig

```python
@dataclass
class TerminalBenchOrchestraConfig:
    main_model: str = "gpt-4o"
    sub_model: str = "gpt-4o-mini"
    max_rounds: int = 5
    sub_max_steps: int = 20
    container_image: str = "terminalbench:latest"
```

### SWEBenchOrchestraConfig

```python
@dataclass
class SWEBenchOrchestraConfig:
    main_model: str                    # MainAgent 模型
    sub_models: List[str]              # SubAgent 可用模型列表
    dataset_name: str = "princeton-nlp/SWE-bench_Verified"
    split: str = "test"
    max_steps: int = 50                # SubAgent 最大步数
    max_attempts: int = 10             # MainAgent 最大尝试次数
    docker_timeout: int = 1800         # 容器超时（秒）
    window_size: int = 100             # ACI 文件查看窗口大小
```

## 向后兼容

为保持兼容性，`OrchestraSubAgent` 作为 `ReActAgent` 的别名继续可用：

```python
# aorchestra/sub_agent.py
from aorchestra.subagents.react_agent import ReActAgent
OrchestraSubAgent = ReActAgent  # 向后兼容别名
```

## 使用示例

### GAIA Benchmark

```python
from aorchestra import ReActAgent, GAIAOrchestraConfig
from aorchestra.runners import GAIARunner

# 配置
config = GAIAOrchestraConfig.load("config/gaia.yaml")

# 创建 Runner
runner = GAIARunner(config)

# 运行 benchmark
results = await runner.run(dataset)
```

### SWE-bench Benchmark

```python
from aorchestra import SWEBenchOrchestraConfig
from aorchestra.runners import SWEBenchOrchestra

# 配置
config = SWEBenchOrchestraConfig.load("config/benchmarks/aorchestra_swebench.yaml")

# 创建 benchmark
benchmark = SWEBenchOrchestra(config)

# 获取任务列表
levels = benchmark.list_levels()

# 运行 benchmark
results = await benchmark.run(levels, max_concurrency=1)
```

### 命令行使用

```bash
# 运行 GAIA
python bench_aorchestra_gaia.py --config config/benchmarks/aorchestra_gaia.yaml

# 运行 SWE-bench
python bench_aorchestra_swebench.py --config config/benchmarks/aorchestra_swebench.yaml

# 运行特定实例
python bench_aorchestra_swebench.py --tasks "django__django-11848,sympy__sympy-12171"
```
