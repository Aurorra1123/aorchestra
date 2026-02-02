# aorchestra Framework Design Document

## Overview

`aorchestra` is a unified multi-agent collaboration framework supporting **GAIA**, **TerminalBench**, and **SWE-bench** benchmarks. The framework adopts a **MainAgent-SubAgent** hierarchical architecture for task planning, delegation, and execution.

## Directory Structure

```
aorchestra/
├── __init__.py              # Module exports
├── config.py                # Config classes (GAIAOrchestraConfig, TerminalBenchOrchestraConfig, SWEBenchOrchestraConfig)
├── main_agent.py            # MainAgent implementation
├── sub_agent.py             # Backward compatibility layer
├── common/
│   └── utils.py             # Common utility functions
├── prompts/
│   ├── gaia.py              # GAIA MainAgent prompt
│   ├── terminalbench.py     # TerminalBench MainAgent prompt
│   └── swebench.py          # SWE-bench MainAgent prompt
├── runners/
│   ├── gaia_runner.py       # GAIA benchmark runner
│   ├── terminalbench_runner.py  # TerminalBench runner
│   └── swebench_runner.py   # SWE-bench runner
├── subagents/               # SubAgent implementations
│   ├── react_agent.py       # ReAct-style Agent (GAIA/TerminalBench)
│   ├── swebench_agent.py    # SWE-bench SubAgent (DISCUSSION + COMMAND format)
└── tools/
    ├── delegate.py          # DelegateTaskTool
    ├── submit.py            # SubmitTool
    ├── complete.py          # CompleteTool
    └── trace_formatter.py   # Trace formatting
```

## Core Abstractions

### 1. Hierarchical Agent Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      MainAgent                          │
│  - Understand original problem                          │
│  - Formulate execution plan                             │
│  - Delegate subtasks to SubAgent                        │
│  - Aggregate results and submit final answer            │
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
│  - Execute specific tasks                               │
│  - Interact with environment (execute code, search)     │
│  - Return execution results to MainAgent                │
└─────────────────────────────────────────────────────────┘
```

### 2. SubAgent Unified Interface

All SubAgents inherit from `BaseAgent` and share a unified interface:

```python
class SubAgentInterface:
    # Core fields
    benchmark_type: str       # "gaia" | "terminalbench"
    task_instruction: str     # Subtask assigned by MainAgent
    context: str              # Context information
    original_question: str    # Original question (for reference)
    allowed_tools: List[str]  # Tool restrictions (optional)
    
    # Core methods
    def reset(self, env_info: BasicInfo) -> None: ...
    async def step(self, observation, history, current_step, max_steps) -> tuple[Action, str, str]: ...
```

### 3. benchmark_type Differentiation

Different behaviors are selected at runtime via the `benchmark_type` parameter:

| Feature | GAIA | TerminalBench | SWE-bench |
|---------|------|---------------|-----------|
| Prompt Template | GAIA_PROMPT | TERMINALBENCH_PROMPT | SWEBENCH_PROMPT |
| SubAgent | ReActAgent (JSON) | ReActAgent (JSON) | SWEBenchSubAgent (DISCUSSION+COMMAND) |
| Toolset | Search, code execution, file ops | execute, finish | ACI commands, bash, finish |
| Completion Action | `complete` | `submit` | `submit` |
| Execution Environment | Local | Docker container | Docker container (SWE-bench image) |

## Key Code

### ReActAgent (subagents/react_agent.py)

Core SubAgent implementation using ReAct (Reasoning + Acting) pattern:

```python
class ReActAgent(BaseAgent):
    """ReAct-style SubAgent supporting GAIA and TerminalBench"""
    
    benchmark_type: str = Field(default="terminalbench")
    task_instruction: str = Field(default="")
    context: str = Field(default="")
    original_question: str = Field(default="")
    allowed_tools: List[str] | None = Field(default=None)
    
    def _build_prompt(self, observation, current_step, max_steps, ...) -> str:
        """Select prompt template based on benchmark_type"""
        if self.benchmark_type == "gaia":
            return GAIA_PROMPT.format(...)
        else:
            return TERMINALBENCH_PROMPT.format(...)
    
    async def step(self, observation, history, current_step, max_steps):
        """Execute one reasoning and action step"""
        prompt = self._build_prompt(...)
        resp = await self.llm(prompt)
        action = self.parse_action(resp)
        return action, resp, prompt
```

**Design Points**:
- Uses Pydantic for configuration management and type validation
- Memory component records execution history for context learning
- Budget warning mechanism to prevent SubAgent timeout

### DelegateTaskTool (tools/delegate.py)

Core tool for MainAgent task delegation:

```python
class DelegateTaskTool:
    """Unified task delegation tool"""
    
    benchmark_type: str           # "gaia" | "terminalbench"
    alias_to_model: Dict[str, str]  # Model alias mapping
    
    async def __call__(
        self,
        task_instruction: str,
        model: str = "default",
        context: str = "",
        original_question: str = "",
        tools: List[str] = None,
        max_steps: int = 30,
    ) -> str:
        # 1. Parse model alias
        real_model = self.alias_to_model.get(model, model)
        
        # 2. Create ReActAgent
        sub_agent = ReActAgent(
            llm=create_llm_instance(...),
            benchmark_type=self.benchmark_type,
            task_instruction=task_instruction,
            context=context,
            ...
        )
        
        # 3. Execute SubAgent loop
        trace = await self._run_sub_agent(sub_agent, max_steps)
        
        # 4. Summarize execution trace
        summary = await self._summarize_trace(trace)
        return summary
    
    async def _summarize_trace(self, trace: str) -> str:
        """Summarize trace using fixed gemini-3-flash-preview model"""
        llm = create_llm_instance(LLMsConfig.default().get("gemini-3-flash-preview"))
        return await llm(SUMMARY_PROMPT.format(trace=trace))
```

**Design Points**:
- Model alias mechanism allows MainAgent to specify different SubAgent models
- Trace summarization uses fixed model (gemini-3-flash-preview) for consistency
- Supports tool filtering to limit SubAgent's available toolset

### TraceFormatter (tools/trace_formatter.py)

Execution trace formatting tool:

```python
class TraceFormatter:
    """Format Agent execution trace"""
    
    def format_step(self, step: int, action: dict, observation: str) -> str:
        """Format single step"""
        return f"[Step {step}]\nAction: {action}\nObservation: {observation}\n"
    
    def format_trace(self, steps: List[dict]) -> str:
        """Format complete trace"""
        return "\n".join(self.format_step(i, s["action"], s["obs"]) for i, s in enumerate(steps))
```

## Extending SubAgent

Adding a new SubAgent type requires:

1. Create new file in `subagents/`
2. Inherit from `BaseAgent` and implement interface
3. Export in `subagents/__init__.py`

Example framework:

```python
# subagents/my_agent.py
class MyAgent(BaseAgent):
    name: str = Field(default="MyAgent")
    benchmark_type: str = Field(default="terminalbench")
    task_instruction: str = Field(default="")
    # ... other fields
    
    def reset(self, env_info: BasicInfo) -> None:
        # Initialization logic
        pass
    
    async def step(self, observation, history, current_step, max_steps):
        # Reasoning and action logic
        pass
```

## Configuration

### GAIAOrchestraConfig

```python
@dataclass
class GAIAOrchestraConfig:
    main_model: str = "gpt-4o"      # MainAgent model
    sub_model: str = "gpt-4o-mini"  # SubAgent default model
    max_rounds: int = 10            # MainAgent max rounds
    sub_max_steps: int = 30         # SubAgent max steps
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
    main_model: str                    # MainAgent model
    sub_models: List[str]              # SubAgent available model list
    dataset_name: str = "princeton-nlp/SWE-bench_Verified"
    split: str = "test"
    max_steps: int = 50                # SubAgent max steps
    max_attempts: int = 10             # MainAgent max attempts
    docker_timeout: int = 1800         # Container timeout (seconds)
    window_size: int = 100             # ACI file view window size
```

## Backward Compatibility

For compatibility, `OrchestraSubAgent` remains available as an alias for `ReActAgent`:

```python
# aorchestra/sub_agent.py
from aorchestra.subagents.react_agent import ReActAgent
OrchestraSubAgent = ReActAgent  # Backward compatibility alias
```

## Usage Examples

### GAIA Benchmark

```python
from aorchestra import ReActAgent, GAIAOrchestraConfig
from aorchestra.runners import GAIARunner

# Configuration
config = GAIAOrchestraConfig.load("config/gaia.yaml")

# Create Runner
runner = GAIARunner(config)

# Run benchmark
results = await runner.run(dataset)
```

### SWE-bench Benchmark

```python
from aorchestra import SWEBenchOrchestraConfig
from aorchestra.runners import SWEBenchOrchestra

# Configuration
config = SWEBenchOrchestraConfig.load("config/benchmarks/aorchestra_swebench.yaml")

# Create benchmark
benchmark = SWEBenchOrchestra(config)

# Get task list
levels = benchmark.list_levels()

# Run benchmark
results = await benchmark.run(levels, max_concurrency=1)
```

### Command Line Usage

```bash
# Run GAIA
python bench_aorchestra_gaia.py --config config/benchmarks/aorchestra_gaia.yaml

# Run SWE-bench
python bench_aorchestra_swebench.py --config config/benchmarks/aorchestra_swebench.yaml

# Run specific instances
python bench_aorchestra_swebench.py --tasks "django__django-11848,sympy__sympy-12171"
```
