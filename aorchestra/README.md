# aorchestra Framework (Current)

This document matches the current implementation in this repository.

## Overview

`aorchestra` is the multi-agent orchestration layer used by:

- GAIA
- TerminalBench
- SWE-bench

It uses a MainAgent + SubAgent design:

- MainAgent plans and calls tools (`delegate_task`, `submit`, or `complete`).
- SubAgent executes concrete actions in the benchmark environment.

## Package Structure

```
aorchestra/
├── __init__.py
├── config.py
├── main_agent.py
├── sub_agent.py
├── common/
│   └── utils.py
├── prompts/
│   ├── gaia.py
│   ├── terminalbench.py
│   └── swebench.py
├── runners/
│   ├── __init__.py
│   ├── gaia_runner.py
│   ├── terminalbench_runner.py
│   └── swebench_runner.py
├── subagents/
│   ├── __init__.py
│   ├── react_agent.py
│   └── swebench_agent.py
└── tools/
    ├── __init__.py
    ├── complete.py
    ├── delegate.py
    ├── submit.py
    └── trace_formatter.py
```

## Runtime Architecture

1. A benchmark runner creates `MainAgent` with benchmark-specific tools and prompt builder.
2. `MainAgent.step()` asks the LLM for one tool action.
3. `delegate_task` creates and runs a SubAgent:
   - GAIA / TerminalBench -> `ReActAgent` (JSON action format)
   - SWE-bench -> `SWEBenchSubAgent` (DISCUSSION + COMMAND format)
4. MainAgent repeats until completion:
   - GAIA completes with `complete(answer=...)`
   - TerminalBench and SWE-bench submit with `submit(reason=...)`

## Benchmark Mapping

| Benchmark | Runner | SubAgent | Completion tool |
|---|---|---|---|
| GAIA | `GAIARunner` | `ReActAgent` | `complete` |
| TerminalBench | `TerminalBenchRunner` | `ReActAgent` | `submit` |
| SWE-bench | `SWEBenchRunner` / `SWEBenchOrchestra` | `SWEBenchSubAgent` | `submit` |

## Public Exports

From `aorchestra`:

- `ReActAgent`
- `SWEBenchSubAgent`
- `OrchestraSubAgent` (backward compatibility alias to `ReActAgent`)
- `GAIAOrchestraConfig`
- `TerminalBenchOrchestraConfig`
- `SWEBenchOrchestraConfig`

From `aorchestra.runners`:

- `GAIARunner`
- `TerminalBenchRunner`
- `SWEBenchRunner`
- `SWEBenchOrchestra`

## Configuration Classes (Actual Fields)

Defined in `aorchestra/config.py`.

### `GAIAOrchestraConfig`

Required:

- `main_model`
- `sub_models` (list)
- `dataset_path`
- `attachments_dir`

Common optional:

- `max_attempts` (default `5`)
- `max_concurrency` (default `1`)
- `max_tasks`
- `level_filter`
- `result_folder`
- `trajectory_folder`

### `TerminalBenchOrchestraConfig`

Required:

- `main_model`
- `sub_models` (list)
- `tasks_dir`

Common optional:

- `max_steps` (default `30`)
- `max_attempts` (default `10`)
- `max_concurrency` (default `1`)
- `sandbox` (`docker|e2b|daytona`, default `docker`)
- `docker_timeout`
- `result_folder`
- `trajectory_dir`
- `csv_summary_path`
- `env_init`
- `e2b_api_key`, `daytona_api_key`, `daytona_api_url`, `daytona_target`

### `SWEBenchOrchestraConfig`

Required:

- `main_model`
- `sub_models` (list)

Common optional:

- `dataset_name` (default `princeton-nlp/SWE-bench_Verified`)
- `split` (default `test`)
- `subset_seed`, `subset_sizes`, `subset_role`
- `selected_ids_file`
- `max_tasks`, `max_steps`, `max_attempts`
- `max_concurrency`, `docker_timeout`
- `result_folder`, `trajectory_dir`, `csv_summary_path`
- `env_init`
- `cache_dir`
- `window_size` (default `100`)

## CLI Entry Points (Repository Root)

Use the root scripts:

```bash
python bench_aorchestra_gaia.py --config config/benchmarks/aorchestra_gaia.yaml
python bench_aorchestra_swebench.py --config config/benchmarks/aorchestra_swebench.yaml
python bench_aorchestra_terminalbench.py --config config/benchmarks/aorchestra_terminalbench.yaml
```

Current CLI options:

- All three: `--config`, `--max_concurrency`, `--tasks`
- GAIA: `--skip_completed <path_to_csv>`
- SWE-bench: `--skip-completed`
- TerminalBench: `--skip_completed`

## Python Usage (SWE-bench Example)

```python
from aorchestra.config import SWEBenchOrchestraConfig
from aorchestra.runners import SWEBenchOrchestra

cfg = SWEBenchOrchestraConfig.load("config/benchmarks/aorchestra_swebench.yaml")
bench = SWEBenchOrchestra(cfg)
levels = bench.list_levels()
results = await bench.run(levels, max_concurrency=cfg.max_concurrency)
```

## Notes

- `DelegateTaskTool` currently summarizes traces with fixed model key `gemini-3-flash-preview`.
- LLM config is loaded by `LLMsConfig.default()` from:
  - `config/global_config.yaml`
  - `config/global_config2.yaml`
  - `config/model_config.yaml`
  - or environment fallback when no config file exists.

## Backward Compatibility

`aorchestra/sub_agent.py` keeps:

```python
OrchestraSubAgent = ReActAgent
```
