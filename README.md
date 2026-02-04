# AOrchestra

MainAgent + SubAgent orchestration framework for **GAIA**, **SWE-bench**, and **TerminalBench**.

> Note: legacy single-agent entry points were removed. Use the AOrchestra runners below.

## Core Idea

AOrchestra separates orchestration and execution:

- **MainAgent**: plans, decomposes, and coordinates subtasks.
- **SubAgent**: executes concrete actions inside each benchmark environment.

This design makes multi-step tasks easier to solve while keeping benchmark-specific tooling modular.

## Repository Layout

```text
.
├── bench_aorchestra_gaia.py
├── bench_aorchestra_swebench.py
├── bench_aorchestra_terminalbench.py
├── aorchestra/                 # MainAgent/SubAgent framework
├── benchmark/                  # Benchmark adapters and datasets
├── config/example/benchmarks/  # Benchmark config templates
└── config/example/model_config.yaml
```

## Quick Start

```bash
# 1) Install
conda create -n orchestra python=3.13 && conda activate orchestra
pip install -r requirements.txt

# 2) Configure
cp .env.example .env
cp config/example/model_config.yaml config/
cp -r config/example/benchmarks config/

# 3) Fill in API keys
vim .env
vim config/model_config.yaml
```

## Dataset Setup

| Benchmark | Download | Put it here |
|---|---|---|
| **GAIA** | https://huggingface.co/datasets/gaia-benchmark/GAIA | `benchmark/gaia/data/Gaia/` (config expects `benchmark/gaia/data/Gaia/2023/validation/metadata.jsonl`) |
| **TerminalBench** | https://www.tbench.ai/leaderboard/terminal-bench/2.0 | `benchmark/terminalbench/terminal-bench/` (config default: `benchmark/terminalbench/terminal-bench/test`) |
| **SWE-bench** | Already prepared in this project setup | Use `config/benchmarks/aorchestra_swebench.yaml` (`dataset_name: princeton-nlp/SWE-bench_Verified`) |

Recommended commands:

```bash
# GAIA (gated Hugging Face dataset; requires accepted access + HF login)
huggingface-cli download gaia-benchmark/GAIA --repo-type dataset --local-dir benchmark/gaia/data/Gaia

# TerminalBench (clone task repo so /test is available)
git clone --depth=1 https://github.com/laude-institute/terminal-bench-2.git benchmark/terminalbench/terminal-bench
```

## API Keys

### By Benchmark

| Benchmark | Required API Keys | Optional |
|---|---|---|
| **GAIA** | `JINA_API_KEY`, `SERPER_API_KEY`, LLM key in `config/model_config.yaml` | - |
| **SWE-bench** | LLM key in `config/model_config.yaml`, Docker | - |
| **TerminalBench** | LLM key in `config/model_config.yaml`, Docker or `E2B_API_KEY` | `DAYTONA_API_KEY` |

### By Tool

| Tool | Environment Variable | Purpose | Obtain From |
|---|---|---|---|
| Jina | `JINA_API_KEY` | Web content extraction | https://jina.ai/ |
| Serper | `SERPER_API_KEY` | Google search | https://serper.dev/ |
| E2B | `E2B_API_KEY` | Cloud sandbox | https://e2b.dev/ |
| Daytona | `DAYTONA_API_KEY` | Cloud sandbox | https://daytona.io/ |
| LLM | in `config/model_config.yaml` | Model calls | OpenAI / Gemini / Claude etc. |

## Run

| Benchmark | Command |
|---|---|
| **GAIA** | `python bench_aorchestra_gaia.py --config config/benchmarks/aorchestra_gaia.yaml` |
| **SWE-bench** | `python bench_aorchestra_swebench.py --config config/benchmarks/aorchestra_swebench.yaml` |
| **TerminalBench** | `python bench_aorchestra_terminalbench.py --config config/benchmarks/aorchestra_terminalbench.yaml` |

Common CLI options:

```bash
--config config/benchmarks/xxx.yaml
--max_concurrency 5
--tasks task1,task2
```

Benchmark-specific options:

- GAIA: `--skip_completed <path/to/results.csv>`
- SWE-bench: `--skip-completed`
- TerminalBench: `--skip_completed`

## Paper

- arXiv: https://arxiv.org/abs/2602.03786

## Citation

```bibtex
@misc{ruan2026aorchestraautomatingsubagentcreation,
      title={AOrchestra: Automating Sub-Agent Creation for Agentic Orchestration},
      author={Jianhao Ruan and Zhihao Xu and Yiran Peng and Fashen Ren and Zhaoyang Yu and Xinbing Liang and Jinyu Xiang and Bang Liu and Chenglin Wu and Yuyu Luo and Jiayi Zhang},
      year={2026},
      eprint={2602.03786},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2602.03786},
}
```
