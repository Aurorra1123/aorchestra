# FoundationAgent-Dev

AI Agent evaluation framework, supporting GAIA, SWE-bench, and TerminalBench benchmarks.

## Quick Start

```bash
# 1. Install
conda create -n orchestra python=3.11 && conda activate orchestra
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
cp config/example/model_config.yaml config/
cp -r config/example/benchmarks config/

# 3. Edit config files, fill in API keys
vim .env
vim config/model_config.yaml

# 4. Run
python bench_gaia.py --config config/benchmarks/gaia.yaml
```

---

## API Keys Configuration Table

### By Benchmark

| Benchmark | Required API Keys | Optional |
|-----------|------------------|----------|
| **GAIA** | `JINA_API_KEY`, `SERPER_API_KEY`, LLM | - |
| **SWE-bench** | LLM, Docker | - |
| **TerminalBench** | LLM, Docker or `E2B_API_KEY` | `DAYTONA_API_KEY` |

### By Tool

| Tool | Environment Variable | Purpose | Obtain From |
|------|---------------------|---------|-------------|
| Jina | `JINA_API_KEY` | Web content extraction | https://jina.ai/ |
| Serper | `SERPER_API_KEY` | Google search | https://serper.dev/ |
| E2B | `E2B_API_KEY` | Cloud sandbox | https://e2b.dev/ |
| Daytona | `DAYTONA_API_KEY` | Cloud sandbox | https://daytona.io/ |
| LLM | Configure in `config/model_config.yaml` | Model calls | OpenAI/Gemini/Claude etc. |

---

## Configuration Files

```
.env                          # Tool API keys (JINA, SERPER, E2B)
config/model_config.yaml      # LLM API keys
config/benchmarks/*.yaml      # Benchmark parameters
```

**Copy from templates:**
```bash
cp .env.example .env
cp config/example/model_config.yaml config/
cp -r config/example/benchmarks config/
```

---

## Run Commands

| Mode | GAIA | SWE-bench | TerminalBench |
|------|------|-----------|---------------|
| Standard | `python bench_gaia.py` | `python bench_swebench.py` | `python bench_terminalbench.py` |
| Orchestra | `python bench_aorchestra_gaia.py` | `python bench_aorchestra_swebench.py` | `python bench_aorchestra_terminalbench.py` |

**Common parameters:**
```bash
--config config/benchmarks/xxx.yaml    # Config file
--max-concurrency 5                     # Concurrency
--tasks task1,task2                     # Specify tasks (optional)
```

