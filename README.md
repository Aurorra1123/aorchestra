# FoundationAgent-Dev

AI Agent 评测框架，支持 GAIA、SWE-bench、TerminalBench 基准测试。

## 快速开始

```bash
# 1. 安装
conda create -n orchestra python=3.11 && conda activate orchestra
pip install -r requirements.txt

# 2. 配置
cp .env.example .env
cp config/example/model_config.yaml config/
cp -r config/example/benchmarks config/

# 3. 编辑配置文件，填入 API keys
vim .env
vim config/model_config.yaml

# 4. 运行
python bench_gaia.py --config config/benchmarks/gaia.yaml
```

---

## API Keys 配置表

### 按 Benchmark 分类

| Benchmark | 必需 API Keys | 可选 |
|-----------|--------------|------|
| **GAIA** | `JINA_API_KEY`, `SERPER_API_KEY`, LLM | - |
| **SWE-bench** | LLM, Docker | - |
| **TerminalBench** | LLM, Docker 或 `E2B_API_KEY` | `DAYTONA_API_KEY` |

### 按工具分类

| 工具 | 环境变量 | 用途 | 获取地址 |
|-----|---------|------|---------|
| Jina | `JINA_API_KEY` | 网页内容提取 | https://jina.ai/ |
| Serper | `SERPER_API_KEY` | Google 搜索 | https://serper.dev/ |
| E2B | `E2B_API_KEY` | 云沙箱 | https://e2b.dev/ |
| Daytona | `DAYTONA_API_KEY` | 云沙箱 | https://daytona.io/ |
| LLM | 在 `config/model_config.yaml` 配置 | 模型调用 | OpenAI/Gemini/Claude 等 |

---

## 配置文件

```
.env                          # 工具 API keys (JINA, SERPER, E2B)
config/model_config.yaml      # LLM API keys
config/benchmarks/*.yaml      # Benchmark 参数
```

**从模板复制：**
```bash
cp .env.example .env
cp config/example/model_config.yaml config/
cp -r config/example/benchmarks config/
```

---

## 运行命令

| 模式 | GAIA | SWE-bench | TerminalBench |
|-----|------|-----------|---------------|
| 普通 | `python bench_gaia.py` | `python bench_swebench.py` | `python bench_terminalbench.py` |
| Orchestra | `python bench_aorchestra_gaia.py` | `python bench_aorchestra_swebench.py` | `python bench_aorchestra_terminalbench.py` |

**通用参数：**
```bash
--config config/benchmarks/xxx.yaml    # 配置文件
--max-concurrency 5                     # 并发数
--tasks task1,task2                     # 指定任务（可选）
```


