"""
SWE-bench Verified Benchmark Runner

Usage:
    python bench_swebench.py [--config CONFIG] [--max-concurrency N]

Example:
    python bench_swebench.py --config config/example/benchmarks/swebench.yaml --max-concurrency 5
"""
from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.bench_swebench import SWEBenchBenchmark, SWEBenchConfig
from benchmark.swebench.result_reporter import print_results
from base.agent.memory import Memory
from base.engine.async_llm import LLMsConfig, create_llm_instance
from base.engine.logs import logger

# Import the new SWEAgent with ACI tools
from agents.swebench_agent import SWEAgent


class SWEBenchAgentWrapper:
    """Wrapper that creates a fresh LLM instance per task to avoid shared cost tracking."""
    
    def __init__(self, model_name: str):
        llm_cfg = LLMsConfig.default().get(model_name)
        llm = create_llm_instance(llm_cfg)
        self._agent = SWEAgent(
            llm=llm,
            memory=Memory(llm=llm, max_memory=20),
        )
        self.llm = llm

    async def step(self, observation, history):
        action, raw_response, raw_input = await self._agent.step(observation, history)
        return action, raw_response, raw_input

    def reset(self, info):
        return self._agent.reset(info)


async def main():
    parser = argparse.ArgumentParser(description="Run SWE-bench Verified benchmark")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/example/benchmarks/swebench.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=5,
        help="Maximum concurrency for running tasks (default: 5).",
    )
    args = parser.parse_args()

    # Load config and setup
    cfg = SWEBenchConfig.load(args.config)
    if not cfg.model:
        logger.error("No model configured")
        return 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg.result_folder) / f"swebench_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    cfg.result_folder = run_dir / "results"
    cfg.result_folder.mkdir(parents=True, exist_ok=True)
    cfg.timestamp = timestamp

    # Create benchmark
    benchmark = SWEBenchBenchmark(cfg)

    levels = benchmark.list_levels()
    if cfg.max_tasks and len(levels) > cfg.max_tasks:
        levels = levels[:cfg.max_tasks]

    logger.info(f"Running SWE-bench Verified with model: {cfg.model}")
    logger.info(f"Dataset: {cfg.dataset_name} (split: {cfg.split})")
    if cfg.subset_sizes and cfg.subset_role:
        logger.info(f"Subset: role={cfg.subset_role}, seed={cfg.subset_seed}, sizes={cfg.subset_sizes}")
    logger.info(f"Instances: {len(levels)}/{len(benchmark.list_levels())}")
    logger.info(f"Max steps per instance: {cfg.max_steps}")
    logger.info(f"Output directory: {run_dir}")

    try:
        results = await benchmark.run(
            agent_cls=SWEBenchAgentWrapper,
            agent_kwargs={"model_name": cfg.model},
            levels=levels,
            max_concurrency=args.max_concurrency,
        )
        print_results(results, run_dir / "results.csv", run_dir / "trajectories")
        return 0
    except KeyboardInterrupt:
        logger.info("\n⚠ Benchmark interrupted by user")
        return 130


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
