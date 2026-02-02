"""
GAIA Benchmark - Single-Layer Baseline

This script runs GAIA benchmark using a single ReAcT Agent directly interacting
with the environment, as a baseline for comparison with orchestra (MainAgent + SubAgent) mode.

Usage:
    python bench_gaia.py --config config/benchmarks/gaia.yaml --max-concurrency 5
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

from benchmark.bench_gaia import GAIABenchmark, GAIAConfig
from agents.gaia_agent import GAIAReAcTAgent
from base.agent.memory import Memory
from base.engine.async_llm import LLMsConfig, create_llm_instance
from base.engine.logs import logger
from benchmark.gaia.tools import (
    GoogleSearchAction,
    ExecuteCodeAction,
    ExtractUrlContentAction,
    ImageAnalysisAction,
    ParseAudioAction,
)


class GAIAAgentWrapper:
    """
    Wrapper that creates a fresh LLM instance per task to avoid shared cost tracking.
    
    This mirrors the pattern used in TerminalBenchAgentWrapper.
    """
    def __init__(self, model_name: str):
        llm_cfg = LLMsConfig.default().get(model_name)
        llm = create_llm_instance(llm_cfg)
        self._agent = GAIAReAcTAgent(
            llm=llm,
            memory=Memory(llm=llm, max_memory=10),
        )
        self.llm = llm

    async def step(self, observation, history, current_step: int = 1, max_steps: int = 30):
        """Delegate step to the wrapped agent."""
        action, raw_response = await self._agent.step(observation, history, current_step, max_steps)
        return action, raw_response, None  # Return 3 values for compatibility

    def reset(self, info):
        """Reset the wrapped agent."""
        return self._agent.reset(info)


async def main():
    parser = argparse.ArgumentParser(description="Run GAIA benchmark (single-layer baseline)")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/benchmarks/gaia.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=5,
        help="Maximum concurrency for running tasks (default: 5).",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated list of task IDs to run (e.g., 'task1,task2,task3').",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("GAIA Benchmark - Single-Layer Baseline")
    logger.info("=" * 60)

    # Load config and setup
    cfg = GAIAConfig.load(args.config)
    
    # Get model from config (use main_model for single-layer mode)
    import yaml
    with open(args.config, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f) or {}
    model = raw_config.get("model") or raw_config.get("main_model")
    
    if not model:
        logger.error("No model configured. Please set 'model' or 'main_model' in config.")
        return 1

    # Check if dataset exists
    if not cfg.dataset_path.exists():
        logger.error(f"Dataset not found: {cfg.dataset_path}")
        logger.info("Expected path: benchmark/gaia/data/Gaia/2023/validation/metadata.jsonl")
        logger.info("Please ensure the GAIA dataset is properly placed.")
        return 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg.timestamp = timestamp

    # Create output directories
    run_dir = Path(cfg.result_folder) / f"gaia_baseline_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    cfg.result_folder = run_dir / "results"
    cfg.result_folder.mkdir(parents=True, exist_ok=True)
    cfg.trajectory_folder = run_dir / "trajectories"
    cfg.trajectory_folder.mkdir(parents=True, exist_ok=True)

    # Create GAIA tools
    gaia_tools = [
        GoogleSearchAction(),
        ExecuteCodeAction(),
        ExtractUrlContentAction(),
        ImageAnalysisAction(),
        ParseAudioAction(),
    ]
    logger.info(f"Loaded {len(gaia_tools)} GAIA tools: {[t.name for t in gaia_tools]}")

    # Create benchmark
    benchmark = GAIABenchmark(cfg, tools=gaia_tools)

    levels = benchmark.list_levels()
    
    # Filter by specific task IDs if provided
    if args.tasks:
        task_ids = [t.strip() for t in args.tasks.split(",") if t.strip()]
        levels = [l for l in levels if (l.get("task_id") or l.get("id")) in task_ids]
        logger.info(f"Filtered to {len(levels)} task(s): {[l.get('task_id') or l.get('id') for l in levels]}")
    elif cfg.max_tasks and len(levels) > cfg.max_tasks:
        levels = levels[:cfg.max_tasks]

    if not levels:
        logger.error("No tasks found in dataset!")
        return 1

    logger.info(f"Running GAIA Baseline with model: {model}")
    logger.info(f"Tasks: {len(levels)}/{len(benchmark.list_levels())}")
    logger.info(f"Max steps per task: {cfg.max_steps}")
    logger.info(f"Output directory: {run_dir}")

    try:
        results = await benchmark.run(
            agent_cls=GAIAAgentWrapper,
            agent_kwargs={"model_name": model},
            levels=levels,
            max_concurrency=args.max_concurrency,
        )
        
        # Print summary
        total = len(results)
        total_reward = sum(r.total_reward for r in results.values())
        correct_count = sum(1 for r in results.values() if r.total_reward > 0.5)
        total_cost = sum(r.cost for r in results.values())
        
        logger.info("\n" + "=" * 60)
        logger.info("GAIA Baseline Summary:")
        logger.info(f"  Total tasks: {total}")
        logger.info(f"  Correct: {correct_count}/{total} ({100*correct_count/total:.1f}%)" if total > 0 else "  Correct: N/A")
        logger.info(f"  Total reward: {total_reward:.2f}")
        logger.info(f"  Average reward: {total_reward/total:.4f}" if total > 0 else "  Average reward: N/A")
        logger.info(f"  Total cost: ${total_cost:.4f}")
        logger.info(f"  Results: {cfg.result_folder}")
        logger.info(f"  Trajectories: {cfg.trajectory_folder}")
        logger.info("=" * 60)
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("\n⚠ Benchmark interrupted by user")
        return 130


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

