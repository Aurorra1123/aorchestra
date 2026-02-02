"""
TerminalBench with aorchestra (MainAgent + SubAgent)

Usage:
    python bench_aorchestra_terminalbench.py --config config/benchmarks/aorchestra_terminalbench.yaml
"""
from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import argparse
import asyncio
import csv
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from base.engine.logs import logger
from aorchestra.config import TerminalBenchOrchestraConfig
from aorchestra.runners.terminalbench_runner import TerminalBenchRunner
from benchmark.bench_terminalbench import TerminalBenchConfig, TerminalBenchBenchmark


DEFAULT_CONFIG = ROOT / "config/benchmarks/aorchestra_terminalbench.yaml"


def load_completed_tasks(csv_path: Path) -> set[str]:
    """从 CSV 文件加载已完成的任务 ID。"""
    if not csv_path.exists():
        return set()
    
    completed = set()
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_id = row.get("task_id") or row.get("id")
            if task_id:
                completed.add(task_id)
    return completed


def build_terminalbench_config(config: TerminalBenchOrchestraConfig) -> TerminalBenchConfig:
    """Build TerminalBenchConfig from aorchestra config."""
    return TerminalBenchConfig(
        tasks_dir=config.tasks_dir,
        max_steps=config.max_steps,
        max_tasks=config.max_tasks,
        docker_timeout=config.docker_timeout,
        model=config.main_model,
        result_folder=config.result_folder,
        trajectory_dir=config.trajectory_dir,
        csv_summary_path=config.csv_summary_path,
        timestamp=config.timestamp,
        env_init=config.env_init,
        sandbox=config.sandbox,
        e2b_api_key=config.e2b_api_key,
        daytona_api_key=config.daytona_api_key,
        daytona_api_url=config.daytona_api_url,
        daytona_target=config.daytona_target,
    )


async def main():
    parser = argparse.ArgumentParser(description="Run TerminalBench with aorchestra.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to config YAML")
    parser.add_argument("--max_concurrency", type=int, default=None)
    parser.add_argument("--tasks", type=str, help="Comma-separated task IDs")
    parser.add_argument("--skip_completed", action="store_true", help="Skip tasks already in results CSV")
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("TerminalBench with aorchestra")
    logger.info("=" * 60)
    
    # Load config
    config = TerminalBenchOrchestraConfig.load(args.config)
    
    if not config.tasks_dir.exists():
        logger.error(f"Dataset not found: {config.tasks_dir}")
        return 1
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.timestamp = timestamp
    
    # Setup directories
    config.result_folder.mkdir(parents=True, exist_ok=True)
    if not config.trajectory_dir:
        config.trajectory_dir = config.result_folder / "trajectories"
    config.trajectory_dir.mkdir(parents=True, exist_ok=True)
    
    if not config.csv_summary_path:
        config.csv_summary_path = config.result_folder / f"terminalbench_aorchestra_{timestamp}.csv"
    
    # Create benchmark environment (TerminalBench baseline env + aorchestra runner)
    tb_config = build_terminalbench_config(config)
    benchmark = TerminalBenchBenchmark(tb_config)
    
    logger.info(f"Trajectory dir: {config.trajectory_dir}")
    logger.info(f"CSV path: {config.csv_summary_path}")
    
    levels = benchmark.list_levels()
    if args.tasks:
        task_ids = [t.strip() for t in args.tasks.split(",") if t.strip()]
        levels = [l for l in levels if l.get("id") in task_ids]
    elif config.max_tasks:
        levels = levels[:config.max_tasks]
    
    # Skip completed tasks
    if args.skip_completed:
        completed = load_completed_tasks(config.csv_summary_path)
        if completed:
            original_count = len(levels)
            levels = [l for l in levels if l.get("id") not in completed]
            skipped = original_count - len(levels)
            logger.info(f"Skipping {skipped} completed tasks")
    
    logger.info(f"aorchestra: main={config.main_model}, subs={config.sub_models}")
    logger.info(f"Running {len(levels)} tasks")
    
    concurrency = args.max_concurrency if args.max_concurrency is not None else config.max_concurrency
    
    # Create runner
    runner = TerminalBenchRunner(
        main_model=config.main_model,
        sub_models=config.sub_models,
        max_attempts=config.max_attempts,
        trajectory_dir=config.trajectory_dir,
        csv_summary_path=config.csv_summary_path,
    )
    
    # Run each level
    results = []
    semaphore = asyncio.Semaphore(concurrency)
    
    async def run_single(level):
        async with semaphore:
            try:
                env = benchmark.make_env(level)
                result = await runner.run(None, env)
                return {"task_id": level.get("id"), "success": result.total_reward > 0, "reward": result.total_reward}
            except Exception as e:
                logger.error(f"Task {level.get('id')} failed: {e}")
                return {"task_id": level.get("id"), "success": False, "error": str(e)}
    
    try:
        tasks = [asyncio.create_task(run_single(level)) for level in levels]
        results = await asyncio.gather(*tasks)
    finally:
        if config.sandbox == "e2b":
            from benchmark.terminalbench.e2b_executor import cleanup_all_sandboxes
            await cleanup_all_sandboxes()
    
    # Summary
    total = len(results)
    success_count = sum(1 for r in results if r.get("success"))
    
    logger.info("\n" + "=" * 60)
    logger.info("TerminalBench Summary:")
    logger.info(f"  Total tasks: {total}")
    logger.info(f"  Successful: {success_count}/{total}")
    logger.info(f"  Results: {config.csv_summary_path}")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
