"""
SWE-bench with aorchestra (MainAgent + SubAgent)

Usage:
    python bench_aorchestra_swebench.py --config config/benchmarks/aorchestra_swebench.yaml
    python bench_aorchestra_swebench.py --config config/benchmarks/aorchestra_swebench.yaml --tasks "django__django-11848"
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
from aorchestra.config import SWEBenchOrchestraConfig
from aorchestra.runners.swebench_runner import SWEBenchOrchestra


DEFAULT_CONFIG_PATH = ROOT / "config/benchmarks/aorchestra_swebench.yaml"


async def main():
    """Run SWE-bench benchmark with aorchestra."""
    parser = argparse.ArgumentParser(description="Run SWE-bench benchmark using aorchestra.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to config YAML.")
    parser.add_argument("--max_concurrency", type=int, default=None, help="Override max_concurrency.")
    parser.add_argument("--tasks", type=str, default=None, help="Comma-separated instance IDs to run.")
    parser.add_argument("--skip-completed", action="store_true", help="Skip tasks that already succeeded in results.csv.")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("SWE-bench with aorchestra")
    logger.info("=" * 60)

    # Load configuration
    config = SWEBenchOrchestraConfig.load(args.config)

    # Setup timestamp and directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.timestamp = timestamp

    config.result_folder.mkdir(parents=True, exist_ok=True)
    if not config.trajectory_dir:
        config.trajectory_dir = config.result_folder / "trajectories"
    config.trajectory_dir.mkdir(parents=True, exist_ok=True)

    if not config.csv_summary_path:
        config.csv_summary_path = config.result_folder / "results.csv"

    # Create benchmark
    benchmark = SWEBenchOrchestra(config)

    # Get levels
    levels = benchmark.list_levels()

    # Filter by specific instance IDs if provided
    if args.tasks:
        task_ids = [t.strip() for t in args.tasks.split(",") if t.strip()]
        levels = [l for l in levels if l.get("id") in task_ids]
        logger.info(f"Filtered to {len(levels)} instance(s): {[l.get('id') for l in levels]}")
    elif config.max_tasks and len(levels) > config.max_tasks:
        levels = levels[:config.max_tasks]
        logger.info(f"Limited to {len(levels)} instance(s) as per config")

    # Skip completed tasks if requested
    if args.skip_completed and config.csv_summary_path and config.csv_summary_path.exists():
        completed_ids = set()
        with config.csv_summary_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Skip if success=True (case-insensitive)
                if row.get("success", "").lower() == "true":
                    completed_ids.add(row.get("task_id"))

        if completed_ids:
            original_count = len(levels)
            levels = [l for l in levels if l.get("id") not in completed_ids]
            skipped = original_count - len(levels)
            logger.info(f"Skipped {skipped} completed task(s): {list(completed_ids)[:5]}{'...' if len(completed_ids) > 5 else ''}")

    if not levels:
        logger.error("No instances to run!")
        return 1

    # Log configuration
    logger.info("=" * 60)
    logger.info(f"Dataset: {config.dataset_name} (split: {config.split})")
    if config.subset_sizes and config.subset_role:
        logger.info(f"Subset: role={config.subset_role}, seed={config.subset_seed}, sizes={config.subset_sizes}")
    logger.info(f"Main model: {config.main_model}")
    logger.info(f"Sub models: {config.sub_models}")
    logger.info(f"Max attempts: {config.max_attempts}")
    logger.info(f"Max steps per SubAgent: {config.max_steps}")
    logger.info(f"Docker timeout: {config.docker_timeout}s")
    logger.info(f"Instances to run: {len(levels)}")
    logger.info(f"Result folder: {config.result_folder}")
    logger.info(f"Trajectory dir: {config.trajectory_dir}")
    logger.info(f"CSV path: {config.csv_summary_path}")
    logger.info("=" * 60)

    # Determine concurrency
    concurrency = args.max_concurrency if args.max_concurrency is not None else config.max_concurrency
    logger.info(f"Running with max_concurrency={concurrency}")

    # Run benchmark
    try:
        results = await benchmark.run(levels=levels, max_concurrency=concurrency)
    except KeyboardInterrupt:
        logger.info("\nBenchmark interrupted by user")
        return 130

    # Print final summary
    total = len(results)
    success_count = sum(1 for r in results.values() if r and r.total_reward > 0)
    total_reward = sum(r.total_reward for r in results.values() if r)

    logger.info("\n" + "=" * 60)
    logger.info("SWE-bench aorchestra Summary:")
    logger.info(f"  Total tasks: {total}")
    logger.info(f"  Successful: {success_count}/{total}")
    logger.info(f"  Total reward: {total_reward:.2f}")
    if total > 0:
        logger.info(f"  Success rate: {success_count/total*100:.1f}%")
    logger.info(f"  Results: {config.csv_summary_path}")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
