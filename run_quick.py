#!/usr/bin/env python3
"""Quick benchmark: just the 8 quality reasoning tests (not tier0)."""
import sys
import asyncio
sys.path.insert(0, '.')

# Force unbuffered output
import functools
print = functools.partial(print, flush=True)

from smf_bench.runner import RunConfig, BenchRunner

config = RunConfig(
    model_id="qwen3.6-35b-a3b-nvfp4",
    endpoint="http://spark-56bc:8888/v1",
    api_key="dummy",
    suites_dir="suites",
    models_dir="models",
    results_db="results/smf-bench.db",
    max_concurrent=4,
    timeout=60,
    engine="vLLM 0.24.0",
)

runner = BenchRunner(config)

# Run ONLY quality/reasoning (8 tests), not tier0 (which adds 30 more)
# The tier0 tests are in a different directory but same category
# We need to filter by dimension or test_id prefix
# Actually, the run() method filters by category. Both quality and tier0
# reasoning tests have category=reasoning. Let me use a custom approach:
# run with no category filter, but only load the quality reasoning suite

# Simplest: temporarily move tier0 dir out
import shutil
import os

tier0_path = "suites/quality/tier0_deterministic"
backup_path = "suites/quality/tier0_deterministic_bak"
if os.path.exists(tier0_path):
    shutil.move(tier0_path, backup_path)
    print(f"Temporarily moved tier0 dir to {backup_path}")

run_id = asyncio.run(runner.run(categories=["reasoning"], verbose=True))
print(f"\nRun ID: {run_id}")

# Restore tier0
if os.path.exists(backup_path):
    shutil.move(backup_path, tier0_path)
    print(f"Restored tier0 dir")