#!/usr/bin/env python3
"""Full end-to-end benchmark run: smf-bench against Qwen3.6-35B on DGX Spark.

Runs the reasoning + tool_calling categories against the live vLLM endpoint.
These are pure text tests — no multimodal inputs needed.
"""
import sys
import asyncio
sys.path.insert(0, '.')

from smf_bench.runner import RunConfig, BenchRunner

config = RunConfig(
    model_id="qwen3.6-35b-a3b-nvfp4",
    endpoint="http://spark-56bc:8888/v1",
    api_key="dummy",
    suites_dir="suites",
    models_dir="models",
    results_db="results/smf-bench.db",
    max_concurrent=4,
    timeout=120,
    engine="vLLM 0.24.0",
    config={"quantization": "NVFP4", "mtp": True},
)

runner = BenchRunner(config)
run_id = asyncio.run(runner.run(categories=["reasoning", "tool_calling"], verbose=True))
print(f"\nRun ID: {run_id}")