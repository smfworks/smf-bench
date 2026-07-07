#!/usr/bin/env python3
"""Run only the 33 tests that errored due to server crash during the resume run."""
import sys
import os
import asyncio
import functools
import json
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
print = functools.partial(print, flush=True)

import yaml
from smf_bench.api_client import APIClient
from smf_bench.evaluators import get_evaluator
from smf_bench.test_registry import TestCase

# The 33 errored test IDs from the crashed resume run
ERRORED_IDS = [
    "v3.prose.frontier.03", "v3.prose.frontier.04", "v3.prose.frontier.05",
    "v3.prose.frontier.06", "v3.prose.frontier.07", "v3.prose.frontier.08",
    "v3.prose.frontier.09", "v3.prose.frontier.10", "v3.prose.frontier.11",
    "v3.prose.frontier.12",
    "writing_article", "writing_summary", "writing_creative",
    "writing_technical", "writing_format",
    "tool_call_weather", "tool_call_calculator",
    "av2-01-compute-write", "av2-02-config-extract", "av2-03-csv-filter",
    "av2-04-log-count", "av2-05-script-and-output", "av2-06-bugfix-run",
    "av2-07-json-spec", "av2-08-multifile-summary", "av2-09-rename",
    "av2-10-reasoning-only", "av2-11-app-counter", "av2-12-app-todo",
    "av2-13-game-pong", "av2-14-game-snake", "av2-15-anim-bounce",
    "av2-16-anim-starfield",
]

SUITE_MAP = {
    "prose": Path(__file__).parent / "suites" / "quality" / "tier0_deterministic" / "prose.yaml",
    "writing": Path(__file__).parent / "suites" / "quality" / "writing" / "writing.yaml",
    "tool_calling": Path(__file__).parent / "suites" / "quality" / "tool_calling" / "tool_calling.yaml",
    "agentic": Path(__file__).parent / "suites" / "quality" / "agentic" / "agentic.yaml",
}


def find_all_tests():
    """Load all tests from suite files, return dict id->(test, suite_name).

    Handles both multi-doc YAML (each doc is a test) and single-doc YAML
    with a 'tests:' list containing test dicts.
    """
    tests = {}
    for suite_name, suite_path in SUITE_MAP.items():
        if not suite_path.exists():
            continue
        with open(suite_path) as f:
            docs = list(yaml.safe_load_all(f))
        for doc in docs:
            if not doc:
                continue
            if isinstance(doc, list):
                # Single-doc with tests: list
                for item in doc:
                    if isinstance(item, dict) and item.get("id"):
                        tests[item["id"]] = (item, suite_name)
            elif isinstance(doc, dict) and doc.get("id"):
                # Multi-doc, each doc is a test
                tests[doc["id"]] = (doc, suite_name)
    return tests


async def run_benchmark(endpoint, model, tag, timeout):
    all_tests = find_all_tests()
    client = APIClient(base_url=endpoint)
    results = []
    pass_count = 0
    fail_count = 0
    err_count = 0

    print(f"Re-running {len(ERRORED_IDS)} errored tests")
    print(f"Endpoint: {endpoint}, Model: {model}, Timeout: {timeout}s")
    print("=" * 70)

    for i, test_id in enumerate(ERRORED_IDS, 1):
        if test_id not in all_tests:
            print(f"  [{i}/{len(ERRORED_IDS)}] {test_id}... NOT FOUND")
            results.append({
                "test_id": test_id, "category": "unknown", "status": "error",
                "passed": False, "score": 0.0, "detail": "Test not found",
                "elapsed": 0, "tokens_used": 0, "evaluator": "unknown",
            })
            err_count += 1
            continue

        test, suite_name = all_tests[test_id]
        print(f"  [{i}/{len(ERRORED_IDS)}] Running {test_id}... ", end="", flush=True)
        t0 = time.time()

        # Build kwargs
        kwargs = {"max_tokens": test.get("max_tokens", 4096),
                  "temperature": test.get("temperature", 0.6)}
        if test.get("metadata", {}).get("tools"):
            kwargs["tools"] = test["metadata"]["tools"]

        messages = test.get("messages")
        if not messages:
            messages = [{"role": "user", "content": test.get("prompt", "")}]

        try:
            resp = await client.chat(messages=messages, **kwargs)

            if resp.error:
                err_count += 1
                results.append({
                    "test_id": test_id, "category": suite_name, "status": "error",
                    "passed": False, "score": 0.0, "detail": resp.error[:200],
                    "elapsed": resp.elapsed, "tokens_used": 0,
                    "evaluator": test.get("evaluator", "text_contains"),
                })
                print(f"ERROR: {resp.error[:80]}")
                continue

            tc = TestCase.from_dict(test)
            evaluator_name = test.get("evaluator", "text_contains")
            try:
                evaluator = get_evaluator(evaluator_name)
            except KeyError:
                print(f"SKIP — unknown evaluator '{evaluator_name}'")
                results.append({
                    "test_id": test_id, "category": suite_name, "status": "skip",
                    "passed": False, "score": 0.0,
                    "detail": f"Unknown evaluator: {evaluator_name}",
                    "elapsed": resp.elapsed, "tokens_used": 0,
                    "evaluator": evaluator_name,
                })
                continue

            result = evaluator(resp, tc)
            status = "pass" if result.passed else "fail"
            if result.passed:
                pass_count += 1
                print(f"PASS ({time.time()-t0:.1f}s)")
            else:
                fail_count += 1
                print(f"FAIL ({time.time()-t0:.1f}s) {result.detail[:60]}")

            results.append({
                "test_id": test_id, "category": suite_name, "status": status,
                "passed": result.passed, "score": result.score,
                "detail": result.detail[:200], "elapsed": resp.elapsed,
                "tokens_used": resp.usage.get("total_tokens", 0) if resp.usage else 0,
                "evaluator": evaluator_name,
            })
        except Exception as e:
            err_count += 1
            results.append({
                "test_id": test_id, "category": suite_name, "status": "error",
                "passed": False, "score": 0.0, "detail": str(e)[:200],
                "elapsed": time.time() - t0, "tokens_used": 0,
                "evaluator": test.get("evaluator", "unknown"),
            })
            print(f"ERROR: {str(e)[:80]}")

        # Save incrementally
        results_file = Path(__file__).parent / "results" / f"stage1_{tag}_errored_rerun.json"
        results_file.parent.mkdir(exist_ok=True)
        with open(results_file, "w") as f:
            json.dump({
                "tag": tag, "endpoint": endpoint, "model": model,
                "total_tests": len(ERRORED_IDS), "tests_run": len(results),
                "results": results,
                "summary": {"pass": pass_count, "fail": fail_count, "error": err_count},
            }, f, indent=2)

    print("=" * 70)
    print(f"RESULTS: {pass_count} pass, {fail_count} fail, {err_count} error out of {len(ERRORED_IDS)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://spark-56bc:8889/v1")
    parser.add_argument("--model", default="model")
    parser.add_argument("--tag", default="nemotron-3-nano-30b")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    asyncio.run(run_benchmark(args.endpoint, args.model, args.tag, args.timeout))