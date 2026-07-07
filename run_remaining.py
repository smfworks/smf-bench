#!/usr/bin/env python3
"""Resume benchmark — run only the remaining suites for Nemotron Stage 1.

Completed in prior run: reasoning (8), math (30), coding (30), reasoning_tier0 (30), instruction easy.01+02.
This runs: instruction (remaining 28), prose (30), writing (5), tool_calling (2), agentic (16) = 81 tests.
"""
import sys
import os
import asyncio
import functools
import json
import time
import argparse
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
print = functools.partial(print, flush=True)

import yaml
from smf_bench.api_client import APIClient
from smf_bench.evaluators import get_evaluator
from smf_bench.test_registry import TestCase

REASONING_MODEL_DEFAULTS = {
    "max_tokens": 4096,
    "temperature": 0.6,
}

# Only the remaining suites
REMAINING_SUITES = [
    ("suites/quality/tier0_deterministic/instruction.yaml", "instruction"),
    ("suites/quality/tier0_deterministic/prose.yaml", "prose"),
    ("suites/quality/writing/writing.yaml", "writing"),
    ("suites/quality/tool_calling/tool_calling.yaml", "tool_calling"),
    ("suites/quality/agentic/agentic.yaml", "agentic"),
]

# Tests already completed in instruction suite
SKIP_IDS = {"v3.instruction.easy.01", "v3.instruction.easy.02"}


def load_tests(path):
    with open(path) as f:
        docs = list(yaml.safe_load_all(f))
    tests = []
    for doc in docs:
        if doc is None:
            continue
        if isinstance(doc, list):
            tests.extend(doc)
        elif isinstance(doc, dict):
            tests.append(doc)
    return tests


def is_reasoning_model(model_name, tag=""):
    combined = (model_name + " " + tag).lower()
    indicators = ["nemotron", "deepseek-r1", "o1", "o3", "qwen3"]
    return any(ind in combined for ind in indicators)


def _save_results(filepath, tag, endpoint, model, reasoning, elapsed, by_cat, all_res):
    total_pass = sum(v["pass"] for v in by_cat.values())
    total_fail = sum(v["fail"] for v in by_cat.values())
    total_error = sum(v["error"] for v in by_cat.values())
    grand_total = total_pass + total_fail + total_error
    grand_rate = total_pass / grand_total * 100 if grand_total > 0 else 0
    output = {
        "tag": tag, "endpoint": endpoint, "model": model,
        "timestamp": datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
        "reasoning_model": reasoning,
        "wall_time_seconds": round(elapsed, 1),
        "summary": {
            "total": grand_total, "passed": total_pass,
            "failed": total_fail, "error": total_error,
            "pass_rate": round(grand_rate, 1),
        },
        "by_category": by_cat,
        "tests": all_res,
    }
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)


async def run_benchmark(endpoint, model, tag, timeout=120.0):
    reasoning = is_reasoning_model(model, tag)
    defaults = REASONING_MODEL_DEFAULTS if reasoning else {}

    # Load all remaining tests
    all_tests = []
    for suite_path, suite_label in REMAINING_SUITES:
        full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), suite_path)
        tests = load_tests(full_path)
        for t in tests:
            if t.get("id") in SKIP_IDS:
                continue
            all_tests.append((t, suite_label))

    print(f"\n{'='*70}")
    print(f"RESUME BENCHMARK — {tag}")
    print(f"Endpoint: {endpoint}")
    print(f"Model: {model}")
    print(f"Remaining tests: {len(all_tests)}")
    print(f"Reasoning model: {reasoning}")
    if reasoning:
        print(f"  (max_tokens={defaults['max_tokens']})")
    print(f"{'='*70}\n")

    results_by_cat = {}
    all_results = []

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"stage1_{tag}_resume_{timestamp}.json")

    async with APIClient(
        base_url=endpoint,
        model=model,
        api_key="dummy",
        timeout=timeout,
    ) as client:
        healthy = await client.health_check()
        if not healthy:
            print("ERROR: Endpoint not reachable")
            return None
        print(f"Endpoint healthy: {healthy}\n")

        start_time = time.perf_counter()

        for i, (test, suite_label) in enumerate(all_tests):
            test_id = test["id"]
            category = test.get("category", suite_label)

            print(f"  [{i+1}/{len(all_tests)}] Running {test_id}...", end=" ", flush=True)

            # Build kwargs — convert 'prompt' string to messages format
            kwargs = {"max_tokens": defaults.get("max_tokens", 2048)}
            if reasoning:
                kwargs["temperature"] = defaults.get("temperature", 0.6)
            # Use per-test overrides if present
            if "max_tokens" in test:
                kwargs["max_tokens"] = test["max_tokens"]
            if "temperature" in test:
                kwargs["temperature"] = test["temperature"]
            # Pass tools if defined in metadata
            if test.get("metadata", {}).get("tools"):
                kwargs["tools"] = test["metadata"]["tools"]

            messages = test.get("messages")
            if not messages:
                prompt = test.get("prompt", "")
                messages = [{"role": "user", "content": prompt}]

            resp = await client.chat(
                messages=messages,
                **kwargs,
            )

            if resp.error:
                print(f"ERROR: {resp.error[:80]}")
                results_by_cat.setdefault(category, {"pass": 0, "fail": 0, "error": 0})
                results_by_cat[category]["error"] += 1
                all_results.append({
                    "test_id": test_id, "category": category, "suite": suite_label,
                    "status": "error", "passed": False,
                    "score": 0.0, "detail": resp.error[:200],
                    "elapsed": resp.elapsed, "tokens_used": 0,
                    "evaluator": test.get("evaluator", "text_contains"),
                })
                continue

            tc = TestCase.from_dict(test)
            evaluator_name = test.get("evaluator", "text_contains")
            try:
                evaluator = get_evaluator(evaluator_name)
            except KeyError:
                print(f"⚠️ SKIP — unknown evaluator '{evaluator_name}'")
                results_by_cat.setdefault(category, {"pass": 0, "fail": 0, "error": 0})
                results_by_cat[category]["error"] += 1
                all_results.append({
                    "test_id": test_id, "category": category, "suite": suite_label,
                    "status": "skip", "passed": False,
                    "score": 0.0, "detail": f"Unknown evaluator: {evaluator_name}",
                    "elapsed": resp.elapsed, "tokens_used": 0,
                    "evaluator": evaluator_name,
                })
                continue
            result = evaluator(resp, tc)

            icon = "✅" if result.passed else "❌"
            status = "PASS" if result.passed else "FAIL"
            print(f"{icon} {status} {resp.elapsed:.1f}s {result.detail[:60]}")

            results_by_cat.setdefault(category, {"pass": 0, "fail": 0, "error": 0})
            if result.passed:
                results_by_cat[category]["pass"] += 1
            else:
                results_by_cat[category]["fail"] += 1

            all_results.append({
                "test_id": test_id, "category": category, "suite": suite_label,
                "status": status.lower(), "passed": result.passed,
                "score": result.score, "detail": result.detail[:200],
                "elapsed": resp.elapsed,
                "tokens_used": resp.usage.get("total_tokens", 0) if resp.usage else 0,
                "evaluator": evaluator_name,
            })

            # Incremental save every 5 tests
            if len(all_results) % 5 == 0:
                _save_results(results_file, tag, endpoint, model, reasoning,
                              time.perf_counter() - start_time, results_by_cat, all_results)

        elapsed_total = time.perf_counter() - start_time

    # Print summary
    print(f"\n{'='*70}")
    print(f"RESULTS BY CATEGORY — {tag} (RESUME)")
    print(f"{'='*70}")
    total_pass = 0
    total_fail = 0
    total_error = 0
    for cat in sorted(results_by_cat.keys()):
        r = results_by_cat[cat]
        total = r["pass"] + r["fail"] + r["error"]
        rate = r["pass"] / total * 100 if total > 0 else 0
        print(f"  {cat:16s}: {r['pass']:3d}/{total:3d} ({rate:5.1f}%)  fail={r['fail']}, err={r['error']}")
        total_pass += r["pass"]
        total_fail += r["fail"]
        total_error += r["error"]

    grand_total = total_pass + total_fail + total_error
    grand_rate = total_pass / grand_total * 100 if grand_total > 0 else 0
    print(f"  {'─'*50}")
    print(f"  {'TOTAL':16s}: {total_pass:3d}/{grand_total:3d} ({grand_rate:5.1f}%)  fail={total_fail}, err={total_error}")
    print(f"  Wall time: {elapsed_total:.1f}s")
    print(f"{'='*70}")

    _save_results(results_file, tag, endpoint, model, reasoning, elapsed_total, results_by_cat, all_results)
    print(f"\nResults saved to: {results_file}")
    return results_file


def main():
    parser = argparse.ArgumentParser(description="Resume remaining benchmark suites")
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    asyncio.run(run_benchmark(args.endpoint, args.model, args.tag, args.timeout))


if __name__ == "__main__":
    main()