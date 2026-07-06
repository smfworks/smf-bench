#!/usr/bin/env python3
"""Multi-category benchmark: reasoning + tool_calling + writing against Qwen."""
import sys
import asyncio
import functools
import yaml
sys.path.insert(0, '.')

print = functools.partial(print, flush=True)

from smf_bench.api_client import APIClient
from smf_bench.evaluators import get_evaluator
from smf_bench.test_registry import TestCase

def load_tests(path):
    with open(path) as f:
        data = yaml.safe_load(f)
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        return [data]
    return []

async def main():
    # Load tests from multiple categories
    all_tests = []
    for path, label in [
        ("suites/quality/reasoning/reasoning.yaml", "reasoning"),
        ("suites/quality/tool_calling/tool_calling.yaml", "tool_calling"),
        ("suites/quality/writing/writing.yaml", "writing"),
    ]:
        tests = load_tests(path)
        print(f"  Loaded {len(tests)} {label} tests")
        all_tests.extend(tests)
    
    print(f"\nTotal: {len(all_tests)} tests across 3 categories")
    print(f"Endpoint: http://spark-56bc:8888/v1")
    print()
    
    async with APIClient(
        base_url="http://spark-56bc:8888/v1",
        model="nvidia/Qwen3.6-35B-A3B-NVFP4",
        api_key="dummy",
        timeout=60.0,
    ) as client:
        healthy = await client.health_check()
        if not healthy:
            print("ERROR: Endpoint not reachable")
            return
        
        results_by_cat = {}
        for test in all_tests:
            test_id = test["id"]
            category = test.get("category", "unknown")
            prompt = test["prompt"]
            messages = [{"role": "user", "content": prompt}]
            
            resp = await client.chat(
                messages,
                max_tokens=test.get("max_tokens", 1024),
                temperature=test.get("temperature", 0.3),
                **({"tools": test["metadata"]["tools"]} if test.get("metadata", {}).get("tools") else {}),
            )
            
            if resp.error:
                print(f"  ❌ ERR  {test_id:40s} [{category:12s}] {resp.error[:50]}")
                results_by_cat.setdefault(category, [0, 0])
                results_by_cat[category][1] += 1
                continue
            
            tc = TestCase.from_dict(test)
            evaluator = get_evaluator(test.get("evaluator", "text_contains"))
            result = evaluator(resp, tc)
            
            icon = "✅" if result.passed else "❌"
            status = "PASS" if result.passed else "FAIL"
            print(f"  {icon} {status:4s} {test_id:40s} [{category:12s}] {resp.elapsed:.1f}s {result.detail[:50]}")
            
            results_by_cat.setdefault(category, [0, 0])
            if result.passed:
                results_by_cat[category][0] += 1
            else:
                results_by_cat[category][1] += 1
        
        print(f"\n{'='*70}")
        print(f"RESULTS BY CATEGORY")
        print(f"{'='*70}")
        total_pass = 0
        total_fail = 0
        for cat in sorted(results_by_cat.keys()):
            p, f = results_by_cat[cat]
            total = p + f
            rate = p / total * 100 if total > 0 else 0
            print(f"  {cat:15s}: {p}/{total} passed ({rate:.0f}%)")
            total_pass += p
            total_fail += f
        
        grand_total = total_pass + total_fail
        grand_rate = total_pass / grand_total * 100 if grand_total > 0 else 0
        print(f"  {'─'*35}")
        print(f"  {'TOTAL':15s}: {total_pass}/{grand_total} passed ({grand_rate:.0f}%)")
        print(f"{'='*70}")

asyncio.run(main())