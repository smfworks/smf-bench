#!/usr/bin/env python3
"""Direct benchmark: load reasoning tests from YAML, send to Qwen, evaluate."""
import sys
import asyncio
import functools
import time
sys.path.insert(0, '.')

print = functools.partial(print, flush=True)

import yaml
from smf_bench.api_client import APIClient
from smf_bench.evaluators import get_evaluator

async def main():
    # Load reasoning tests — YAML has a list as the top-level document
    with open("suites/quality/reasoning/reasoning.yaml") as f:
        data = yaml.safe_load(f)
    if isinstance(data, list):
        tests = data
    elif isinstance(data, dict):
        tests = [data]
    else:
        tests = list(yaml.safe_load_all(f))
    print(f"Loaded {len(tests)} reasoning tests")
    
    async with APIClient(
        base_url="http://spark-56bc:8888/v1",
        model="nvidia/Qwen3.6-35B-A3B-NVFP4",
        api_key="dummy",
        timeout=60.0,
    ) as client:
        healthy = await client.health_check()
        print(f"Endpoint healthy: {healthy}")
        if not healthy:
            return
        
        passed = 0
        failed = 0
        for test in tests:
            test_id = test["id"]
            prompt = test["prompt"]
            max_tokens = test.get("max_tokens", 1024)
            temperature = test.get("temperature", 0.3)
            evaluator_name = test.get("evaluator", "regex_match")
            expected = test.get("expected", "")
            
            messages = [{"role": "user", "content": prompt}]
            resp = await client.chat(messages, max_tokens=max_tokens, temperature=temperature)
            
            if resp.error:
                print(f"  ❌ ERR  {test_id:40s} {resp.error[:60]}")
                failed += 1
                continue
            
            # Evaluate
            # Evaluate — use TestCase.from_dict for proper construction
            from smf_bench.test_registry import TestCase
            tc = TestCase.from_dict(test)
            evaluator = get_evaluator(evaluator_name)
            result = evaluator(resp, tc)
            
            icon = "✅" if result.passed else "❌"
            status = "PASS" if result.passed else "FAIL"
            print(f"  {icon} {status:4s} {test_id:40s} {resp.elapsed:.1f}s {result.detail[:60]}")
            if result.passed:
                passed += 1
            else:
                failed += 1
        
        total = passed + failed
        rate = passed / total if total > 0 else 0
        print(f"\n{'='*60}")
        print(f"Results: {passed}/{total} passed ({rate:.1%}), {failed} failed")
        print(f"{'='*60}")

asyncio.run(main())