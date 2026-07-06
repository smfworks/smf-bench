#!/usr/bin/env python3
"""Minimal test: just the API client health check + one chat call."""
import sys
import asyncio
sys.path.insert(0, '.')

from smf_bench.api_client import APIClient

async def main():
    print("Creating client...", flush=True)
    async with APIClient(
        base_url="http://spark-56bc:8888/v1",
        model="nvidia/Qwen3.6-35B-A3B-NVFP4",
        api_key="dummy",
        timeout=30.0,
    ) as client:
        print("Health check...", flush=True)
        healthy = await client.health_check()
        print(f"  Healthy: {healthy}", flush=True)
        
        if healthy:
            print("Sending chat...", flush=True)
            resp = await client.chat(
                messages=[{"role": "user", "content": "What is 2+2? Reply with just the number."}],
                max_tokens=16,
                temperature=0.0,
            )
            print(f"  Text: {resp.text}", flush=True)
            print(f"  Elapsed: {resp.elapsed:.2f}s", flush=True)
            print(f"  Error: {resp.error}", flush=True)

asyncio.run(main())