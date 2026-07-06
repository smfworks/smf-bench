#!/usr/bin/env python3
"""End-to-end test: run a few reasoning tests against Qwen3.6-35B on DGX Spark."""
import sys
import time
import json
sys.path.insert(0, '.')

import urllib.request

# --- 1. Send a simple reasoning prompt to Qwen and verify the response ---

def qwen_chat(prompt, max_tokens=256, temperature=0.0):
    """Send a chat completion to the Qwen vLLM endpoint."""
    url = "http://spark-56bc:8888/v1/chat/completions"
    payload = {
        "model": "nvidia/Qwen3.6-35B-A3B-NVFP4",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    elapsed = time.perf_counter() - t0
    return data, elapsed

# Test 1: Math reasoning
print("=== Test 1: Math Reasoning ===")
resp, t = qwen_chat("What is 17 * 23 + 9? Reply with the number only.")
text = resp["choices"][0]["message"]["content"]
print(f"Prompt: 17 * 23 + 9")
print(f"Response: {text[:200]}")
print(f"Expected: 400")
print(f"Correct: {'400' in text}")
print(f"Latency: {t:.2f}s")
print(f"Tokens: {resp.get('usage', {})}")
print()

# Test 2: Logic reasoning
print("=== Test 2: Logic Reasoning ===")
resp, t = qwen_chat("If all bloops are razzies and all razzies are lazzies, are all bloops lazzies? Answer yes or no.")
text = resp["choices"][0]["message"]["content"]
print(f"Response: {text[:200]}")
print(f"Expected: yes")
print(f"Correct: {'yes' in text.lower()}")
print(f"Latency: {t:.2f}s")
print()

# Test 3: Writing
print("=== Test 3: Writing ===")
resp, t = qwen_chat("Write a haiku about mountains.")
text = resp["choices"][0]["message"]["content"]
print(f"Response: {text[:200]}")
lines = [l for l in text.strip().split("\n") if l.strip()]
print(f"Line count: {len(lines)} (expected 3 for haiku)")
print(f"Latency: {t:.2f}s")
print()

# Test 4: Instruction following
print("=== Test 4: Instruction Following ===")
resp, t = qwen_chat("Reply with exactly the word PONG.")
text = resp["choices"][0]["message"]["content"]
print(f"Response: {text!r}")
print(f"Expected: PONG")
print(f"Correct: {text.strip().lower() == 'pong'}")
print(f"Latency: {t:.2f}s")
print()

print("=== Summary ===")
print("All 4 tests sent to Qwen3.6-35B via vLLM endpoint at http://spark-56bc:8888/v1")
print("Framework successfully connects to the model and evaluates responses.")