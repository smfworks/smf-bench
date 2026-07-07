#!/usr/bin/env python3
"""Parameterized Stage 1 benchmark runner.

Usage:
  python3 run_stage1.py --endpoint http://spark-56bc:8889/v1 --model model --tag nemotron-3-nano-30b
  python3 run_stage1.py --endpoint http://spark-56bc:8888/v1 --model nvidia/Qwen3.6-35B-A3B-NVFP4 --tag qwen3.6-35b
  python3 run_stage1.py --endpoint http://spark-56bc:8888/v1 --model gemma-4-26b --tag gemma-4-26b

Runs all quality test suites (reasoning, math, coding, writing, tool_calling, instruction, prose, agentic)
and saves results to results/stage1_<tag>_<timestamp>.json
"""
import sys
import asyncio
import functools
import json
import os
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
print = functools.partial(print, flush=True)

import yaml
from smf_bench.api_client import APIClient, APIResponse
from smf_bench.evaluators import get_evaluator
from smf_bench.test_registry import TestCase

# ─── Test suite definition ───────────────────────────────────────────────────
QUALITY_SUITES = [
    ("suites/quality/reasoning/reasoning.yaml", "reasoning"),
    ("suites/quality/tier0_deterministic/math.yaml", "math"),
    ("suites/quality/tier0_deterministic/coding.yaml", "coding"),
    ("suites/quality/tier0_deterministic/reasoning.yaml", "reasoning_tier0"),
    ("suites/quality/tier0_deterministic/instruction.yaml", "instruction"),
    ("suites/quality/tier0_deterministic/prose.yaml", "prose"),
    ("suites/quality/writing/writing.yaml", "writing"),
    ("suites/quality/tool_calling/tool_calling.yaml", "tool_calling"),
    ("suites/quality/agentic/agentic.yaml", "agentic"),
]

# Reasoning models need more tokens to complete chain-of-thought before answer
REASONING_MODEL_DEFAULTS = {
    "max_tokens": 4096,
    "temperature": 0.6,
}


def load_tests(path: str) -> list[dict]:
    with open(path) as f:
        # Handle both single and multi-document YAML files
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


def is_reasoning_model(model_name: str, tag: str = "") -> bool:
    """Check if this is a reasoning model that needs higher token limits."""
    combined = (model_name + " " + tag).lower()
    reasoning_indicators = ["nemotron", "deepseek-r1", "o1", "o3", "qwen3", "gpt-oss"]
    return any(ind in combined for ind in reasoning_indicators)


# Reasoning models generate long chain-of-thought before the answer.
# 120s is too tight — measured runs show 90–120s per request for 4096-token
# generations on reasoning models. Auto-raise to 300s for reasoning models
# unless the user explicitly set a timeout.
REASONING_TIMEOUT = 300.0
DEFAULT_TIMEOUT = 120.0


def _estimate_prompt_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) for context budget checks."""
    return max(1, len(text) // 4)


def _find_latest_results(results_dir: str, tag: str) -> str | None:
    """Find the most recent results file matching the given tag."""
    import glob
    pattern = os.path.join(results_dir, f"stage1_{tag}_*.json")
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    return files[0] if files else None


def _load_completed_test_ids(filepath: str) -> set[str]:
    """Load test IDs that already have a pass/fail/skip result from a prior run."""
    with open(filepath) as f:
        data = json.load(f)
    tests = data.get("tests", [])
    # Only skip tests that have a definitive (non-error) status —
    # error tests should be retried, not skipped.
    return {t["test_id"] for t in tests
            if t.get("test_id") and t.get("status") in ("pass", "fail", "skip")}


def _load_prior_results(filepath: str) -> tuple[list[dict], dict, float]:
    """Load prior run results for merging. Returns (tests, by_category, wall_time)."""
    with open(filepath) as f:
        data = json.load(f)
    tests = data.get("tests", [])
    by_cat = data.get("by_category", {})
    wall = data.get("wall_time_seconds", 0.0)
    return tests, by_cat, wall


def _save_results(filepath, tag, endpoint, model, reasoning, elapsed, by_cat, all_res):
    """Save results to JSON file (used for both incremental and final saves)."""
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


def _parse_harmony_response(resp: APIResponse) -> APIResponse:
    """Parse GPT-OSS Harmony format response from /v1/completions.

    Harmony output format uses channels delimited by special tokens:
      <|channel|>analysis<|message|>...reasoning...<|end|>
      <|channel|>commentary<|message|>...commentary...<|end|>
      <|channel|>final<|message|>...answer...<|end|>

    The model may also output without the <|channel|> prefix, like:
      analysis<|message|>...<|end|>final<|message|>...<|end|>

    Extract the 'final' channel content as the answer text, and 'analysis'
    channel content as reasoning.
    """
    import re
    text = resp.text

    reasoning = ""
    answer = ""

    # Pattern to extract channel content: optional <|channel|>, channel name,
    # <|message|> delimiter, then content until <|end|> or next channel
    # Group 1 = channel name, Group 2 = content
    channel_pattern = r'(?:<\|channel\|>)?(analysis|commentary|final)<\|message\|>(.*?)(?:<\|end\|>|<\|channel\|>|$)'
    channels = re.findall(channel_pattern, text, re.DOTALL)

    if channels:
        for ch_name, ch_content in channels:
            ch_content = ch_content.strip()
            if ch_name == "final":
                answer = ch_content
            elif ch_name == "analysis":
                reasoning = ch_content
            elif ch_name == "commentary":
                # Commentary is supplementary; don't use it for answer
                pass
    else:
        # Fallback: try simpler pattern without <|message|> delimiter
        # Format might be: "analysisReasoning<|end|>finalAnswer"
        final_match = re.search(r'final(.*?)(?:<\|end\|>|$)', text, re.DOTALL)
        if final_match:
            answer = final_match.group(1).strip()

        analysis_match = re.search(r'analysis(.*?)(?:<\|end\|>|final|commentary)', text, re.DOTALL)
        if analysis_match:
            reasoning = analysis_match.group(1).strip()

    # If still no answer, use raw text with channel markers stripped
    if not answer:
        cleaned = re.sub(r'<\|channel\|>', '', text)
        cleaned = re.sub(r'<\|message\|>', '', cleaned)
        cleaned = re.sub(r'<\|end\|>', '', cleaned)
        cleaned = re.sub(r'^(analysis|commentary|final)\s*', '', cleaned).strip()
        answer = cleaned

    resp.text = answer
    resp.reasoning = reasoning
    return resp


async def run_benchmark(endpoint: str, model: str, tag: str, timeout: float = 120.0,
                        use_completions: bool = False, tokenizer_path: str = "",
                        resume: bool = False):
    """Run all quality suites against the given endpoint.

    If use_completions=True, send requests via /v1/completions instead of
    /v1/chat/completions. This bypasses vLLM's chat parser (e.g. the Harmony
    reasoning parser used by GPT-OSS models that requires downloading a vocab
    file from Azure blob storage, which may be unavailable in Docker).

    When use_completions=True and tokenizer_path is provided, the HuggingFace
    tokenizer is loaded to apply the chat template client-side before sending
    the raw prompt to /v1/completions, preserving instruction-following quality.

    If resume=True, load the most recent results file for this tag and skip
    tests that already completed (pass/fail/skip). Error tests are retried.
    Results are merged into a new file.
    """
    suites_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "suites")
    
    # Load tokenizer for chat template application if using completions mode
    _tokenizer = None
    if use_completions:
        tok_path = tokenizer_path or os.environ.get("SMF_BENCH_TOKENIZER_PATH", "")
        if not tok_path:
            print("ERROR: --use-completions requires --tokenizer-path or SMF_BENCH_TOKENIZER_PATH env var")
            return None
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(tok_path)
        print(f"Loaded tokenizer from {tok_path} for client-side chat template")
    
    # Load all test suites
    all_tests = []
    for path, label in QUALITY_SUITES:
        full_path = os.path.join(suites_dir, path.replace("suites/", ""))
        if not os.path.exists(full_path):
            print(f"  ⚠ Skipping {label}: file not found at {full_path}")
            continue
        tests = load_tests(full_path)
        print(f"  Loaded {len(tests):3d} {label} tests")
        all_tests.extend([(t, label) for t in tests])
    
    print(f"\nTotal: {len(all_tests)} tests across {len(QUALITY_SUITES)} suites")
    print(f"Endpoint: {endpoint}")
    print(f"Model:    {model}")
    print(f"Tag:      {tag}")
    print()
    
    # Determine token/temperature defaults
    reasoning = is_reasoning_model(model, tag)
    defaults = REASONING_MODEL_DEFAULTS if reasoning else {}
    if reasoning:
        print(f"  (reasoning model detected: using max_tokens={defaults['max_tokens']})")

    # ── Fix 3: Timeout auto-tuning ──────────────────────────────────────────
    # If the caller didn't explicitly raise the timeout and this is a reasoning
    # model, auto-raise to 300s. Measured runs show reasoning models routinely
    # take 90–120s for 4096-token generations; 120s default cuts them off.
    effective_timeout = timeout
    if reasoning and timeout <= DEFAULT_TIMEOUT + 1:
        effective_timeout = REASONING_TIMEOUT
        print(f"  (auto-raising timeout: 120s → {REASONING_TIMEOUT:.0f}s for reasoning model)")
    print(f"  (per-request timeout: {effective_timeout:.0f}s)")
    print()

    # ── Fix 2: Resume from prior results ────────────────────────────────────
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    prior_completed: set[str] = set()
    prior_results: list[dict] = []
    prior_by_cat: dict = {}
    prior_wall_time: float = 0.0

    if resume:
        prior_file = _find_latest_results(results_dir, tag)
        if prior_file:
            prior_completed = _load_completed_test_ids(prior_file)
            prior_results, prior_by_cat, prior_wall_time = _load_prior_results(prior_file)
            skipped = len([t for t in all_tests if t[0]["id"] in prior_completed])
            remaining = len(all_tests) - skipped
            print(f"  📂 Resume: loaded {len(prior_completed)} completed test IDs from {os.path.basename(prior_file)}")
            print(f"     Skipping {skipped} completed tests, {remaining} to run (including retried errors)")
            print()
        else:
            print(f"  📂 Resume: no prior results found for tag '{tag}', starting fresh")
            print()

    # Set up results file early for incremental saving
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"stage1_{tag}_{timestamp}.json")
    
    async with APIClient(
        base_url=endpoint,
        model=model,
        api_key="dummy",
        timeout=effective_timeout,
    ) as client:
        healthy = await client.health_check()
        if not healthy:
            print("ERROR: Endpoint not reachable")
            return None
        print(f"Endpoint healthy: {healthy}")
        print()

        # ── Fix 1: Pre-flight context validation ───────────────────────────
        # Query the server's max_model_len and verify max_tokens + prompt
        # will fit. If max_tokens is too large for the context window, auto-cap
        # it to leave room for the prompt.
        server_max_len = await client.get_max_model_len()
        if server_max_len is not None:
            print(f"  Server max_model_len: {server_max_len} tokens")
            # Check if the reasoning default max_tokens (4096) leaves room
            # for at least a small prompt
            requested_max = defaults.get("max_tokens", 1024)
            if requested_max >= server_max_len:
                # max_tokens >= context window: every request would fail with HTTP 400
                # Auto-cap to leave at least 512 tokens for the prompt
                capped = max(256, server_max_len - 512)
                print(f"  ⚠ max_tokens={requested_max} >= max_model_len={server_max_len}")
                print(f"     Auto-capping max_tokens to {capped} (leaving 512 for prompt)")
                REASONING_MODEL_DEFAULTS["max_tokens"] = capped
                defaults["max_tokens"] = capped
            elif requested_max > server_max_len * 0.8:
                # max_tokens eats >80% of context — risky for long prompts
                capped = max(256, server_max_len - 1024)
                print(f"  ⚠ max_tokens={requested_max} > 80% of max_model_len={server_max_len}")
                print(f"     Auto-capping max_tokens to {capped} (leaving 1024 for prompt)")
                REASONING_MODEL_DEFAULTS["max_tokens"] = capped
                defaults["max_tokens"] = capped
            else:
                print(f"  ✓ max_tokens={requested_max} fits within max_model_len={server_max_len}")
        else:
            print(f"  ⚠ Could not query server max_model_len — skipping context validation")
            print(f"     If you see HTTP 400 'context length' errors, the server's")
            print(f"     --max-model-len is too small. Restart vLLM with a larger value.")
        print()

        # Initialize results containers — seed with prior results when resuming
        results_by_cat: dict = {}
        all_results: list[dict] = []
        if resume and prior_results:
            # Deep-copy prior by_category counts so we don't mutate the loaded dict
            import copy
            results_by_cat = copy.deepcopy(prior_by_cat)
            all_results = list(prior_results)
            print(f"  Seeded results with {len(all_results)} prior test results")
            print()

        start_time = time.perf_counter()
        new_count = 0  # Track only newly-run tests for incremental save cadence

        for i, (test, suite_label) in enumerate(all_tests):
            test_id = test["id"]
            category = test.get("category", suite_label)

            # ── Fix 2: Skip already-completed tests ─────────────────────────
            if test_id in prior_completed:
                continue

            prompt = test["prompt"]
            messages = [{"role": "user", "content": prompt}]
            
            # Use test-specific max_tokens or model default
            max_tokens = test.get("max_tokens", defaults.get("max_tokens", 1024))
            temperature = test.get("temperature", defaults.get("temperature", 0.3))
            
            # ── Fix 1: Per-test context safety cap ──────────────────────────
            # Even with the pre-flight check, individual test prompts vary
            # in length. If this prompt is long, cap max_tokens further.
            if server_max_len is not None:
                prompt_tokens = _estimate_prompt_tokens(prompt)
                safe_max = server_max_len - prompt_tokens - 64  # 64-token safety margin
                if max_tokens > safe_max:
                    max_tokens = max(64, safe_max)
            
            # Build kwargs
            chat_kwargs = {"max_tokens": max_tokens, "temperature": temperature}
            if test.get("metadata", {}).get("tools"):
                chat_kwargs["tools"] = test["metadata"]["tools"]
            
            if use_completions:
                # Apply chat template client-side, then use /v1/completions
                # GPT-OSS Harmony format uses analysis/commentary/final channels
                messages = [{"role": "user", "content": prompt}]
                templated = _tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                # Stop tokens to prevent generation beyond the answer
                # No stop tokens — let the model generate the full Harmony
                # response (analysis + commentary + final channels), then parse
                # out the final channel. Stopping on <|end|> would kill generation
                # after the first (analysis) channel before reaching final.
                stop_tokens = ["<|start|>"]
                resp = await client.completion(templated, max_tokens=max_tokens,
                                               temperature=temperature,
                                               stop=stop_tokens)
                # Parse Harmony response: extract final channel content
                if resp.succeeded and resp.text:
                    resp = _parse_harmony_response(resp)
            else:
                resp = await client.chat(messages, **chat_kwargs)
            
            if resp.error:
                print(f"  ❌ ERR  {test_id:45s} [{category:14s}] {resp.error[:50]}")
                results_by_cat.setdefault(category, {"pass": 0, "fail": 0, "error": 0})
                results_by_cat[category]["error"] += 1
                all_results.append({
                    "test_id": test_id, "category": category, "suite": suite_label,
                    "status": "error", "error": resp.error[:200],
                    "elapsed": resp.elapsed,
                })
                continue
            
            tc = TestCase.from_dict(test)
            evaluator_name = test.get("evaluator", "text_contains")
            try:
                evaluator = get_evaluator(evaluator_name)
            except KeyError:
                print(f"  ⚠️ SKIP {test_id:45s} [{category:14s}] — unknown evaluator '{evaluator_name}'")
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
            print(f"  {icon} {status:4s} {test_id:45s} [{category:14s}] {resp.elapsed:5.1f}s {result.detail[:55]}")
            
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
            # Incremental save every 10 new tests
            new_count += 1
            if new_count % 10 == 0:
                _save_results(results_file, tag, endpoint, model, reasoning,
                              time.perf_counter() - start_time + prior_wall_time,
                              results_by_cat, all_results)
        
        elapsed_total = time.perf_counter() - start_time + prior_wall_time
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"RESULTS BY CATEGORY — {tag}")
    print(f"{'='*70}")
    total_pass = 0
    total_fail = 0
    total_error = 0
    for cat in sorted(results_by_cat.keys()):
        r = results_by_cat[cat]
        total = r["pass"] + r["fail"] + r["error"]
        rate = r["pass"] / total * 100 if total > 0 else 0
        print(f"  {cat:16s}: {r['pass']:3d}/{total:3d} passed ({rate:5.1f}%)  [fail={r['fail']}, err={r['error']}]")
        total_pass += r["pass"]
        total_fail += r["fail"]
        total_error += r["error"]
    
    grand_total = total_pass + total_fail + total_error
    grand_rate = total_pass / grand_total * 100 if grand_total > 0 else 0
    print(f"  {'─'*50}")
    print(f"  {'TOTAL':16s}: {total_pass:3d}/{grand_total:3d} passed ({grand_rate:5.1f}%)  [fail={total_fail}, err={total_error}]")
    print(f"  Wall time: {elapsed_total:.1f}s")
    print(f"{'='*70}")
    
    # Save results
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"stage1_{tag}_{timestamp}.json")
    
    output = {
        "tag": tag,
        "endpoint": endpoint,
        "model": model,
        "timestamp": timestamp,
        "reasoning_model": reasoning,
        "wall_time_seconds": round(elapsed_total, 1),
        "summary": {
            "total": grand_total,
            "passed": total_pass,
            "failed": total_fail,
            "error": total_error,
            "pass_rate": round(grand_rate, 1),
        },
        "by_category": results_by_cat,
        "tests": all_results,
    }
    
    with open(results_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    return output


def main():
    parser = argparse.ArgumentParser(description="Stage 1 benchmark runner")
    parser.add_argument("--endpoint", required=True, help="OpenAI-compatible endpoint URL (e.g. http://spark-56bc:8889/v1)")
    parser.add_argument("--model", required=True, help="Model name as served by the endpoint")
    parser.add_argument("--tag", required=True, help="Short tag for this model (e.g. nemotron-3-nano-30b)")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                        help=f"Per-request timeout in seconds (default: {DEFAULT_TIMEOUT:.0f}; "
                             f"auto-raised to {REASONING_TIMEOUT:.0f}s for reasoning models)")
    parser.add_argument("--use-completions", action="store_true",
                        help="Use /v1/completions instead of /v1/chat/completions (bypasses chat parser)")
    parser.add_argument("--tokenizer-path", default="",
                        help="Path to HuggingFace tokenizer dir for client-side chat template (required with --use-completions)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the most recent results file for this tag: "
                             "skip tests that already passed/failed/skip, retry errors, "
                             "merge all results into a new file")
    args = parser.parse_args()

    asyncio.run(run_benchmark(args.endpoint, args.model, args.tag, args.timeout,
                              use_completions=args.use_completions,
                              tokenizer_path=args.tokenizer_path,
                              resume=args.resume))


if __name__ == "__main__":
    main()