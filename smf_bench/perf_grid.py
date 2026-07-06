"""
perf_grid — Performance grid benchmark methodology.

Extracted from Aeon-Bench-Pod (MIT License, Copyright 2026 AEON-7).
Adapted for smf-bench: uses our api_client instead of aeon.targets.OpenAITarget.

Methodology:
  - Concurrency ladder (1, 4, 8, 16, 32) — one category at a time per level
  - Per-category isolation: "Math @ c4" means exactly 4 concurrent Math streams,
    never a mixed-category soup (which would contaminate per-category numbers)
  - Cache-busting tags: each replica gets a unique prefix to prevent vLLM prefix
    cache from inflating prefill measurements
  - Metrics per stream: TTFT (ms), decode tok/s, prefill tps (prompt_tokens / ttft_sec),
    TPOT (ms/token), e2e (ms)
  - Aggregate per cell: total output tokens / cell wall clock = aggregate decode tok/s
  - Percentiles: p50, p95 via hand-checkable linear interpolation (no numpy)

Public API:
    run_direct_grid(client, alias, conc_levels=(1,4,8,16,32), max_tokens=256,
                    temperature=0.0, repeats=1, progress_cb=None) -> grid dict
    to_results(grid) -> list of result rows
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

# ---------------------------------------------------------------- prompt sets
# Deterministic long prompts (~1500 tokens each) so prefill throughput is a
# meaningful measurement, built from pure f-strings over fixed ranges.


def _long_math():
    lines = [
        f"2026-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}  txn-{i:04d}  vendor-{i % 17:02d}"
        f"  amount: {(i * 37) % 995 + 5}.{i % 100:02d} USD  memo: recurring service charge unit {i % 9}"
        for i in range(1, 61)
    ]
    return (
        "Below is a transaction ledger.\n" + "\n".join(lines)
        + "\nHow many transactions are listed above? Reply with the number only."
    )


def _long_reasoning():
    lines = [
        f"Fact {i}: person P{i % 23} was observed in room R{(i * 7) % 12} at hour {(i * 3) % 24}"
        f" holding badge B{i % 7} and wearing tag T{(i * 5) % 31}."
        for i in range(1, 66)
    ]
    return (
        "Consider the following facts.\n" + "\n".join(lines)
        + "\nBased only on the facts above, name one room id that person P3 appears in."
        " Reply with the room id only."
    )


def _long_coding():
    chunks = [
        f"def util_{i:03d}(x):\n"
        f'    """helper {i}: scales the input by {i % 13} then offsets by {i % 7}."""\n'
        f"    return x * {i % 13} + {i % 7}\n"
        for i in range(1, 56)
    ]
    return (
        "Here is a Python module.\n```python\n" + "\n".join(chunks)
        + "```\nHow many function definitions appear in the module above?"
        " Reply with the number only."
    )


def _long_prose():
    lines = [
        f"In the {i}th hour the harbor town kept its slow watch, and lamplighter {i % 9}"
        f" counted {(i * 3) % 40 + 1} boats returning under a copper sky while bell {i % 5} tolled."
        for i in range(1, 46)
    ]
    return (
        "Read the passage below.\n" + " ".join(lines)
        + "\nSummarize the passage above in exactly one sentence."
    )


def _long_instruction():
    lines = [
        f"Rule {i}: when the input index equals {i}, respond in lowercase, keep the reply under"
        f" {(i % 9) + 3} words, and never mention the number {(i * 11) % 97}."
        for i in range(1, 56)
    ]
    return (
        "Here is a rulebook.\n" + "\n".join(lines)
        + "\nFollowing only Rule 7, write the single word ok."
    )


PROMPTS = {
    "Math": [
        "Compute 847 * 63. Reply with the number only.",
        "What is 15% of 2400? Reply with the number only.",
        "Solve for x: 3x + 11 = 47. Reply with the number only.",
        _long_math(),
    ],
    "Reasoning": [
        "If all bloops are razzies and all razzies are lazzies, are all bloops lazzies? Answer yes or no.",
        "A farmer has 17 sheep; all but 9 run away. How many are left? Reply with the number only.",
        "Which is heavier: a kilogram of steel or a kilogram of feathers? Answer in one word.",
        _long_reasoning(),
    ],
    "Coding": [
        "Write a Python function that reverses a string.",
        "Write a one-line Python list comprehension that squares the numbers 1 through 10.",
        "What does this print? print(sum(range(5))) Reply with the number only.",
        _long_coding(),
    ],
    "Prose": [
        "Write a haiku about mountains.",
        "Write one sentence describing rain on a tin roof.",
        "Give a two-sentence opening for a mystery novel set in a lighthouse.",
        _long_prose(),
    ],
    "Instruction": [
        "Reply with exactly the word PONG.",
        "List three primary colors, one per line, with no other text.",
        "Write the word echo exactly five times, separated by commas.",
        _long_instruction(),
    ],
}

CATEGORIES = list(PROMPTS.keys())

# ---------------------------------------------------------------- aggregation


def _mean(xs):
    return (sum(xs) / len(xs)) if xs else None


def _pct(xs, p):
    """Percentile with linear interpolation (hand-checkable, no numpy)."""
    if not xs:
        return None
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    k = (p / 100.0) * (len(s) - 1)
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _r(x, nd=2):
    return round(x, nd) if isinstance(x, (int, float)) else None


def _agg(reqs, wall_clock_s, n_errors=0):
    ttfts = [r["ttft_ms"] for r in reqs if r.get("ttft_ms") is not None]
    dtps = [r["decode_tps"] for r in reqs if r.get("decode_tps") is not None]
    ptps = [r["prefill_tps"] for r in reqs if r.get("prefill_tps") is not None]
    e2es = [r["e2e_ms"] for r in reqs if r.get("e2e_ms") is not None]
    tpots = [r["tpot_ms"] for r in reqs if r.get("tpot_ms") is not None]
    out_sum = sum(r.get("output_tokens") or 0 for r in reqs)
    in_sum = sum(r.get("input_tokens") or 0 for r in reqs)
    return {
        "n": len(reqs),
        "n_errors": n_errors,
        "ttft_ms_mean": _r(_mean(ttfts)),
        "ttft_ms_p50": _r(_pct(ttfts, 50)),
        "ttft_ms_p95": _r(_pct(ttfts, 95)),
        "decode_tps_mean": _r(_mean(dtps)),
        "prefill_tps_mean": _r(_mean(ptps)),
        "e2e_ms_mean": _r(_mean(e2es)),
        "tpot_ms_mean": _r(_mean(tpots), 3),
        "tpot_ms_p50": _r(_pct(tpots, 50), 3),
        "tpot_ms_p95": _r(_pct(tpots, 95), 3),
        "output_tokens_total": out_sum,
        "input_tokens_total": in_sum,
        "agg_decode_tps": _r(out_sum / wall_clock_s) if wall_clock_s and wall_clock_s > 0 else None,
    }


# ---------------------------------------------------------------- direct grid


def _one_request(client, category, prompt, temperature, max_tokens):
    """Send one request to the model via our api_client, collect perf metrics."""
    t0 = time.perf_counter()
    resp = client.chat(
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    e2e_ms = (time.perf_counter() - t0) * 1000.0

    ttft = resp.ttft_ms
    in_tok = resp.prompt_tokens
    in_est = False
    if in_tok is None:
        in_tok = max(1, len(prompt) // 4)
        in_est = True
    prefill = _r(in_tok / (ttft / 1000.0)) if (ttft and ttft > 0) else None

    out_tok = resp.completion_tokens or 0
    if e2e_ms is not None and ttft is not None and out_tok > 1 and e2e_ms > ttft:
        tpot = _r((e2e_ms - ttft) / (out_tok - 1), 3)
    elif resp.decode_tps:
        tpot = _r(1000.0 / resp.decode_tps, 3)
    else:
        tpot = None

    decode_tps = None
    if out_tok and e2e_ms and ttft is not None:
        decode_time_s = (e2e_ms - ttft) / 1000.0
        if decode_time_s > 0:
            decode_tps = _r(out_tok / decode_time_s)

    return {
        "category": category,
        "ttft_ms": _r(ttft) if ttft else None,
        "decode_tps": decode_tps,
        "prefill_tps": prefill,
        "e2e_ms": _r(e2e_ms),
        "tpot_ms": tpot,
        "output_tokens": out_tok,
        "input_tokens": in_tok,
        "input_tokens_estimated": in_est,
    }


def _bust(prompt, i):
    """Cache-busting tag: changes first tokens so every stream pays real prefill.

    Without this, vLLM's prefix cache would serve cached prefills and measure
    a fantasy prefill throughput."""
    return f"[measurement {i:04d}] {prompt}"


def run_direct_grid(
    client,
    alias: str,
    *,
    conc_levels=(1, 4, 8, 16, 32),
    max_tokens: int = 256,
    temperature: float = 0.0,
    repeats: int = 1,
    progress_cb: Optional[Callable] = None,
):
    """Direct-to-model perf grid, ONE CATEGORY AT A TIME per level.

    'Math @ c4' means exactly 4 concurrent streams of Math prompts and nothing
    else in flight — never a mixed-category soup. Each cell's aggregates use
    ITS OWN wall clock. Prompts are tiled to >= conc tasks with cache-busting
    tags so the level is actually saturated.

    Returns {kind:'direct', suite_id, alias, conc_levels, levels: {c: {overall, categories, requests, errors}}}.
    """
    grid = {
        "kind": "direct",
        "suite_id": "smf-perf-v1",
        "alias": alias,
        "conc_levels": list(conc_levels),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "repeats": repeats,
        "isolation": "per_category",
        "levels": {},
    }

    for conc in conc_levels:
        cats, all_reqs, all_errs = {}, [], []
        base_counts = {c: max(1, int(repeats)) * len(PROMPTS[c]) for c in CATEGORIES}
        total = sum(max(base_counts[c], int(conc)) for c in CATEGORIES)
        done, wall_sum = 0, 0.0

        for cat in CATEGORIES:
            base = [p for _ in range(max(1, int(repeats))) for p in PROMPTS[cat]]
            n = max(len(base), int(conc))
            tasks = [_bust(base[i % len(base)], i) for i in range(n)]
            reqs, errors = [], []
            t0 = time.perf_counter()

            with ThreadPoolExecutor(max_workers=int(conc)) as ex:
                futs = [
                    (p, ex.submit(_one_request, client, cat, p, temperature, max_tokens))
                    for p in tasks
                ]
                for p, fut in futs:
                    try:
                        reqs.append(fut.result())
                    except Exception as e:
                        errors.append(
                            {
                                "category": cat,
                                "error": f"{type(e).__name__}: {e}"[:300],
                                "prompt_head": p[:80],
                            }
                        )
                    done += 1
                    if progress_cb:
                        progress_cb(conc, done, total)

            cw = time.perf_counter() - t0
            wall_sum += cw
            cell = _agg(reqs, cw, n_errors=len(errors))
            cell["cell_wall_s"] = round(cw, 3)
            cats[cat] = cell
            all_reqs.extend(reqs)
            all_errs.extend(errors)

        grid["levels"][int(conc)] = {
            "conc": int(conc),
            "wall_clock_s": round(wall_sum, 3),
            "overall": _agg(all_reqs, wall_sum, n_errors=len(all_errs)),
            "categories": cats,
            "requests": all_reqs,
            "errors": all_errs,
        }

    return grid


def to_results(grid: dict) -> list[dict]:
    """Convert grid to flat result rows for storage/leaderboard."""
    rows = []
    for conc, level in grid["levels"].items():
        for cat, cell in level["categories"].items():
            row = {
                "suite_id": grid["suite_id"],
                "alias": grid["alias"],
                "conc": conc,
                "category": cat,
                "wall_clock_s": cell.get("cell_wall_s"),
                "n": cell["n"],
                "n_errors": cell["n_errors"],
                "ttft_ms_mean": cell["ttft_ms_mean"],
                "ttft_ms_p50": cell["ttft_ms_p50"],
                "ttft_ms_p95": cell["ttft_ms_p95"],
                "decode_tps_mean": cell["decode_tps_mean"],
                "prefill_tps_mean": cell["prefill_tps_mean"],
                "tpot_ms_mean": cell["tpot_ms_mean"],
                "agg_decode_tps": cell["agg_decode_tps"],
            }
            rows.append(row)
    return rows