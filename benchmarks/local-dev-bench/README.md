# local-dev-bench

Developer-usability harness for local coding models, first run as
[EXP-2026-11-04-ripple-65b-feasibility](https://www.smfclearinghouse.com/blog/ripple-65b-on-the-amd-box-honest-scoreboard/)
(Ripple-65B Preview vs Qwen3-Coder-30B-A3B on an AMD Radeon 8060S box, llama.cpp Vulkan).

Unlike the main smf-bench suites (broad capability batteries), this one is small
and focused: the tasks a developer hands a local model in a real afternoon,
scored by pytest pass/fail rather than opinions. It exists so the same ruler
can be run on any hardware — a 92 GB Strix-class AIO, a Mac Studio, a 4090 —
and the numbers compared.

## What's in it

| Script | Measures |
|---|---|
| `harness/local_dev_harness.py` | A. 3 planted-bug fixes (stats variance, LRU eviction/KeyError, consecutive dedup) — each with a deterministic test. B. 1 feature-from-spec (ring buffer). C. 10 tool-call fidelity prompts (broken-JSON rate). Writes `results_<tag>.json`. |
| `harness/context_needle.py` | ~118k-token synthetic repo, 4 facts buried at controlled depths, 4 needle questions + 2 no-answer controls (correct answer: "NOT IN EXCERPT"). |

Both run against any OpenAI-compatible endpoint (llama.cpp, vLLM, SGLang, Ollama):

```bash
# serve your model, then:
python harness/local_dev_harness.py --endpoint http://localhost:8484 --model ripple --tag my-model
python harness/context_needle.py --endpoint http://localhost:8484 --tag my-model
```

Requirements: Python 3.10+, `pytest` on the path (the harness shells out to it),
and a served model. Greedy decoding throughout (temperature 0).

## Validating the ruler

Before trusting any run, verify the harness measures in both directions:

1. Ship the reference implementations of each task — all 4 test suites must PASS.
2. Ship the planted-bug versions — all 3 must FAIL.

This was done before the published run (see PROGRESS log below). If your endpoint
mangles code blocks or tool-call JSON, the ruler will show it as task failures —
check `test_tail` and `detail` fields in the results file before quoting scores.

## Published results (2026-11-05)

GMKtec EVO-X2 (Ryzen AI MAX+ 395, Radeon 8060S, 92 GB unified memory), llama.cpp
with the community `ricursive` architecture port (commit `9e2a04a` + AtomicChat
patch), Vulkan backend, Q4_K_M quants, greedy decoding, server-measured timings.

| | Ripple-65B-Preview (Q4_K_M, 38.8 GB) | Qwen3-Coder-30B-A3B (Q4_K_M, 18.6 GB) |
|---|---|---|
| Coding tasks | 4/4 | 4/4 |
| Clean tool calls | 10/10 (0 broken JSON) | 10/10 (0 broken JSON) |
| 118k-context needle | 4/4 + 2/2 refusals | 4/4 + 2/2 refusals |
| Generation | 5.7 tok/s | 92 tok/s |
| Cold prefill, 124k tokens | 23.0 min (89.7 tok/s mean) | 12.0 min (175.9 tok/s) |

Verdict: quality tie on every measured axis; the MoE control was ~16x faster at
generation and ~1.9x faster at cold prefill, at half the disk. Full narrative in
[the blog post](https://www.smfclearinghouse.com/blog/ripple-65b-on-the-amd-box-honest-scoreboard/).

Raw artifacts in this directory:

- `results/results_ripple-65b-q4km.json` — task + tool-call records (per-task completion tokens, wall time, reasoning-flag)
- `results/results_qwen3-coder-30b-a3b-q4km.json` — same, control arm
- `results/results_ctx_ripple-65b-q4km.json`, `results/results_ctx_qwen3-coder-30b-a3b-q4km.json` — long-context arm
- `logs/` — server timing logs (the ground-truth prefill lines live here), harness run logs

## Notes for cross-hardware runs

- Serve with `--jinja` on llama.cpp so tool-call templates render.
- One model served at a time; thermal state affects dense-model prefill more than
  token-generation speed.
- If your run reports an implausibly fast long-context read, check for slot-cache
  warmth from a prior attempt — cold-start the server before quoting prefill numbers.
- DFlash2 speculative decoding (Ripple's draft model) has no GGUF; it was not
  testable in the published run. If a GGUF draft ships, expect the generation gap
  to narrow — rerun then.

## Provenance

Extracted from SMF Works internal experiment EXP-2026-11-04-ripple-65b-feasibility
(2026-11-04/05). Same license as smf-bench (MIT).
