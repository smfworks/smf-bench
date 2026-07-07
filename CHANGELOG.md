# smf-bench Changelog

## [Unreleased] — 2026-07-07

### Framework Fixes: Robustness Improvements

Three fixes addressing root cause of failed/partial benchmark runs. All three
were identified during root cause analysis of 5 failed GPT-OSS-120B runs and
1 crashed Nemotron-3-Super-120B run.

#### Fix 1: Context Validation (Pre-flight + Per-test)

**Root cause:** `run_stage1.py` sent `max_tokens=4096` for reasoning models
without checking the server's `max_model_len`. When vLLM was started with
`--max-model-len 4096`, every request exceeded the context window → HTTP 400
errors on 166/181 tests (92% failure rate).

**Changes:**
- `smf_bench/api_client.py`: Added `get_max_model_len()` method to `APIClient`.
  Queries `/v1/models` and extracts `max_model_len` from the server's response.
  Works with vLLM's top-level format and nested config/parameters formats.
- `run_stage1.py`: Added pre-flight context validation after health check.
  If `max_tokens >= max_model_len`, auto-caps to leave 512 tokens for prompts.
  If `max_tokens > 80% of max_model_len`, auto-caps to leave 1024 tokens.
  Prints a clear diagnostic in all cases (capped, fits, or query failed).
- `run_stage1.py`: Added per-test context safety cap. For each test, estimates
  prompt token count (~4 chars/token) and caps `max_tokens` to
  `max_model_len - prompt_tokens - 64` if the prompt is long enough to
  overflow with the requested `max_tokens`.

**Files modified:**
- `smf_bench/api_client.py` — new method `get_max_model_len()` (21 lines added)
- `run_stage1.py` — pre-flight check block (30 lines) + per-test cap (6 lines)

#### Fix 2: Resume Capability (--resume flag)

**Root cause:** No resume capability. If a run was interrupted (Ctrl-C, SSH
session dropped, server crash), the entire 181-test suite had to restart from
test 0. This caused 3 partial GPT-OSS runs (60, 90, and 160 tests) where the
completed work was lost. Manual band-aid scripts `run_errored.py` and
`run_remaining.py` were created to work around this, but they hardcoded
model-specific test IDs and suite lists.

**Changes:**
- `run_stage1.py`: Added `--resume` CLI flag.
- `run_stage1.py`: Added helper functions:
  - `_find_latest_results(results_dir, tag)` — finds most recent results file
  - `_load_completed_test_ids(filepath)` — extracts test IDs with pass/fail/skip
    status (error tests are NOT skipped — they get retried)
  - `_load_prior_results(filepath)` — loads tests, by_category, and wall_time
- `run_stage1.py`: When `--resume` is set, loads the most recent results file
  for the given tag, seeds `all_results` and `results_by_cat` with prior data,
  and skips already-completed tests in the main loop.
- `run_stage1.py`: Incremental save now counts only newly-run tests (`new_count`
  instead of `len(all_results)`) so the save cadence isn't affected by the
  prior results seed.
- `run_stage1.py`: Wall time accumulates prior + new run time for accurate
  total wall time reporting.

**Files modified:**
- `run_stage1.py` — helper functions (35 lines) + resume logic in run_benchmark
  (25 lines) + main loop skip check (2 lines) + save counter fix (4 lines)

#### Fix 3: Timeout Auto-tuning

**Root cause:** Default per-request timeout was 120s for all models. Reasoning
models generating 4096 tokens routinely take 90–120s. The GPT-OSS 18:33 run
showed 24/60 tests (40%) taking ≥110s, with math tests averaging 98.6s and only
7/30 passing (23%) — reasoning chains were cut off before reaching the answer.

**Changes:**
- `run_stage1.py`: Added constants `REASONING_TIMEOUT = 300.0` and
  `DEFAULT_TIMEOUT = 120.0`.
- `run_stage1.py`: If the model is detected as a reasoning model (via existing
  `is_reasoning_model()`) AND the user didn't explicitly set a timeout above the
  default, auto-raises the per-request timeout to 300s.
- `run_stage1.py`: Updated `--timeout` help text to document the auto-raise
  behavior.

**Files modified:**
- `run_stage1.py` — constants (4 lines) + auto-tuning logic (6 lines) + help text

### Verification

All three fixes verified against the live GPT-OSS-120B server on spark-56bc:
- `get_max_model_len()` returns 16384 from the running vLLM container ✅
- Pre-flight capping logic correctly identifies 4096 < 16384 (no cap needed) ✅
- Per-test cap correctly reduces max_tokens for long prompts ✅
- Resume logic: 60 completed tests loaded, 121 remaining, 60+121=181 ✅
- Timeout auto-tuning: 120s → 300s for gpt-oss model ✅
- Both files pass `ast.parse()` syntax check ✅
- `--help` output shows new `--resume` flag and updated timeout description ✅

### Deprecated Workarounds

`run_errored.py` and `run_remaining.py` are now superseded by the `--resume`
flag and can be safely deleted in a future cleanup.