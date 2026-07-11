# smf-bench

> A unified, capability-gated benchmark framework for evaluating LLMs and multimodal models across a consistent test suite.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

## Why smf-bench?

Most LLM benchmarks are either (a) tied to a specific evaluation harness that evolves independently of your needs, or (b) single-purpose test suites that don't cover multimodal, agentic, or performance dimensions. **smf-bench** solves this by providing one framework you own, with one consistent test battery that every model runs — getting `N/A` on capabilities it lacks rather than a zero or a skip.

### Design Principles

1. **One consistent suite across all models** — every model runs the same comprehensive test battery
2. **Capability-gated testing** — models get `N/A` on tests they can't perform, not a zero or skip
3. **You own the standard** — not dependent on any external project's evolving suite
4. **Pluggable evaluators** — text matching, regex, keyword threshold, tool-call validation, programmatic checks, LLM judging
5. **Reproducible** — every run tagged by model, engine version, config hash, and timestamp in SQLite


## Canonical tool-calling score (SMF Eval Doctrine)

**smf-bench is not the canonical tool-calling benchmark for SMF Works.**

| Role | Suite |
|------|--------|
| **Tool-calling quality (canonical)** | [tool-eval-bench](https://github.com/MiaAI-Lab/tool-eval-bench) — **69** multi-turn scenarios (+ hard mode), safety gate, Pass@k |
| **Pinned SHA** | `8eca976167dfe925c125edd5a289433e78ee54e0` |
| **Doctrine** | `AionaVault/Research/evaluation/SMF-EVAL-DOCTRINE.md` (internal) |
| **Research** | `AionaVault/Research/evaluation/2026-07-10-smf-bench-and-eval-frameworks-research.md` |

The in-repo `suites/quality/tool_calling/` suite (**2** tests: weather + calculator) is a **smoke / capability probe only**. Do **not** use it to claim “strong tool use” in reports, blogs, or model cards.

### Quick TEB recipe (OpenAI-compatible endpoint)

```bash
# Install pinned tool-eval-bench
uv tool install git+https://github.com/MiaAI-Lab/tool-eval-bench.git@8eca976167dfe925c125edd5a289433e78ee54e0

# Full tool quality (canonical)
tool-eval-bench --base-url "$ENDPOINT" --model "$MODEL" --seed 42 --json-file teb-full.json

# Accuracy plugins (optional same day)
tool-eval-bench --base-url "$ENDPOINT" --model "$MODEL" --skip-tool-eval \
  --gsm8k --ifeval --mmlu --gsm8k-limit 200 --mmlu-limit 500 --json-file teb-acc.json
```

**smf-bench owns:** capability-gated multimodal, performance grid (TTFT/concurrency/context), SMF custom quality suites, and Aeon-style agentic file/shell tasks.  
**tool-eval-bench owns:** serious tool-call quality scoring for serving stacks (vLLM, SGLang, llama.cpp, LiteLLM).

Phase 2 (optional): ingest TEB JSON into smf-bench SQLite via an external adapter so one leaderboard shows both.

## Architecture

```
smf-bench/
├── smf_bench/
│   ├── model_registry.py     # Capability manifests + gating logic
│   ├── test_registry.py       # Test loader, capability matching, N/A handling
│   ├── api_client.py          # OpenAI-compatible client (vLLM, Ollama, TGI, ...)
│   ├── evaluators/__init__.py # Pluggable scoring (regex, keyword, tool-call, ...)
│   ├── results_store.py       # SQLite persistence
│   ├── reporting.py           # Leaderboard, comparison tables, N/A-aware scoring
│   ├── runner.py              # Main orchestrator, parallel execution
│   ├── cli.py                 # CLI entry point
│   ├── perf_grid.py           # Concurrency-ladder performance methodology
│   └── adapters/__init__.py   # Agentic task adapter pattern
├── suites/                    # YAML test cases (239 tests, 15 categories)
│   ├── quality/               # Reasoning, coding, math, writing, agentic, tool calling
│   ├── multimodal/            # Vision (20), video (17), audio (3)
│   └── performance/           # Latency, TTFT, concurrency, context scaling
├── models/                    # Model manifests (YAML)
├── results/                   # SQLite database (gitignored)
└── smoke_test.py              # Quick validation script
```

## Test Suite (239 tests, 15 categories)

| Category | Source | Tests |
|----------|--------|------:|
| Reasoning | SMF Works | 8 |
| Coding | SMF Works | 30 |
| Math | SMF Works | 30 |
| Instruction Following | SMF Works | 30 |
| Prose Quality | SMF Works | 30 |
| Writing | SMF Works | 5 |
| Tool Calling | SMF Works | 2 (**smoke only** — use tool-eval-bench for canonical tool score) |
| Agentic Tasks | Extracted from Aeon-Bench-Pod (MIT) | 16 |
| Tier-0 Deterministic | Extracted from Aeon-Bench-Pod (MIT) | 150 |
| Vision Understanding | SMF Works | 20 |
| Video Understanding | SMF Works | 17 |
| Audio Understanding | SMF Works | 3 |
| Latency & Throughput | SMF Works | 5 |
| TTFT (Time to First Token) | SMF Works | 3 |
| Concurrency | SMF Works + Aeon-Bench-Pod methodology | 4 |
| Context Length Scaling | SMF Works | 6 |

## Quick Start

```bash
# Clone
git clone https://github.com/smfworks/smf-bench.git
cd smf-bench

# Install
pip install -e .

# Verify setup
python3 smoke_test.py

# Run a benchmark
python3 run_direct.py          # Quick: 8 reasoning tests
python3 run_multi.py           # Multi-category: reasoning + tool_calling + writing

# Or use the CLI
smf-bench run \
  --model models/qwen3.6-35b-a3b-nvfp4.yaml \
  --endpoint http://localhost:8888/v1 \
  --categories reasoning,tool_calling,writing
```

## Model Manifests

Each model declares its capabilities in a YAML manifest. The framework uses this to gate tests — a text-only model gets `N/A` on vision/video/audio tests rather than failing them.

```yaml
model_id: "qwen3.6-35b-a3b-nvfp4"
name: "Qwen3.6-35B-A3B-NVFP4"
provider: "qwen"
served_name: "nvidia/Qwen3.6-35B-A3B-NVFP4"
input_modalities: [text]
output_modalities: [text]
capabilities:
  - reasoning
  - coding
  - writing
  - tool_calling
  - context_long
  - streaming
context_length: 262144
```

## Evaluators

| Evaluator | Description |
|-----------|-------------|
| `text_contains` | Substring match (supports single string, list, or keyword threshold with `min_matches`) |
| `text_match` | Exact match (trimmed, case-insensitive) |
| `regex_match` | Regex pattern match |
| `tool_call` | Validates tool name + arguments in API response |
| `json_contains` | Parse JSON output, check for key/value |
| `code_compiles` | Check that Python code compiles |
| `html_valid` | Check that HTML has required tags |
| `performance` | Latency, TTFT, throughput metrics |

## Results & Reporting

Every run is stored in SQLite with:
- Run metadata (model, endpoint, engine version, config hash, timestamp)
- Per-test results (status, score, elapsed, tokens, TTFT, raw response)
- N/A-aware scoring (N/A tests excluded from pass rate calculation)

```sql
SELECT test_id, status, score, elapsed
FROM results
WHERE run_id = 'run_1783300074_qwen3.6-35b-a3b-nvfp4'
ORDER BY category;
```

## Attribution

smf-bench extracts and adapts components from [Aeon-Bench-Pod](https://github.com/AEON-7/Aeon-Bench-Pod) (MIT License, Copyright 2026 AEON-7):

- **150 Tier-0 deterministic test cases** — ported to YAML
- **16 agentic tasks** (games, animations, apps) — ported to YAML
- **Performance grid methodology** (`perf_grid.py`) — concurrency ladder, cache-busting, TTFT/decode/TPOT metrics
- **Adapter pattern** (`adapters/__init__.py`) — abstract agentic task execution

These components retain their MIT attribution in source headers.

## License

MIT — SMF Works. See [LICENSE](LICENSE) for details.