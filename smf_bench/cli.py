"""
CLI entry point for smf-bench.

Usage:
  smf-bench run --model qwen3.6-35b-a3b-nvfp4 --endpoint http://spark-56bc:8888/v1
  smf-bench run --model qwen3.6-35b-a3b-nvfp4 --categories reasoning,tool_calling
  smf-bench report --run-id run_1234567890_qwen3.6-35b-a3b-nvfp4
  smf-bench compare --run-ids run_123,run_456
  smf-bench list-models
  smf-bench list-tests
  smf-bench list-runs
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from .model_registry import ModelRegistry
from .reporting import generate_markdown_report, generate_comparison_table
from .results_store import ResultsStore
from .runner import BenchRunner, RunConfig
from .test_registry import TestRegistry


def cmd_run(args: argparse.Namespace) -> int:
    """Execute a benchmark run."""
    config = RunConfig(
        model_id=args.model,
        endpoint=args.endpoint,
        api_key=args.api_key,
        suites_dir=args.suites_dir,
        models_dir=args.models_dir,
        results_db=args.results_db,
        max_concurrent=args.concurrency,
        timeout=args.timeout,
        engine=args.engine,
    )

    runner = BenchRunner(config)
    categories = args.categories.split(",") if args.categories else None
    difficulties = args.difficulty.split(",") if args.difficulty else None

    run_id = asyncio.run(runner.run(categories=categories, difficulties=difficulties, verbose=True))

    # Generate report
    run = runner.store.get_run(run_id)
    if run:
        report = generate_markdown_report(run, runner.store)
        report_path = Path(args.results_db).parent / f"{run_id}.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report)
        print(f"\nReport saved to: {report_path}")

    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Show report for a specific run."""
    store = ResultsStore(args.results_db)
    run = store.get_run(args.run_id)
    if not run:
        print(f"Run '{args.run_id}' not found")
        return 1

    report = generate_markdown_report(run, store)
    if args.output:
        Path(args.output).write_text(report)
        print(f"Report saved to: {args.output}")
    else:
        print(report)
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Compare multiple runs."""
    store = ResultsStore(args.results_db)
    run_ids = args.run_ids.split(",")
    runs = []
    for rid in run_ids:
        run = store.get_run(rid.strip())
        if run:
            runs.append(run)
        else:
            print(f"Warning: Run '{rid}' not found")

    if not runs:
        print("No valid runs to compare")
        return 1

    report = generate_comparison_table(runs, store)
    if args.output:
        Path(args.output).write_text(report)
        print(f"Comparison saved to: {args.output}")
    else:
        print(report)
    return 0


def cmd_list_models(args: argparse.Namespace) -> int:
    """List registered models."""
    registry = ModelRegistry()
    count = registry.load_dir(args.models_dir)
    print(f"Registered models ({count}):")
    for mid in registry.list_models():
        m = registry.get(mid)
        if m:
            caps = ", ".join(sorted(c.value for c in m.capabilities))
            mods_in = ", ".join(sorted(m.value for m in m.input_modalities))
            print(f"  {mid:40s}  in=[{mods_in}]  caps=[{caps}]")
    return 0


def cmd_list_tests(args: argparse.Namespace) -> int:
    """List loaded test cases."""
    registry = TestRegistry()
    count = registry.load_dir(args.suites_dir)
    print(f"Loaded {count} test(s) from {args.suites_dir}")
    print(f"Categories: {', '.join(registry.categories())}")
    print(f"Dimensions: {', '.join(registry.dimensions())}")
    if args.category:
        tests = registry.by_category(args.category)
        print(f"\nTests in '{args.category}' ({len(tests)}):")
        for t in tests:
            print(f"  {t.test_id:50s}  eval={t.evaluator:15s}  difficulty={t.difficulty:10s}  weight={t.weight}")
    else:
        for cat in registry.categories():
            tests = registry.by_category(cat)
            print(f"\n{cat} ({len(tests)}):")
            for t in tests:
                print(f"  {t.test_id:50s}  eval={t.evaluator:15s}  difficulty={t.difficulty:10s}")
    return 0


def cmd_list_runs(args: argparse.Namespace) -> int:
    """List recent runs."""
    store = ResultsStore(args.results_db)
    runs = store.list_runs(limit=args.limit)
    if not runs:
        print("No runs found")
        return 0
    print(f"{'Run ID':50s} {'Model':30s} {'Pass':>6s} {'Fail':>6s} {'N/A':>6s} {'Rate':>8s} {'When'}")
    print("-" * 120)
    for r in runs:
        applicable = r.passed + r.failed + r.errors
        rate = r.passed / applicable if applicable > 0 else 0.0
        print(f"{r.run_id:50s} {r.model_id:30s} {r.passed:6d} {r.failed:6d} {r.na_count:6d} {rate:7.1%} {r.timestamp}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="smf-bench",
        description="SMF Works unified benchmark framework",
    )
    parser.add_argument("--models-dir", default="models", help="Model registry directory")
    parser.add_argument("--suites-dir", default="suites", help="Test suites directory")
    parser.add_argument("--results-db", default="results/smf-bench.db", help="Results database path")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # run
    run_p = subparsers.add_parser("run", help="Execute benchmark suite")
    run_p.add_argument("--model", required=True, help="Model ID to test")
    run_p.add_argument("--endpoint", required=True, help="OpenAI-compatible API endpoint")
    run_p.add_argument("--api-key", default="dummy", help="API key (if needed)")
    run_p.add_argument("--engine", default="", help="Engine version (e.g. 'vLLM 0.24.0')")
    run_p.add_argument("--categories", default=None, help="Comma-separated categories to run")
    run_p.add_argument("--difficulty", default=None,
                       help="Comma-separated difficulty levels (easy,medium,hard,expert,frontier)")
    run_p.add_argument("--concurrency", type=int, default=4, help="Max concurrent requests")
    run_p.add_argument("--timeout", type=int, default=300, help="Per-test timeout (seconds)")
    run_p.set_defaults(func=cmd_run)

    # report
    report_p = subparsers.add_parser("report", help="Show report for a run")
    report_p.add_argument("--run-id", required=True, help="Run ID to report")
    report_p.add_argument("--output", default=None, help="Save report to file")
    report_p.set_defaults(func=cmd_report)

    # compare
    compare_p = subparsers.add_parser("compare", help="Compare multiple runs")
    compare_p.add_argument("--run-ids", required=True, help="Comma-separated run IDs")
    compare_p.add_argument("--output", default=None, help="Save comparison to file")
    compare_p.set_defaults(func=cmd_compare)

    # list-models
    lm_p = subparsers.add_parser("list-models", help="List registered models")
    lm_p.set_defaults(func=cmd_list_models)

    # list-tests
    lt_p = subparsers.add_parser("list-tests", help="List loaded test cases")
    lt_p.add_argument("--category", default=None, help="Filter by category")
    lt_p.set_defaults(func=cmd_list_tests)

    # list-runs
    lr_p = subparsers.add_parser("list-runs", help="List recent runs")
    lr_p.add_argument("--limit", type=int, default=20, help="Number of runs to show")
    lr_p.set_defaults(func=cmd_list_runs)

    args = parser.parse_args()
    sys.exit(args.func(args))