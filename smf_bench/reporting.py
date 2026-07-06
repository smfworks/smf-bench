"""
Reporting — N/A-aware scoring, comparison tables, and output formatting.

Generates:
- Per-run summary with pass/fail/NA breakdown by category
- Multi-model comparison tables
- Performance metrics tables (throughput, TTFT, latency)
- Markdown reports
- Console output via rich
"""

from __future__ import annotations

from typing import Any

from .results_store import ResultsStore, RunSummary


def format_run_summary(run: RunSummary, store: ResultsStore) -> dict:
    """Build a summary dict for a single run."""
    cats = store.get_results_by_category(run.run_id)
    applicable = run.passed + run.failed + run.errors
    pass_rate = run.passed / applicable if applicable > 0 else 0.0

    return {
        "run_id": run.run_id,
        "model_id": run.model_id,
        "timestamp": run.timestamp,
        "endpoint": run.endpoint,
        "engine": run.engine,
        "total_tests": run.total_tests,
        "passed": run.passed,
        "failed": run.failed,
        "na": run.na_count,
        "errors": run.errors,
        "applicable": applicable,
        "pass_rate": pass_rate,
        "duration_s": round(run.duration_s, 1),
        "categories": cats,
    }


def generate_markdown_report(run: RunSummary, store: ResultsStore) -> str:
    """Generate a full Markdown report for a single run."""
    summary = format_run_summary(run, store)
    lines = [
        f"# smf-bench Report: {run.model_id}",
        "",
        f"**Run ID:** {run.run_id}  ",
        f"**Timestamp:** {run.timestamp}  ",
        f"**Endpoint:** {run.endpoint}  ",
        f"**Engine:** {run.engine}  ",
        f"**Duration:** {run.duration_s:.1f}s  ",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total Tests | {summary['total_tests']} |",
        f"| Applicable | {summary['applicable']} |",
        f"| Passed | {summary['passed']} |",
        f"| Failed | {summary['failed']} |",
        f"| N/A | {summary['na']} |",
        f"| Errors | {summary['errors']} |",
        f"| **Pass Rate (applicable only)** | **{summary['pass_rate']:.1%}** |",
        "",
        "## By Category",
        "",
        f"| Category | Pass | Fail | N/A | Error | Total | Pass Rate |",
        f"|----------|:----:|:----:|:---:|:-----:|:-----:|:---------:|",
    ]

    for cat in sorted(summary["categories"].keys()):
        c = summary["categories"][cat]
        applicable_cat = c["PASS"] + c["FAIL"] + c["ERROR"]
        rate = c["PASS"] / applicable_cat if applicable_cat > 0 else 0.0
        lines.append(
            f"| {cat} | {c['PASS']} | {c['FAIL']} | {c['N/A']} | {c['ERROR']} | {c['total']} | {rate:.1%} |"
        )

    lines.append("")
    lines.append("## Detailed Results")
    lines.append("")
    results = store.get_results(run.run_id)
    lines.append(f"| Test | Category | Status | Score | Time | Detail |")
    lines.append(f"|------|----------|--------|:-----:|:----:|--------|")
    for r in results:
        status_icon = {"PASS": "✅", "FAIL": "❌", "N/A": "⬜", "ERROR": "⚠️"}.get(r["status"], "?")
        detail = (r["detail"] or "")[:80]
        lines.append(
            f"| {r['test_id']} | {r['category']} | {status_icon} {r['status']} | {r['score']:.2f} | {r['elapsed']:.1f}s | {detail} |"
        )

    return "\n".join(lines)


def generate_comparison_table(runs: list[RunSummary], store: ResultsStore) -> str:
    """Generate a side-by-side comparison table for multiple runs."""
    summaries = [format_run_summary(r, store) for r in runs]

    lines = ["# smf-bench Comparison", ""]

    # Summary table
    header = "| Metric | " + " | ".join(s["model_id"] for s in summaries) + " |"
    sep = "|--------|" + "|".join(["--------"] * len(summaries)) + "|"
    lines.append(header)
    lines.append(sep)

    def row(label: str, key: str, fmt: str = "d") -> str:
        vals = []
        for s in summaries:
            v = s[key]
            if fmt == "d":
                vals.append(str(v))
            elif fmt == "pct":
                vals.append(f"{v:.1%}")
            elif fmt == "f1":
                vals.append(f"{v:.1f}")
        return f"| {label} | " + " | ".join(vals) + " |"

    lines.append(row("Total Tests", "total_tests"))
    lines.append(row("Applicable", "applicable"))
    lines.append(row("Passed", "passed"))
    lines.append(row("Failed", "failed"))
    lines.append(row("N/A", "na"))
    lines.append(row("Errors", "errors"))
    lines.append(row("Pass Rate", "pass_rate", "pct"))
    lines.append(row("Duration (s)", "duration_s", "f1"))

    # Per-category comparison
    all_cats = sorted({c for s in summaries for c in s["categories"]})
    if all_cats:
        lines.append("")
        lines.append("## By Category")
        lines.append("")
        cat_header = "| Category | " + " | ".join(s["model_id"] for s in summaries) + " |"
        cat_sep = "|----------|" + "|".join(["--------"] * len(summaries)) + "|"
        lines.append(cat_header)
        lines.append(cat_sep)

        for cat in all_cats:
            vals = []
            for s in summaries:
                c = s["categories"].get(cat, {})
                p = c.get("PASS", 0)
                total = c.get("total", 0)
                na = c.get("N/A", 0)
                applicable = total - na
                rate = p / applicable if applicable > 0 else 0.0
                if na == total and total > 0:
                    vals.append("N/A")
                else:
                    vals.append(f"{p}/{applicable} ({rate:.0%})")
            lines.append(f"| {cat} | " + " | ".join(vals) + " |")

    lines.append("")
    return "\n".join(lines)