"""
Results Store — SQLite persistence for benchmark runs.

Every test execution is stored with:
- run_id, timestamp, model_id, endpoint, engine_version, config_hash
- test_id, category, dimension
- status: PASS, FAIL, N/A, ERROR
- score, detail, elapsed, tokens, ttft
- raw response (JSON)

This enables apples-to-apples comparison across runs, models, and configs.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    timestamp   TEXT NOT NULL,
    model_id    TEXT NOT NULL,
    endpoint    TEXT NOT NULL,
    engine      TEXT,
    config_hash TEXT,
    config_json TEXT,
    total_tests INTEGER,
    passed      INTEGER,
    failed      INTEGER,
    na_count    INTEGER,
    errors      INTEGER,
    duration_s  REAL
);

CREATE TABLE IF NOT EXISTS results (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL REFERENCES runs(run_id),
    test_id     TEXT NOT NULL,
    category    TEXT NOT NULL,
    dimension   TEXT NOT NULL,
    status      TEXT NOT NULL,  -- PASS, FAIL, N/A, ERROR
    score       REAL,
    detail      TEXT,
    elapsed     REAL,
    tokens_in   INTEGER,
    tokens_out  INTEGER,
    ttft_ms     REAL,
    text_output TEXT,
    raw_json    TEXT
);

CREATE INDEX IF NOT EXISTS idx_results_run ON results(run_id);
CREATE INDEX IF NOT EXISTS idx_results_test ON results(test_id);
CREATE INDEX IF NOT EXISTS idx_results_model ON runs(model_id);
"""


@dataclass
class RunSummary:
    run_id: str
    timestamp: str
    model_id: str
    endpoint: str
    engine: str
    config_hash: str
    total_tests: int
    passed: int
    failed: int
    na_count: int
    errors: int
    duration_s: float


class ResultsStore:
    """SQLite-backed results storage."""

    def __init__(self, db_path: str | Path = "results/smf-bench.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(SCHEMA)

    def _config_hash(self, config: dict) -> str:
        return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]

    def create_run(
        self,
        model_id: str,
        endpoint: str,
        engine: str = "",
        config: dict | None = None,
    ) -> str:
        """Create a new run record and return the run_id."""
        run_id = f"run_{int(time.time())}_{model_id.replace('/', '_')}"
        config = config or {}
        config_hash = self._config_hash(config)
        config_json = json.dumps(config)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO runs (run_id, timestamp, model_id, endpoint, engine,
                   config_hash, config_json, total_tests, passed, failed, na_count, errors, duration_s)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0, 0, 0, 0, 0)""",
                (run_id, time.strftime("%Y-%m-%dT%H:%M:%S"), model_id, endpoint,
                 engine, config_hash, config_json),
            )
        return run_id

    def store_result(
        self,
        run_id: str,
        test_id: str,
        category: str,
        dimension: str,
        status: str,
        score: float = 0.0,
        detail: str = "",
        elapsed: float = 0.0,
        tokens_in: int = 0,
        tokens_out: int = 0,
        ttft_ms: float | None = None,
        text_output: str = "",
        raw: dict | None = None,
    ) -> None:
        """Store a single test result."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO results
                   (run_id, test_id, category, dimension, status, score, detail,
                    elapsed, tokens_in, tokens_out, ttft_ms, text_output, raw_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (run_id, test_id, category, dimension, status, score, detail,
                 elapsed, tokens_in, tokens_out, ttft_ms, text_output[:2000],
                 json.dumps(raw) if raw else None),
            )
            # Update run counters
            if status == "PASS":
                conn.execute("UPDATE runs SET passed = passed + 1 WHERE run_id = ?", (run_id,))
            elif status == "FAIL":
                conn.execute("UPDATE runs SET failed = failed + 1 WHERE run_id = ?", (run_id,))
            elif status == "N/A":
                conn.execute("UPDATE runs SET na_count = na_count + 1 WHERE run_id = ?", (run_id,))
            elif status == "ERROR":
                conn.execute("UPDATE runs SET errors = errors + 1 WHERE run_id = ?", (run_id,))
            conn.execute("UPDATE runs SET total_tests = total_tests + 1 WHERE run_id = ?", (run_id,))

    def finalize_run(self, run_id: str, duration_s: float) -> None:
        """Set the total duration for a completed run."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE runs SET duration_s = ? WHERE run_id = ?",
                (duration_s, run_id),
            )

    def get_run(self, run_id: str) -> RunSummary | None:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            if row:
                return RunSummary(**dict(row))
        return None

    def list_runs(self, limit: int = 20) -> list[RunSummary]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
            return [RunSummary(**dict(r)) for r in rows]

    def get_results(self, run_id: str) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM results WHERE run_id = ? ORDER BY id", (run_id,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_results_by_category(self, run_id: str) -> dict[str, dict]:
        """Aggregate results by category: {category: {pass, fail, na, error, total}}"""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """SELECT category, status, COUNT(*) as count
                   FROM results WHERE run_id = ?
                   GROUP BY category, status""",
                (run_id,),
            ).fetchall()
        result: dict[str, dict] = {}
        for cat, status, count in rows:
            if cat not in result:
                result[cat] = {"PASS": 0, "FAIL": 0, "N/A": 0, "ERROR": 0, "total": 0}
            result[cat][status] = count
            result[cat]["total"] += count
        return result

    def compare_runs(self, run_ids: list[str]) -> dict:
        """Compare multiple runs side by side."""
        comparison = {}
        for run_id in run_ids:
            run = self.get_run(run_id)
            if not run:
                continue
            cats = self.get_results_by_category(run_id)
            comparison[run_id] = {
                "model_id": run.model_id,
                "timestamp": run.timestamp,
                "summary": {
                    "total": run.total_tests,
                    "passed": run.passed,
                    "failed": run.failed,
                    "na": run.na_count,
                    "errors": run.errors,
                    "pass_rate": run.passed / max(run.passed + run.failed, 1),
                    "duration_s": run.duration_s,
                },
                "categories": cats,
            }
        return comparison