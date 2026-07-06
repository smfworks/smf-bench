"""
Runner — main orchestrator for smf-bench.

Responsibilities:
1. Load model registry + test registry
2. Partition tests into applicable / N/A based on model capabilities
3. Execute applicable tests against the endpoint
4. Record N/A for inapplicable tests
5. Store all results in SQLite
6. Generate report

Execution modes:
- sequential: one test at a time (default for quality tests)
- parallel: concurrent requests (for performance/concurrency tests)
- streaming: use streaming API (for TTFT measurement)
"""

from __future__ import annotations

import asyncio
import functools
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Force unbuffered print for real-time progress in long runs
_print = print
print = functools.partial(_print, flush=True)

from .api_client import APIClient, APIResponse
from .evaluators import get_evaluator, EvalResult
from .model_registry import ModelRegistry, ModelManifest
from .results_store import ResultsStore
from .test_registry import TestRegistry, TestCase


@dataclass
class RunConfig:
    """Configuration for a benchmark run."""
    model_id: str
    endpoint: str
    api_key: str = "dummy"
    suites_dir: str = "suites"
    models_dir: str = "models"
    results_db: str = "results/smf-bench.db"
    max_concurrent: int = 4
    timeout: int = 300
    engine: str = ""               # e.g. "vLLM 0.24.0", "Ollama 0.5.0"
    config: dict = field(default_factory=dict)  # extra config for hashing


class BenchRunner:
    """Main benchmark runner — orchestrates test execution and result storage."""

    def __init__(self, config: RunConfig) -> None:
        self.config = config
        self.models = ModelRegistry()
        self.tests = TestRegistry()
        self.store = ResultsStore(config.results_db)
        self._stop = False

    def load(self) -> tuple[int, int]:
        """Load model manifests and test cases. Returns (model_count, test_count)."""
        mc = self.models.load_dir(self.config.models_dir)
        tc = self.tests.load_dir(self.config.suites_dir)
        return mc, tc

    async def run(self, categories: list[str] | None = None, verbose: bool = False) -> str:
        """Execute the benchmark suite.

        Args:
            categories: If provided, only run tests in these categories.
                        If None, run all tests.
            verbose: Print per-test results to console.

        Returns:
            run_id of the completed run.
        """
        mc, tc = self.load()
        if verbose:
            print(f"Loaded {mc} model(s), {tc} test(s)")

        model = self.models.get(self.config.model_id)
        if not model:
            raise KeyError(
                f"Model '{self.config.model_id}' not found in {self.config.models_dir}. "
                f"Available: {self.models.list_models()}"
            )

        # Filter by category if requested
        all_tests = self.tests.all_tests()
        if categories:
            all_tests = [t for t in all_tests if t.category in categories]

        # Partition into applicable / N/A
        applicable, not_applicable = self._partition(model, all_tests)

        if verbose:
            print(f"\nModel: {model.name} ({model.model_id})")
            print(f"  Applicable tests: {len(applicable)}")
            print(f"  N/A tests:        {len(not_applicable)}")
            print(f"  Endpoint: {self.config.endpoint}")
            print()

        # Create run record
        run_id = self.store.create_run(
            model_id=model.model_id,
            endpoint=self.config.endpoint,
            engine=self.config.engine,
            config={
                "model": model.model_id,
                "context_length": model.context_length,
                "max_output_tokens": model.max_output_tokens,
                "input_modalities": [m.value for m in model.input_modalities],
                "output_modalities": [m.value for m in model.output_modalities],
                "capabilities": [c.value for c in model.capabilities],
                **self.config.config,
            },
        )

        start_time = time.perf_counter()

        # Record N/A tests first (instant, no API calls)
        for test in not_applicable:
            self.store.store_result(
                run_id=run_id,
                test_id=test.test_id,
                category=test.category,
                dimension=test.dimension,
                status="N/A",
                score=0.0,
                detail=f"Model lacks required capabilities for this test",
                elapsed=0.0,
            )
            if verbose:
                print(f"  ⬜ N/A  {test.test_id:40s} [{test.category}]")

        # Execute applicable tests
        sem = asyncio.Semaphore(self.config.max_concurrent)

        async def run_one(client: APIClient, test: TestCase) -> None:
            async with sem:
                await self._execute_test(client, test, run_id, verbose)

        async with APIClient(
            base_url=self.config.endpoint,
            model=model.api_model_name,
            api_key=self.config.api_key,
            timeout=float(self.config.timeout),
        ) as client:
            # Health check
            healthy = await client.health_check()
            if not healthy:
                print(f"ERROR: Endpoint {self.config.endpoint} is not reachable")
                self.store.finalize_run(run_id, time.perf_counter() - start_time)
                return run_id

            if verbose:
                print(f"  Endpoint healthy ✓\n")

            # Run tests with controlled concurrency
            tasks = [asyncio.create_task(run_one(client, t)) for t in applicable]
            await asyncio.gather(*tasks)

        duration = time.perf_counter() - start_time
        self.store.finalize_run(run_id, duration)

        if verbose:
            run = self.store.get_run(run_id)
            if run:
                applicable_count = run.passed + run.failed + run.errors
                rate = run.passed / applicable_count if applicable_count > 0 else 0.0
                print(f"\n{'='*60}")
                print(f"Run complete: {run_id}")
                print(f"  Total:    {run.total_tests}")
                print(f"  Passed:   {run.passed}")
                print(f"  Failed:   {run.failed}")
                print(f"  N/A:      {run.na_count}")
                print(f"  Errors:   {run.errors}")
                print(f"  Pass Rate (applicable only): {rate:.1%}")
                print(f"  Duration: {duration:.1f}s")
                print(f"{'='*60}")

        return run_id

    def _partition(
        self, model: ModelManifest, tests: list[TestCase]
    ) -> tuple[list[TestCase], list[TestCase]]:
        """Split tests into (applicable, not_applicable) based on model capabilities."""
        applicable = []
        not_applicable = []
        for test in tests:
            if model.can_run(test.required_modalities, test.required_capabilities):
                applicable.append(test)
            else:
                not_applicable.append(test)
        return applicable, not_applicable

    async def _execute_test(
        self,
        client: APIClient,
        test: TestCase,
        run_id: str,
        verbose: bool,
    ) -> None:
        """Execute a single test case and store the result."""
        try:
            # Build the message payload
            messages = self._build_messages(test)

            # Determine if we need streaming (for TTFT tests)
            stream = test.category in ("ttft", "concurrency") or test.metadata.get("stream", False)

            # Call the API
            kwargs: dict[str, Any] = {
                "max_tokens": test.max_tokens,
                "temperature": test.temperature,
            }
            if test.metadata.get("tools"):
                kwargs["tools"] = test.metadata["tools"]

            resp = await client.chat(messages, stream=stream, **kwargs)

            if resp.error:
                self.store.store_result(
                    run_id, test.test_id, test.category, test.dimension,
                    status="ERROR", score=0.0, detail=resp.error,
                    elapsed=resp.elapsed, text_output=resp.text[:500],
                )
                if verbose:
                    print(f"  ⚠️ ERR  {test.test_id:40s} [{test.category}] {resp.error[:60]}")
                return

            # Evaluate the response
            evaluator = get_evaluator(test.evaluator)
            result = evaluator(resp, test)

            status = "PASS" if result.passed else "FAIL"
            tokens_in = resp.usage.get("prompt_tokens", 0)
            tokens_out = resp.usage.get("completion_tokens", 0)
            ttft_ms = (resp.ttft * 1000) if resp.ttft else None

            self.store.store_result(
                run_id, test.test_id, test.category, test.dimension,
                status=status, score=result.score, detail=result.detail,
                elapsed=resp.elapsed, tokens_in=tokens_in, tokens_out=tokens_out,
                ttft_ms=ttft_ms, text_output=resp.text[:2000],
                raw={"tool_calls": resp.tool_calls, "usage": resp.usage},
            )

            if verbose:
                icon = "✅" if result.passed else "❌"
                print(f"  {icon} {status:4s} {test.test_id:40s} [{test.category}] {resp.elapsed:.1f}s {result.detail[:50]}")

        except Exception as e:
            self.store.store_result(
                run_id, test.test_id, test.category, test.dimension,
                status="ERROR", score=0.0, detail=f"Exception: {e}",
                elapsed=0.0,
            )
            if verbose:
                print(f"  ⚠️ ERR  {test.test_id:40s} [{test.category}] {e}")

    def _build_messages(self, test: TestCase) -> list[dict]:
        """Build the chat messages for a test case.

        If test.messages is populated (multimodal), use those directly.
        Otherwise, build a simple user message from test.prompt.
        """
        if test.messages:
            return test.messages
        return [{"role": "user", "content": test.prompt}]

    def stop(self) -> None:
        """Signal the runner to stop after current tests."""
        self._stop = True