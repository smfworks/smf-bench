"""smf-bench — SMF Works unified benchmark framework."""

__version__ = "0.1.0"

from .model_registry import ModelRegistry, ModelManifest, Capability, Modality
from .test_registry import TestRegistry, TestCase
from .api_client import APIClient, APIResponse
from .results_store import ResultsStore
from .runner import BenchRunner, RunConfig
from .reporting import generate_markdown_report, generate_comparison_table
from .adapters import Adapter, DirectAdapter, HarnessAdapter, score_agentic
from .perf_grid import run_direct_grid, CATEGORIES as PERF_CATEGORIES

__all__ = [
    "ModelRegistry",
    "ModelManifest",
    "Capability",
    "Modality",
    "TestRegistry",
    "TestCase",
    "APIClient",
    "APIResponse",
    "ResultsStore",
    "BenchRunner",
    "RunConfig",
    "generate_markdown_report",
    "generate_comparison_table",
    "Adapter",
    "DirectAdapter",
    "HarnessAdapter",
    "score_agentic",
    "run_direct_grid",
    "PERF_CATEGORIES",
]