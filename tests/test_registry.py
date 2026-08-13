"""Offline registry and capability-gating contracts."""
from smf_bench import ModelRegistry
from smf_bench import TestRegistry as SuiteRegistry


def test_models_and_suites_load():
    models = ModelRegistry()
    tests = SuiteRegistry()
    assert models.load_dir("models/") >= 1
    assert tests.load_dir("suites/") >= 1
    assert models.list_models()
    assert tests.categories()


def test_capability_gating_returns_two_partitions():
    models = ModelRegistry()
    tests = SuiteRegistry()
    models.load_dir("models/")
    tests.load_dir("suites/")
    mid = models.list_models()[0]
    applicable, na = models.applicable_tests(tests, mid)
    assert isinstance(applicable, list)
    assert isinstance(na, list)
    assert len(applicable) + len(na) == tests.count()
