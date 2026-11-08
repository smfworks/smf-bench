"""Offline ruler validation for local-dev-bench.

Verifies the harness measures in both directions WITHOUT any model:
  1. Reference solutions pass every task test.
  2. The shipped planted bugs fail those same tests.
Run from the repo root:  pytest tests/test_local_dev_bench_ruler.py -q
"""
import importlib.util
import sys
from pathlib import Path

HARNESS = Path(__file__).parent.parent / "benchmarks" / "local-dev-bench" / "harness" / "local_dev_harness.py"
spec = importlib.util.spec_from_file_location("ldh", HARNESS)
ldh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ldh)

# Reference solutions (what a correct fix looks like)
REF_STATS = '''def mean(xs):
    return sum(xs) / len(xs)

def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n % 2 == 1:
        return xs[n // 2]
    return (xs[n // 2 - 1] + xs[n // 2]) / 2

def variance(xs):
    if len(xs) < 2:
        raise ValueError("need at least 2 points")
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
'''

REF_LRU = '''class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.data = {}
        self.order = []
    def get(self, key):
        if key not in self.data:
            raise KeyError(key)
        self.order.remove(key)
        self.order.append(key)
        return self.data[key]
    def put(self, key, value):
        if key in self.data:
            self.order.remove(key)
        self.data[key] = value
        self.order.append(key)
        if len(self.data) > self.capacity:
            old = self.order.pop(0)
            del self.data[old]
'''

REF_DEDUP = '''def dedup_consecutive(items, key=None):
    out = []
    prev = object()
    for x in items:
        k = x if key is None else key(x)
        if not out or k != prev:
            out.append(x)
            prev = k
    return out
'''

REF_RINGBUFFER = '''class RingBuffer:
    def __init__(self, capacity):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self._buf = []
    def append(self, x):
        if len(self._buf) >= self.capacity:
            self._buf.pop(0)
        self._buf.append(x)
    def to_list(self):
        return list(self._buf)
    def __len__(self):
        return len(self._buf)
    def is_empty(self):
        return len(self._buf) == 0
    def is_full(self):
        return len(self._buf) >= self.capacity
'''


def _refs():
    return {
        "bug-stats": {"stats.py": REF_STATS},
        "bug-lru": {"lru.py": REF_LRU},
        "bug-dedup": {"dedup.py": REF_DEDUP},
        "feat-ringbuffer": {"ringbuffer.py": REF_RINGBUFFER},
    }


def test_reference_solutions_pass():
    refs = _refs()
    for task in ldh.BUGS + [ldh.FEATURE]:
        files = dict(refs[task["id"]])
        files["test_task.py"] = task["test"]
        passed, tail = ldh.run_tests(files)
        assert passed, f"reference solution for {task['id']} must pass its tests:\n{tail}"


def test_planted_bugs_fail():
    for task in ldh.BUGS:
        files = dict(task["files"])
        files["test_task.py"] = task["test"]
        passed, tail = ldh.run_tests(files)
        assert not passed, f"planted bug in {task['id']} must fail its tests:\n{tail}"


def test_feature_task_requires_implementation():
    # No implementation -> test import fails -> harness scores a miss.
    passed, _ = ldh.run_tests({"test_task.py": ldh.FEATURE["test"]})
    assert not passed
