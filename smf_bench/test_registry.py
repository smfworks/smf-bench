"""
Test Registry — test definitions, loader, and capability matching.

Tests are defined as YAML case files. Each test declares:
- id, name, category, dimension
- required modalities and capabilities
- evaluator type and expected output
- the prompt/message to send

The registry loads all test files from the suites/ directory tree and
matches them against model manifests at run time.
"""

from __future__ import annotations

import json
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .model_registry import Capability, Modality


@dataclass
class TestCase:
    """A single test case definition."""
    test_id: str
    name: str
    category: str          # e.g. "vision", "reasoning", "latency_throughput"
    dimension: str         # e.g. "vision", "performance", "quality"
    difficulty: str = "unspecified"  # easy, medium, hard, expert, frontier
    required_modalities: set[Modality] = field(default_factory=set)
    required_capabilities: set[Capability] = field(default_factory=set)
    evaluator: str = "text_contains"  # evaluator type key
    prompt: str = ""                   # text prompt (for text-only tests)
    messages: list[dict] = field(default_factory=list)  # full message array (for multimodal)
    expected: Any = None               # expected output (string, regex, dict, etc.)
    max_tokens: int = 1024
    temperature: float = 0.6
    timeout: int = 120
    weight: float = 1.0               # for weighted scoring
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "TestCase":
        req_mods = {Modality(m) for m in data.get("requires_modalities", [])}
        req_caps = {Capability(c) for c in data.get("requires_capabilities", [])}
        # Collect extra fields (expected_keywords, min_matches, tools, etc.) into metadata
        known_keys = {"id", "name", "category", "dimension", "difficulty",
                      "requires_modalities",
                      "requires_capabilities", "evaluator", "prompt", "messages",
                      "expected", "max_tokens", "temperature", "timeout", "weight",
                      "metadata"}
        extra = {k: v for k, v in data.items() if k not in known_keys}
        meta = {**extra, **data.get("metadata", {})}
        return cls(
            test_id=data["id"],
            name=data.get("name", data["id"]),
            category=data.get("category", "general"),
            dimension=data.get("dimension", "quality"),
            difficulty=data.get("difficulty", "unspecified"),
            required_modalities=req_mods,
            required_capabilities=req_caps,
            evaluator=data.get("evaluator", "text_contains"),
            prompt=data.get("prompt", ""),
            messages=data.get("messages", []),
            expected=data.get("expected"),
            max_tokens=data.get("max_tokens", 1024),
            temperature=data.get("temperature", 0.6),
            timeout=data.get("timeout", 120),
            weight=data.get("weight", 1.0),
            metadata=meta,
        )

    @classmethod
    def from_yaml_file(cls, path: Path) -> list["TestCase"]:
        """Load test cases from a YAML file. Supports both single-doc and multi-doc."""
        with open(path) as f:
            # safe_load_all handles both single-doc and multi-doc YAML
            docs = list(yaml.safe_load_all(f))
        if not docs:
            return []
        # If first doc is a list, flatten it
        cases = []
        for doc in docs:
            if isinstance(doc, list):
                cases.extend(doc)
            elif isinstance(doc, dict):
                cases.append(doc)
            # else: skip None/empty docs
        return [cls.from_dict(d) for d in cases]

    @classmethod
    def from_json_file(cls, path: Path) -> list["TestCase"]:
        """Load test cases from a JSON file. Can be single case or list."""
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [data]
        return [cls.from_dict(d) for d in data]


class TestRegistry:
    """Registry of all test cases, loaded from suite directories."""
    def __init__(self) -> None:
        self._tests: list[TestCase] = []
        self._by_category: dict[str, list[TestCase]] = {}

    def add(self, test: TestCase) -> None:
        self._tests.append(test)
        if test.category not in self._by_category:
            self._by_category[test.category] = []
        self._by_category[test.category].append(test)

    def load_dir(self, dir_path: str | Path) -> int:
        """Recursively load all .yaml and .json test files from a directory tree."""
        dir_path = Path(dir_path)
        count = 0
        for ext in ("*.yaml", "*.yml", "*.json"):
            for p in sorted(dir_path.rglob(ext)):
                try:
                    if p.suffix == ".json":
                        tests = TestCase.from_json_file(p)
                    else:
                        tests = TestCase.from_yaml_file(p)
                    for t in tests:
                        self.add(t)
                        count += 1
                except Exception as e:
                    print(f"  WARN: Failed to load {p}: {e}")
        return count

    def all_tests(self) -> list[TestCase]:
        return list(self._tests)

    def by_category(self, category: str) -> list[TestCase]:
        return self._by_category.get(category, [])

    def by_dimension(self, dimension: str) -> list[TestCase]:
        return [t for t in self._tests if t.dimension == dimension]

    def count(self) -> int:
        return len(self._tests)

    def categories(self) -> list[str]:
        return sorted(self._by_category.keys())

    def dimensions(self) -> list[str]:
        return sorted({t.dimension for t in self._tests})