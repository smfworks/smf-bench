"""
Evaluators — pluggable scoring functions for test outputs.

Each evaluator takes an APIResponse and a TestCase and returns an EvalResult.
Evaluators are registered by name so test cases can reference them by string.

Built-in evaluators:
- text_contains:  substring match (case-insensitive)
- text_match:     exact match (case-insensitive, trimmed)
- regex_match:    regex pattern match
- json_contains:  parse JSON output, check for key/value
- code_compiles:  check that Python code compiles
- html_valid:     check that HTML has required tags
- tool_call:      check tool call structure, name, and arguments
- programmatic:   run a custom Python checker function
- llm_judge:      use a judge model to score on a rubric (placeholder)
- perceptual_clip: CLIP score between prompt and generated image (requires vision extras)
- binary_exists:  binary output file was produced (for media generation)
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any, Callable

from ..api_client import APIResponse
from ..test_registry import TestCase


@dataclass
class EvalResult:
    """Result of evaluating a test case against a response."""
    passed: bool
    score: float          # 0.0 - 1.0 (partial credit allowed)
    detail: str = ""      # human-readable explanation
    expected: Any = None
    actual: Any = None


# ─── Evaluator registry ──────────────────────────────────────────────────────

EvaluatorFn = Callable[[APIResponse, TestCase], EvalResult]
_EVALUATORS: dict[str, EvaluatorFn] = {}


def register(name: str) -> Callable[[EvaluatorFn], EvaluatorFn]:
    """Decorator to register an evaluator."""
    def decorator(fn: EvaluatorFn) -> EvaluatorFn:
        _EVALUATORS[name] = fn
        return fn
    return decorator


def get_evaluator(name: str) -> EvaluatorFn:
    """Look up a registered evaluator by name."""
    if name not in _EVALUATORS:
        raise KeyError(f"No evaluator registered as '{name}'. Available: {list(_EVALUATORS.keys())}")
    return _EVALUATORS[name]


def list_evaluators() -> list[str]:
    return sorted(_EVALUATORS.keys())


# ─── Built-in evaluators ─────────────────────────────────────────────────────

@register("text_contains")
def eval_text_contains(resp: APIResponse, test: TestCase) -> EvalResult:
    """Case-insensitive substring match.
    
    Supports three modes:
    1. test.expected is a list → all items must be found
    2. test.expected is a string → substring must be found
    3. test.metadata has 'expected_keywords' + 'min_matches' → at least N keywords must be found
    """
    haystack = (resp.text + " " + resp.reasoning).lower()
    expected = test.expected
    
    # Mode 3: keyword-based scoring with min_matches threshold
    if expected is None and "expected_keywords" in test.metadata:
        keywords = test.metadata["expected_keywords"]
        min_matches = test.metadata.get("min_matches", len(keywords))
        found = [k for k in keywords if k.lower() in haystack]
        passed = len(found) >= min_matches
        missing = [k for k in keywords if k.lower() not in haystack]
        detail = f"Matched {len(found)}/{len(keywords)} keywords (need {min_matches})"
        if missing:
            detail += f"; missing: {missing}"
        return EvalResult(passed, len(found) / len(keywords) if keywords else 0.0, detail, keywords, resp.text[:200])
    
    # Mode 1: list of expected substrings
    if isinstance(expected, list):
        missing = [e for e in expected if e.lower() not in haystack]
        if missing:
            return EvalResult(False, 0.0, f"Missing expected substrings: {missing}", expected, resp.text[:200])
        return EvalResult(True, 1.0, "All expected substrings found", expected, resp.text[:200])
    
    # Mode 2: single expected string
    if isinstance(expected, str):
        if expected.lower() in haystack:
            return EvalResult(True, 1.0, "Expected substring found", expected, resp.text[:200])
        return EvalResult(False, 0.0, f"Expected '{expected}' not found in output", expected, resp.text[:200])
    
    return EvalResult(False, 0.0, f"Invalid expected type: {type(expected)}", expected, resp.text[:200])


@register("text_match")
def eval_text_match(resp: APIResponse, test: TestCase) -> EvalResult:
    """Exact match (trimmed, case-insensitive)."""
    actual = resp.text.strip().lower()
    expected = str(test.expected).strip().lower()
    passed = actual == expected
    return EvalResult(
        passed,
        1.0 if passed else 0.0,
        "Exact match" if passed else f"Expected '{expected}', got '{actual[:100]}'",
        test.expected,
        resp.text[:200],
    )


@register("regex_match")
def eval_regex_match(resp: APIResponse, test: TestCase) -> EvalResult:
    """Regex pattern match against output."""
    pattern = str(test.expected)
    match = re.search(pattern, resp.text, re.IGNORECASE | re.DOTALL)
    if match:
        return EvalResult(True, 1.0, f"Regex matched: {match.group()[:100]}", pattern, resp.text[:200])
    return EvalResult(False, 0.0, f"Regex '{pattern}' did not match", pattern, resp.text[:200])


@register("json_contains")
def eval_json_contains(resp: APIResponse, test: TestCase) -> EvalResult:
    """Parse JSON from output and check for expected key/value pairs."""
    try:
        # Try to extract JSON from the response (may be wrapped in markdown)
        text = resp.text.strip()
        if text.startswith("```"):
            # Strip markdown code fences
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        data = json.loads(text)
    except json.JSONDecodeError:
        return EvalResult(False, 0.0, "Output is not valid JSON", test.expected, resp.text[:200])

    expected = test.expected  # dict of key → value (or key → "*present*")
    if not isinstance(expected, dict):
        return EvalResult(False, 0.0, f"Expected dict, got {type(expected)}", expected, resp.text[:200])

    missing = []
    for key, expected_val in expected.items():
        if key not in data:
            missing.append(f"{key} (missing)")
        elif expected_val != "*" and data[key] != expected_val:
            missing.append(f"{key}={data[key]} (expected {expected_val})")

    if missing:
        return EvalResult(False, 0.0, f"JSON mismatches: {missing}", expected, resp.text[:200])
    return EvalResult(True, 1.0, "All expected JSON keys/values match", expected, resp.text[:200])


@register("code_compiles")
def eval_code_compiles(resp: APIResponse, test: TestCase) -> EvalResult:
    """Check that the output is valid Python code that compiles."""
    code = resp.text.strip()
    # Strip markdown code fences if present
    if code.startswith("```"):
        code = re.sub(r"^```(?:python)?\s*", "", code)
        code = re.sub(r"\s*```$", "", code)
    try:
        ast.parse(code)
        # If expected is a string, also check it appears in the code
        if isinstance(test.expected, str) and test.expected.lower() not in code.lower():
            return EvalResult(False, 0.0, f"Code compiles but missing '{test.expected}'", test.expected, resp.text[:200])
        return EvalResult(True, 1.0, "Code compiles successfully", test.expected, resp.text[:200])
    except SyntaxError as e:
        return EvalResult(False, 0.0, f"Syntax error: {e}", test.expected, resp.text[:200])


@register("html_valid")
def eval_html_valid(resp: APIResponse, test: TestCase) -> EvalResult:
    """Check that the output is valid HTML with required tags."""
    html = resp.text.strip()
    if html.startswith("```"):
        html = re.sub(r"^```(?:html)?\s*", "", html)
        html = re.sub(r"\s*```$", "", html)

    required_tags = test.expected if isinstance(test.expected, list) else [test.expected]
    missing = []
    for tag in required_tags:
        if f"<{tag}" not in html.lower():
            missing.append(tag)

    if missing:
        return EvalResult(False, 0.0, f"Missing HTML tags: {missing}", required_tags, resp.text[:200])

    # Basic well-formedness: has <html> or is a fragment with tags
    if "<" not in html or ">" not in html:
        return EvalResult(False, 0.0, "Output does not appear to be HTML", required_tags, resp.text[:200])

    return EvalResult(True, 1.0, "HTML contains all required tags", required_tags, resp.text[:200])


@register("tool_call")
def eval_tool_call(resp: APIResponse, test: TestCase) -> EvalResult:
    """Check tool call structure, tool name, and argument values."""
    if not resp.tool_calls:
        return EvalResult(False, 0.0, "No tool call in response", test.expected, resp.text[:200])

    expected = test.expected if isinstance(test.expected, dict) else {}
    tc = resp.tool_calls[0]
    func = tc.get("function", {})
    tool_name = func.get("name", "")
    args_str = func.get("arguments", "{}")

    checks = []
    score = 0.0
    total_checks = 0

    # Check tool name
    if "tool" in expected:
        total_checks += 1
        if tool_name == expected["tool"]:
            checks.append(f"tool={tool_name} ✓")
            score += 1.0
        else:
            checks.append(f"tool={tool_name} ✗ (expected {expected['tool']})")

    # Check arguments contain expected substrings
    if "args_contains" in expected:
        for key, val in expected["args_contains"].items():
            total_checks += 1
            if str(val).lower() in args_str.lower():
                checks.append(f"arg {key}={val} ✓")
                score += 1.0
            else:
                checks.append(f"arg {key}={val} ✗")

    # Check arguments match expected JSON exactly
    if "args_json" in expected:
        total_checks += 1
        try:
            actual_args = json.loads(args_str)
            if actual_args == expected["args_json"]:
                checks.append("args JSON ✓")
                score += 1.0
            else:
                checks.append(f"args JSON ✗ (got {actual_args})")
        except json.JSONDecodeError:
            checks.append(f"args not valid JSON: {args_str[:100]}")

    if total_checks == 0:
        # Just check that a tool call was made
        return EvalResult(True, 1.0, f"Tool call made: {tool_name}", test.expected, resp.tool_calls)

    passed = score == total_checks
    return EvalResult(passed, score / total_checks, "; ".join(checks), test.expected, resp.tool_calls)


@register("binary_exists")
def eval_binary_exists(resp: APIResponse, test: TestCase) -> EvalResult:
    """Check that a binary output file was produced (for media generation)."""
    from pathlib import Path
    if resp.binary_path and Path(resp.binary_path).exists():
        size = Path(resp.binary_path).stat().st_size
        min_size = test.metadata.get("min_file_size", 100)
        if size >= min_size:
            return EvalResult(True, 1.0, f"Binary output produced ({size} bytes)", test.expected, resp.binary_path)
        return EvalResult(False, 0.0, f"Binary too small ({size} bytes, min {min_size})", test.expected, resp.binary_path)
    return EvalResult(False, 0.0, "No binary output produced", test.expected, resp.binary_path)


@register("programmatic")
def eval_programmatic(resp: APIResponse, test: TestCase) -> EvalResult:
    """Run a custom Python checker function defined in the test metadata.

    The checker function is specified in test.metadata['checker'] as a string
    of Python code. It receives `output` (str) and `response` (APIResponse)
    and must return a dict with 'passed' (bool), 'score' (float), 'detail' (str).
    """
    checker_code = test.metadata.get("checker")
    if not checker_code:
        return EvalResult(False, 0.0, "No checker function defined in metadata")

    try:
        local_ns: dict[str, Any] = {}
        exec(checker_code, {"__builtins__": __builtins__}, local_ns)
        checker = local_ns.get("check")
        if not callable(checker):
            return EvalResult(False, 0.0, "Checker code did not define a 'check' function")

        result = checker(output=resp.text, response=resp)
        if not isinstance(result, dict):
            return EvalResult(False, 0.0, f"Checker returned {type(result)}, expected dict")

        return EvalResult(
            passed=result.get("passed", False),
            score=result.get("score", 0.0),
            detail=result.get("detail", ""),
            expected=test.expected,
            actual=resp.text[:200],
        )
    except Exception as e:
        return EvalResult(False, 0.0, f"Checker error: {e}", test.expected, resp.text[:200])


@register("llm_judge")
def eval_llm_judge(resp: APIResponse, test: TestCase) -> EvalResult:
    """Use a judge model to score the output on a rubric.

    Placeholder — requires a judge endpoint to be configured.
    The rubric is defined in test.metadata['rubric'].
    """
    rubric = test.metadata.get("rubric", "Output quality and correctness")
    # TODO: implement LLM judge using a configured judge endpoint
    # For now, this is a pass-through that always passes with a note
    return EvalResult(
        True,
        1.0,
        f"LLM judge not yet configured (rubric: {rubric}). Auto-pass.",
        test.expected,
        resp.text[:200],
    )


@register("perceptual_clip")
def eval_perceptual_clip(resp: APIResponse, test: TestCase) -> EvalResult:
    """CLIP score between prompt and generated image (requires vision extras)."""
    if not resp.binary_path:
        return EvalResult(False, 0.0, "No image produced for CLIP evaluation")

    try:
        from transformers import CLIPModel, CLIPProcessor
        from PIL import Image
        import torch
    except ImportError:
        return EvalResult(
            False, 0.0,
            "CLIP evaluation requires: pip install smf-bench[vision]",
            test.expected, resp.binary_path,
        )

    # TODO: load CLIP model, compute similarity, threshold
    # For now, fall back to binary_exists
    return eval_binary_exists(resp, test)