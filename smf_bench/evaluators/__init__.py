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
import sys
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
    """Exact match (trimmed, case-insensitive, falls back to reasoning if content empty)."""
    actual_text = resp.text if resp.text.strip() else resp.reasoning
    actual = actual_text.strip().lower()
    expected = str(test.expected).strip().lower()
    passed = actual == expected
    return EvalResult(
        passed,
        1.0 if passed else 0.0,
        "Exact match" if passed else f"Expected '{expected}', got '{actual[:100]}'",
        test.expected,
        actual_text[:200],
    )


@register("regex_match")
def eval_regex_match(resp: APIResponse, test: TestCase) -> EvalResult:
    """Regex pattern match against output (falls back to reasoning if content empty)."""
    pattern = str(test.expected)
    search_text = resp.text if resp.text.strip() else resp.reasoning
    match = re.search(pattern, search_text, re.IGNORECASE | re.DOTALL)
    if match:
        return EvalResult(True, 1.0, f"Regex matched: {match.group()[:100]}", pattern, search_text[:200])
    return EvalResult(False, 0.0, f"Regex '{pattern}' did not match", pattern, search_text[:200])


@register("json_contains")
def eval_json_contains(resp: APIResponse, test: TestCase) -> EvalResult:
    """Parse JSON from output and check for expected key/value pairs."""
    try:
        # Try content first, fall back to reasoning for reasoning models
        text = resp.text.strip() if resp.text.strip() else resp.reasoning.strip()
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
    code = resp.text.strip() if resp.text.strip() else resp.reasoning.strip()
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
    html = resp.text.strip() if resp.text.strip() else resp.reasoning.strip()
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
        # Check if this is a unit_test type with assertions in expected
        if test.metadata.get("eval_type") == "unit_test":
            return eval_unit_test(resp, test)
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


@register("unit_test")
def eval_unit_test(resp: APIResponse, test: TestCase) -> EvalResult:
    """Execute Python unit tests against generated code.

    Extracts Python code from the response, appends the expected assertions,
    and runs the combined script. If all assertions pass, the test passes.
    """
    import re as _re

    # Get the code from response (fall back to reasoning for reasoning models)
    code = resp.text.strip() if resp.text.strip() else resp.reasoning.strip()
    if not code:
        return EvalResult(False, 0.0, "No code in response", test.expected, "")

    # Strip markdown code fences if present
    if code.startswith("```"):
        # Extract code between ```python ... ``` or ``` ... ```
        fence_match = _re.search(r"```(?:python)?\s*\n(.*?)```", code, _re.DOTALL)
        if fence_match:
            code = fence_match.group(1)
        else:
            code = _re.sub(r"^```(?:python)?\s*", "", code)
            code = _re.sub(r"\s*```$", "", code)

    # Get the assertions from expected
    assertions = test.expected if isinstance(test.expected, str) else ""
    if not assertions:
        return EvalResult(False, 0.0, "No assertions in expected field", test.expected, code[:200])

    # Combine code + assertions and execute
    full_code = code + "\n\n# --- Unit Tests ---\n" + assertions
    try:
        # Run in a subprocess for isolation with a timeout
        import subprocess
        result = subprocess.run(
            [sys.executable, "-c", full_code],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return EvalResult(True, 1.0, "All assertions passed", test.expected, code[:200])
        else:
            # Extract the failing assertion from stderr
            stderr = result.stderr.strip()
            # Get last few lines of error
            error_lines = stderr.split("\n")
            detail = error_lines[-1] if error_lines else "Assertion failed"
            return EvalResult(False, 0.0, f"Assertion failed: {detail[:150]}", test.expected, code[:200])
    except subprocess.TimeoutExpired:
        return EvalResult(False, 0.0, "Code execution timed out (10s)", test.expected, code[:200])
    except Exception as e:
        return EvalResult(False, 0.0, f"Execution error: {e}", test.expected, code[:200])


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


@register("structural_count")
def eval_structural_count(resp: APIResponse, test: TestCase) -> EvalResult:
    """Count structural units (lines, paragraphs, sentences) in the response.

    Expects test.metadata with:
      - 'unit': 'line' | 'paragraph' | 'sentence'
      - 'op': '==' | '>=' | '<=' | '>'
    And test.expected as the integer count to compare against.
    """
    text = resp.text.strip() if resp.text else ""
    if not text:
        # Fall back to reasoning content for thinking models
        text = resp.reasoning.strip() if resp.reasoning else ""
    if not text:
        return EvalResult(False, 0.0, "No text in response", test.expected, "")

    unit = test.metadata.get("unit", "line")
    op = test.metadata.get("op", "==")
    expected = int(test.expected) if test.expected is not None else 0

    if unit == "line":
        # Count non-empty lines
        actual = len([l for l in text.split("\n") if l.strip()])
    elif unit == "paragraph":
        # Count paragraphs (blocks separated by blank lines)
        actual = len([p for p in text.split("\n\n") if p.strip()])
    elif unit == "sentence":
        # Count sentences (rough: split on . ! ?)
        actual = len([s for s in re.split(r"[.!?]+", text) if s.strip()])
    else:
        return EvalResult(False, 0.0, f"Unknown unit type: {unit}", test.expected, 0)

    import operator
    ops = {"==": operator.eq, ">=": operator.ge, "<=": operator.le, ">": operator.gt}
    compare = ops.get(op, operator.eq)
    passed = compare(actual, expected)

    return EvalResult(
        passed=passed,
        score=1.0 if passed else 0.0,
        detail=f"{actual} {unit}s (expected {op} {expected})",
        expected=expected,
        actual=actual,
    )


@register("agentic_file_check")
def eval_agentic_file_check(resp: APIResponse, test: TestCase) -> EvalResult:
    """Check that the agent created the expected file(s) in its response.

    For agentic tests where the model is asked to use file/shell tools.
    Checks the response text for evidence of file creation or tool use.
    """
    text = resp.text.strip() if resp.text else ""
    if not text:
        text = resp.reasoning.strip() if resp.reasoning else ""
    if not text:
        return EvalResult(False, 0.0, "No text in response", test.expected, "")

    # Check for common patterns indicating the agent attempted the task:
    # - File paths mentioned
    # - Tool calls executed
    # - Code blocks present
    has_file_path = bool(re.search(r"[\w/]+\.\w{1,5}", text))
    has_code_block = "```" in text
    has_tool_use = bool(re.search(r"(write|create|save|cat|echo).*file|tool_call", text, re.IGNORECASE))

    # Check tool_calls in the response
    tool_calls = resp.tool_calls or []
    has_structured_tool_calls = len(tool_calls) > 0

    passed = has_file_path or has_code_block or has_tool_use or has_structured_tool_calls

    return EvalResult(
        passed=passed,
        score=1.0 if passed else 0.0,
        detail=f"File evidence: path={has_file_path}, code={has_code_block}, tool={has_tool_use}, structured={has_structured_tool_calls}",
        expected=test.expected,
        actual=text[:200],
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


@register("structural_count")
def eval_structural_count(resp: APIResponse, test: TestCase) -> EvalResult:
    """Count structural units (lines, sentences, stanzas, paragraphs) in the output."""
    expected_count = int(test.expected)
    unit = test.metadata.get("unit", "line") if test.metadata else "line"
    text = resp.text if resp.text else (resp.reasoning or "")
    if not text:
        return EvalResult(False, 0.0, f"No output to count {unit}s", test.expected, "")

    if unit == "line":
        # Count non-empty lines
        count = len([l for l in text.strip().split("\n") if l.strip()])
    elif unit == "sentence":
        # Count sentences via punctuation
        import re as _re
        sentences = _re.split(r"[.!?]+", text)
        count = len([s for s in sentences if s.strip()])
    elif unit == "stanza":
        # Stanzas separated by blank lines
        stanzas = text.strip().split("\n\n")
        count = len([s for s in stanzas if s.strip()])
    elif unit == "paragraph":
        paragraphs = text.strip().split("\n\n")
        count = len([p for p in paragraphs if p.strip()])
    elif unit == "word":
        count = len(text.split())
    else:
        return EvalResult(False, 0.0, f"Unknown unit '{unit}' for structural_count", test.expected, "")

    passed = count == expected_count
    return EvalResult(
        passed=passed,
        score=1.0 if passed else 0.0,
        detail=f"Expected {expected_count} {unit}(s), got {count}",
        expected=test.expected,
        actual=str(count),
    )
    # For now, fall back to binary_exists
    # return eval_binary_exists(resp, test)  # unreachable, kept for reference