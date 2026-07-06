"""
Adapter pattern for smf-bench — harness adapters for agentic tasks.

Extracted from Aeon-Bench-Pod (MIT License, Copyright 2026 AEON-7).
Adapted for smf-bench: simplified, uses our api_client.

An adapter is the small amount of harness-specific code that turns an agentic
task into a scored outcome. The contract is one method:

    run_task(task, model_base_url, served_alias, *, timeout=300) -> {"files": ..., "answer": ...}

where `task` is an agentic case dict with:
    {"prompt": str,
     "setup_files": {relpath: content},   # written to workdir before the run
     "success": {                          # the deterministic success spec
         "files": {relpath: {"contains": [...]} | {"equals": text}},
         "answer_contains": [needle, ...],
     },
     "timeout_s": int}

The adapter:
  1. Creates a temporary workdir and writes setup_files into it
  2. Sends the prompt to the model (via api_client, harness, or subagent)
  3. The model uses its tools to write files / produce an answer
  4. Returns the observable outcome: files produced + final answer text

Scoring (score_agentic) checks:
  - Every file `contains` needle → one criterion per needle
  - Every file `equals` text → one criterion
  - Every `answer_contains` needle → one criterion
  - score = passed / total (0..1)
"""
from __future__ import annotations

import os
import re
import tempfile
import shutil
from typing import Any, Optional

# ---------------------------------------------------------------- reasoning scrubber
# Strip <think>...</think> reasoning blocks from harness answers.
# Reasoning content is NEVER part of an answer.

_THINK_RE = re.compile(r"<think(?:ing)?\b[^>]*>.*?</think(?:ing)?\s*>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"^\s*<think(?:ing)?\b[^>]*>.*\Z", re.DOTALL | re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """Drop <think>...</think> reasoning blocks and a leading unclosed trace."""
    if not text:
        return text
    text = _THINK_RE.sub("", text)
    text = _THINK_OPEN_RE.sub("", text)
    return text.strip()


# ---------------------------------------------------------------- file helpers


def _norm(s) -> str:
    """Whitespace-stripped, lowercased canonical form for `contains` matching."""
    return re.sub(r"\s+", "", str(s)).lower()


def _canon_text(s: str) -> str:
    """Line-ending-normalised, per-line-rstripped, outer-stripped form for `equals`."""
    s = str(s).replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in s.split("\n")).strip()


def _read_file(workdir: str, relpath: str) -> Optional[str]:
    """File content (utf-8, tolerant) or None if missing/unreadable."""
    path = os.path.join(workdir, relpath)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return None


# ---------------------------------------------------------------- scoring


def score_agentic(task: dict, workdir: str, answer: str) -> tuple[float, list[dict]]:
    """Score one agentic task from the OBSERVABLE OUTCOME.

    Returns (score, evidence):
      score    — 1.0 iff every criterion passed, else passed/total (0..1)
      evidence — [{"criterion": str, "ok": bool, "detail": str}, ...]
    """
    spec = task.get("success") or {}
    evidence = []

    # File checks
    for rel, check in (spec.get("files") or {}).items():
        content = _read_file(workdir, rel)
        if not isinstance(check, dict):
            check = {"contains": [str(check)]}

        if "equals" in check:
            want = check["equals"]
            if content is None:
                ok, detail = False, "file missing"
            else:
                ok = _canon_text(content) == _canon_text(want)
                detail = "exact match" if ok else f"content mismatch (got {content[:80]!r})"
            evidence.append({"criterion": f"file:{rel} equals", "ok": ok, "detail": detail})

        if "contains" in check:
            for needle in check["contains"]:
                if content is None:
                    ok, detail = False, "file missing"
                else:
                    ok = _norm(needle) in _norm(content)
                    detail = "contains" if ok else f"missing {needle!r}"
                evidence.append(
                    {"criterion": f"file:{rel} contains {needle!r}", "ok": ok, "detail": detail}
                )

    # Answer checks
    clean_answer = strip_reasoning(answer or "")
    for needle in (spec.get("answer_contains") or []):
        ok = _norm(needle) in _norm(clean_answer)
        detail = "answer contains" if ok else f"answer missing {needle!r}"
        evidence.append(
            {"criterion": f"answer contains {needle!r}", "ok": ok, "detail": detail}
        )

    passed = sum(1 for e in evidence if e["ok"])
    total = len(evidence)
    score = passed / total if total > 0 else 0.0
    return score, evidence


# ---------------------------------------------------------------- adapter base


class Adapter:
    """Abstract base — each harness adapter implements run_task."""

    name: str = ""

    def run_task(
        self,
        task: dict,
        model_base_url: str,
        served_alias: str,
        *,
        timeout: int = 300,
    ) -> dict:
        """Run one agentic task. Returns {"files": {relpath: content}, "answer": str}."""
        raise NotImplementedError

    def _create_workdir(self, task: dict) -> str:
        """Create a temp workdir and populate setup_files."""
        workdir = tempfile.mkdtemp(prefix=f"smf_bench_{self.name}_")
        for rel, content in (task.get("setup_files") or {}).items():
            path = os.path.join(workdir, rel)
            os.makedirs(os.path.dirname(path) or workdir, exist_ok=True)
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(content)
        return workdir

    def _collect_files(self, workdir: str) -> dict:
        """Read all files in workdir (non-recursive top level + one level deep)."""
        files = {}
        for entry in os.listdir(workdir):
            path = os.path.join(workdir, entry)
            if os.path.isfile(path):
                try:
                    with open(path, encoding="utf-8", errors="replace") as f:
                        files[entry] = f.read()
                except OSError:
                    pass
        return files

    def _cleanup(self, workdir: str):
        """Remove the workdir."""
        shutil.rmtree(workdir, ignore_errors=True)


# ---------------------------------------------------------------- direct adapter
# The simplest adapter: sends the prompt to the model via api_client, no tools.
# The model must produce the answer in its response. Files are not written
# (no tool access), so file-based criteria will fail — this is the baseline.


class DirectAdapter(Adapter):
    """Direct API adapter — no tools, no file I/O. Only answer-based criteria can pass."""

    name = "direct"

    def __init__(self, client):
        self.client = client

    def run_task(
        self,
        task: dict,
        model_base_url: str,
        served_alias: str,
        *,
        timeout: int = 300,
    ) -> dict:
        workdir = self._create_workdir(task)
        try:
            resp = self.client.chat(
                messages=[{"role": "user", "content": task["prompt"]}],
                max_tokens=task.get("max_tokens", 4096),
                temperature=task.get("temperature", 0.3),
            )
            answer = resp.text or ""
            files = self._collect_files(workdir)
            return {"files": files, "answer": answer}
        finally:
            self._cleanup(workdir)


# ---------------------------------------------------------------- harness adapter
# Harness adapters delegate to an external agent system (Hermes, OpenClaw, etc.)
# that has file/shell tools. The adapter sends the prompt and collects results.
# These would be implemented per-harness in adapters/hermes.py, adapters/openclaw.py, etc.


class HarnessAdapter(Adapter):
    """Base for harness-based adapters (Hermes, OpenClaw, OpenCode).

    Subclasses must implement _run_in_harness(prompt, workdir, timeout) -> answer.
    """

    name = "harness"

    def run_task(
        self,
        task: dict,
        model_base_url: str,
        served_alias: str,
        *,
        timeout: int = 300,
    ) -> dict:
        workdir = self._create_workdir(task)
        try:
            answer = self._run_in_harness(
                prompt=task["prompt"],
                workdir=workdir,
                model_base_url=model_base_url,
                served_alias=served_alias,
                timeout=task.get("timeout_s", timeout),
            )
            answer = strip_reasoning(answer or "")
            files = self._collect_files(workdir)
            return {"files": files, "answer": answer}
        finally:
            self._cleanup(workdir)

    def _run_in_harness(
        self,
        prompt: str,
        workdir: str,
        model_base_url: str,
        served_alias: str,
        timeout: int,
    ) -> str:
        """Run the prompt in the harness, return the final answer text."""
        raise NotImplementedError


__all__ = [
    "Adapter",
    "DirectAdapter",
    "HarnessAdapter",
    "score_agentic",
    "strip_reasoning",
]