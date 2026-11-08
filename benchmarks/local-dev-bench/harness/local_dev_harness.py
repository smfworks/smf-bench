#!/usr/bin/env python3
"""
Ripple-65B agentic harness — EXP-2026-11-04-ripple-65b-feasibility phase 2.

Tests a local OpenAI-compatible endpoint for developer usability on agentic coding:
  A. Planted-bug fixes (3 tasks, each with a deterministic test)
  B. Feature-from-spec (1 task, deterministic test)
  C. Tool-call fidelity (10 calls, broken-JSON rate)
  D. Instruction following + budget behavior (cap-hit at fixed budget)

Usage (any OpenAI-compatible endpoint, e.g. llama.cpp llama-server on :8484):
  python local_dev_harness.py --endpoint http://localhost:8484 --model ripple --tag ripple
  python local_dev_harness.py --endpoint http://localhost:PORT --model NAME --tag control

Writes: results_<tag>.json  (one file per run; aggregate/compare across tags)
Greedy decoding (temp 0) for determinism.
"""
import argparse, json, os, re, subprocess, sys, tempfile, time, urllib.request

def chat(endpoint, model, messages, tools=None, max_tokens=2048, timeout=600):
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": 0}
    if tools: body["tools"] = tools; body["tool_choice"] = "auto"
    req = urllib.request.Request(endpoint.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=timeout))
    return {"msg": r["choices"][0]["message"], "usage": r.get("usage", {}), "wall": time.time() - t0}

def run_tests(files):
    """files: {path: content}. Write to temp dir, run pytest, return (all_pass, output)."""
    d = tempfile.mkdtemp(prefix="ripple_h_")
    for p, c in files.items():
        fp = os.path.join(d, p); os.makedirs(os.path.dirname(fp), exist_ok=True); open(fp, "w").write(c)
    r = subprocess.run([sys.executable, "-m", "pytest", d, "-x", "-q", "--no-header"],
                       capture_output=True, text=True, timeout=120)
    return (r.returncode == 0), (r.stdout + r.stderr)[-1500:]

# ---------------- Task definitions ----------------
# Each: buggy files + test file. The test passes ONLY when the bug is fixed correctly.

BUGGY_STATS = '''def mean(xs):
    return sum(xs) / len(xs)

def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n % 2 == 1:
        return xs[n // 2]
    return (xs[n // 2] + xs[n // 2 + 1]) / 2

def variance(xs):
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs)  # BUG: population variance; spec says sample
'''

STATS_SPEC = """The module stats.py must provide, for a non-empty list of numbers xs:
- mean(xs): arithmetic mean.
- median(xs): middle value for odd-length; average of the two middle values for even-length.
- variance(xs): the SAMPLE variance, i.e. sum((x-mean)^2) / (len(xs)-1). Raise ValueError on len(xs) < 2.
Fix whatever is wrong in the existing implementation. Do not change the function signatures."""

STATS_TEST = '''from stats import mean, median, variance
import pytest

def test_mean():
    assert mean([1, 2, 3]) == 2

def test_median_odd():
    assert median([3, 1, 2]) == 2

def test_median_even():
    assert median([4, 1, 3, 2]) == 2.5

def test_variance_sample():
    assert variance([1, 2, 3, 4]) == pytest.approx(5/3)

def test_variance_requires_two():
    with pytest.raises(ValueError):
        variance([5])
'''

BUGGY_LRU = '''class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.data = {}
        self.order = []

    def get(self, key):
        if key not in self.data:
            return None  # BUG: spec requires KeyError on miss
        self.order.remove(key)  # BUG: should move to END (most recent), not delete
        self.data[key]
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

LRU_SPEC = """The module lru.py must implement class LRUCache(capacity) with:
- get(key): returns the value, or raises KeyError if absent. Marks key as most recently used.
- put(key, value): inserts/updates; marks key as most recently used.
- Evicts the LEAST recently used key when capacity would be exceeded.
Fix whatever is wrong in the existing implementation. Keep the constructor signature."""

LRU_TEST = '''import pytest
from lru import LRUCache

def test_miss_raises():
    c = LRUCache(2)
    with pytest.raises(KeyError):
        c.get("x")

def test_eviction_order():
    c = LRUCache(2)
    c.put("a", 1); c.put("b", 2)
    c.get("a")          # a now most recent
    c.put("c", 3)       # evicts b
    assert c.get("a") == 1
    with pytest.raises(KeyError):
        c.get("b")

def test_update_no_evict():
    c = LRUCache(2)
    c.put("a", 1); c.put("b", 2)
    c.put("a", 9)       # update, no evict
    c.put("c", 3)       # evicts b
    assert c.get("a") == 9
'''

BUGGY_DEDUP = '''def dedup_consecutive(items, key=None):
    """Remove consecutive duplicates, keeping first occurrence order.
    BUG: drops ALL duplicates anywhere in the list, not just consecutive."""
    seen = set()
    out = []
    for x in items:
        k = x if key is None else key(x)
        if k not in seen:
            out.append(x)
            seen.add(k)
    return out
'''

DEDUP_SPEC = """The module dedup.py must provide dedup_consecutive(items, key=None):
removes CONSECUTIVE duplicate items (compared by key(x) if key given, else the item itself),
keeping the first of each run. Non-consecutive repeats MUST be kept.
Example: dedup_consecutive([1,1,2,2,1]) == [1,2,1]. Fix the existing implementation."""

DEDUP_TEST = '''from dedup import dedup_consecutive

def test_basic():
    assert dedup_consecutive([1, 1, 2, 2, 3]) == [1, 2, 3]

def test_non_consecutive_kept():
    assert dedup_consecutive([1, 1, 2, 2, 1]) == [1, 2, 1]

def test_key_function():
    words = ["a", "A", "b", "B", "a"]
    assert dedup_consecutive(words, key=str.lower) == ["a", "b", "a"]

def test_empty():
    assert dedup_consecutive([]) == []
'''

FEATURE_QUEUE = '''# Write a new module ringbuffer.py implementing class RingBuffer(capacity).
# Requirements:
# - RingBuffer(n) holds at most n items, n >= 1 (raise ValueError on n < 1).
# - append(x): adds x; when full, silently overwrites the OLDEST item.
# - to_list(): returns contents oldest -> newest, as a list.
# - __len__: current number of items.
# - is_empty(), is_full(): bools.
'''

RINGBUFFER_TEST = '''import pytest
from ringbuffer import RingBuffer

def test_len_and_list():
    b = RingBuffer(3)
    assert b.is_empty()
    b.append(1); b.append(2)
    assert len(b) == 2 and b.to_list() == [1, 2]
    assert not b.is_full()

def test_overwrite():
    b = RingBuffer(2)
    b.append("a"); b.append("b"); b.append("c")
    assert b.is_full()
    assert b.to_list() == ["b", "c"]

def test_invalid_capacity():
    with pytest.raises(ValueError):
        RingBuffer(0)
'''

TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "run_tests",
        "description": "Run the project's test suite and return pass/fail output.",
        "parameters": {"type": "object", "properties": {
            "verbose": {"type": "boolean", "description": "show individual test names"},
            "pattern": {"type": "string", "description": "only run tests matching this pattern"}
        }, "required": []},
    },
}

TOOL_PROMPTS = [
    "The tests are failing and I cannot see why. Run the test suite for me.",
    "Please execute the project tests, but only the ones matching 'test_login', with verbose output.",
    "Run the full test suite quietly.",
    "Can you check if the tests pass? Use the test runner.",
    "I just refactored auth.py. Run the tests to verify nothing broke, verbose please.",
    "Execute the test suite with pattern 'test_cache'.",
    "Verify the build by running all tests. Not verbose.",
    "Run only tests whose names contain 'integration', verbose mode on.",
    "The CI is red. Run our tests and tell me the result.",
    "Quietly run every test in the project.",
]

BUGS = [
    {"id": "bug-stats", "spec": STATS_SPEC, "files": {"stats.py": BUGGY_STATS}, "test": STATS_TEST},
    {"id": "bug-lru", "spec": LRU_SPEC, "files": {"lru.py": BUGGY_LRU}, "test": LRU_TEST},
    {"id": "bug-dedup", "spec": DEDUP_SPEC, "files": {"dedup.py": BUGGY_DEDUP}, "test": DEDUP_TEST},
]
FEATURE = {"id": "feat-ringbuffer", "spec": FEATURE_QUEUE, "files": {}, "test": RINGBUFFER_TEST}

def extract_code(msg_content):
    """Pull python code blocks out of an assistant message."""
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", msg_content or "", re.S)
    return blocks

def extract_first_file(blocks, fname):
    """Return the code block that defines the target module, or None."""
    mod = fname.split(".")[0]
    for b in blocks:
        if re.search(rf"\b(def |class |{mod}\b)", b):
            return b
    return blocks[0] if blocks else None

def score_task(endpoint, model, task, tag):
    # baseline: does the buggy version fail tests? (sanity)
    if task["files"]:
        pre_pass, _ = run_tests({**task["files"], "test_task.py": task["test"]})
    else:
        pre_pass = False
    messages = [{"role": "user", "content": task["spec"]}]
    r = chat(endpoint, model, messages, max_tokens=3000)
    content = r["msg"].get("content") or ""
    blocks = extract_code(content)
    wrote = {}
    if task["files"]:
        fname = list(task["files"].keys())[0]
        code = extract_first_file(blocks, fname)
        if code: wrote[fname] = code
    else:
        for b in blocks:
            if "class RingBuffer" in b: wrote["ringbuffer.py"] = b; break
        if "ringbuffer.py" not in wrote and blocks: wrote["ringbuffer.py"] = blocks[-1]
    post_pass = False; test_out = ""
    if wrote:
        wrote["test_task.py"] = task["test"]
        post_pass, test_out = run_tests(wrote)
    rec = {
        "id": task["id"], "tag": tag,
        "prefail": (not pre_pass),  # True if the shipped bug actually failed tests
        "post_pass": post_pass,
        "ntok": r["usage"].get("completion_tokens", 0),
        "wall": round(r["wall"], 1),
        "reasoning": bool(r["msg"].get("reasoning_content") or r["msg"].get("reasoning")),
        "code_found": bool(wrote),
        "test_tail": test_out[-400:],
    }
    return rec

def score_tools(endpoint, model, tag):
    recs = []
    for p in TOOL_PROMPTS:
        try:
            r = chat(endpoint, model, [{"role": "user", "content": p}], tools=[TOOL_SPEC], max_tokens=400)
            m = r["msg"]
            tcs = m.get("tool_calls") or []
            ok = False; detail = ""
            if tcs:
                fn = tcs[0]["function"]["name"]
                try:
                    args = json.loads(tcs[0]["function"]["arguments"] or "{}")
                    ok = fn == "run_tests"
                    detail = f"{fn}({json.dumps(args)})"
                except Exception as e:
                    detail = f"JSON-BROKEN: {e}"
            rec = {"prompt": p, "tool_called": bool(tcs), "valid": ok, "detail": detail,
                   "ntok": r["usage"].get("completion_tokens", 0)}
        except Exception as e:
            rec = {"prompt": p, "tool_called": False, "valid": False, "detail": f"ERROR {str(e)[:80]}"}
        recs.append(rec)
    # expected behaviors per prompt index: 0 T/v None, 1 T/v pattern, 2 T/-, 3 T/-, 4 T/v,
    # 5 T/- pattern, 6 T/-, 7 T/v pattern, 8 T/-, 9 T/-
    return {"tag": tag, "calls": recs,
            "broken_json": sum(1 for r in recs if "JSON-BROKEN" in r["detail"]),
            "called": sum(1 for r in recs if r["tool_called"]),
            "valid": sum(1 for r in recs if r["valid"])}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", default="ripple")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    out = {"tag": args.tag, "endpoint": args.endpoint, "started": time.strftime("%F %T"), "tasks": [], "tools": None}
    for task in BUGS + [FEATURE]:
        print(f"[{args.tag}] scoring {task['id']} ...", flush=True)
        try:
            out["tasks"].append(score_task(args.endpoint, args.model, task, args.tag))
        except Exception as e:
            out["tasks"].append({"id": task["id"], "tag": args.tag, "error": str(e)[:200]})
    print(f"[{args.tag}] tool-calling fidelity ...", flush=True)
    try:
        out["tools"] = score_tools(args.endpoint, args.model, args.tag)
    except Exception as e:
        out["tools"] = {"tag": args.tag, "error": str(e)[:200]}
    fn = f"results_{args.tag}.json"
    json.dump(out, open(fn, "w"), indent=1)
    # summary
    ok = sum(1 for t in out["tasks"] if t.get("post_pass"))
    print(f"\n[{args.tag}] TASKS: {ok}/{len(out['tasks'])} passed")
    if out["tools"] and "error" not in out["tools"]:
        print(f"[{args.tag}] TOOLS: called {out['tools']['called']}/10, valid {out['tools']['valid']}/10, broken JSON {out['tools']['broken_json']}")

if __name__ == "__main__":
    main()
