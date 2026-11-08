#!/usr/bin/env python3
"""
128k long-context needle test for Ripple-65B vs Qwen3-Coder-30B-A3B.
Same harness family as local_dev_harness.py: deterministic scoring, greedy decoding.

Design:
- Build a synthetic "repo" of ~60 plausible Python files with realistic names/content.
- Bury 4 distinct facts at different depths (beginning / middle / scattered / end).
- Ask 6 questions (4 needle + 2 no-answer controls to catch hallucination).
- Serve at -c 131072 on both models. Score by exact/substring match of expected answer.

Usage:
  python ctx_test.py --endpoint http://localhost:8484 --tag ripple --out results_ctx_ripple.json
"""
import argparse, json, time, urllib.request

def chat(endpoint, messages, max_tokens=300, timeout=5400):
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": 0}
    req = urllib.request.Request(endpoint.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=timeout))
    return {"content": r["choices"][0]["message"].get("content") or "",
            "ntok": r.get("usage", {}).get("completion_tokens", 0),
            "ptok": r.get("usage", {}).get("prompt_tokens", 0),
            "wall": time.time() - t0}

# ---------- synthetic repo ----------
HEADER = """# Project: forge-ci (synthetic). A fictional CI orchestration tool.
# Files are small excerpts from a larger repo, provided for context.
"""

FILE_TEMPLATE = """# --- {name} ---
{body}
"""

def filler_file(idx):
    bodies = [
        "def retry(func, attempts=3, backoff=1.5):\n    for i in range(attempts):\n        try:\n            return func()\n        except Exception:\n            if i == attempts - 1:\n                raise\n            time.sleep(backoff ** i)\n",
        "class Config:\n    def __init__(self, **kw):\n        self.__dict__.update(kw)\n    def get(self, k, d=None):\n        return self.__dict__.get(k, d)\n",
        "def chunk(items, size):\n    return [items[i:i+size] for i in range(0, len(items), size)]\n",
        "def deep_merge(a, b):\n    out = dict(a)\n    for k, v in b.items():\n        if isinstance(v, dict) and isinstance(out.get(k), dict):\n            out[k] = deep_merge(out[k], v)\n        else:\n            out[k] = v\n    return out\n",
        "class RateLimiter:\n    def __init__(self, max_per_sec):\n        self.max = max_per_sec\n        self.window = []\n    def allow(self):\n        now = time.time()\n        self.window = [t for t in self.window if now - t < 1]\n        if len(self.window) < self.max:\n            self.window.append(now)\n            return True\n        return False\n",
    ]
    name = f"forge_ci/{['utils','config','scheduler','workers','api','authx','logs','cache'][idx % 8]}_{idx // 8}.py"
    return FILE_TEMPLATE.format(name=name, body=bodies[idx % len(bodies)])

# The four buried facts (each in its own file, at controlled positions):
FACTS = [
    ("beginning", "forge_ci/constants_0.py",
     "MAX_RETRY_BACKOFF_SECONDS = 47\n# This value was chosen by the reliability team after the March incident.\n"),
    ("middle", "forge_ci/queue_manager_9.py",
     "QUEUE_HEARTBEAT_INTERVAL = 17  # seconds; do not change without reading RFC-114\n"),
    ("scattered", "forge_ci/auth_22.py",
     "LEGACY_TOKEN_PREFIX = 'fci-v1-'\n# Old tokens (fci-v1-) expire after 30 days; new ones after 90.\n"),
    ("end", "forge_ci/deploy_41.py",
     "DEPLOY_LOCK_NAME = 'forge-deploy-global-lock'\n# Only one deploy lock may exist cluster-wide.\n"),
]

NEEDLE_QUESTIONS = [
    {"q": "What is MAX_RETRY_BACKOFF_SECONDS set to, and who chose it?", "expect": "47", "also": "reliability"},
    {"q": "What is the queue heartbeat interval in seconds, and what document must be read before changing it?", "expect": "17", "also": "RFC-114"},
    {"q": "What is the legacy token prefix string, and how long do old tokens last?", "expect": "fci-v1-", "also": "30"},
    {"q": "What is the name of the deploy lock, and how many may exist cluster-wide?", "expect": "forge-deploy-global-lock", "also": "one"},
]

NOANSWER_QUESTIONS = [
    {"q": "What is the default database connection pool size in this project?", "expect_none": True},
    {"q": "Which license does forge-ci use?", "expect_none": True},
]

def build_context(n_filler=52):
    files = []
    # insert facts at controlled positions: 0 (beginning), 1/3, 2/3, end
    fact_positions = {0: 2, 1: n_filler // 3, 2: 2 * n_filler // 3, 3: n_filler - 2}
    fact_by_pos = {fact_positions[i]: FACTS[i][1:] for i in range(len(FACTS))}
    used_names = set()
    fi = 0
    for i in range(n_filler):
        if i in fact_by_pos:
            files.append(fact_by_pos[i])
        else:
            while True:
                f = filler_file(fi); fi += 1
                fname = f.split("# --- ")[1].split(" ---")[0]
                if fname not in used_names:
                    body = f.split("# --- " + fname + " ---\n")[1]
                    files.append((fname, body)); used_names.add(fname); break
    ctx = HEADER + "\n".join(FILE_TEMPLATE.format(name=fn, body=b) for fn, b in files)
    return ctx

def score_answer(ans, q):
    a = ans.lower()
    hit_main = q["expect"].lower() in a
    hit_also = q.get("also", "").lower() in a if q.get("also") else None
    return {"main": hit_main, "also": hit_also}

def run_model(endpoint, tag, n_filler=2200):
    ctx = build_context(n_filler=n_filler)
    print(f"[{tag}] context chars: {len(ctx)} (~{len(ctx)//4} tokens)", flush=True)
    results = {"tag": tag, "ctx_chars": len(ctx), "items": []}
    for q in NEEDLE_QUESTIONS + NOANSWER_QUESTIONS:
        prompt = f"""You are given an excerpt of a software project below. Answer the question using ONLY the excerpt. If the excerpt does not contain the answer, reply exactly: NOT IN EXCERPT.

--- EXCERPT BEGIN ---
{ctx}
--- EXCERPT END ---

Question: {q['q']}

Answer concisely."""
        r = chat(endpoint, [{"role": "user", "content": prompt}])
        rec = {"q": q["q"][:60], "ntok": r["ntok"], "ptok": r["ptok"], "wall": round(r["wall"], 1),
               "answer": r["content"][:300]}
        if "expect" in q:
            s = score_answer(r["content"], q)
            rec.update({"type": "needle", "main": s["main"], "also": s["also"]})
            print(f"  [{'HIT' if s['main'] else 'MISS'}] {q['q'][:50]} -> {r['content'][:80]!r}", flush=True)
        else:
            refused = "not in excerpt" in r["content"].lower()
            rec.update({"type": "noanswer", "correctly_refused": refused})
            print(f"  [{'OK-REFUSED' if refused else 'HALLUCINATED'}] {q['q'][:50]}", flush=True)
        results["items"].append(rec)
    n_hit = sum(1 for r in results["items"] if r.get("type") == "needle" and r["main"])
    n_ref = sum(1 for r in results["items"] if r.get("type") == "noanswer" and r["correctly_refused"])
    results["needle_hits"] = n_hit; results["needle_total"] = len(NEEDLE_QUESTIONS)
    results["refusals"] = n_ref; results["noanswer_total"] = len(NOANSWER_QUESTIONS)
    print(f"[{tag}] needle {n_hit}/4, refusals {n_ref}/2", flush=True)
    return results

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n_filler", type=int, default=2200)
    args = ap.parse_args()
    out = run_model(args.endpoint, args.tag, n_filler=args.n_filler)
    json.dump(out, open(f"results_ctx_{args.tag}.json", "w"), indent=1)
    print("saved", f"results_ctx_{args.tag}.json")
