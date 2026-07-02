"""Reproducible converter: BFCL v4 real tasks -> 150-instance samples.json.

Source (real data, no fabrication) — gorilla Berkeley-Function-Call-Leaderboard
v4 checkout (pass --bfcl-raw; see DATA.md). Questions live under `bfcl_eval/data`
and the matching ground truth under its `possible_answer/` subdirectory.

  questions     : <bfcl-raw>/BFCL_v4_<cat>.json
                  (each line: {id, question:[[{role,content}...]], function:[...]})
  ground truth  : <bfcl-raw>/possible_answer/BFCL_v4_<cat>.json
                  (each line: {id, ground_truth:[{func:{arg:[allowed...]}}, ...]})

We draw from the HARDER multi-call categories (every selected instance has a
>=2-call gold), preferring parallel_multiple (parallel + multi-tool selection),
then parallel (parallel calls to one tool), then live_parallel_multiple (real
user queries). Deterministic: take the first N entries of each category in
upstream id order.

Mapping to the repo unified schema (matches data/bfcl/samples.json template):
  id              = "bfcl_<n>"  (1-based, in selection order)
  query           = concatenation of the user-turn contents of question[0]
  available_tools = [{name, description, parameters:{argname: description}}, ...]
                    (BFCL `function` specs flattened to the template's param shape)
  gold_plan       = [{tool: func, args:{arg: FIRST allowed value}}, ...]
                    (first allowed value per arg; pure-optional args omitted)
  gold_match      = the raw BFCL ground_truth list (drives the official metric)
  gold_answer     = None
  mock_outputs    = {}

The official metric (src/eval_bfcl.bfcl_success) AST-matches gold_plan against
gold_match; we KEEP only instances whose gold_plan passes it (gold is correct by
construction) and stop at 150.

Run:
  python scripts/data/build_bfcl.py --bfcl-raw /path/to/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data
"""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "data", "bfcl", "samples.json")
BFCL_RAW = None  # set in main() from --bfcl-raw / BFCL_RAW_DIR
BFCL_GT = None   # set in main() from --bfcl-gt (defaults to <bfcl-raw>/possible_answer)

sys.path.insert(0, REPO)
from src.eval_bfcl import bfcl_success  # noqa: E402

TARGET = 150

# (category, how many to take from the FIRST entries of that category, in id order)
# All three categories are 100% multi-call (>=2 gold calls). 90+40+20 = 150.
PLAN = [
    ("parallel_multiple", 90),
    ("parallel", 40),
    ("live_parallel_multiple", 20),
]


def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_query(question) -> str:
    """Join the user-turn contents of the (single-turn) BFCL question."""
    # question is [[{role, content}, ...]] (a list of conversation turns).
    convo = question[0] if question and isinstance(question[0], list) else question
    users = [m.get("content", "") for m in convo if m.get("role") == "user"]
    if not users:  # defensive: fall back to all message contents
        users = [m.get("content", "") for m in convo]
    return " ".join(u for u in users if u).strip()


def flatten_tools(function_specs: list[dict]) -> list[dict]:
    """BFCL `function` spec -> template tool shape {name, description, parameters:{arg: desc}}."""
    tools = []
    for fn in function_specs:
        params = fn.get("parameters", {}) or {}
        props = params.get("properties", {}) or {}
        flat = {arg: (pv.get("description", "") or "") for arg, pv in props.items()}
        tools.append({
            "name": fn.get("name", ""),
            "description": fn.get("description", "") or "",
            "parameters": flat,
        })
    return tools


def build_gold_plan(ground_truth: list[dict]) -> list[dict]:
    """One {tool, args} per gold func, using the FIRST allowed value per arg.

    The "" sentinel marks an optional arg; if it is the first allowed value we
    take the first concrete (non-"") value if one exists, else omit the arg.
    """
    plan = []
    for spec in ground_truth:
        (func, args), = spec.items()
        out_args = {}
        for arg, allowed in args.items():
            if not allowed:
                continue
            first = allowed[0]
            if first == "":
                nonempty = [v for v in allowed if v != ""]
                if nonempty:
                    out_args[arg] = nonempty[0]
                # else: optional arg, omit it
                continue
            out_args[arg] = first
        plan.append({"tool": func, "args": out_args})
    return plan


def main():
    global BFCL_RAW, BFCL_GT
    ap = argparse.ArgumentParser()
    ap.add_argument("--bfcl-raw", default=os.environ.get("BFCL_RAW_DIR"),
                    help="path to bfcl_eval/data inside a clone of "
                         "https://github.com/ShishirPatil/gorilla (berkeley-function-call-leaderboard)")
    ap.add_argument("--bfcl-gt", default=None,
                    help="path to the possible_answer dir (default: <bfcl-raw>/possible_answer)")
    args = ap.parse_args()
    if not args.bfcl_raw or not os.path.isdir(args.bfcl_raw):
        ap.error("point --bfcl-raw (or the BFCL_RAW_DIR env var) at the bfcl_eval/data dir of a "
                 "clone of https://github.com/ShishirPatil/gorilla (see DATA.md)")
    BFCL_RAW = args.bfcl_raw
    BFCL_GT = args.bfcl_gt or os.path.join(BFCL_RAW, "possible_answer")
    if not os.path.isdir(BFCL_GT):
        ap.error(f"ground-truth dir not found: {BFCL_GT} (see DATA.md)")

    selected = []
    kept_per_cat = {}
    for cat, n in PLAN:
        q = load_jsonl(os.path.join(BFCL_RAW, f"BFCL_v4_{cat}.json"))
        a = load_jsonl(os.path.join(BFCL_GT, f"BFCL_v4_{cat}.json"))
        gt_by_id = {x["id"]: x["ground_truth"] for x in a}

        taken = 0
        for entry in q:
            if taken >= n:
                break
            qid = entry["id"]
            gt = gt_by_id.get(qid)
            if gt is None or len(gt) < 2:  # require real multi-call gold
                continue

            inst = {
                "id": f"bfcl_{len(selected) + 1}",
                "query": extract_query(entry["question"]),
                "available_tools": flatten_tools(entry.get("function", [])),
                "gold_plan": build_gold_plan(gt),
                "gold_match": gt,
                "gold_answer": None,
                "mock_outputs": {},
            }
            # gold MUST pass the official AST metric (gold correct by construction)
            if not bfcl_success({"gold_match": gt}, inst["gold_plan"]):
                continue
            selected.append(inst)
            taken += 1
        kept_per_cat[cat] = taken
        print(f"  {cat:24} kept {taken}/{n}")

    print(f"  total selected = {len(selected)}")
    assert len(selected) == TARGET, f"expected {TARGET}, got {len(selected)}"

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2)
    print(f"  wrote {len(selected)} instances -> {OUT}")


if __name__ == "__main__":
    main()
