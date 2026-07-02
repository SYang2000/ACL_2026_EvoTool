"""Recover the REAL required output strings (r_outputs) for each tau_retail_<n>.

Spec §3 (SPEC.md): the official tau-bench reward is DB-state equality AND
the task's required output strings being present in the agent's answer. The upstream
retail Task literals carry an ``outputs=[...]`` field (the strings the answer MUST
contain, e.g. a refund amount), but the 150-instance ``data/taubench/samples.json``
dropped it. This script recovers it into a SIDECAR file
``data/taubench/required_outputs.json`` (samples.json is NOT touched), which
``src/benchmarks.load_instances`` attaches to each instance as ``instance['outputs']``
so ``src/envs/taubench_env.taubench_success`` can enforce the r_outputs check.

Recovery is by EXACT instruction-text match: every instance in samples.json carries
the upstream ``instruction`` as its ``query``; we load the upstream retail task files
with light shims (no tau_bench import) to build ``instruction -> outputs`` and look up
each instance. This is robust to the exact selection order used to build samples.json.
A task whose upstream ``outputs`` is empty maps to ``[]`` (reward then reduces to
DB-state equality alone). NO string is fabricated — empty stays empty.

Output schema (data/taubench/required_outputs.json):
  { "tau_retail_<n>": ["<required string>", ...] }

Run:
  python scripts/data/build_taubench_outputs.py --tau-repo /path/to/tau-bench
"""

from __future__ import annotations

import argparse
import json
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAMPLES = os.path.join(REPO, "data", "taubench", "samples.json")
OUT = os.path.join(REPO, "data", "taubench", "required_outputs.json")
RETAIL = None  # set in main() from --tau-repo / TAU_BENCH_REPO

# Canonical upstream source files (same set build_taubench.py drew from). tasks.py is
# a concatenation of these, so we read the split files to avoid double-loading and
# prefer a non-empty outputs list on the rare duplicate instruction.
SOURCES = ["tasks_test.py", "tasks_train.py", "tasks_dev.py"]


class Action:
    def __init__(self, name=None, kwargs=None, **extra):
        pass


class Task:
    """Collects (instruction, outputs) as the upstream literals are exec'd."""

    collected: list = []

    def __init__(self, annotator=None, user_id=None, instruction=None,
                 actions=None, outputs=None, **extra):
        Task.collected.append((instruction, list(outputs or [])))


def load_instruction_outputs() -> dict:
    Task.collected = []
    ns = {"Task": Task, "Action": Action}
    for fn in SOURCES:
        path = os.path.join(RETAIL, fn)
        if not os.path.exists(path):
            continue
        src = open(path, "r", encoding="utf-8").read()
        src = re.sub(r"^from tau_bench\.types import .*$", "", src, flags=re.MULTILINE)
        exec(compile(src, path, "exec"), dict(ns))
    instr2out: dict = {}
    for instruction, outputs in Task.collected:
        if instruction is None:
            continue
        key = instruction.strip()
        # prefer a non-empty outputs list if the same instruction appears twice
        if key not in instr2out or (outputs and not instr2out[key]):
            instr2out[key] = outputs
    return instr2out


def main():
    global RETAIL
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau-repo", default=os.environ.get("TAU_BENCH_REPO"),
                    help="path to a clone of https://github.com/sierra-research/tau-bench")
    args = ap.parse_args()
    if not args.tau_repo or not os.path.isdir(args.tau_repo):
        ap.error("point --tau-repo (or the TAU_BENCH_REPO env var) at a clone of "
                 "https://github.com/sierra-research/tau-bench")
    RETAIL = os.path.join(args.tau_repo, "tau_bench", "envs", "retail")

    instr2out = load_instruction_outputs()
    with open(SAMPLES, "r", encoding="utf-8") as f:
        samples = json.load(f)

    result: dict = {}
    matched = nonempty = unmatched = 0
    for it in samples:
        q = (it.get("query") or "").strip()
        outs = instr2out.get(q)
        if outs is None:
            unmatched += 1
            outs = []          # no upstream match -> no required strings (DB-only)
        else:
            matched += 1
            if outs:
                nonempty += 1
        result[it["id"]] = outs

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  samples={len(samples)} matched={matched} unmatched={unmatched} "
          f"with-required-strings={nonempty}")
    print(f"  wrote {len(result)} entries -> {OUT}")


if __name__ == "__main__":
    main()
