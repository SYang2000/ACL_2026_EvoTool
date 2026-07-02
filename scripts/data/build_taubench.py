"""Reproducible converter: tau-bench RETAIL real tasks -> 150-instance samples.json.

Source (real data, no fabrication):
  tau-bench checkout retail task files (Task/Action python literals):
    tau_bench/envs/retail/tasks_train.py   (TASKS_TRAIN, ~500 tasks)
    tau_bench/envs/retail/tasks_test.py    (TASKS_TEST,  ~115 tasks)
    tau_bench/envs/retail/tasks_dev.py     (TASKS_DEV,   ~20 tasks)

Each upstream Task has: user_id, instruction, actions=[Action(name=..., kwargs=...)].
We map it to the repo unified schema:
  id              = "tau_retail_<n>"
  query           = task.instruction
  available_tools = the 15 retail tools (copied verbatim from the vendored
                    scripts/data/taubench_tools.json template)
  gold_plan       = [{"tool": action.name, "args": action.kwargs}, ...]
  gold_answer     = None
  mock_outputs    = {}

The official metric (src/envs/taubench_env.taubench_success) replays gold_plan
against the vendored retail DB and requires a valid final DB state. Read-only
tool calls are no-ops on state, so a gold_plan that contains ONLY the upstream
write/lookup actions is sufficient. We KEEP every task whose gold_plan passes the
official metric (gold is correct by construction) and select the first 150 such
tasks in deterministic source order (train -> test -> dev).

Run:
  python scripts/data/build_taubench.py --tau-repo /path/to/tau-bench
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "data", "taubench", "samples.json")
# canonical 15-tool list, vendored so the build is self-contained
TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "taubench_tools.json")
RETAIL = None  # set in main() from --tau-repo / TAU_BENCH_REPO

sys.path.insert(0, REPO)
from src.envs.taubench_env import taubench_success  # noqa: E402

TARGET = 150


# --------------------------------------------------------------------------- #
# Minimal Task/Action shims so we can exec the upstream task literals WITHOUT
# importing the tau_bench package (which transitively pulls in litellm).
# --------------------------------------------------------------------------- #
class Action:
    def __init__(self, name=None, kwargs=None, **extra):
        self.name = name
        self.kwargs = kwargs if kwargs is not None else {}


class Task:
    def __init__(self, annotator=None, user_id=None, instruction=None,
                 actions=None, outputs=None, **extra):
        self.user_id = user_id
        self.instruction = instruction
        self.actions = actions or []


def load_tasks(filename: str, list_var: str):
    """Exec one upstream task file with the shims and return its task list."""
    path = os.path.join(RETAIL, filename)
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    # drop the `from tau_bench.types import Task, Action` line; we inject shims
    src = re.sub(r"^from tau_bench\.types import .*$", "", src, flags=re.MULTILINE)
    ns = {"Task": Task, "Action": Action}
    exec(compile(src, path, "exec"), ns)
    return ns[list_var]


def load_template_tools():
    with open(TEMPLATE, "r", encoding="utf-8") as f:
        return json.load(f)


def task_to_instance(task, idx, tools):
    gold_plan = [{"tool": a.name, "args": a.kwargs} for a in task.actions]
    return {
        "id": f"tau_retail_{idx}",
        "query": task.instruction or "",
        "available_tools": tools,
        "gold_plan": gold_plan,
        "gold_answer": None,
        "mock_outputs": {},
    }


def main():
    global RETAIL
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau-repo", default=os.environ.get("TAU_BENCH_REPO"),
                    help="path to a clone of https://github.com/sierra-research/tau-bench")
    args = ap.parse_args()
    if not args.tau_repo or not os.path.isdir(args.tau_repo):
        ap.error("point --tau-repo (or the TAU_BENCH_REPO env var) at a clone of "
                 "https://github.com/sierra-research/tau-bench (see DATA.md)")
    RETAIL = os.path.join(args.tau_repo, "tau_bench", "envs", "retail")

    tools = load_template_tools()
    assert len(tools) == 15, f"expected 15 retail tools, got {len(tools)}"

    # Deterministic source order: test (canonical eval set) -> train -> dev.
    sources = [
        ("tasks_test.py", "TASKS_TEST"),
        ("tasks_train.py", "TASKS_TRAIN"),
        ("tasks_dev.py", "TASKS_DEV"),
    ]
    all_tasks = []
    for fn, var in sources:
        ts = load_tasks(fn, var)
        all_tasks.extend(ts)
        print(f"  loaded {len(ts):4d} from {fn}")
    print(f"  total upstream tasks: {len(all_tasks)}")

    selected = []
    skipped = 0
    seen_keys = set()
    for task in all_tasks:
        if len(selected) >= TARGET:
            break
        # require a non-empty write/lookup action list
        if not task.actions:
            skipped += 1
            continue
        # dedupe identical (instruction, gold action) tasks
        key = (
            task.instruction,
            tuple((a.name, json.dumps(a.kwargs, sort_keys=True)) for a in task.actions),
        )
        if key in seen_keys:
            continue
        inst = task_to_instance(task, len(selected) + 1, tools)
        # gold MUST reach a valid DB state under the official metric
        if not taubench_success(inst, inst["gold_plan"]):
            skipped += 1
            continue
        seen_keys.add(key)
        selected.append(inst)

    print(f"  selected={len(selected)}  skipped(no-actions/gold-fail)={skipped}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2)
    print(f"  wrote {len(selected)} instances -> {OUT}")


if __name__ == "__main__":
    main()
