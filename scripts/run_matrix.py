"""Run a (preset x benchmark) matrix, each in an isolated process (so the
episode cache never leaks across runs), and emit a Markdown results table.

Examples:
  python scripts/run_matrix.py --table main --benchmarks toolbench restbench taubench bfcl
  python scripts/run_matrix.py --table main --benchmarks dummy
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

PY = sys.executable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# preset path (relative to configs/, without .yaml) -> display label
TABLES = {
    "main": [
        ("evotool", "EvoTool"),
    ],
}


def _cmd(preset: str, benchmark: str, overrides: list[str]) -> list[str]:
    name = os.path.basename(preset)
    cmd = [PY, "run.py", "--config", f"configs/{preset}.yaml", "--benchmark", benchmark]
    if overrides:
        cmd += ["--override"] + overrides
    return cmd


def _result(preset: str, benchmark: str) -> dict:
    name = os.path.basename(preset)
    with open(os.path.join(ROOT, "results", f"{benchmark}__{name}.json")) as f:
        return json.load(f)


def run_jobs(jobs: list[tuple], overrides: list[str], concurrency: int) -> None:
    """Run (preset, benchmark) jobs with bounded concurrency; vLLM batches the
    concurrent request streams, so this is much faster than sequential."""
    running: list[tuple] = []
    pending = list(jobs)
    while pending or running:
        while pending and len(running) < concurrency:
            preset, _, b = pending.pop(0)
            p = subprocess.Popen(_cmd(preset, b, overrides), cwd=ROOT,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            running.append((p, preset, b))
        done = [(p, pr, b) for (p, pr, b) in running if p.poll() is not None]
        for p, pr, b in done:
            running.remove((p, pr, b))
            tag = "ok" if p.returncode == 0 else f"FAIL({p.returncode})"
            print(f"  finished {os.path.basename(pr):20s} {b:12s} [{tag}]", flush=True)
        if running and not done:
            running[0][0].wait()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True, choices=list(TABLES))
    ap.add_argument("--benchmarks", nargs="+", required=True)
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--out", default="results/SUMMARY.md")
    args = ap.parse_args()

    presets = TABLES[args.table]
    jobs = [(preset, label, b) for (preset, label) in presets for b in args.benchmarks]
    print(f"running {len(jobs)} jobs at concurrency {args.concurrency} ...", flush=True)
    run_jobs(jobs, args.override, args.concurrency)

    grid: dict = {}
    for preset, label in presets:
        for b in args.benchmarks:
            grid[(label, b)] = _result(preset, b)

    # build markdown table: rows = presets, cols = benchmarks (+ avg)
    header = "| Method | " + " | ".join(args.benchmarks) + " | **Avg** |"
    sep = "|" + "---|" * (len(args.benchmarks) + 2)
    lines = [f"### Table: {args.table}  (score = mean reward x100)", "", header, sep]
    for _, label in presets:
        scores = [grid[(label, b)]["score"] for b in args.benchmarks]
        avg = sum(scores) / len(scores)
        cells = " | ".join(f"{s:.1f}" for s in scores)
        bold = "**" if "EvoTool" in label and "ours" in label or label.startswith("EvoTool") else ""
        lines.append(f"| {bold}{label}{bold} | {cells} | {bold}{avg:.1f}{bold} |")
    table_md = "\n".join(lines)

    print("\n" + table_md)
    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "a") as f:
        f.write("\n\n" + table_md + "\n")
    print(f"\nappended -> {out}")


if __name__ == "__main__":
    main()
