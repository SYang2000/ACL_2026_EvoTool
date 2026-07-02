"""Run a (preset x benchmark) table across multiple SEEDS and average, to get
low-variance numbers (single small-sample runs are noisy under vLLM batching +
stochastic search). Reports mean success rate per cell + Avg column.

  python scripts/run_seeds.py --table ablation_blame --benchmarks restbench toolbench taubench bfcl --seeds 42 43 44
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

PY = sys.executable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import importlib.util
spec = importlib.util.spec_from_file_location("rm", os.path.join(ROOT, "scripts", "run_matrix.py"))
rm = importlib.util.module_from_spec(spec); spec.loader.exec_module(rm)
TABLES = rm.TABLES


def run_jobs(jobs, concurrency):
    running, pending = [], list(jobs)
    while pending or running:
        while pending and len(running) < concurrency:
            preset, b, seed, out = pending.pop(0)
            cmd = [PY, "run.py", "--config", f"configs/{preset}.yaml", "--benchmark", b,
                   "--override", f"seed={seed}", "--out", out]
            running.append((subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.DEVNULL,
                                             stderr=subprocess.DEVNULL), out))
        done = [(p, o) for (p, o) in running if p.poll() is not None]
        for p, o in done:
            running.remove((p, o)); print(f"  done {os.path.basename(o)}", flush=True)
        if running and not done:
            running[0][0].wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True, choices=list(TABLES))
    ap.add_argument("--benchmarks", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--out", default="results/SUMMARY.md")
    args = ap.parse_args()

    presets = TABLES[args.table]
    jobs, paths = [], {}
    for preset, label in presets:
        for b in args.benchmarks:
            for s in args.seeds:
                out = os.path.join(ROOT, "results", f"seed_{os.path.basename(preset)}_{b}_{s}.json")
                jobs.append((preset, b, s, out))
                paths[(label, b, s)] = out
    print(f"running {len(jobs)} jobs ({len(presets)}x{len(args.benchmarks)}x{len(args.seeds)} seeds) ...", flush=True)
    run_jobs(jobs, args.concurrency)

    def cell(label, b):
        vals = []
        for s in args.seeds:
            try:
                vals.append(json.load(open(paths[(label, b, s)]))["success_rate"])
            except Exception:
                pass
        return sum(vals) / len(vals) if vals else 0.0

    header = "| Method | " + " | ".join(args.benchmarks) + " | **Avg** |"
    sep = "|" + "---|" * (len(args.benchmarks) + 2)
    lines = [f"### Table: {args.table} (mean success rate over seeds {args.seeds})", "", header, sep]
    for _, label in presets:
        cells = [cell(label, b) for b in args.benchmarks]
        avg = sum(cells) / len(cells)
        row = " | ".join(f"{c:.1f}" for c in cells)
        lines.append(f"| {label} | {row} | **{avg:.1f}** |")
    md = "\n".join(lines)
    print("\n" + md)
    with open(os.path.join(ROOT, args.out), "a") as f:
        f.write("\n\n" + md + "\n")


if __name__ == "__main__":
    main()
