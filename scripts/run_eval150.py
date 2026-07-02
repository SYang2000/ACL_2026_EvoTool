"""Run a (preset x benchmark) matrix across MULTIPLE vLLM servers (one per GPU),
writing results + per-job logs to --outdir (default: results/).

Jobs are round-robined over --ports so the 4 GPUs are all kept busy; per-job
stderr is captured for debugging. Builds a Markdown success-rate table at the end.

  python scripts/run_eval150.py --table ablation_blame --benchmarks restbench bfcl taubench toolbench
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time

PY = sys.executable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = "results/"
SEED = 42
GENS = None  # optional evolve.generations override (budget ablation)

spec = importlib.util.spec_from_file_location("rm", os.path.join(ROOT, "scripts", "run_matrix.py"))
rm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rm)
TABLES = rm.TABLES


def run_jobs(jobs, ports, concurrency):
    os.makedirs(os.path.join(OUTDIR, "logs"), exist_ok=True)
    running, pending = [], list(enumerate(jobs))
    while pending or running:
        while pending and len(running) < concurrency:
            idx, (preset, _label, b) = pending.pop(0)
            name = os.path.basename(preset)
            port = ports[idx % len(ports)]
            errf = open(os.path.join(OUTDIR, "logs", f"{b}__{name}.err"), "w")
            cmd = [PY, "run.py", "--config", f"configs/{preset}.yaml", "--benchmark", b,
                   "--override", f"output_path={OUTDIR}",
                   f"llm.base_url=http://127.0.0.1:{port}/v1", f"seed={SEED}"]
            if GENS is not None:
                cmd.append(f"evolve.generations={GENS}")
            p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=errf)
            running.append((p, preset, b, port, errf, time.time()))
        done = [r for r in running if r[0].poll() is not None]
        for r in done:
            p, preset, b, port, errf, t0 = r
            errf.close()
            running.remove(r)
            tag = "ok" if p.returncode == 0 else f"FAIL({p.returncode})"
            print(f"  [{tag}] {b}__{os.path.basename(preset)} (gpu:{port}) "
                  f"{time.time()-t0:.0f}s", flush=True)
        if running and not done:
            time.sleep(5)


def cell(label_preset, b):
    name = os.path.basename(label_preset)
    try:
        return json.load(open(os.path.join(OUTDIR, f"{b}__{name}.json")))["success_rate"]
    except Exception:
        return None


def main():
    global OUTDIR, SEED, GENS
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True, choices=list(TABLES))
    ap.add_argument("--benchmarks", nargs="+", required=True)
    ap.add_argument("--ports", nargs="+", type=int, default=[8000, 8001, 8002, 8003])
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--outdir", default=OUTDIR)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--gens", type=int, default=None)
    args = ap.parse_args()
    OUTDIR = args.outdir
    SEED = args.seed
    GENS = args.gens

    presets = TABLES[args.table]
    jobs = [(preset, label, b) for (preset, label) in presets for b in args.benchmarks]
    print(f"{args.table}: {len(jobs)} jobs over {len(args.ports)} GPUs, concurrency={args.concurrency}",
          flush=True)
    t0 = time.time()
    run_jobs(jobs, args.ports, args.concurrency)
    print(f"all jobs done in {(time.time()-t0)/60:.1f} min", flush=True)

    header = "| Method | " + " | ".join(args.benchmarks) + " | **Avg** |"
    sep = "|" + "---|" * (len(args.benchmarks) + 2)
    lines = [f"### {args.table} (held-out test success rate, 150-sample / 90-30-30 split)",
             "", header, sep]
    for preset, label in presets:
        cells = [cell(preset, b) for b in args.benchmarks]
        vals = [c for c in cells if c is not None]
        avg = sum(vals) / len(vals) if vals else 0.0
        row = " | ".join(f"{c:.1f}" if c is not None else "ERR" for c in cells)
        lines.append(f"| {label} | {row} | **{avg:.1f}** |")
    md = "\n".join(lines)
    print("\n" + md)
    with open(os.path.join(OUTDIR, f"TABLE_{args.table}.md"), "w") as f:
        f.write(md + "\n")
    print(f"\nsaved -> {OUTDIR}/TABLE_{args.table}.md")


if __name__ == "__main__":
    main()
