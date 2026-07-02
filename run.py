"""Single entry point.

    python run.py --config configs/evotool.yaml --benchmark dummy
    python run.py --config configs/baselines/static.yaml --benchmark bfcl --override evolve.generations=4

Layers the chosen config on top of configs/base.yaml, evolves a policy (a no-op
for static), evaluates it on the eval split, and writes a result JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

import yaml

from src.benchmarks import load_instances, split_instances
from src.config import EvoToolConfig, _merge_into_dataclass, apply_overrides
from src.evolve.loop import evolve
from src.evolve.select import best_policy
from src.llm.client import LLMClient
from src.metrics import headline_score, success_rate
from src.policy.agent import route_episode, run_episode
from src.runlog import RunLog

BASE_CONFIG = "configs/base.yaml"


def build_config(config_path: str) -> EvoToolConfig:
    cfg = EvoToolConfig()
    for path in (BASE_CONFIG, config_path):
        if path and os.path.exists(path):
            with open(path, "r") as f:
                raw = yaml.safe_load(f) or {}
            _merge_into_dataclass(cfg, raw)
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--benchmark", default=None)
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = build_config(args.config)
    if args.benchmark:
        cfg.benchmark = args.benchmark
    apply_overrides(cfg, args.override)
    random.seed(cfg.seed)

    name = os.path.splitext(os.path.basename(args.config))[0]
    print(f"=== {name} on {cfg.benchmark} "
          f"(mutation={cfg.evolve.mutation_target}, selection={cfg.evolve.selection}) ===")

    client = LLMClient(cfg.llm)
    instances = load_instances(cfg.benchmark, cfg.data_path)
    train, sel, test = split_instances(instances, cfg)
    # held-out test must be disjoint from train/sel
    _ids = lambda xs: {x["id"] for x in xs}
    leak = _ids(test) & (_ids(train) | _ids(sel))
    print(f"split: train={len(train)} sel={len(sel)} test={len(test)} (held-out leak={len(leak)})")

    # capture per-generation (epoch) log
    log_lines: list[str] = []
    def epoch_log(msg: str) -> None:
        log_lines.append(msg.strip())
        if cfg.verbose:
            print(msg)

    # Structured run log (SPEC.md §9): one JSONL record per generation
    # (incl. rejected) + a final summary with learning-curve arrays for plotting.
    # Purely observational — it never touches the reward/decision path.
    runlog_path = os.path.join(cfg.output_path, "logs", f"{cfg.benchmark}__{name}.runlog.jsonl")
    run_log = RunLog(runlog_path, meta={
        "preset": name, "benchmark": cfg.benchmark,
        "mutation_target": cfg.evolve.mutation_target, "selection": cfg.evolve.selection,
        "seed": cfg.seed, "epochs": cfg.evolve.epochs, "batch_size": cfg.evolve.batch_size,
        "n_train": len(train), "n_sel": len(sel), "n_test": len(test),
    })

    t0 = time.time()
    population = evolve(client, cfg, train, sel, log=epoch_log, test=test, runlog=run_log)

    # Test-time deployment (SPEC.md §7): default = the SINGLE best policy
    # Theta* = argmax mean S_sel reward. The per-instance route_episode ensemble is
    # kept ONLY as the named optional ablation evolve.test_ensemble (never headline).
    if cfg.evolve.test_ensemble:
        deployed_ids = [p.policy_id for p in population]
        episodes = [route_episode(client, population, x, cfg.max_steps) for x in test]
    else:
        theta_star = best_policy(client, cfg, population, sel)
        deployed_ids = [theta_star.policy_id]
        episodes = [run_episode(client, theta_star, x, cfg.max_steps) for x in test]
    success = success_rate([e.success for e in episodes])
    mean_reward = headline_score([e.reward for e in episodes])
    elapsed = time.time() - t0

    log_dir = os.path.join(cfg.output_path, "logs")
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, f"{cfg.benchmark}__{name}.log"), "w") as lf:
        lf.write(f"# {name} on {cfg.benchmark}  (mutation={cfg.evolve.mutation_target}, "
                 f"selection={cfg.evolve.selection})\n")
        lf.write(f"# split: train={len(train)} sel={len(sel)} test={len(test)} leak={len(leak)}\n")
        lf.write("\n".join(log_lines))
        lf.write(f"\n# FINAL held-out test success={round(success,2)} "
                 f"mean_reward={round(mean_reward,2)} pop={len(population)} "
                 f"ids={[p.policy_id for p in population]}\n")
        lf.write(f"# deploy={'ensemble' if cfg.evolve.test_ensemble else 'single'} "
                 f"deployed_ids={deployed_ids}\n")

    result = {
        "preset": name,
        "benchmark": cfg.benchmark,
        "mutation_target": cfg.evolve.mutation_target,
        "selection": cfg.evolve.selection,
        "score": round(success, 2),          # headline = task success rate (paper metric)
        "success_rate": round(success, 2),
        "mean_reward": round(mean_reward, 2),
        "tokens": client.total_tokens,
        "seconds": round(elapsed, 1),
        "n_train": len(train),
        "n_sel": len(sel),
        "n_test": len(test),
        "held_out_leak": len(leak),
        "population_size": len(population),
        "population_ids": [p.policy_id for p in population],
        "test_ensemble": cfg.evolve.test_ensemble,        # §7: default single Theta*
        "deployed_policy_ids": deployed_ids,              # what actually ran at test
    }
    print(f"--> success={result['success_rate']} mean_reward={result['mean_reward']} "
          f"tokens={result['tokens']} time={result['seconds']}s")

    # Close the structured run log with a final per-run summary (learning curves as
    # arrays + the deployed-policy held-out test result), for offline plotting (§9).
    summary = run_log.finalize(extra={
        "final_test_success": round(success, 2),
        "final_test_mean_reward": round(mean_reward, 2),
        "deploy": "ensemble" if cfg.evolve.test_ensemble else "single",
        "deployed_policy_ids": deployed_ids,
        "population_ids": [p.policy_id for p in population],
        "total_tokens": client.total_tokens,
        "seconds": round(elapsed, 1),
    })
    result["runlog"] = runlog_path
    result["runlog_summary"] = getattr(run_log, "summary_path", None)
    print(f"runlog -> {runlog_path} ({summary['n_generations']} generations)")

    out = args.out or os.path.join(cfg.output_path, f"{cfg.benchmark}__{name}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
