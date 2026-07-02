"""Error-progression / blame-dynamics analysis (paper Figure 4): run EvoTool on a
benchmark, capture per-generation blame targets and mini-batch reward, and show
how blame shifts across modules while reward rises over iterations.

  python scripts/analysis.py taubench
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml

from src.benchmarks import load_instances, split_instances
from src.config import EvoToolConfig, _merge_into_dataclass
from src.evolve.loop import evolve
from src.llm.client import LLMClient
from src.metrics import success_rate
from src.policy.agent import route_episode
from src.policy.modules import MODULES, initial_policy


def cfg_for(benchmark):
    c = EvoToolConfig()
    for p in ("configs/base.yaml", "configs/evotool.yaml"):
        _merge_into_dataclass(c, yaml.safe_load(open(p)) or {})
    c.benchmark = benchmark
    return c


def main():
    b = sys.argv[1] if len(sys.argv) > 1 else "taubench"
    cfg = cfg_for(b)
    client = LLMClient(cfg.llm)
    train, sel, eval_ = split_instances(load_instances(b, "data/"), cfg)

    lines = []
    pop = evolve(client, cfg, train, sel, log=lines.append)

    print(f"=== EvoTool blame dynamics on {b} ===")
    blame_hist = {m: 0 for m in MODULES}
    print(f"{'gen':>3}  {'target':<12} {'parent_avg':>10} {'child_avg':>10}  result")
    for ln in lines:
        m = re.search(r"gen (\d+): parent=\S+ target=(\S+) parent_avg=([\d.]+) child_avg=([\d.]+) (\w+)", ln)
        if not m:
            continue
        g, tgt, pa, ca, res = m.groups()
        base = tgt.split(":")[-1]
        if base in blame_hist:
            blame_hist[base] += 1
        print(f"{int(g):>3}  {tgt:<12} {float(pa):>10.3f} {float(ca):>10.3f}  {res}")

    print(f"\nblame histogram (which module got fixed): {blame_hist}")
    init_succ = success_rate([route_episode(client, [initial_policy()], x, cfg.max_steps).success for x in eval_])
    final_succ = success_rate([route_episode(client, pop, x, cfg.max_steps).success for x in eval_])
    print(f"eval success: initial {init_succ:.1f} -> evolved {final_succ:.1f}  (Δ {final_succ - init_succ:+.1f})")


if __name__ == "__main__":
    main()
