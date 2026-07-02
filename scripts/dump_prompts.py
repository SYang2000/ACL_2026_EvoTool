"""Evolve a policy on a benchmark and print INITIAL vs EVOLVED module prompts
(the paper's Appendix A.7 vs A.8 comparison).

  python scripts/dump_prompts.py restbench
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml

from src.benchmarks import load_instances, split_instances
from src.config import EvoToolConfig, _merge_into_dataclass
from src.evolve.loop import evolve
from src.llm.client import LLMClient
from src.policy.modules import MODULES, initial_policy


def main():
    b = sys.argv[1] if len(sys.argv) > 1 else "restbench"
    cfg = EvoToolConfig()
    for p in ("configs/base.yaml", "configs/evotool.yaml"):
        _merge_into_dataclass(cfg, yaml.safe_load(open(p)) or {})
    cfg.benchmark = b
    client = LLMClient(cfg.llm)
    train, sel, _ = split_instances(load_instances(b, "data/"), cfg)
    pop = evolve(client, cfg, train, sel, log=lambda *_: None)
    init = initial_policy()
    best = max(pop, key=lambda p: sum(p.spec(m) != init.spec(m) for m in MODULES))

    out = [f"benchmark: {b}   final population ids: {[p.policy_id for p in pop]}",
           f"showing most-evolved policy: {best.policy_id}", "=" * 80]
    for m in MODULES:
        changed = best.spec(m) != init.spec(m)
        out.append(f"\n######### MODULE: {m}  {'[EVOLVED]' if changed else '[unchanged]'} #########")
        out.append(f"--- INITIAL (paper A.7) ---\n{init.spec(m)}")
        if changed:
            out.append(f"\n--- EVOLVED (paper A.8 analog) ---\n{best.spec(m)}")
    text = "\n".join(out)
    print(text)
    with open(f"results/evolved_prompts_{b}.txt", "w") as f:
        f.write(text)


if __name__ == "__main__":
    main()
