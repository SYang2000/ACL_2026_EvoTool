"""Transferability analysis (paper Figure 5): evolve a policy on benchmark A,
then evaluate the resulting population on benchmark B (whose tools it never saw).
The evolved module *specs* are tool-agnostic, so improvements should transfer.

  python scripts/transfer.py restbench taubench
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml

from src.benchmarks import load_instances, split_instances
from src.config import EvoToolConfig, _merge_into_dataclass
from src.evolve.loop import evolve
from src.llm.client import LLMClient
from src.metrics import success_rate
from src.policy.agent import route_episode
from src.policy.modules import initial_policy


def cfg_for(benchmark):
    c = EvoToolConfig()
    for p in ("configs/base.yaml", "configs/evotool.yaml"):
        _merge_into_dataclass(c, yaml.safe_load(open(p)) or {})
    c.benchmark = benchmark
    return c


def succ_on(client, population, instances, max_steps):
    return success_rate([route_episode(client, population, x, max_steps).success for x in instances])


def main():
    A, B = sys.argv[1], sys.argv[2]
    cfgA, cfgB = cfg_for(A), cfg_for(B)
    client = LLMClient(cfgA.llm)

    trainA, selA, _ = split_instances(load_instances(A, "data/"), cfgA)
    _, _, evalB = split_instances(load_instances(B, "data/"), cfgB)

    print(f"evolving on {A} ...")
    popA = evolve(client, cfgA, trainA, selA, log=lambda *_: None)

    transfer = succ_on(client, popA, evalB, cfgB.max_steps)
    static = succ_on(client, [initial_policy()], evalB, cfgB.max_steps)
    print(f"\n=== Transfer {A} -> {B} ===")
    print(f"  static on {B}:               {static:.1f}")
    print(f"  {A}-evolved policy on {B}:   {transfer:.1f}   (Δ = {transfer - static:+.1f})")


if __name__ == "__main__":
    main()
