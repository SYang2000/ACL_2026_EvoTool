"""Per-generation evolution trace: for each benchmark, show which module was
blamed/mutated at every generation and whether the child was accepted, plus the
final retained population and the evolved prompt text.

  python scripts/evolution_trace.py            # all four real benchmarks
  python scripts/evolution_trace.py restbench  # one
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


def run(b: str) -> None:
    cfg = EvoToolConfig()
    for p in ("configs/base.yaml", "configs/evotool.yaml"):
        _merge_into_dataclass(cfg, yaml.safe_load(open(p)) or {})
    cfg.benchmark = b
    client = LLMClient(cfg.llm)
    train, sel, _ = split_instances(load_instances(b, "data/"), cfg)

    lines: list[str] = []
    pop = evolve(client, cfg, train, sel, log=lambda s: lines.append(s.strip()))

    print(f"\n========== {b}  (G={cfg.evolve.generations}) ==========")
    print(f"{'gen':>3}  {'target':12} {'parent_avg':>10} {'child_avg':>9}  verdict")
    for ln in lines:
        # "gen g: parent=ID target=T parent_avg=.. child_avg=.. ACCEPT/reject | pop=N"
        d = dict(tok.split("=", 1) for tok in ln.replace(":", "").split()
                 if "=" in tok)
        g = ln.split(":", 1)[0].replace("gen", "").strip()
        verdict = "ACCEPT" if "ACCEPT" in ln else "reject"
        print(f"{g:>3}  {d.get('target',''):12} {float(d.get('parent_avg',0)):>10.3f} "
              f"{float(d.get('child_avg',0)):>9.3f}  {verdict}")

    init = initial_policy()
    changed = [m for m in MODULES if any(p.spec(m) != init.spec(m) for p in pop)]
    print(f"  retained population : {[p.policy_id for p in pop]}")
    print(f"  modules evolved     : {changed or '(none)'}")

    # initial vs evolved prompt text for every module any retained policy changed
    for m in changed:
        evolved_p = next(p for p in pop if p.spec(m) != init.spec(m))
        print(f"\n  ----- MODULE {m.upper()} : INITIAL -> EVOLVED ({evolved_p.policy_id}) -----")
        print(f"  [initial] {init.spec(m)}")
        print(f"  [evolved] {evolved_p.spec(m)}")


def main():
    benches = sys.argv[1:] or ["restbench", "taubench", "bfcl", "toolbench"]
    for b in benches:
        run(b)


if __name__ == "__main__":
    main()
