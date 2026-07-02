"""Data quality gate: a well-formed samples.json must have its GOLD plan pass the
benchmark's OFFICIAL success metric (gold is correct by construction). A low pass
rate means the data is malformed (wrong tool names, wrong gold args, broken schema).

  python scripts/verify_data.py restbench bfcl taubench toolbench
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.benchmarks import load_instances
from src.evaluators import is_success


class GoldStep:
    """Minimal Step stand-in: replays each gold tool call as a successful step."""
    def __init__(self, tool, args):
        self.tool = tool
        self.args = args or {}
        self.observation = {"status": "success", "output": "ok"}


def verify(benchmark: str, data_path: str = "data/") -> tuple[int, int]:
    insts = load_instances(benchmark, data_path)
    ok = 0
    bad = []
    for x in insts:
        steps = [GoldStep(s.get("tool"), s.get("args")) for s in x.get("gold_plan", [])]
        # answer=None selects gold-validation mode: tau-bench checks DB-state only
        # (the gold plan has no natural-language answer for the r_outputs check), and
        # toolbench falls back to the offline call-subsequence heuristic (no judge
        # client here). bfcl/restbench ignore the answer argument entirely.
        if is_success(x, steps, None):
            ok += 1
        else:
            bad.append(x.get("id"))
    pct = 100.0 * ok / len(insts) if insts else 0.0
    print(f"{benchmark:10}: n={len(insts):3}  gold-passes-official-metric = {ok}/{len(insts)} = {pct:.1f}%")
    if bad:
        print(f"            failing ids (first 10): {bad[:10]}")
    return ok, len(insts)


def main():
    benches = sys.argv[1:] or ["restbench", "bfcl", "taubench", "toolbench"]
    for b in benches:
        try:
            verify(b)
        except Exception as e:  # noqa: BLE001
            print(f"{b:10}: ERROR {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
