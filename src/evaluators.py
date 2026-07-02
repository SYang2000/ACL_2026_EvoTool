"""Per-benchmark success criteria for the SYNTHESIZER's final output (SPEC.md §3).

Dispatched by benchmark name. Not every lane is the benchmark's *official* metric —
each is labelled honestly:

- bfcl      : OFFICIAL gorilla AST match (exact call-count set-match + name match +
              reject unexpected params + strict per-type value match)   [src/eval_bfcl_ast.py]
- taubench  : OFFICIAL reward = final DB-state equality AND all required output
              strings present in the answer                             [src/envs/taubench_env.py]
- restbench : OFFICIAL automatic "Correct Path" — the gold endpoint path is an
              ordered subsequence of the successfully-executed endpoints (arg-agnostic).
- toolbench : OFFLINE APPROXIMATION — a ToolEval-style LLM judge over the synthesizer's
              answer (the official ToolEval pass-rate needs live RapidAPI + a GPT judge,
              not reproducible offline). NOT official.                  [src/eval_toolbench_judge.py]
"""

from __future__ import annotations

from src.metrics import _is_subsequence


def _predicted(steps) -> list[dict]:
    return [{"tool": s.tool, "args": s.args} for s in steps]


def is_success(instance: dict, steps, answer: str, client=None) -> bool:
    b = instance.get("benchmark", "")
    if b == "bfcl":
        # OFFICIAL BFCL paradigm: AST matching of the assembled call list (no execution).
        from src.eval_bfcl_ast import bfcl_ast_success
        return bfcl_ast_success(instance, _predicted(steps))
    if b == "taubench":
        # OFFICIAL tau-bench reward: final DB-state equality AND every required output
        # string present in the synthesizer's answer (SPEC.md §3).
        from src.envs.taubench_env import taubench_success
        return taubench_success(instance, _predicted(steps), answer)
    if b == "toolbench":
        # ToolEval-style LLM judge (OFFLINE APPROX, NOT official): our served model
        # judges whether the synthesizer's answer solves the query given the recorded
        # tool outputs. Needs a live client; when none is available (e.g. offline data
        # verification) fall back to the recorded-call-subsequence heuristic below.
        if client is not None:
            from src.eval_toolbench_judge import toolbench_judge
            return toolbench_judge(instance, answer, steps, client)
        gold = [s.get("tool") for s in instance.get("gold_plan", [])]
        succeeded = [s.tool for s in steps if s.observation.get("status") == "success"]
        return _is_subsequence(gold, succeeded)

    # restbench (Correct Path) + the toy dummy/diverse datasets: the gold endpoint
    # path must be an ordered subsequence of the endpoints that actually executed
    # successfully (arg-agnostic). Only ``status == "success"`` steps count — an
    # "unavailable" or "error" step is never a success (SPEC.md §2/§3).
    gold = [s.get("tool") for s in instance.get("gold_plan", [])]
    succeeded = [s.tool for s in steps if s.observation.get("status") == "success"]
    return _is_subsequence(gold, succeeded)
