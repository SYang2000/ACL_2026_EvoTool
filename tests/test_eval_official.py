"""Negative-control unit checks for the per-benchmark evaluators (SPEC.md §3).

Server-free: every check is pure logic (the toolbench LLM judge is exercised with a
stub client). Run directly::

    python tests/test_eval_official.py

The point is the NEGATIVE controls: a wrong argument, an extra call, an unexpected
parameter, an out-of-order path, and a missing required output string must each make
the official check return False. A clean gold-style input must return True.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval_bfcl_ast import bfcl_ast_success
from src.envs.taubench_env import taubench_success
from src.evaluators import is_success


# --------------------------------------------------------------------------- #
# tiny stand-ins for agent.Step and the LLM client
# --------------------------------------------------------------------------- #
class FakeStep:
    def __init__(self, tool, args=None, status="success", output="ok"):
        self.tool = tool
        self.args = args or {}
        self.observation = {"tool": tool, "args": self.args,
                            "status": status, "output": output}


class FakeJudgeClient:
    """Stub LLM client: returns a fixed judge verdict (no server)."""
    def __init__(self, solved):
        self._solved = solved
    def generate_json(self, messages):
        return {"solved": self._solved, "reason": "stub"}


_PASS = 0
_FAIL = 0


def check(name, cond):
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f"  PASS  {name}")
    else:
        _FAIL += 1
        print(f"  FAIL  {name}")


# --------------------------------------------------------------------------- #
# BFCL: gorilla AST checker
# --------------------------------------------------------------------------- #
def test_bfcl():
    print("[bfcl] gorilla AST checker")
    # gold: f(a=number 1, b=string "x"); g(n=number 5)
    gold = {"gold_match": [
        {"f": {"a": [1], "b": ["x"]}},
        {"g": {"n": [5]}},
    ]}
    ok = [{"tool": "f", "args": {"a": 1, "b": "x"}},
          {"tool": "g", "args": {"n": 5}}]
    check("gold-style calls PASS", bfcl_ast_success(gold, ok) is True)

    # NEGATIVE: wrong argument value
    bad_val = [{"tool": "f", "args": {"a": 2, "b": "x"}},
               {"tool": "g", "args": {"n": 5}}]
    check("wrong arg value -> FAIL", bfcl_ast_success(gold, bad_val) is False)

    # NEGATIVE: extra call (count mismatch)
    extra = ok + [{"tool": "g", "args": {"n": 5}}]
    check("extra call -> FAIL", bfcl_ast_success(gold, extra) is False)

    # NEGATIVE: missing call (count mismatch)
    check("missing call -> FAIL", bfcl_ast_success(gold, ok[:1]) is False)

    # NEGATIVE: unexpected parameter
    unexpected = [{"tool": "f", "args": {"a": 1, "b": "x", "c": 9}},
                  {"tool": "g", "args": {"n": 5}}]
    check("unexpected param -> FAIL", bfcl_ast_success(gold, unexpected) is False)

    # NEGATIVE: missing required parameter
    missing_req = [{"tool": "f", "args": {"a": 1}},
                   {"tool": "g", "args": {"n": 5}}]
    check("missing required param -> FAIL", bfcl_ast_success(gold, missing_req) is False)

    # NEGATIVE: loose str/int coercion must NOT match ("1" != 1, "5" != 5)
    coerced = [{"tool": "f", "args": {"a": "1", "b": "x"}},
               {"tool": "g", "args": {"n": "5"}}]
    check("str/int coercion -> FAIL", bfcl_ast_success(gold, coerced) is False)

    # NEGATIVE: bool must NOT match int 1
    gold_bool = {"gold_match": [{"h": {"flag": [True]}}]}
    check("int 1 vs bool True -> FAIL",
          bfcl_ast_success(gold_bool, [{"tool": "h", "args": {"flag": 1}}]) is False)
    check("bool True vs bool True -> PASS",
          bfcl_ast_success(gold_bool, [{"tool": "h", "args": {"flag": True}}]) is True)

    # POSITIVE: optional param (allowed contains "") may be omitted; int==float ok
    gold_opt = {"gold_match": [{"f": {"a": [1.0], "b": ["x", ""]}}]}
    check("optional param omitted + int/float number match -> PASS",
          bfcl_ast_success(gold_opt, [{"tool": "f", "args": {"a": 1}}]) is True)
    check("optional param wrong value -> FAIL",
          bfcl_ast_success(gold_opt, [{"tool": "f", "args": {"a": 1, "b": "y"}}]) is False)


# --------------------------------------------------------------------------- #
# tau-bench: DB-state AND required output strings
# --------------------------------------------------------------------------- #
def test_taubench():
    print("[taubench] DB-state AND required output strings (r_outputs)")
    # empty gold plan + empty predicted actions -> DB state trivially equal, so the
    # r_outputs check is what decides here (isolates the NEW logic).
    inst = {"gold_plan": [], "outputs": ["10"]}
    check("DB-equal + required string present -> PASS",
          taubench_success(inst, [], "the total is 10 dollars") is True)
    # NEGATIVE: required output string missing from the answer
    check("DB-equal + required string MISSING -> FAIL",
          taubench_success(inst, [], "no number in this answer") is False)
    # NEGATIVE: multiple required strings, one missing
    inst2 = {"gold_plan": [], "outputs": ["54.04", "41.64"]}
    check("one of two required strings missing -> FAIL",
          taubench_success(inst2, [], "only 54.04 here") is False)
    check("both required strings present -> PASS",
          taubench_success(inst2, [], "54.04 and 41.64") is True)
    # no required strings -> reduces to DB-state equality (empty answer OK)
    check("no required strings -> DB-state only PASS",
          taubench_success({"gold_plan": [], "outputs": []}, [], "") is True)
    # gold-validation mode (answer=None) -> DB-state only, ignores r_outputs
    check("answer=None (gold validation) -> DB-state only PASS",
          taubench_success(inst, []) is True)


# --------------------------------------------------------------------------- #
# restbench: Correct-Path subsequence (via is_success dispatch)
# --------------------------------------------------------------------------- #
def test_restbench():
    print("[restbench] Correct-Path ordered subsequence")
    inst = {"benchmark": "restbench",
            "gold_plan": [{"tool": "GET /a"}, {"tool": "GET /b"}]}
    in_order = [FakeStep("GET /a"), FakeStep("GET /x"), FakeStep("GET /b")]
    check("gold path is an ordered subsequence -> PASS",
          is_success(inst, in_order, "") is True)
    # NEGATIVE: out of order
    out_of_order = [FakeStep("GET /b"), FakeStep("GET /a")]
    check("out-of-order path -> FAIL", is_success(inst, out_of_order, "") is False)
    # NEGATIVE: missing endpoint
    check("missing endpoint -> FAIL",
          is_success(inst, [FakeStep("GET /a")], "") is False)
    # NEGATIVE: an unsuccessful (errored) step does not count toward the path
    errored = [FakeStep("GET /a"), FakeStep("GET /b", status="error")]
    check("errored step does not satisfy path -> FAIL",
          is_success(inst, errored, "") is False)


# --------------------------------------------------------------------------- #
# toolbench: ToolEval-style LLM judge (offline approx) via stub client
# --------------------------------------------------------------------------- #
def test_toolbench_judge():
    print("[toolbench] ToolEval-style LLM judge (offline approx)")
    inst = {"benchmark": "toolbench", "query": "do the thing",
            "gold_plan": [{"tool": "t"}]}
    steps = [FakeStep("t", output="some recorded output")]
    check("judge says solved -> PASS",
          is_success(inst, steps, "answer", client=FakeJudgeClient(True)) is True)
    # NEGATIVE: judge says not solved
    check("judge says not solved -> FAIL",
          is_success(inst, steps, "answer", client=FakeJudgeClient(False)) is False)
    # no client -> offline subsequence fallback (gold tool present & succeeded)
    check("no-client fallback: gold call succeeded -> PASS",
          is_success(inst, steps, "answer", client=None) is True)
    check("no-client fallback: gold call missing -> FAIL",
          is_success(inst, [FakeStep("other")], "answer", client=None) is False)


def main():
    test_bfcl()
    test_taubench()
    test_restbench()
    test_toolbench_judge()
    print(f"\n{_PASS} passed, {_FAIL} failed")
    sys.exit(1 if _FAIL else 0)


if __name__ == "__main__":
    main()
