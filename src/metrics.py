"""Reward and reporting metrics.

The reported headline is the **mean per-benchmark success** — binary per instance,
where "success" is each benchmark's own criterion (see src/evaluators.is_success),
NOT a single exact-sequence rule:

  - bfcl      : gorilla AST match (exact call-count set-match + per-arg value match)
  - restbench : automatic "Correct Path" (gold endpoints an ordered subsequence)
  - tau-bench : final DB-state equality AND all required output strings present
  - toolbench : ToolEval-style LLM judge over the answer (offline approx, NOT official)

`episode_reward` (which drives evolution + selection) is `success` plus a small
continuous "progress" term, so the loop still has a usable gradient before it
reaches full success.
"""

from __future__ import annotations


def _lcs(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def _seq_f1(pred: list[str], gold: list[str]) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    lcs = _lcs(pred, gold)
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(pred), lcs / len(gold)
    return 2 * precision * recall / (precision + recall)


def _answer_match(pred: str, gold: str) -> float:
    if not gold:
        return 1.0 if (pred or "").strip() else 0.0
    p, g = (pred or "").lower(), gold.lower()
    if g in p:
        return 1.0
    gt = set(g.split())
    return len(set(p.split()) & gt) / len(gt) if gt else 0.0


def _tools(steps: list[dict]) -> list[str]:
    return [s.get("tool") for s in steps]


def _is_subsequence(sub: list, seq: list) -> bool:
    it = iter(seq)
    return all(x in it for x in sub)


def progress(steps: list[dict], instance: dict) -> float:
    """Continuous progress signal in [0, 1] (ordered-sequence × error-free
    fraction). Combined with the per-benchmark binary success in agent.run_episode
    to give evolution a gradient before any child fully succeeds."""
    gold = [s.get("tool") for s in instance.get("gold_plan", [])]
    seq = _seq_f1(_tools(steps), gold)
    exec_frac = (sum(1 for s in steps if s.get("status") == "success") / len(steps)) if steps else 0.0
    return 0.5 * seq + 0.5 * exec_frac


def success_rate(successes: list[bool]) -> float:
    return 100.0 * sum(successes) / len(successes) if successes else 0.0


def headline_score(rewards: list[float]) -> float:
    return 100.0 * (sum(rewards) / len(rewards)) if rewards else 0.0
