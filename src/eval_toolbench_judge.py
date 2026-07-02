"""ToolEval-style LLM judge for ToolBench — OFFLINE APPROXIMATION (SPEC.md §3).

NOT the official metric. The official ToolBench reward (ToolEval) runs a GPT judge
over **live RapidAPI** executions and reports a pass-rate; that is not reproducible
offline (the APIs are paywalled / unstable, and we only have the recorded DFSDT
responses). This module is an honestly-labelled approximation:

  our served model judges whether the synthesizer's final answer SOLVES the user
  query, given the query + the recorded tool outputs the rollout actually saw.

So the graded object is the SYNTHESIZER's answer text (SPEC.md §3), scored
by an LLM judge rather than by call-sequence matching. Everywhere this is surfaced
it is labelled "ToolEval-style LLM judge (offline approx)", never "official".

Public API::

    toolbench_judge(instance, answer, steps, client) -> bool
"""

from __future__ import annotations


_JUDGE_SYSTEM = (
    "You are a strict evaluator for a tool-use agent (ToolEval-style). You are given "
    "a user query, the tool calls an agent made together with the tool outputs it "
    "observed, and the agent's FINAL ANSWER. Decide whether the final answer actually "
    "and completely solves the user's query, grounded in the observed tool outputs. "
    "An answer that is empty, refuses, hallucinates content not supported by the tool "
    "outputs, or leaves part of the request unaddressed is NOT solved. "
    'Return JSON only: {"solved": true|false, "reason": "<one short sentence>"}.'
)


def _render_evidence(steps) -> str:
    """Render the executed trajectory (tool, args, observed/recorded output) as the
    evidence the judge reasons over. Steps with no recorded output are shown honestly
    as unavailable so the judge never assumes content the agent never saw."""
    lines = []
    for s in steps or []:
        obs = getattr(s, "observation", None) or {}
        out = obs.get("output")
        status = obs.get("status")
        if out is None or status == "unavailable":
            out = "(no recorded API output available)"
        lines.append(f"- {s.tool}({s.args}) -> {out}")
    return "\n".join(lines) or "(the agent made no tool calls)"


def toolbench_judge(instance: dict, answer: str, steps, client) -> bool:
    """ToolEval-style LLM judge (offline approx). ``True`` iff our served model judges
    the synthesizer's ``answer`` to solve ``instance['query']`` given the observed
    tool outputs. Requires a live ``client``; callers without one must use their own
    offline fallback (see ``evaluators.is_success``)."""
    if client is None:
        raise ValueError("toolbench_judge requires an LLM client (offline approx).")

    query = instance.get("query", "")
    evidence = _render_evidence(steps)
    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {"role": "user", "content": (
            f"USER_QUERY:\n{query}\n\n"
            f"AGENT_TOOL_CALLS_AND_OUTPUTS:\n{evidence}\n\n"
            f"AGENT_FINAL_ANSWER:\n{answer or '(empty answer)'}"
        )},
    ]
    verdict = client.generate_json(messages)
    return _parse_verdict(verdict)


def _parse_verdict(verdict: dict) -> bool:
    """Read a boolean ``solved`` out of the judge's JSON, tolerant of string forms."""
    if not isinstance(verdict, dict):
        return False
    v = verdict.get("solved")
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "yes", "1", "solved", "pass")
    if isinstance(v, (int, float)):
        return v >= 1
    return False
