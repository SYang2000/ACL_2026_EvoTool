"""Trajectory-Grounded Blame Attribution (paper Section 4.1 / §11).

The Blamer LLM is the DECISION-MAKER. It reads a single representative episode
`e` — the query, the FULL per-substep trajectory (sub_i, tool_i, args_i,
output_i/status), and the reward (pass/no-pass) — plus, as ADVISORY HINTS ONLY,
the gold reference (we are on the training set, so this is allowed) and the
rule-based diagnostic events. It judges which step(s) are wrong (or that all are
correct) and names ONE module pi* = argmax_pi b_pi(e) from
{planner, selector, caller, synthesizer}.

We take the LLM's choice DIRECTLY. The gold reference and the rule events never
override it (they are hints, not a gate). `rule_blame` is consulted ONLY as a
last-resort fallback when the LLM output is unparseable.
"""

from __future__ import annotations

from src.evolve.diagnostics import render_events, rule_blame
from src.llm.client import LLMClient
from src.policy.agent import Episode
from src.policy.modules import MODULES
from src.prompts import BLAMER_META_PROMPT


def _trajectory_text(episode: Episode) -> str:
    """Render the FULL per-substep trajectory: for each substep, the subgoal, the
    selected tool, the caller's args, and the execution output/status. This is the
    primary evidence the Blamer reasons over (not the rule events)."""
    lines = [f"PLAN: {episode.plan}"]
    for i, s in enumerate(episode.steps, 1):
        lines.append(
            f"step {i}: subgoal={s.subgoal!r} tool={s.tool!r} args={s.args} "
            f"-> [{s.observation.get('status')}] {s.observation.get('output')}"
        )
    lines.append(f"FINAL ANSWER (graded object): {episode.answer!r}")
    return "\n".join(lines)


def _gold_text(instance: dict) -> str:
    """Gold plan/answer as a HINT for the Blamer (allowed — this is the training
    set). It is reference context, never the decision rule."""
    gold_tools = [s.get("tool") for s in instance.get("gold_plan", [])]
    gold_answer = instance.get("gold_answer")
    lines = []
    if gold_tools:
        lines.append(f"gold tool sequence: {gold_tools}")
    if gold_answer:
        lines.append(f"gold answer: {gold_answer!r}")
    return "\n".join(lines) if lines else "(no gold reference available)"


def blame(client: LLMClient, episode: Episode, diagnostics: dict,
          rationale_out: dict | None = None) -> str:
    """Name the module most responsible for `episode` (the single representative
    episode e). The LLM's arg-max pick is taken DIRECTLY; rule_blame is only the
    unparseable-output fallback.

    If `rationale_out` is given it is populated IN PLACE with the Blamer LLM's full
    parsed JSON rationale plus how the decision resolved (for the run log, §9). This
    is purely observational and never affects the returned module."""
    reward = diagnostics["reward"]
    # pass/no-pass is the TRUE task outcome (episode.success from the official metric),
    # not a threshold on the blended reward — a genuine pass with partial progress
    # (success=True, reward 0.7-0.99) must read as pass for the Blamer (prompts.py 0/1).
    outcome = 1 if episode.success else 0
    packet = (
        f"# TASK\n{episode.instance['query']}\n\n"
        f"# TRAJECTORY (per substep: subgoal, tool, args -> [status] output)\n"
        f"{_trajectory_text(episode)}\n\n"
        f"# OUTCOME\n{outcome} ({'pass' if outcome else 'no-pass'}, reward={reward:.2f})\n\n"
        f"# GOLD REFERENCE (advisory hint only — diagnose the trajectory, do not just copy it)\n"
        f"{_gold_text(episode.instance)}\n\n"
        f"# DIAGNOSTIC HINTS (advisory rule-based events; you may override them)\n"
        f"{render_events(diagnostics)}"
    )
    messages = [
        {"role": "system", "content": BLAMER_META_PROMPT},
        {"role": "user", "content": packet},
    ]
    out = client.generate_json(messages)
    primary = str(out.get("primary", "")).strip().lower()
    if rationale_out is not None:
        rationale_out.clear()
        if isinstance(out, dict):
            rationale_out.update(out)
        rationale_out["_parsed_primary"] = primary
        rationale_out["_resolved_by"] = "llm" if primary in MODULES else "rule_fallback"
    if primary in MODULES:
        return primary  # take the LLM's arg-max choice DIRECTLY (no gold/event gate)
    # Fallback ONLY when the LLM output is unparseable.
    fallback = rule_blame(diagnostics)
    if rationale_out is not None:
        rationale_out["_fallback_module"] = fallback
    return fallback
