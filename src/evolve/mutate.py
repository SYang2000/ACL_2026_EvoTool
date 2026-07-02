"""Feedback-Guided Targeted Mutation (paper Section 4.2).

A Mutator LLM rewrites ONLY the blamed module's specification, grounded in the
failure episode, keeping the other three modules fixed.

The two mutation-guidance signals are toggleable for the Table 3 ablation:
  - use_trajectory (tau): include the trajectory + diagnostic events.
  - use_feedback (F):      include the explicit critique framing (diagnosed error
                           mode + heuristic edit patterns). Without it the mutator
                           is only asked to "rewrite to do better".
"""

from __future__ import annotations

import json

from src.evolve.blame import _trajectory_text
from src.evolve.diagnostics import render_events
from src.llm.client import LLMClient
from src.policy.agent import Episode
from src.policy.modules import MODULES, Policy
from src.prompts import MUTATOR_META_PROMPT

_PLAIN_SYSTEM = (
    "You are a prompt editor for one module of a modular tool-using agent. "
    "Rewrite the given module specification so the agent performs better. "
    "Preserve the module's interface and output format.\n\n"
    'Return JSON only: {"target_module": "<module>", "revised_spec": "<new spec text>"}'
)


def mutate(
    client: LLMClient,
    policy: Policy,
    target: str,
    episode: Episode,
    diagnostics: dict,
    use_trajectory: bool,
    use_feedback: bool,
    child_id: str,
) -> Policy:
    parts = [
        f"# TARGET MODULE\n{target}",
        f"# CURRENT SPECIFICATION\n{policy.spec(target)}",
        f"# TASK\n{episode.instance['query']}",
        f"# OUTCOME\nreward={episode.reward:.2f}",
    ]
    if use_trajectory:
        parts.insert(3, f"# TRAJECTORY (evidence)\n{_trajectory_text(episode)}")
        parts.append(f"# MODULE EVENTS\n{render_events(diagnostics)}")

    system = MUTATOR_META_PROMPT if use_feedback else _PLAIN_SYSTEM
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n\n".join(parts)},
    ]
    out = client.generate_json(messages)
    revised = str(out.get("revised_spec", "")).strip()
    if not revised:
        return policy  # no-op: keep parent spec if the mutator produced nothing
    return policy.with_module(target, revised, child_id)


_MONOLITHIC_SYSTEM = (
    "You are a black-box prompt optimizer (OPRO / PromptBreeder style). You are "
    "given the current prompts of a tool-using agent's four modules and the fact "
    "that the agent scored poorly. Without any per-step diagnosis, propose an "
    "improved full prompt for ALL FOUR modules at once.\n\n"
    'Return JSON only: {"planner":"...","selector":"...","caller":"...","synthesizer":"..."}'
)


def monolithic_mutate(client: LLMClient, policy: Policy, episode: Episode, child_id: str) -> Policy:
    """Monolithic baseline: a single global rewrite of the whole prompt, guided
    only by the score (no trajectory, no blame). This is the defining property of
    monolithic optimizers and the reason they entangle behaviours: they cannot
    localize which module to fix, so the hidden, feedback-only requirements stay
    undiscovered."""
    current = {m: policy.spec(m) for m in MODULES}
    user = (
        f"# TASK\n{episode.instance['query']}\n\n"
        f"# CURRENT PROMPTS\n{json.dumps(current, indent=2)}\n\n"
        f"# OUTCOME\nThe agent scored only reward={episode.reward:.2f}. Improve the prompts."
    )
    out = client.generate_json([
        {"role": "system", "content": _MONOLITHIC_SYSTEM},
        {"role": "user", "content": user},
    ])
    new = {m: (str(out.get(m)).strip() if out.get(m) else current[m]) for m in MODULES}
    return Policy(planner=new["planner"], selector=new["selector"],
                  caller=new["caller"], synthesizer=new["synthesizer"], policy_id=child_id)
