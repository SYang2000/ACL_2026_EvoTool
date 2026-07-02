"""The four-module tool-use policy.

A `Policy` is just four prompt specifications (planner/selector/caller/synthesizer).
Each `run_<module>` function turns a spec + inputs into a structured output via one
LLM call. The spec is the *evolvable* part; the JSON I/O contract appended here is
fixed so that evolving the spec never breaks parsing.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from src.env import registry
from src.llm.client import LLMClient
from src.prompts import INITIAL_SPECS

MODULES = ["planner", "selector", "caller", "synthesizer"]


@dataclass
class Policy:
    planner: str
    selector: str
    caller: str
    synthesizer: str
    policy_id: str = "init"

    def spec(self, module: str) -> str:
        return getattr(self, module)

    def with_module(self, module: str, new_spec: str, new_id: str) -> "Policy":
        return replace(self, **{module: new_spec, "policy_id": new_id})


def initial_policy() -> Policy:
    return Policy(**INITIAL_SPECS, policy_id="init")


# --- per-module calls -------------------------------------------------------

def run_planner(client: LLMClient, spec: str, query: str) -> list[str]:
    # The planner does NOT see the tool index: it produces ABSTRACT subgoals
    # (what to accomplish), leaving the actual tool choice to the Selector. This
    # keeps the four modules as balanced, independent levers (planner=decompose,
    # selector=choose tool, caller=build args) rather than the planner dictating
    # the whole tool sequence.
    fmt = (
        '\n\nReturn JSON only: {"plan": ["subgoal 1", "subgoal 2", ...]}. '
        "Each subgoal is one short ABSTRACT step describing WHAT to accomplish "
        "(e.g. 'find the movie', 'get its credits'). Do NOT name specific tools or APIs."
    )
    messages = [
        {"role": "system", "content": spec + fmt},
        {"role": "user", "content": f"USER_TASK:\n{query}"},
    ]
    out = client.generate_json(messages)
    plan = out.get("plan") or []
    return [str(s).strip() for s in plan if str(s).strip()]


def run_selector(client: LLMClient, spec: str, query: str, substep: str,
                 state: dict, tools: list[dict]) -> str:
    # STRICT SEQUENCE (paper §1): the Selector picks the SINGLE tool for the CURRENT
    # substep (or NONE if the substep needs no tool call). It is invoked once per
    # planner substep, so the plan length governs the tool sequence.
    fmt = (
        '\n\nChoose the single best tool to accomplish the CURRENT_SUBGOAL, using '
        "the steps done so far for context. "
        'Return JSON only: {"tool": "<EXACT name from AVAILABLE_TOOLS, or NONE if this '
        'subgoal needs no tool>"}.'
    )
    messages = [
        {"role": "system", "content": spec + fmt},
        {"role": "user", "content": (
            f"USER_TASK:\n{query}\n\nCURRENT_SUBGOAL:\n{substep}\n\n"
            f"STEPS DONE SO FAR:\n{_state_brief(state)}\n\nAVAILABLE_TOOLS:\n{registry.tools_brief(tools)}"
        )},
    ]
    out = client.generate_json(messages)
    return str(out.get("tool") or out.get("next_tool") or "NONE").strip()


def run_caller(client: LLMClient, spec: str, tool: dict, query: str, substep: str,
               state: dict) -> dict:
    # The Caller constructs valid arguments for the selected tool of the CURRENT
    # substep, resolving values from the task and from earlier observations in STATE.
    fmt = '\n\nReturn JSON only: {"arguments": {"<param>": "<value>", ...}}.'
    messages = [
        {"role": "system", "content": spec + fmt},
        {"role": "user", "content": (
            f"USER_TASK:\n{query}\n\nCURRENT_SUBGOAL:\n{substep}\n\n"
            f"SELECTED_TOOL:\n{tool.get('name')}: {tool.get('description', '')}\n\n"
            f"TOOL_DOC:\n{_tool_doc(tool)}\n\nOBSERVATIONS SO FAR:\n{_state_brief(state)}"
        )},
    ]
    out = client.generate_json(messages)
    args = out.get("arguments")
    return args if isinstance(args, dict) else {}


def run_synthesizer(client: LLMClient, spec: str, query: str, steps: list) -> str:
    # The Synthesizer answers the original query from ALL per-substep tuples
    # (sub_i, tool_i, args_i, output_i) + the query (paper §1). Some benchmarks have
    # no tool outputs (bfcl = AST matching only; toolbench steps with no recorded
    # response) — those are surfaced honestly so the answer never invents content.
    fmt = "\n\nReturn the final answer as plain text, grounded only in the tool outputs above."
    history = "\n".join(_render_step(s) for s in steps) or "(no tool calls)"
    messages = [
        {"role": "system", "content": spec + fmt},
        {"role": "user", "content": f"USER_TASK:\n{query}\n\nSUBGOAL_STEPS:\n{history}"},
    ]
    return client.generate(messages).strip()


def _render_step(s) -> str:
    obs = s.observation or {}
    out = obs.get("output")
    if out is None:
        if obs.get("status") == "no_execution":
            out = "(not executed — call recorded for AST matching)"
        else:
            out = "(no recorded output available)"
    sub = f"[{s.subgoal}] " if s.subgoal else ""
    return f"- {sub}{s.tool}({s.args}) -> {out}"


# --- small helpers ----------------------------------------------------------

def _state_brief(state: dict) -> str:
    if not state:
        return "(nothing done yet)"
    return "\n".join(f"{k}: {v}" for k, v in state.items())


def _tool_doc(tool: dict) -> str:
    params = tool.get("parameters") or tool.get("inputs") or []
    if isinstance(params, dict):
        return ", ".join(f"{k}: {v}" for k, v in params.items())
    parts = []
    for p in params:
        if isinstance(p, dict):
            parts.append(f"{p.get('name')} ({p.get('type', 'any')})")
    return ", ".join(parts) or "(no documented parameters)"
