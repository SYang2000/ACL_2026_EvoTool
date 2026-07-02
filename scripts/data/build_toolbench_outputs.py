"""Recover the REAL recorded ToolBench API responses for each tb_<n> instance.

Spec §2 (SPEC.md): toolbench tool outputs must REPLAY the real recorded
API responses from the source DFSDT data (`Yhyu13/ToolBench_toolllama_G123_dfs`).
The 150-instance `data/toolbench/samples.json` ships with `mock_outputs={}` (the
recorded responses were dropped at build time). This script recovers them into a
SIDECAR file `data/toolbench/recorded_outputs.json` (samples.json is NOT touched),
which `src/benchmarks.load_instances` attaches to each instance so the rollout's
toolbench executor can replay them.

Source structure (see scripts/data/build_toolbench.py for the full contract):
  conversations = [system, user, assistant, function, assistant, function, ...]
  * each `assistant` turn holds "Action: <tool>\nAction Input: <json>" — one real
    API call (these become the gold_plan, with the terminal "Finish" dropped);
  * the `function` turn IMMEDIATELY FOLLOWING an action turn holds that call's
    recorded API response, a JSON object `{"error": ..., "response": ...}`.

We re-run build_toolbench's EXACT deterministic selection (same file order, same
filters) so the emitted ids / gold ordering match samples.json byte-for-byte, and
for every selected record we walk conversations index-wise pairing each non-Finish
action with the function turn that follows it. A gold step whose action has no
following function turn (e.g. the trajectory was truncated) is recorded as
`{"output": null}` — explicitly UNAVAILABLE, never a fabricated success.

Output schema (data/toolbench/recorded_outputs.json):
  { "tb_<n>": [ {"tool": str, "args": {...}, "output": <str|null>,
                 "error": <str|null>}, ... ] }   # aligned 1:1 with gold_plan order

Run:
  python scripts/data/build_toolbench_outputs.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import re

from huggingface_hub import hf_hub_download

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HF_REPO_ID = "Yhyu13/ToolBench_toolllama_G123_dfs"
SRC_FILE = "toolllama_G123_dfs_eval.json"
SAMPLES = os.path.join(REPO, "data", "toolbench", "samples.json")
OUT = os.path.join(REPO, "data", "toolbench", "recorded_outputs.json")

TARGET = 150
_ACTION_RE = re.compile(r"Action:\s*(.+)")


def _load_build_module():
    """Import scripts/data/build_toolbench.py by path (scripts is not a package)
    so we reuse its EXACT parse_tools / normalize_tool / parse_gold_plan /
    first_user_query helpers — guaranteeing identical selection to samples.json."""
    path = os.path.join(REPO, "scripts", "data", "build_toolbench.py")
    spec = importlib.util.spec_from_file_location("build_toolbench", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _action_name(turn: dict) -> str | None:
    """The tool name from an assistant Action turn (None if no action / Finish)."""
    if turn.get("from") != "assistant":
        return None
    m = _ACTION_RE.search(turn.get("value", ""))
    if not m:
        return None
    name = m.group(1).strip().splitlines()[0].strip()
    return name or None


def _recorded_responses(convs: list) -> list[dict | None]:
    """For each non-Finish action turn (in order), the parsed function response
    that immediately follows it, or None if there is no following function turn.

    Aligned 1:1 with parse_gold_plan (which also walks assistant Action turns in
    order, dropping Finish), so position i here matches gold_plan[i]."""
    out: list[dict | None] = []
    for i, turn in enumerate(convs):
        name = _action_name(turn)
        if not name or name == "Finish":
            continue
        nxt = convs[i + 1] if i + 1 < len(convs) else None
        if not nxt or nxt.get("from") != "function":
            out.append(None)
            continue
        raw = nxt.get("value", "")
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            # Some responses are not valid JSON; keep the raw text as the output.
            out.append({"error": None, "response": raw})
            continue
        if isinstance(parsed, dict):
            out.append({"error": parsed.get("error", ""), "response": parsed.get("response")})
        else:
            out.append({"error": None, "response": raw})
    return out


def main() -> None:
    bt = _load_build_module()

    src = hf_hub_download(HF_REPO_ID, SRC_FILE, repo_type="dataset")
    with open(src, "r", encoding="utf-8") as f:
        records = json.load(f)
    print(f"  loaded {len(records)} source records from {HF_REPO_ID}/{SRC_FILE}")

    with open(SAMPLES, "r", encoding="utf-8") as f:
        samples = {it["id"]: it for it in json.load(f)}

    recovered: dict[str, list] = {}
    seen_queries: set[str] = set()
    n = 0
    for rec in records:
        if n >= TARGET:
            break
        convs = rec.get("conversations") or []
        if not convs or convs[0].get("from") != "system":
            continue
        raw_tools = bt.parse_tools(convs[0]["value"])
        if not raw_tools:
            continue
        tools = [t for t in (bt.normalize_tool(x) for x in raw_tools if isinstance(x, dict)) if t]
        tool_names = {t["name"] for t in tools}
        if not tool_names:
            continue
        gold_plan = bt.parse_gold_plan(convs)
        if not gold_plan:
            continue
        if not all(step["tool"] in tool_names for step in gold_plan):
            continue
        query = bt.first_user_query(convs)
        if not query or query in seen_queries:
            continue
        seen_queries.add(query)
        n += 1
        inst_id = f"tb_{n}"

        responses = _recorded_responses(convs)
        # Alignment sanity: responses are 1:1 with gold_plan order.
        if len(responses) != len(gold_plan):
            # Truncate/pad defensively so positions still align with gold_plan.
            responses = (responses + [None] * len(gold_plan))[: len(gold_plan)]

        entries = []
        for step, resp in zip(gold_plan, responses):
            if resp is None:
                entries.append({"tool": step["tool"], "args": step.get("args", {}),
                                "output": None, "error": None})
            else:
                out_val = resp.get("response")
                entries.append({"tool": step["tool"], "args": step.get("args", {}),
                                "output": None if out_val is None else str(out_val),
                                "error": (resp.get("error") or None)})
        recovered[inst_id] = entries

        # Cross-check against the shipped samples.json (id + gold ordering).
        gold = samples.get(inst_id)
        if gold is not None:
            sg = [s["tool"] for s in gold["gold_plan"]]
            mg = [e["tool"] for e in entries]
            assert sg == mg, f"{inst_id}: gold order drift {sg} != {mg}"

    assert n == TARGET, f"expected {TARGET} selected, got {n}"
    assert set(recovered) == set(samples), "recovered ids != samples.json ids"

    n_have = sum(1 for v in recovered.values() for e in v if e["output"] is not None)
    n_steps = sum(len(v) for v in recovered.values())
    print(f"  recovered responses for {n_have}/{n_steps} gold steps "
          f"({n_steps - n_have} steps have no recorded response -> unavailable)")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(recovered, f, indent=2, ensure_ascii=False)
    print(f"  wrote sidecar -> {OUT}")


if __name__ == "__main__":
    main()
