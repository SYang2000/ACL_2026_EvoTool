"""Reproducible converter: ToolBench (OpenBMB) real solved queries -> 150-instance samples.json.

Source (real data, no fabrication)
-----------------------------------
OpenBMB ToolBench, in its public processed ShareGPT DFS answer set
`Yhyu13/ToolBench_toolllama_G123_dfs` (`toolllama_G123_dfs_eval.json`), which is
the official ToolBench DFSDT answer trees flattened to conversations. Each record
is one solved ToolBench query (a mix of G1/G2/G3):

  conversations = [system, user, assistant, function, assistant, ..., assistant]

  * system   : after the marker "you have access to the following APIs:" holds the
               candidate API list -> a Python-literal list of
               {name, description, parameters}, terminated by the pseudo-tool
               "Finish". This is the ToolBench api_list / answer_generation.function.
  * user     : the natural-language query.
  * assistant: "Thought: ...\nAction: <tool_name>\nAction Input: <json args>"
               -> one real gold API call. The terminal "Finish" action is dropped.

Mapping to the repo unified schema (matches data/toolbench/samples.json template):
  id              = "tb_<n>"  (1-based, in selection order)
  query           = the user-turn content
  available_tools = [{name, description, parameters:{argname: description}}, ...]
                    (ToolBench param JSON-Schema flattened to the template's shape)
  gold_plan       = [{tool, args}, ...]  (Finish dropped)
  gold_answer     = None
  mock_outputs    = {}

Tool NAMES are identical between available_tools and gold_plan by construction
(both come from the same record). The ToolBench official metric -- gold APIs are an
ordered subsequence of the successfully-executed APIs (src/evaluators.is_success)
-- therefore passes for every emitted gold.

Selection (deterministic, no hand-editing)
------------------------------------------
Walk the source in upstream file order and KEEP an instance iff
  (1) it has >=1 non-Finish gold call,
  (2) every gold tool name is a candidate api_list tool (name consistency),
  (3) its query is not a duplicate of one already kept,
until TARGET (150) instances are collected, then stop.

Run:
  python scripts/data/build_toolbench.py
"""

from __future__ import annotations

import ast
import json
import os
import re

from huggingface_hub import hf_hub_download

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HF_REPO_ID = "Yhyu13/ToolBench_toolllama_G123_dfs"
SRC_FILE = "toolllama_G123_dfs_eval.json"
OUT = os.path.join(REPO, "data", "toolbench", "samples.json")

TARGET = 150
_API_MARKER = "you have access to the following APIs:"
_ACTION_RE = re.compile(r"Action:\s*(.+)")
_ACTION_INPUT_RE = re.compile(r"Action Input:\s*(\{.*\}|\{\s*\})", re.S)


def parse_tools(system_value: str):
    """Pull the candidate API list (a Python literal) out of the system prompt."""
    idx = system_value.find(_API_MARKER)
    if idx < 0:
        return None
    blk = system_value[idx + len(_API_MARKER):].strip()
    start, end = blk.find("["), blk.rfind("]")
    if start < 0 or end < 0:
        return None
    try:
        tools = ast.literal_eval(blk[start:end + 1])
    except (ValueError, SyntaxError):
        return None
    return tools if isinstance(tools, list) else None


def flatten_params(parameters: dict) -> dict:
    """{type, properties:{name:{description,...}}, ...} -> {name: description}.

    Matches the existing data/toolbench/samples.json convention.
    """
    out: dict = {}
    props = (parameters or {}).get("properties") or {}
    for pname, spec in props.items():
        out[pname] = (spec.get("description", "") or "") if isinstance(spec, dict) else ""
    return out


def normalize_tool(tool: dict):
    name = tool.get("name")
    if not name or name == "Finish":
        return None
    return {
        "name": name,
        "description": tool.get("description", "") or "",
        "parameters": flatten_params(tool.get("parameters") or {}),
    }


def parse_gold_plan(conversations: list) -> list:
    """Ordered list of real API calls from the assistant turns (Finish dropped)."""
    plan = []
    for turn in conversations:
        if turn.get("from") != "assistant":
            continue
        value = turn.get("value", "")
        m = _ACTION_RE.search(value)
        if not m:
            continue
        name = m.group(1).strip().splitlines()[0].strip()
        if not name or name == "Finish":
            continue
        args: dict = {}
        mi = _ACTION_INPUT_RE.search(value)
        if mi:
            try:
                parsed = json.loads(mi.group(1))
                if isinstance(parsed, dict):
                    args = parsed
            except json.JSONDecodeError:
                args = {}
        plan.append({"tool": name, "args": args})
    return plan


def first_user_query(conversations: list) -> str:
    for turn in conversations:
        if turn.get("from") == "user":
            q = (turn.get("value") or "").strip()
            # Drop the ToolBench prompt scaffolding suffix ("...\nBegin!").
            if q.endswith("Begin!"):
                q = q[: -len("Begin!")].rstrip()
            return q
    return ""


def main():
    src = hf_hub_download(HF_REPO_ID, SRC_FILE, repo_type="dataset")
    with open(src, "r", encoding="utf-8") as f:
        records = json.load(f)
    print(f"  loaded {len(records)} source records from {HF_REPO_ID}/{SRC_FILE}")

    selected = []
    seen_queries = set()
    for rec in records:
        if len(selected) >= TARGET:
            break
        convs = rec.get("conversations") or []
        if not convs or convs[0].get("from") != "system":
            continue

        raw_tools = parse_tools(convs[0]["value"])
        if not raw_tools:
            continue
        tools = [t for t in (normalize_tool(x) for x in raw_tools if isinstance(x, dict)) if t]
        tool_names = {t["name"] for t in tools}
        if not tool_names:
            continue

        gold_plan = parse_gold_plan(convs)
        if not gold_plan:
            continue
        # Name consistency: every gold call must be a candidate tool, else the
        # offline executor can't mark it success and the subsequence metric fails.
        if not all(step["tool"] in tool_names for step in gold_plan):
            continue

        query = first_user_query(convs)
        if not query or query in seen_queries:
            continue
        seen_queries.add(query)

        selected.append({
            "id": f"tb_{len(selected) + 1}",
            "query": query,
            "available_tools": tools,
            "gold_plan": gold_plan,
            "gold_answer": None,
            "mock_outputs": {},
        })

    print(f"  total selected = {len(selected)}")
    assert len(selected) == TARGET, f"expected {TARGET}, got {len(selected)}"

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2, ensure_ascii=False)
    print(f"  wrote {len(selected)} instances -> {OUT}")


if __name__ == "__main__":
    main()
