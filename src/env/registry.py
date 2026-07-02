"""Tool index helpers: naming, lookup, compact rendering, and lightweight
keyword retrieval (used to cap the tool index for large-inventory benchmarks
like ToolBench so the prompt stays small).
"""

from __future__ import annotations


def tool_names(tools: list[dict]) -> list[str]:
    return [t["name"] for t in tools]


def find_tool(tools: list[dict], name: str) -> dict | None:
    return next((t for t in tools if t.get("name") == name), None)


def tools_brief(tools: list[dict]) -> str:
    """One line per tool: `name: description`, plus its parameter keys."""
    lines = []
    for t in tools:
        params = t.get("parameters") or t.get("inputs") or []
        if isinstance(params, dict):
            keys = list(params.keys())
        else:
            keys = [p.get("name") for p in params if isinstance(p, dict) and p.get("name")]
        key_str = f" (args: {', '.join(k for k in keys if k)})" if keys else ""
        lines.append(f"- {t['name']}: {t.get('description', '')}{key_str}")
    return "\n".join(lines)


def retrieve_tools(query: str, tools: list[dict], k: int) -> list[dict]:
    """Return all tools if the set is small; else top-k by keyword overlap."""
    if len(tools) <= k:
        return tools
    q = set(query.lower().split())

    def score(t: dict) -> int:
        text = (t.get("name", "") + " " + t.get("description", "")).lower()
        return len(q & set(text.split()))

    return sorted(tools, key=score, reverse=True)[:k]
