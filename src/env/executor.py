"""Per-benchmark offline tool executors (SPEC.md §2).

Tool calls are never sent to a live API; each benchmark gets the closest-to-real
offline observation the rollout's Synthesizer can ground on:

  - taubench  : REAL execution against the vendored retail DB (TauBenchEnv, picked
                in agent.env_for — not here).
  - toolbench : REPLAY of the real recorded DFSDT API responses (recovered into
                ``instance['recorded_outputs']`` by scripts/data/build_toolbench_outputs.py
                and attached by benchmarks.load_instances). See ReplayExecutor.
  - restbench : a realistic, TYPED mock derived from the OAS endpoint shape — NOT a
                "fake success + empty output". See MockOASExecutor.
  - bfcl      : NO execution at all (the official paradigm is AST matching of the
                generated calls). executor_for returns None and agent.run_episode
                records the chosen calls with no observation.

The synthetic ``OfflineExecutor`` (confirm-token dialect + per-tool mock strings)
remains ONLY for the toy ``dummy`` / ``diverse`` datasets used in smoke tests.

Status (success / error / unavailable) is itself a diagnostic signal the Blamer reads.
"""

from __future__ import annotations

from dataclasses import dataclass, field

CONFIRM_KEY = "confirm"


# --------------------------------------------------------------------------- #
# Synthetic toy executor (dummy / diverse only)
# --------------------------------------------------------------------------- #
@dataclass
class OfflineExecutor:
    """Confirm-token synthetic dialect + per-tool mock strings. Toys only."""

    mock_outputs: dict              # {tool_name: output_string}
    available_names: set            # names present in the instance's tool index
    required_token: str | None = None  # if set, enforce a confirm token (synthetic dialects only)

    def execute(self, tool: str, args: dict) -> dict:
        if tool not in self.available_names:
            return {"tool": tool, "args": args,
                    "output": f"ERROR: unknown tool '{tool}' (not in the tool index)",
                    "status": "error"}
        if self.required_token is not None and \
                str((args or {}).get(CONFIRM_KEY, "")).lower() != self.required_token:
            return {"tool": tool, "args": args,
                    "output": f"ERROR: every tool call must include argument {CONFIRM_KEY}='{self.required_token}'",
                    "status": "error"}
        if tool in self.mock_outputs:
            return {"tool": tool, "args": args, "output": self.mock_outputs[tool], "status": "success"}
        return {"tool": tool, "args": args, "output": "(ok)", "status": "success"}


# --------------------------------------------------------------------------- #
# ToolBench: replay the real recorded DFSDT API responses (§2)
# --------------------------------------------------------------------------- #
@dataclass
class ReplayExecutor:
    """Replay the real recorded ToolBench API responses (SPEC.md §2).

    ``recorded`` is the per-instance sidecar list (aligned to the gold plan):
    ``[{"tool", "args", "output": <str|None>, "error": <str|None>}, ...]``. We build
    a per-tool FIFO queue of the recorded *turns* (in recorded order) and pop them as
    the policy calls each tool — the live rollout may call tools in a different
    order/args than the gold trajectory, so a per-tool queue is the most robust
    offline replay.

    Faithfulness (§2 "REPLAY the real recorded API responses … no fake success"):
    we NEVER fabricate output content, and we NEVER report success the source did
    not record. Each recorded turn is replayed by its REAL recorded outcome:

      * a non-empty recorded response payload  -> ``status="success"`` (the payload);
      * an empty/absent response that carries a recorded API ``error`` (503, timeout,
        rate-limit, …)                         -> ``status="error"`` replaying that
        recorded error text (NOT a fake "success" with empty output, and NOT dropped
        — the Synthesizer/Blamer see the real recorded failure);
      * a recognized tool with NO recorded turn left -> ``status="unavailable"`` with
        an explicit marker (never fabricated content, never "success");
      * an unknown tool                        -> ``status="error"``.

    ``status`` (success / error / unavailable) is itself the diagnostic signal the
    Blamer reads and the canonical outcome marker the metrics read, so it must
    reflect the real recorded outcome — an unavailable or errored call is never
    counted as a success.
    """

    recorded: list = field(default_factory=list)   # the instance's recorded_outputs
    available_names: set = field(default_factory=set)
    _queue: dict = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        for entry in self.recorded or []:
            out = entry.get("output")
            err = entry.get("error")
            if out:                       # non-empty recorded response payload
                rec = {"output": str(out), "status": "success"}
            elif err:                     # recorded API error (response empty/absent)
                rec = {"output": str(err), "status": "error"}
            else:
                continue                  # no recorded content -> stays unavailable
            self._queue.setdefault(entry.get("tool"), []).append(rec)

    def execute(self, tool: str, args: dict) -> dict:
        if tool not in self.available_names:
            return {"tool": tool, "args": args,
                    "output": f"ERROR: unknown tool '{tool}' (not in the tool index)",
                    "status": "error", "output_available": False}
        q = self._queue.get(tool)
        if q:
            rec = q.pop(0)
            # Real recorded content (a success payload OR a recorded API error).
            return {"tool": tool, "args": args, "output": rec["output"],
                    "status": rec["status"], "output_available": True}
        # Recognized call, but no recorded turn remains -> explicitly UNAVAILABLE
        # (never a fake "success"; §2). Surfaced honestly to the Synthesizer/Blamer.
        return {"tool": tool, "args": args,
                "output": "(no recorded API response available for this call)",
                "status": "unavailable", "output_available": False}


# --------------------------------------------------------------------------- #
# RestBench: realistic typed mock from the OAS endpoint shape (§2)
# --------------------------------------------------------------------------- #
@dataclass
class MockOASExecutor:
    """Realistic offline mock for RestBench REST endpoints.

    RestBench's source (RestGPT) ships no recorded HTTP responses, so we synthesize
    a TYPED example response from the real OAS endpoint shape (method + path + the
    tool's documented parameters) — explicitly NOT "fake success + empty output".
    The mock echoes the resolved request and returns representative typed fields
    (ids as ints, names as strings, list-valued ``results``) keyed off the path's
    resource, so the Synthesizer has realistically-shaped data to ground on.

    OFFLINE APPROXIMATION (documented, not "official"): values are illustrative
    examples, not live API data; only the response *shape* is real-OAS-derived.
    """

    tools: list = field(default_factory=list)
    available_names: set = field(default_factory=set)

    def execute(self, tool: str, args: dict) -> dict:
        if tool not in self.available_names:
            return {"tool": tool, "args": args,
                    "output": f"ERROR: unknown tool '{tool}' (not in the tool index)",
                    "status": "error"}
        import json as _json
        body = _mock_oas_response(tool, args or {})
        return {"tool": tool, "args": args, "output": _json.dumps(body),
                "status": "success", "output_available": True}


def _split_endpoint(name: str) -> tuple[str, str]:
    """'GET /search/person' -> ('GET', '/search/person')."""
    parts = name.split(" ", 1)
    if len(parts) == 2 and parts[0].isupper():
        return parts[0], parts[1]
    return "GET", name


def _resolve_path(path: str, args: dict) -> str:
    """Substitute {placeholder} path params from args where available."""
    out = path
    for k, v in (args or {}).items():
        out = out.replace("{" + str(k) + "}", str(v))
    return out


def _mock_oas_response(endpoint: str, args: dict) -> dict:
    """A typed, realistically-shaped example response for a REST endpoint.

    Derived only from the endpoint name + supplied args (no network, no fabricated
    'official' data). Read endpoints (GET) return an example resource / results
    list; write endpoints return a typed acknowledgement.
    """
    method, path = _split_endpoint(endpoint)
    resolved = _resolve_path(path, args)
    resource = next((seg for seg in reversed(resolved.strip("/").split("/"))
                     if seg and not seg.startswith("{")), "resource")
    query_args = {k: v for k, v in (args or {}).items() if "{" + str(k) + "}" not in path}

    if method in ("POST", "PUT", "PATCH", "DELETE"):
        return {"status": 200, "method": method, "endpoint": resolved,
                "request": query_args or args or {},
                "result": {"id": 1001, "status": "ok", "modified": True}}

    # GET: an example resource plus a small typed results list.
    example = {"id": 1001, "name": f"Example {resource}", "popularity": 42.0}
    return {"status": 200, "method": method, "endpoint": resolved,
            "request": query_args,
            "page": 1, "total_results": 1, "total_pages": 1,
            "results": [example]}


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
def executor_for(instance: dict):
    """Return the executor for an instance, or None for benchmarks with NO
    execution (bfcl). Dispatched by benchmark (SPEC.md §2)."""
    benchmark = instance.get("benchmark", "")

    if benchmark == "bfcl":
        return None  # official BFCL paradigm is AST matching — no execution

    available = set(t["name"] for t in instance.get("available_tools", []))

    if benchmark == "toolbench":
        return ReplayExecutor(
            recorded=instance.get("recorded_outputs") or [],
            available_names=available,
        )

    if benchmark == "restbench":
        return MockOASExecutor(
            tools=instance.get("available_tools", []),
            available_names=available,
        )

    # Toy datasets (dummy / diverse): synthetic confirm-token dialect.
    tok = instance.get("required_token")
    return OfflineExecutor(
        mock_outputs=instance.get("mock_outputs") or {},
        available_names=available,
        required_token=str(tok).lower() if tok is not None else None,
    )
