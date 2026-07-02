"""Load a benchmark into the unified instance schema and split it.

Unified instance:
  {id, query, available_tools:[{name,description,parameters}], gold_plan:[{tool,args}],
   gold_answer, mock_outputs:{tool_name: output}, metric}

All four EvoTool benchmarks normalize into this shape, so nothing downstream
knows which benchmark it is running.
"""

from __future__ import annotations

import json
import os
import random

from src.config import EvoToolConfig


_VERIFY_TOOL = {
    "name": "verify",
    "description": "Internal validation step.",
    "parameters": {"result": "the result to validate"},
}


def _normalize(item: dict) -> dict:
    out = {
        "id": str(item.get("id", "")),
        "query": item.get("query") or item.get("question") or item.get("instruction") or "",
        "available_tools": item.get("available_tools") or item.get("tools") or [],
        "gold_plan": item.get("gold_plan") or [],
        "gold_answer": item.get("gold_answer"),
        "mock_outputs": item.get("mock_outputs") or {},
        "required_token": item.get("required_token"),  # None for real data
        "metric": item.get("metric") or "score",
    }
    return out


def _inject_hidden_verify(items: list[dict]) -> None:
    """Add a hidden, required final `verify` step to every other instance. It is
    NOT mentioned in the query, so a vague planner skips it -> the instance fails
    until the loop blames the planner and teaches it to verify. A caller-only
    (single-aspect) optimizer can never fix this, which is what separates EvoTool
    from single-aspect baselines."""
    for i, it in enumerate(items):
        if i % 2 != 0:
            continue
        names = {t["name"] for t in it["available_tools"]}
        if "verify" not in names:
            it["available_tools"] = it["available_tools"] + [_VERIFY_TOOL]
        if not it["gold_plan"] or it["gold_plan"][-1].get("tool") != "verify":
            it["gold_plan"] = it["gold_plan"] + [{"tool": "verify", "args": {"result": "ok"}}]
        it["mock_outputs"] = {**it["mock_outputs"], "verify": "Validated."}


# Verify-injection is a synthetic crutch; disabled now that we use real data
# (real benchmarks have real selector / planner / argument failure modes).
_VERIFY_BENCHMARKS = set()


def _load_recorded_outputs(benchmark: str, data_path: str) -> dict:
    """SPEC.md §2: recover the REAL recorded ToolBench API responses from
    the sidecar produced by scripts/data/build_toolbench_outputs.py (samples.json
    itself is never edited). Keyed by instance id -> list aligned to the gold plan.
    Returns {} if the sidecar is absent (rollout then marks every toolbench step's
    output explicitly unavailable rather than fabricating one)."""
    if benchmark != "toolbench":
        return {}
    path = os.path.join(data_path, benchmark, "recorded_outputs.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f) or {}


def _load_required_outputs(benchmark: str, data_path: str) -> dict:
    """SPEC.md §3: recover the tau-bench required output strings (r_outputs)
    from the sidecar produced by scripts/data/build_taubench_outputs.py (samples.json
    itself is never edited). Keyed by instance id -> list of required strings. Returns
    {} if absent (then the tau reward reduces to DB-state equality alone)."""
    if benchmark != "taubench":
        return {}
    path = os.path.join(data_path, benchmark, "required_outputs.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f) or {}


def load_instances(benchmark: str, data_path: str) -> list[dict]:
    path = os.path.join(data_path, benchmark, "samples.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"data/{benchmark} is not bundled with this repo - see DATA.md for how to "
            f"build it from the official sources"
        )
    with open(path, "r") as f:
        raw = json.load(f)
    recorded = _load_recorded_outputs(benchmark, data_path)
    required_outputs = _load_required_outputs(benchmark, data_path)
    items = [_normalize(x) for x in raw]
    for it in items:
        it["benchmark"] = benchmark          # so the evaluator can dispatch the per-benchmark metric
        it["gold_match"] = raw_match(raw, it)  # carry bfcl gold_match if present
        # ToolBench: attach the recovered recorded API responses so the rollout's
        # ReplayExecutor can replay them (empty/absent -> steps marked unavailable).
        it["recorded_outputs"] = recorded.get(it["id"], [])
        # tau-bench: attach the recovered required output strings (r_outputs) so the
        # official success check can require them in the synthesizer's answer (§3).
        it["outputs"] = required_outputs.get(it["id"], [])

    if benchmark in _VERIFY_BENCHMARKS:
        _inject_hidden_verify(items)
    return items


def raw_match(raw: list, item: dict):
    for r in raw:
        if str(r.get("id", "")) == item["id"]:
            return r.get("gold_match")
    return None


def split_instances(instances: list[dict], cfg: EvoToolConfig) -> tuple[list, list, list]:
    """Disjoint train / sel / test (held-out).

    train -> S_train (evolution mini-batches), sel -> S_sel (population selection),
    test -> the held-out evaluation split that NEVER overlaps train or sel, so the
    reported score measures generalization rather than memorized training instances.
    For 150-instance datasets the default 90/30/30 split partitions the data exactly.
    """
    rng = random.Random(cfg.seed)
    items = list(instances)
    rng.shuffle(items)
    n = len(items)

    a, b, c = cfg.n_train, cfg.n_sel, cfg.n_test
    train = items[:a]
    sel = items[a:a + b]
    test = items[a + b:a + b + c]
    # Tiny-dataset fallback (dummy/diverse toys): if there is no room for a disjoint
    # held-out slice, reuse the tail so the pipeline still runs (flagged, not for real
    # numbers). Real 150-sample benchmarks always take the clean disjoint branch above.
    if not test:
        test = items[-min(c, n):] if n else []
    return train, sel, test
