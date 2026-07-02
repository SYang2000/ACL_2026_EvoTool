"""Offline tau-bench (retail) environment + official success check.

This is a self-contained re-implementation of tau-bench's retail-domain success
criterion (SPEC.md §3). The official reward has TWO required halves:

  1. **Database-state equality** — applying the candidate tool-call actions to a
     fresh copy of the retail DB must yield the same final mutable DB state as
     applying the gold actions.
  2. **Required output strings (r_outputs)** — every required output string for the
     task must appear in the agent's final answer text.

A task is solved iff BOTH hold. The required output strings are recovered from the
upstream tau-bench retail Task ``outputs`` fields (recovered offline into
``data/taubench/required_outputs.json`` by scripts/data/build_taubench_outputs.py and
attached to each instance as ``instance['outputs']`` by benchmarks.load_instances).
Many retail tasks have an empty ``outputs`` list, in which case the reward reduces to
DB-state equality alone — that is correct, not a defect.

The tool invoke-logic is vendored in ``taubench_tools.TOOLS`` (pure dict
manipulation, no tau_bench package import). The retail DB JSONs live in
``taubench_data/{users,orders,products}.json``.
"""

import copy
import json
import os
from typing import Any, Dict, List

try:  # absolute import when used as `src.envs.taubench_env`
    from src.envs.taubench_tools import TOOLS
except ImportError:  # fallback for relative import
    from .taubench_tools import TOOLS


_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "taubench_data")


def load_db() -> Dict[str, Any]:
    """Deep-load the 3 retail JSONs into a single DB dict.

    Returns a fresh dict ``{"users": ..., "orders": ..., "products": ...}``.
    Every call returns an independent (deep) copy so callers can mutate freely.
    """
    db: Dict[str, Any] = {}
    for part in ("users", "orders", "products"):
        path = os.path.join(_DATA_DIR, f"{part}.json")
        with open(path, "r", encoding="utf-8") as f:
            db[part] = json.load(f)
    # json.load already produces fresh objects; deepcopy is belt-and-suspenders
    # so that any module-level reference cannot accidentally be shared.
    return copy.deepcopy(db)


class TauBenchEnv:
    """A stateful offline tau-bench retail environment over an in-memory DB."""

    def __init__(self) -> None:
        # Fresh, independently-deep-copied DB for this environment instance.
        self.db: Dict[str, Any] = load_db()

    def execute(self, tool: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single tool call against the current DB state.

        Returns ``{"tool", "args", "output", "status"}``. ``status`` is
        ``"success"`` if the tool ran without raising (note: tools return
        ``"Error: ..."`` strings for domain-level failures, which still count
        as a successful *execution*). ``status`` is ``"error"`` if the tool
        name is unknown or the underlying call raises an exception.
        """
        args = args or {}
        if tool not in TOOLS:
            return {
                "tool": tool,
                "args": args,
                "output": f"Error: unknown tool '{tool}'",
                "status": "error",
            }
        try:
            output = TOOLS[tool](self.db, **args)
            return {
                "tool": tool,
                "args": args,
                "output": output,
                "status": "success",
            }
        except Exception as e:  # noqa: BLE001 - surface any tool error
            return {
                "tool": tool,
                "args": args,
                "output": f"Error: {type(e).__name__}: {e}",
                "status": "error",
            }


def _apply_actions(db: Dict[str, Any], actions: List[Dict[str, Any]]) -> None:
    """Apply a list of ``{tool/name, args/kwargs}`` actions to ``db`` in place.

    Each action is applied best-effort: a failed call (unknown tool, raised
    exception) is simply skipped, matching tau-bench's behaviour where invalid
    write actions leave the DB unchanged. Read-only tools are no-ops on state.
    """
    for action in actions or []:
        tool = action.get("tool") or action.get("name")
        args = action.get("args")
        if args is None:
            args = action.get("kwargs", {})
        if not tool or tool not in TOOLS:
            continue
        try:
            TOOLS[tool](db, **(args or {}))
        except Exception:  # noqa: BLE001 - skip failed calls, leave DB as-is
            continue


def _mutable_state(db: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the mutable portion of the DB for comparison.

    Only ``orders`` and ``users`` can be mutated by the retail write tools
    (order status/items/payment_history/address/exchange/return fields, and
    user address / gift-card balances). ``products`` is read-only, so it is
    ignored in the success comparison.
    """
    return {"orders": db["orders"], "users": db["users"]}


def _required_outputs(instance: Dict[str, Any]) -> List[str]:
    """Required output strings (r_outputs) recovered from the upstream Task.outputs
    and attached to the instance as ``instance['outputs']``. Returns ``[]`` when the
    task has no required strings (then the reward reduces to DB-state equality)."""
    outs = instance.get("outputs") or instance.get("r_outputs") or []
    return [str(o) for o in outs]


def _outputs_satisfied(instance: Dict[str, Any], answer: str) -> bool:
    """True iff every required output string for the task appears (as an exact
    substring) in the agent's final ``answer`` (tau-bench's r_outputs check)."""
    required = _required_outputs(instance)
    if not required:
        return True
    text = answer or ""
    return all(req in text for req in required)


def taubench_success(
    instance: Dict[str, Any],
    predicted_actions: List[Dict[str, Any]],
    answer: str | None = None,
) -> bool:
    """Official tau-bench success: DB-state equality AND required output strings.

    Builds two fresh DBs, applies the gold plan (``instance["gold_plan"]``) to one
    and ``predicted_actions`` to the other, and requires the mutable parts
    (``orders`` and ``users``) to be deeply equal. When grading a synthesizer output
    (``answer`` is a string, possibly empty), ALSO requires every required output
    string to be present in ``answer`` (SPEC.md §3).

    ``answer is None`` selects gold-validation mode (no answer to check), so the
    result is DB-state equality alone — used by data-build/verification scripts whose
    gold trajectory has no natural-language answer to score.
    """
    gold_actions = instance.get("gold_plan") or instance.get("gold_actions") or []

    gold_db = load_db()
    _apply_actions(gold_db, gold_actions)

    pred_db = load_db()
    _apply_actions(pred_db, predicted_actions)

    db_equal = _mutable_state(gold_db) == _mutable_state(pred_db)
    if not db_equal:
        return False
    if answer is None:
        return True  # gold-validation mode: DB-state only (no answer to grade)
    return _outputs_satisfied(instance, answer)
