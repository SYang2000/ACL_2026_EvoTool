"""Real BFCL gorilla **AST checker** semantics (SPEC.md §3, bfcl lane).

This is a faithful re-implementation of the gorilla ``ast_checker`` decision logic
for the BFCL ``parallel`` / ``multiple`` / ``parallel_multiple`` categories — the
*official* BFCL paradigm is AST matching of the generated calls against the
``ground_truth`` possible-answer set (there is NO execution).

What "AST match" means here (and why it is stricter than the old proxy in
``eval_bfcl.py`` was):

  1. **Exact call-count set-match.** The number of predicted calls must equal the
     number of ground-truth calls. Extra calls FAIL; missing calls FAIL. (The old
     proxy ignored extra predicted calls — that is removed.)
  2. **Function-name match.** Each gold call has exactly one function name; a
     predicted call only matches it if the tool name is identical.
  3. **Reject unexpected parameters.** If a predicted call passes any argument that
     is not a declared parameter of the matched gold function, that call FAILS.
     (The old proxy silently ignored unexpected params — removed.)
  4. **Per-type value checking, NO loose coercion.** A predicted value matches an
     allowed value only when their *type families* agree (string / number / bool /
     list / dict / null) AND they are equal. ``"5"`` (str) does NOT match ``5``
     (number); ``1`` (int) does NOT match ``True`` (bool). int/float are the single
     "number" family (gorilla treats them as one numeric type), so ``5`` matches
     ``5.0`` — but no string<->number<->bool coercion is allowed. (The old proxy
     coerced str/int/float/bool into each other — removed.)
  5. **Optional parameters.** A gold parameter whose allowed-values list contains the
     empty-string sentinel ``""`` is optional: the prediction MAY omit it; if it is
     provided it must still match one of the concrete (non-``""``) allowed values.

Public API::

    bfcl_ast_success(instance: dict, predicted_calls: list[dict]) -> bool

where ``predicted_calls = [{"tool": name, "args": {...}}, ...]`` is the synthesizer's
assembled call list and ``instance["gold_match"]`` is the raw BFCL ``ground_truth``
list of ``{func: {arg: [allowed values ...]}}``.
"""

from __future__ import annotations


def _type_family(v) -> str:
    """Coarse type family for strict (no-coercion) value matching.

    int and float collapse to one ``"number"`` family (gorilla's numeric type);
    bool is kept distinct from number so ``1`` never matches ``True``; strings,
    lists, dicts and null stay separate so no string<->number coercion is possible.
    """
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, (int, float)):
        return "number"
    if isinstance(v, str):
        return "string"
    if isinstance(v, list):
        return "list"
    if isinstance(v, dict):
        return "dict"
    if v is None:
        return "null"
    return type(v).__name__


def _is_optional(allowed_values) -> bool:
    """An arg is optional iff its allowed-values list carries the ``""`` sentinel."""
    return any(isinstance(v, str) and v == "" for v in allowed_values)


def _value_matches(predicted, allowed_values) -> bool:
    """True iff ``predicted`` equals one concrete allowed value under STRICT,
    type-family-aware matching (no str/int/bool coercion). The ``""`` sentinel is an
    optionality marker, not a concrete match target, so it is skipped here."""
    pk = _type_family(predicted)
    for allowed in allowed_values:
        if isinstance(allowed, str) and allowed == "":
            continue  # optional-parameter sentinel, not a real value to match
        if _type_family(allowed) != pk:
            continue  # strict per-type check: refuse cross-family coercion
        if predicted == allowed:
            return True
    return False


def _call_matches_gold(pred_call: dict, gold_spec: dict) -> bool:
    """True iff one predicted call satisfies one gold spec ``{func: {arg: [vals]}}``."""
    (gold_func, gold_args), = gold_spec.items()

    if pred_call.get("tool") != gold_func:
        return False

    pred_args = pred_call.get("args", {}) or {}

    # (3) reject unexpected parameters: every supplied arg must be declared in gold.
    for arg in pred_args:
        if arg not in gold_args:
            return False

    # (4)/(5) per-arg value + optionality checking.
    for arg, allowed_values in gold_args.items():
        if arg in pred_args:
            if not _value_matches(pred_args[arg], allowed_values):
                return False
        elif not _is_optional(allowed_values):
            return False  # a required parameter was omitted
    return True


def bfcl_ast_success(instance: dict, predicted_calls: list) -> bool:
    """Official BFCL AST match (parallel / multiple / parallel_multiple).

    Returns ``True`` iff the predicted call list is a perfect, distinct assignment
    onto the gold call set: same number of calls (exact count set-match), and every
    gold call matched by exactly one predicted call (name + per-arg value + no
    unexpected params). Extra, missing, or mismatched calls all FAIL.
    """
    gold = instance.get("gold_match") or []
    predicted = list(predicted_calls or [])

    # (1) exact call-count set-match — extra OR missing calls FAIL. (When gold is
    # empty this requires predicted to be empty too.)
    if len(predicted) != len(gold):
        return False
    if not gold:
        return True

    # Perfect bipartite assignment: each gold call -> a distinct predicted call.
    used = [False] * len(predicted)

    def assign(gold_idx: int) -> bool:
        if gold_idx == len(gold):
            return True
        gold_spec = gold[gold_idx]
        for pi, pred_call in enumerate(predicted):
            if used[pi]:
                continue
            if _call_matches_gold(pred_call, gold_spec):
                used[pi] = True
                if assign(gold_idx + 1):
                    return True
                used[pi] = False
        return False

    return assign(0)
