"""Backward-compatible shim for the BFCL evaluation entry point.

The real gorilla AST-checker semantics now live in :mod:`src.eval_bfcl_ast`
(``bfcl_ast_success``): exact call-count set-match, function-name match, rejection
of unexpected parameters, and strict per-type value matching with no loose
str/int/bool coercion (SPEC.md §3).

This module used to contain a *simplified proxy* (value-membership with light type
coercion that ignored extra calls and unexpected params). That proxy is removed;
``bfcl_success`` here is just an alias for the strict checker so existing callers
(e.g. scripts/data/build_bfcl.py) keep working with the correct semantics.
"""

from __future__ import annotations

from src.eval_bfcl_ast import bfcl_ast_success

# Public alias kept for backward compatibility. Delegates to the strict AST checker.
bfcl_success = bfcl_ast_success

__all__ = ["bfcl_success", "bfcl_ast_success"]
