"""Pin the pure parent-sampling-weight logic for the two OPT-IN evolve switches.

Only `_parent_weights` carries non-trivial branching that can be tested without
any LLM/vLLM, so that is what we pin here:
  - mode="win"     -> returns the win-count weights object unchanged.
  - mode="val_acc" -> LINEAR-proportional to per-policy mean S_sel reward.
  - mode="val_acc" with all-zero means -> uniform fallback.
"""

import math
from dataclasses import dataclass

from src.evolve.loop import _parent_weights


@dataclass
class _FakePolicy:
    policy_id: str


def _close(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(math.isclose(a[k], b[k]) for k in a)


def test_win_mode_returns_win_weights_unchanged():
    win_weights = {"a": 0.3, "b": 0.7}
    pop = [_FakePolicy("a"), _FakePolicy("b")]
    out = _parent_weights("win", win_weights, {"a": 0.2, "b": 0.6}, pop)
    assert out is win_weights  # identity: default path untouched


def test_val_acc_mode_is_linear_proportional():
    pop = [_FakePolicy("a"), _FakePolicy("b")]
    out = _parent_weights("val_acc", {"a": 1.0}, {"a": 0.2, "b": 0.6}, pop)
    assert _close(out, {"a": 0.25, "b": 0.75})


def test_val_acc_all_zero_means_falls_back_to_uniform():
    pop = [_FakePolicy("a"), _FakePolicy("b")]
    out = _parent_weights("val_acc", {"a": 1.0}, {"a": 0.0, "b": 0.0}, pop)
    assert _close(out, {"a": 0.5, "b": 0.5})


if __name__ == "__main__":  # allow running without pytest
    test_win_mode_returns_win_weights_unchanged()
    test_val_acc_mode_is_linear_proportional()
    test_val_acc_all_zero_means_falls_back_to_uniform()
    print("all tests passed")
