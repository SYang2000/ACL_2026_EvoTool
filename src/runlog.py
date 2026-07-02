"""Structured run log for the self-evolving optimization (SPEC.md §9).

Writes ONE JSONL record per generation — including REJECTED ones — under the run's
output dir, so the whole optimization trajectory can be inspected and the learning
curves replotted offline. Nothing here is on the reward/decision path: it only
*observes* the loop, so disabling it cannot change a run's result.

Each generation record captures:
  - gen index, epoch, parent_id, mini-batch instance ids
  - the blamed module + the Blamer LLM's FULL rationale (its parsed JSON)
  - the mutated module's prompt BEFORE and AFTER (full text + a unified diff)
  - child_id, accepted/rejected, parent-vs-child mini-batch reward
  - learning-curve points: train-batch reward, S_sel validation score, and the
    periodic S_test score (epoch boundaries only); per-module cumulative mutation
    counts; cumulative + per-gen token cost; population snapshot (ids + per-id
    S_sel win-counts / weights / mean reward)

A final per-run `summary` record (also written to a sibling `.summary.json`) holds
the learning curves as arrays so they can be plotted without re-parsing the JSONL.
"""

from __future__ import annotations

import difflib
import json
import os

from src.policy.modules import MODULES, Policy

# Keys we track cumulative mutation counts for: the four modules plus "all"
# (the monolithic rewrite, which touches every module at once).
_MUT_KEYS = MODULES + ["all"]


def unified_diff(before: str, after: str, module: str) -> str:
    """A standard, newline-separated unified diff of one module's spec text
    (parent -> child)."""
    return "\n".join(
        difflib.unified_diff(
            (before or "").splitlines(),
            (after or "").splitlines(),
            fromfile=f"{module}@parent",
            tofile=f"{module}@child",
            lineterm="",
        )
    )


def _mutation_view(parent: Policy, child: Policy, target: str) -> dict:
    """Render the BEFORE/AFTER prompt(s) and unified diff for the mutated module.

    For a single-module mutation (blame / random / fixed) `before`/`after` are the
    plain spec strings. For the monolithic "all" rewrite every module changes, so
    `before`/`after` are JSON maps of all four specs and the diff concatenates the
    per-module diffs."""
    if target == "all":
        before = {m: parent.spec(m) for m in MODULES}
        after = {m: child.spec(m) for m in MODULES}
        diff = "\n".join(unified_diff(before[m], after[m], m) for m in MODULES)
        return {
            "prompt_before": json.dumps(before, indent=2),
            "prompt_after": json.dumps(after, indent=2),
            "prompt_diff": diff,
            "changed_modules": [m for m in MODULES if before[m] != after[m]],
        }
    before = parent.spec(target)
    after = child.spec(target) if target in MODULES else ""
    return {
        "prompt_before": before,
        "prompt_after": after,
        "prompt_diff": unified_diff(before, after, target),
        "changed_modules": [target] if before != after else [],
    }


class RunLog:
    """Append-only JSONL writer + in-memory learning-curve accumulator."""

    def __init__(self, path: str, meta: dict | None = None) -> None:
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._f = open(path, "w")
        self.meta = dict(meta or {})
        self.records: list[dict] = []
        # cumulative per-module mutation counts (attempted and accepted)
        self.mut_attempts: dict[str, int] = {k: 0 for k in _MUT_KEYS}
        self.mut_accepted: dict[str, int] = {k: 0 for k in _MUT_KEYS}
        self.n_accepted = 0
        self.n_rejected = 0
        self._last_tokens = 0
        # learning-curve arrays (one point per generation, test only at epoch ends)
        self.curve: dict[str, list] = {
            "gen": [], "epoch": [],
            "parent_batch_reward": [], "child_batch_reward": [],
            "train_batch_reward": [], "val_sel_score": [],
            "tokens_cumulative": [],
        }
        self.test_curve: dict[str, list] = {"gen": [], "epoch": [], "score": []}
        self._write({"type": "run_meta", **self.meta})

    # -- low level -----------------------------------------------------------
    def _write(self, obj: dict) -> None:
        self._f.write(json.dumps(obj, default=str) + "\n")
        self._f.flush()

    # -- per-generation ------------------------------------------------------
    def log_generation(
        self,
        *,
        gen: int,
        epoch: int,
        parent: Policy,
        batch_ids: list,
        target: str,
        blame_rationale: dict | None,
        child: Policy,
        accepted: bool,
        parent_batch_reward: float,
        child_batch_reward: float,
        val_sel_score: float | None,
        test_score: float | None,
        population: list[Policy],
        weights: dict,
        sel_means: dict,
        sel_wins: dict,
        total_tokens: int,
    ) -> None:
        # cumulative mutation bookkeeping (count the attempt by target module)
        key = target if target in self.mut_attempts else "all"
        self.mut_attempts[key] += 1
        if accepted:
            self.mut_accepted[key] += 1
            self.n_accepted += 1
        else:
            self.n_rejected += 1

        tokens_this_gen = max(0, total_tokens - self._last_tokens)
        self._last_tokens = total_tokens

        # the reward the run actually "keeps" on this mini-batch (drives the curve)
        train_batch_reward = child_batch_reward if accepted else parent_batch_reward

        snapshot = [
            {
                "id": p.policy_id,
                "sel_wins": sel_wins.get(p.policy_id, 0),
                "weight": round(float(weights.get(p.policy_id, 0.0)), 4),
                "sel_mean": round(float(sel_means.get(p.policy_id, 0.0)), 4),
            }
            for p in population
        ]

        mut = _mutation_view(parent, child, target)
        record = {
            "type": "generation",
            "gen": gen,
            "epoch": epoch,
            "parent_id": parent.policy_id,
            "batch_ids": list(batch_ids),
            "blamed_module": target,
            "blame_rationale": blame_rationale or {},
            "mutated_module": target,
            "prompt_before": mut["prompt_before"],
            "prompt_after": mut["prompt_after"],
            "prompt_diff": mut["prompt_diff"],
            "changed_modules": mut["changed_modules"],
            "child_id": child.policy_id,
            "accepted": bool(accepted),
            "parent_batch_reward": round(float(parent_batch_reward), 4),
            "child_batch_reward": round(float(child_batch_reward), 4),
            "train_batch_reward": round(float(train_batch_reward), 4),
            "val_sel_score": (None if val_sel_score is None else round(float(val_sel_score), 4)),
            "test_score": (None if test_score is None else round(float(test_score), 4)),
            "cumulative_mutations": dict(self.mut_attempts),
            "cumulative_accepted_mutations": dict(self.mut_accepted),
            "tokens_cumulative": total_tokens,
            "tokens_this_gen": tokens_this_gen,
            "population_size": len(population),
            "population_snapshot": snapshot,
        }
        self.records.append(record)
        self._write(record)

        # accumulate learning curves
        self.curve["gen"].append(gen)
        self.curve["epoch"].append(epoch)
        self.curve["parent_batch_reward"].append(record["parent_batch_reward"])
        self.curve["child_batch_reward"].append(record["child_batch_reward"])
        self.curve["train_batch_reward"].append(record["train_batch_reward"])
        self.curve["val_sel_score"].append(record["val_sel_score"])
        self.curve["tokens_cumulative"].append(total_tokens)
        if test_score is not None:
            self.test_curve["gen"].append(gen)
            self.test_curve["epoch"].append(epoch)
            self.test_curve["score"].append(record["test_score"])

    # -- final summary -------------------------------------------------------
    def summary(self, extra: dict | None = None) -> dict:
        final_pop = self.records[-1]["population_snapshot"] if self.records else []
        s = {
            "type": "summary",
            **self.meta,
            "n_generations": len(self.records),
            "n_accepted": self.n_accepted,
            "n_rejected": self.n_rejected,
            "cumulative_mutations": dict(self.mut_attempts),
            "cumulative_accepted_mutations": dict(self.mut_accepted),
            "total_tokens": self._last_tokens,
            "final_population": final_pop,
            "learning_curves": {**self.curve, "test": dict(self.test_curve)},
        }
        if extra:
            s.update(extra)
        return s

    def finalize(self, extra: dict | None = None) -> dict:
        """Write the summary record (to the JSONL and to a sibling `.summary.json`),
        close the file, and return the summary dict."""
        s = self.summary(extra)
        self._write(s)
        summary_path = self.path.replace(".jsonl", "") + ".summary.json"
        with open(summary_path, "w") as f:
            json.dump(s, f, indent=2, default=str)
        self.summary_path = summary_path
        self.close()
        return s

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:  # noqa: BLE001
            pass
