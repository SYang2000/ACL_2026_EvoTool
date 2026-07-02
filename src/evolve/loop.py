"""EvoTool self-evolving optimization loop (Algorithm 1).

One config drives every baseline via two switches:
  cfg.evolve.mutation_target -> how the module to mutate is chosen
  cfg.evolve.selection       -> how parents survive across generations

Each generation: sample a parent, run a mini-batch, pick the worst episode,
choose the target module, mutate it into a child, accept the child if it beats
the parent on the mini-batch, then run population selection on S_sel.

Budget is DATA-DERIVED (SPEC.md §8): the loop runs `epochs` full passes
over S_train, each a fresh shuffled WITHOUT-replacement pass chunked into
mini-batches of B, so each epoch yields ceil(n_train / B) generations and every
train instance is used exactly once per epoch.
"""

from __future__ import annotations

import math
import random

from src.config import EvoToolConfig
from src.evolve import select
from src.evolve.blame import blame
from src.evolve.diagnostics import extract_diagnostics
from src.evolve.mutate import monolithic_mutate, mutate
from src.llm.client import LLMClient
from src.policy.agent import Episode, mean_reward, run_episode
from src.policy.modules import MODULES, Policy, initial_policy


def _choose_target(mode: str, episode, diagnostics: dict,
                   client: LLMClient, rng: random.Random) -> tuple[str, dict]:
    # Blame operates on the single representative episode e (paper §4.1, §11):
    # pi* = argmax_pi b_pi(e), with the Blamer LLM as the decision-maker.
    # Returns (target_module, rationale) where rationale is the Blamer LLM's full
    # parsed JSON (for the run log §9); empty for the non-blame baselines.
    if mode == "blame":
        rationale: dict = {}
        return blame(client, episode, diagnostics, rationale_out=rationale), rationale
    if mode == "random":
        return rng.choice(MODULES), {"mode": "random"}
    if mode.startswith("fixed:"):
        return mode.split(":", 1)[1], {"mode": mode}
    if mode == "all":
        return "all", {"mode": "all (monolithic, no blame)"}
    return rng.choice(MODULES), {"mode": "default-random"}


def _representative(episodes: list, diag_list: list):
    """The single representative episode e (paper §4.1): the lowest-reward episode
    in the mini-batch — the clearest failure for the Blamer to diagnose and the
    Mutator to fix. Chosen BEFORE blame so the same e drives both."""
    return min(zip(episodes, diag_list), key=lambda p: p[0].reward)


def _make_child(client, parent, target, episode, diagnostics, cfg, child_id) -> Policy:
    ev = cfg.evolve
    if target == "all":  # monolithic: one score-guided global rewrite (no blame)
        return monolithic_mutate(client, parent, episode, child_id)
    return mutate(client, parent, target, episode, diagnostics,
                  ev.use_trajectory, ev.use_feedback, child_id)


def _sample_parent(population, weights, rng) -> Policy:
    by_id = {p.policy_id: p for p in population}
    ids = [pid for pid in weights if pid in by_id] or [p.policy_id for p in population]
    ws = [weights.get(pid, 1.0) for pid in ids]
    return by_id[rng.choices(ids, weights=ws, k=1)[0]]


def _parent_weights(mode: str, win_weights: dict, sel_means: dict, population) -> dict:
    """Parent-sampling weights keyed by policy_id (CHANGE 1, OPT-IN):
      - "win"     -> the unchanged win-count weights from select_population.
      - "val_acc" -> LINEAR-proportional to per-policy MEAN S_sel reward (max(0, mean)),
                     normalized to sum 1; if all means are 0 fall back to UNIFORM.
    Pure function (no RNG, no LLM); retention is unaffected — this only reweights
    which surviving policy is sampled as the parent to mutate."""
    if mode == "val_acc":
        ids = [p.policy_id for p in population]
        vals = [max(0.0, sel_means.get(pid, 0.0)) for pid in ids]
        total = sum(vals)
        if total > 0:
            return {pid: v / total for pid, v in zip(ids, vals)}
        return {pid: 1.0 / len(ids) for pid in ids}  # all-zero fallback: uniform
    return win_weights


def _epoch_batches(train: list[dict], batch_size: int, epochs: int, rng: random.Random):
    """Yield (epoch_idx, mini-batch) over `epochs` fresh shuffled WITHOUT-replacement
    passes of S_train (SPEC.md §8). Each epoch reshuffles ALL of S_train and
    chunks it into mini-batches of `batch_size`, so every train instance is used once
    per epoch and no instance repeats within an epoch."""
    for epoch in range(epochs):
        order = list(train)
        rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            yield epoch, order[start:start + batch_size]


def evolve(client: LLMClient, cfg: EvoToolConfig, train: list[dict], sel: list[dict],
           log=print, test: list[dict] | None = None, runlog=None) -> list[Policy]:
    """Returns the RETAINED population (the selection strategy's artifact):
    diversity keeps complementary specialists, greedy keeps one, top-k keeps k.
    At TEST time the default deploys the SINGLE best policy Theta* (SPEC.md
    §7); the retained population still matters because Theta* is picked from it and
    the per-instance ensemble remains available as a named optional ablation.

    `test` and `runlog` are OPTIONAL and purely for the structured run log (§9):
    when both are present the loop records a periodic S_test learning-curve point for
    the current best policy at each epoch boundary. Passing neither leaves behaviour
    (and the RNG stream / results) unchanged."""
    ev = cfg.evolve
    population = [initial_policy()]
    weights = {population[0].policy_id: 1.0}

    if ev.mutation_target == "none" or ev.epochs <= 0 or not train:
        return population

    rng = random.Random(cfg.seed)
    batch_size = max(1, ev.batch_size)
    # Data-derived budget (SPEC.md §8): one without-replacement pass over
    # S_train = ceil(n_train / B) generations; default total = epochs * that.
    gens_per_epoch = max(1, math.ceil(len(train) / batch_size))
    if ev.generations and ev.generations > 0:
        total = ev.generations                        # explicit override (budget ablation)
        epochs = math.ceil(total / gens_per_epoch)
    else:
        epochs = ev.epochs
        total = epochs * gens_per_epoch

    # Per-policy S_sel means for the CURRENT population (CHANGE 1 + CHANGE 2, OPT-IN).
    # Computed ONLY when an opt-in switch needs it; the default path leaves it empty
    # and never calls sel_stats here, so it adds ZERO extra LLM calls. sel_stats uses
    # the same cached episodes select_population reads, so even when on it is cost-neutral.
    need_sel_means = (ev.parent_select == "val_acc") or (ev.accept_on == "sel")
    if need_sel_means:
        sel_means, _ = select.sel_stats(client, cfg, population, sel)
    else:
        sel_means = {}

    g = 0
    count_in_epoch = 0
    prev_epoch = None
    # NOTE: keep the lazy _epoch_batches generator (do NOT materialise it) so the RNG
    # is consumed in exactly the original order (per-epoch shuffle interleaved with
    # parent sampling); materialising would shuffle all epochs up front and change
    # which parents get sampled, i.e. change results.
    for epoch, batch in _epoch_batches(train, batch_size, epochs, rng):
        if g >= total:
            break
        if epoch != prev_epoch:
            count_in_epoch = 0
            prev_epoch = epoch
        count_in_epoch += 1

        parent = _sample_parent(
            population, _parent_weights(ev.parent_select, weights, sel_means, population), rng)
        episodes = [run_episode(client, parent, x, cfg.max_steps) for x in batch]
        diag_list = [extract_diagnostics(e, e.instance) for e in episodes]

        rep, rep_diag = _representative(episodes, diag_list)
        target, blame_rationale = _choose_target(ev.mutation_target, rep, rep_diag, client, rng)
        child_id = f"g{g}-{target}"
        child = _make_child(client, parent, target, rep, rep_diag, cfg, child_id)

        if ev.accept_on == "sel":
            # CHANGE 2 (OPT-IN): compare on held-out S_sel. Parent baseline reads the
            # maintained sel_means (cached); the child's S_sel episodes computed here are
            # cached and reused by the select_population call below (no extra dedup needed).
            parent_avg = sel_means.get(parent.policy_id)
            if parent_avg is None:
                parent_avg = mean_reward(client, parent, sel, cfg.max_steps)
            child_avg = mean_reward(client, child, sel, cfg.max_steps)
        else:
            parent_avg = sum(e.reward for e in episodes) / len(episodes)
            child_avg = mean_reward(client, child, batch, cfg.max_steps)
        accepted = child_avg > parent_avg
        if accepted:
            population.append(child)

        population, weights = select.select_population(client, cfg, population, sel)
        if need_sel_means:  # refresh for next generation's parent sampling / accept gate
            sel_means, _ = select.sel_stats(client, cfg, population, sel)
        log(f"  gen {g} (epoch {epoch}): parent={parent.policy_id} target={target} "
            f"parent_avg={parent_avg:.3f} child_avg={child_avg:.3f} "
            f"{'ACCEPT' if accepted else 'reject'} | pop={len(population)}")

        if runlog is not None:
            # Population snapshot + S_sel learning point. sel_stats reuses the cached
            # selection episodes (no new LLM calls). The validation score is the mean
            # S_sel reward of the would-be-deployed best policy Theta* (argmax mean).
            sel_means, sel_wins = select.sel_stats(client, cfg, population, sel)
            val_sel_score = max(sel_means.values()) if sel_means else None
            # Periodic S_test (epoch boundary only): the last generation of this epoch
            # is count_in_epoch == gens_per_epoch, or the global budget end.
            is_epoch_end = (count_in_epoch >= gens_per_epoch) or (g + 1 >= total)
            test_score = None
            if test and ev.log_test_eval and is_epoch_end and sel_means:
                best_id = max(sel_means, key=lambda k: sel_means[k])
                best_pol = next((p for p in population if p.policy_id == best_id), population[0])
                # This periodic S_test probe is LOGGING ONLY (§9): snapshot/restore the
                # token counter so it cannot inflate the headline result['tokens'] —
                # disabling logging must not change a run's reported numbers.
                _tok = client.total_tokens
                test_score = mean_reward(client, best_pol, test, cfg.max_steps)
                client.total_tokens = _tok
            runlog.log_generation(
                gen=g, epoch=epoch, parent=parent, batch_ids=[x.get("id") for x in batch],
                target=target, blame_rationale=blame_rationale, child=child, accepted=accepted,
                parent_batch_reward=parent_avg, child_batch_reward=child_avg,
                val_sel_score=val_sel_score, test_score=test_score,
                population=population, weights=weights, sel_means=sel_means, sel_wins=sel_wins,
                total_tokens=client.total_tokens,
            )
        g += 1

    return population
