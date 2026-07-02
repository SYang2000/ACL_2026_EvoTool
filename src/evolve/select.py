"""Population selection strategies (paper Section 4.3 + Table 4 ablation).

Given the current population evaluated on the held-out selection set S_sel,
return (surviving policies, parent-sampling weights keyed by policy_id):

  - diversity : keep every policy that wins >=1 instance (instance-wise winner);
                sample parents proportional to win frequency. (EvoTool)
  - greedy    : keep only the single best-average policy.
  - topk      : keep the top-k by average reward (uniform weights).
  - static    : no selection (used with mutation_target="none").
"""

from __future__ import annotations

from src.config import EvoToolConfig
from src.llm.client import LLMClient
from src.policy.agent import run_episode
from src.policy.modules import Policy


def _reward_matrix(client, population, sel_set, max_steps) -> list[list[float]]:
    # rows = policies, cols = selection instances (episodes are cached, so this
    # only actually runs LLM calls for policy/instance pairs not seen before).
    return [[run_episode(client, p, x, max_steps).reward for x in sel_set] for p in population]


def _avg(row: list[float]) -> float:
    return sum(row) / len(row) if row else 0.0


def select_population(
    client: LLMClient,
    cfg: EvoToolConfig,
    population: list[Policy],
    sel_set: list[dict],
) -> tuple[list[Policy], dict]:
    strategy = cfg.evolve.selection
    if strategy == "static" or len(population) == 1:
        return population, {population[0].policy_id: 1.0}

    R = _reward_matrix(client, population, sel_set, cfg.max_steps)

    if strategy == "greedy":
        best = max(range(len(population)), key=lambda i: _avg(R[i]))
        return [population[best]], {population[best].policy_id: 1.0}

    if strategy == "topk":
        ranked = sorted(range(len(population)), key=lambda i: _avg(R[i]), reverse=True)
        keep = ranked[: cfg.evolve.topk]
        return [population[i] for i in keep], {population[i].policy_id: 1.0 / len(keep) for i in keep}

    # diversity (default): instance-wise winners + win-frequency weights
    win_count = {i: 0 for i in range(len(population))}
    for j in range(len(sel_set)):
        col = [R[i][j] for i in range(len(population))]
        winner = max(range(len(population)), key=lambda i: col[i])
        win_count[winner] += 1
    winners = [i for i in range(len(population)) if win_count[i] > 0]
    if not winners:  # degenerate: all-zero rewards
        winners = [max(range(len(population)), key=lambda i: _avg(R[i]))]
    total = sum(win_count[i] for i in winners)
    weights = {population[i].policy_id: (win_count[i] / total if total else 1.0 / len(winners))
               for i in winners}
    return [population[i] for i in winners], weights


def best_policy(client: LLMClient, cfg: EvoToolConfig, population: list[Policy],
                sel_set: list[dict]) -> Policy:
    if len(population) == 1:
        return population[0]
    R = _reward_matrix(client, population, sel_set, cfg.max_steps)
    best = max(range(len(population)), key=lambda i: _avg(R[i]))
    return population[best]


def sel_stats(client: LLMClient, cfg: EvoToolConfig, population: list[Policy],
              sel_set: list[dict]) -> tuple[dict, dict]:
    """Per-policy S_sel mean reward and instance-wise win counts, keyed by policy_id
    (for the run log's population snapshot — SPEC.md §9). Episodes are cached,
    so right after selection this adds NO new LLM calls; it only reads the same matrix.
    Win counts mirror the diversity rule: W(x) = argmax_Theta r_x(Theta)."""
    if not population:
        return {}, {}
    R = _reward_matrix(client, population, sel_set, cfg.max_steps)
    means = {p.policy_id: _avg(R[i]) for i, p in enumerate(population)}
    wins = {p.policy_id: 0 for p in population}
    for j in range(len(sel_set)):
        col = [R[i][j] for i in range(len(population))]
        winner = max(range(len(population)), key=lambda i: col[i])
        wins[population[winner].policy_id] += 1
    return means, wins
