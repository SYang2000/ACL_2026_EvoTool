"""Configuration dataclasses for EvoTool.

A single config object drives everything. Each baseline / ablation in the paper
is just a different setting of `evolve.mutation_target` x `evolve.selection`
(see configs/baselines/*.yaml), so we never branch on "which baseline" anywhere
else in the code.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any

import yaml


@dataclass
class LLMConfig:
    model_name: str = "Qwen3-8B"
    temperature: float = 0.0
    max_tokens: int = 1024
    api_key: str = ""
    base_url: str = ""


@dataclass
class EvolveConfig:
    # How the mutation target module is chosen each generation:
    #   "blame"  -> Blamer LLM picks the most responsible module (EvoTool)
    #   "random" -> pick a module uniformly at random
    #   "all"    -> mutate every module together (monolithic baseline)
    #   "fixed:<module>" -> always mutate one module (single-aspect baseline)
    #   "none"   -> no evolution at all (static / hand-crafted baseline)
    mutation_target: str = "blame"
    # How parents survive / are sampled across generations:
    #   "diversity" -> instance-wise winner keep + win-frequency sampling (EvoTool)
    #   "greedy"    -> keep only the single best-average policy
    #   "topk"      -> keep the top-k by average reward
    #   "static"    -> no selection (used with mutation_target="none")
    selection: str = "diversity"
    # Budget is DATA-DERIVED (SPEC.md §8): each epoch is one fresh shuffled
    # WITHOUT-replacement pass over S_train in mini-batches of B, i.e.
    # ceil(n_train / batch_size) generations per epoch; total = epochs * that.
    epochs: int = 3               # number of full passes over S_train (paper: 3)
    batch_size: int = 3           # B  (paper: 3)
    # Optional explicit total-generation override for the budget ablation ONLY:
    #   0 (default) -> derive from data (epochs * ceil(n_train/batch_size));
    #   >0          -> pin the total generation count (still reshuffled epochs).
    generations: int = 0
    topk: int = 2                 # k for selection="topk"
    # Test-time deployment (SPEC.md §7): default deploys the SINGLE best
    # policy Theta* (argmax mean S_sel reward). True selects the per-instance
    # route_episode ensemble — a NAMED optional ablation only, never the headline.
    test_ensemble: bool = False
    # Mutation-guidance ablation (paper Table 3): what the Mutator LLM receives.
    use_trajectory: bool = True   # tau: trajectory evidence
    use_feedback: bool = True     # F : explicit natural-language critique
    # Run-log learning curve (SPEC.md §9): when True and a held-out S_test is
    # passed to evolve(), the run log records a PERIODIC S_test score for the current
    # best policy at each epoch boundary. This costs extra eval calls (it is the only
    # part of logging that does), so it can be disabled to keep token counts comparable.
    log_test_eval: bool = True
    # Parent-sampling source (OPT-IN ablation): which weights pick the parent to mutate.
    #   "win"     -> win-count weights from select_population (EvoTool default, UNCHANGED)
    #   "val_acc" -> sample parents LINEAR-proportional to per-policy MEAN S_sel reward
    # Retention (which policies survive) is UNCHANGED — still the win-count diversity rule.
    parent_select: str = "win"
    # Accept-gate evaluation set (OPT-IN ablation): where child is compared to parent.
    #   "batch" -> same mini-batch as run this generation (EvoTool default, UNCHANGED)
    #   "sel"   -> held-out S_sel (parent baseline read from maintained sel_means)
    accept_on: str = "batch"


@dataclass
class EvoToolConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    evolve: EvolveConfig = field(default_factory=EvolveConfig)
    benchmark: str = "dummy"
    data_path: str = "data/"
    output_path: str = "results/"
    seed: int = 42
    max_steps: int = 6            # max tool-call steps per episode
    n_train: int = 12             # |S_train|
    n_sel: int = 8                # |S_sel|
    n_test: int = 20              # |S_test| held-out, DISJOINT from train/sel
    n_eval: int = 20              # (legacy alias; test is the reported split)
    verbose: bool = False


def _merge_into_dataclass(instance: Any, data: dict) -> Any:
    if not is_dataclass(instance) or not isinstance(data, dict):
        return instance
    for f in fields(instance):
        if f.name not in data:
            continue
        value = data[f.name]
        current = getattr(instance, f.name)
        if is_dataclass(current) and isinstance(value, dict):
            _merge_into_dataclass(current, value)
        else:
            setattr(instance, f.name, value)
    return instance


def load_config(yaml_path: str) -> EvoToolConfig:
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f) or {}
    cfg = EvoToolConfig()
    _merge_into_dataclass(cfg, raw)
    return cfg


def apply_overrides(cfg: EvoToolConfig, overrides: list[str]) -> None:
    """Apply CLI overrides like `evolve.generations=4` or `benchmark=bfcl`."""
    for item in overrides or []:
        key, _, value = item.partition("=")
        target = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            target = getattr(target, p)
        leaf = parts[-1]
        current = getattr(target, leaf)
        # cast the string value to the type of the current field
        if isinstance(current, bool):
            cast = value.strip().lower() in ("1", "true", "yes")
        elif isinstance(current, int) and not isinstance(current, bool):
            cast = int(value)
        elif isinstance(current, float):
            cast = float(value)
        else:
            cast = value
        setattr(target, leaf, cast)
