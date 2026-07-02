"""Offline wiring test: exercise the whole pipeline (agent + blame + mutate +
select + evolve loop + metrics) with a fake LLM, so we catch import/shape bugs
without spending any GPU time. Run from the repo root:  python scripts/offline_smoke.py
"""

import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.benchmarks import load_instances, split_instances
from src.config import EvoToolConfig
from src.evolve.loop import evolve
from src.metrics import success_rate
from src.policy.agent import route_episode


class FakeClient:
    """Returns plausibly-shaped structured outputs; selector is stochastic so
    children differ from parents and the accept / selection paths get exercised."""

    def __init__(self):
        self.total_tokens = 0
        self.n = 0

    def generate(self, messages, **kw):
        self.n += 1
        self.total_tokens += 50
        return "final answer grounded in tool outputs"

    def generate_json(self, messages):
        self.n += 1
        self.total_tokens += 50
        system = messages[0]["content"]
        user = messages[-1]["content"]
        if "diagnostic judge" in system:
            return {"planner": 0.1, "selector": 0.6, "caller": 0.2, "synthesizer": 0.1,
                    "primary": "selector", "diagnosis": "wrong tool chosen"}
        if "prompt editor" in system:
            return {"target_module": "selector", "revised_spec": f"REVISED spec v{self.n}"}
        if '"plan"' in system or "planning agent" in system.lower():
            return {"plan": ["first subgoal", "second subgoal"]}
        if "tool_name" in system:
            names = re.findall(r"- (\w+):", user)
            rng = random.Random(self.n)
            return {"tool_name": rng.choice(names) if names else "none"}
        if "arguments" in system:
            return {"arguments": {"query": "x"}}
        return {}


def main():
    cfg = EvoToolConfig()
    cfg.benchmark = "dummy"
    cfg.evolve.mutation_target = "blame"
    cfg.evolve.selection = "diversity"
    cfg.evolve.generations = 3
    cfg.evolve.batch_size = 2
    cfg.n_train, cfg.n_sel, cfg.n_eval, cfg.max_steps = 4, 4, 6, 3

    client = FakeClient()
    instances = load_instances(cfg.benchmark, cfg.data_path)
    train, sel, eval_ = split_instances(instances, cfg)
    print(f"loaded {len(instances)} dummy instances; split {len(train)}/{len(sel)}/{len(eval_)}")

    population = evolve(client, cfg, train, sel)
    eps = [route_episode(client, population, x, cfg.max_steps) for x in eval_]
    print(f"OK: pop={len(population)} success={success_rate([e.success for e in eps]):.1f} "
          f"llm_calls={client.n}")

    # sanity: also run the static path
    cfg.evolve.mutation_target = "none"
    cfg.evolve.selection = "static"
    cfg.evolve.generations = 0
    pop2 = evolve(client, cfg, train, sel)
    print(f"OK static path: pop={len(pop2)} ids={[p.policy_id for p in pop2]}")


if __name__ == "__main__":
    main()
