"""Run a policy on a task instance to produce an episode record e = (x, tau, y-hat, R).

The composition is the paper's fixed modular policy run in STRICT SEQUENCE
(SPEC.md §1):

    Pi = synthesizer o (caller o selector)*  o planner

  1. Planner   : query -> ordered substeps [sub_1 ... sub_k]   (decomposition ONLY).
  2. For each sub_i (in order):
        Selector picks the single tool for THAT substep;
        Caller fills that tool's arguments;
        the tool is executed (per §2 tool-output wiring) -> output_i.
     This yields the per-substep tuples (sub_i, tool_i, args_i, output_i).
  3. Synthesizer: given ALL (sub_i, tool_i, args_i, output_i) tuples + the original
     query -> the FINAL ANSWER. That answer is the graded object (SPEC.md §3).

Tool-output wiring (§2): taubench = REAL execution (TauBenchEnv); toolbench = REPLAY
of recorded DFSDT responses; restbench = realistic typed OAS mock; bfcl = NO
execution (AST matching only — the chosen calls are recorded with no observation).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.env import registry
from src.env.executor import executor_for
from src.evaluators import is_success
from src.llm.client import LLMClient
from src.metrics import progress
from src.policy import modules
from src.policy.modules import Policy


def env_for(instance: dict):
    """Tool environment for an instance (SPEC.md §2):

      - taubench  : REAL stateful retail DB env (vendored tools over the JSON DB).
      - toolbench : ReplayExecutor over the recovered recorded API responses.
      - restbench : MockOASExecutor (realistic typed mock from the OAS shape).
      - bfcl      : None  -> NO execution (official paradigm is AST matching).
    """
    if instance.get("benchmark") == "taubench":
        from src.envs.taubench_env import TauBenchEnv
        return TauBenchEnv()
    return executor_for(instance)  # may be None for bfcl


@dataclass
class Step:
    subgoal: str            # the planner substep this call serves (sub_i)
    tool: str
    args: dict
    observation: dict       # {tool, args, output, status[, output_available]}


@dataclass
class Episode:
    instance: dict
    plan: list[str]
    steps: list[Step]
    answer: str
    reward: float
    success: bool = False
    predicted_plan: list[dict] = field(default_factory=list)


# Episodes are deterministic (temperature=0), so identical (policy, instance)
# pairs are memoized. The selection phase re-evaluates surviving policies every
# generation; without this cache that would re-run hundreds of identical episodes.
_EPISODE_CACHE: dict = {}


def _policy_key(policy: Policy) -> int:
    return hash((policy.planner, policy.selector, policy.caller, policy.synthesizer))


# Selector tokens that mean "this substep needs no tool" (so we skip it).
_NO_TOOL = ("STOP", "NONE", "DONE", "FINISH", "SKIP", "NO_TOOL", "")


def _step_summary(sub: str, tool: str, obs: dict) -> str:
    out = obs.get("output")
    status = obs.get("status")
    if status == "unavailable":
        # toolbench: a recognized call with no recorded API response to replay (§2).
        out = "(unavailable: no recorded API response)"
    elif out is None:
        # bfcl: the chosen call is recorded but never executed (AST-matching
        # paradigm produces no observation) -> status "no_execution".
        out = "(no output)"
    return f"[{sub}] {tool} -> {out}"


def run_episode(client: LLMClient, policy: Policy, instance: dict, max_steps: int) -> Episode:
    key = (_policy_key(policy), instance.get("id"))
    if key in _EPISODE_CACHE:
        return _EPISODE_CACHE[key]

    query = instance["query"]
    tools = instance["available_tools"]
    executor = env_for(instance)          # None for bfcl (no execution)

    # 1. Planner -> ordered abstract substeps (decomposition only).
    plan = modules.run_planner(client, policy.planner, query)

    # 2. STRICT SEQUENCE: for each substep, Selector chooses its tool, Caller fills
    #    its args, and the tool is executed per the benchmark's §2 output wiring.
    steps: list[Step] = []
    state: dict = {}
    for sub in plan[:max_steps]:
        tool = modules.run_selector(client, policy.selector, query, sub, state, tools)
        if tool.upper() in _NO_TOOL:
            continue                      # this substep requires no tool call
        card = registry.find_tool(tools, tool)
        args = modules.run_caller(client, policy.caller, card, query, sub, state) if card else {}
        if executor is None:
            # bfcl: NO execution — record the chosen call, no observation produced.
            obs = {"tool": tool, "args": args, "output": None,
                   "status": "no_execution", "output_available": False}
        else:
            obs = executor.execute(tool, args)
        steps.append(Step(sub, tool, args, obs))
        state[f"step_{len(steps)}"] = _step_summary(sub, tool, obs)

    # 3. Synthesizer -> final answer from the (sub, tool, args, output) tuples + query.
    answer = modules.run_synthesizer(client, policy.synthesizer, query, steps)

    predicted_plan = [
        {"tool": s.tool, "args": s.args, "status": s.observation.get("status")} for s in steps
    ]
    # The graded object is the synthesizer's output (answer + the assembled call
    # list); is_success dispatches the per-benchmark success check (see evaluators).
    # The client is threaded through for benchmarks whose grader is itself an LLM
    # (toolbench: ToolEval-style LLM judge, offline approx).
    success = is_success(instance, steps, answer, client)
    reward = 0.7 * (1.0 if success else 0.0) + 0.3 * progress(predicted_plan, instance)
    episode = Episode(instance, plan, steps, answer, reward, success, predicted_plan)
    _EPISODE_CACHE[key] = episode
    return episode


def mean_reward(client: LLMClient, policy: Policy, instances: list[dict], max_steps: int) -> float:
    if not instances:
        return 0.0
    return sum(run_episode(client, policy, x, max_steps).reward for x in instances) / len(instances)


def _error_count(ep: Episode) -> int:
    return sum(1 for s in ep.steps if s.observation.get("status") == "error")


def route_episode(client: LLMClient, policies: list[Policy], instance: dict, max_steps: int) -> Episode:
    """Optional best-of-population routing (ablation only — SPEC.md §7 keeps
    the single Theta* at test time). Runs every retained policy and keeps the rollout
    with the fewest execution errors, breaking ties by reward."""
    eps = [run_episode(client, p, instance, max_steps) for p in policies]
    return min(eps, key=lambda e: (_error_count(e), -e.reward))
