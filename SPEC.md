# EvoTool — Implementation Specification

Design contract for this implementation of **EvoTool: Self-Evolving Tool-Use Policy
Optimization in LLM Agents via Blame-Aware Mutation and Diversity-Aware Selection**
(ACL 2026, https://aclanthology.org/2026.acl-long.2016/). Source files reference the section numbers below (`SPEC.md §n`)
wherever a design decision is enforced in code.

Conventions:
- The bundled `data/*/samples.json` files are fixed inputs: the loaders never edit them
  (sidecar files carry recovered outputs; see §2-§3).
- Where the offline setting forces an approximation of an official metric or
  environment, the code says so explicitly in comments and never labels a proxy
  "official".

---

## 1. Rollout — policy = 4 modules in STRICT sequence
Input per instance: `(query, available_tools, tool arg-schemas)`. Pipeline:
1. **Planner**: query → decompose into substeps `[sub_1 … sub_k]`. ONLY decomposition.
2. For each `sub_i`: **Selector** picks the tool for that substep; **Caller** fills that
   tool's arguments. Produces tuples `(sub_i, tool_i, args_i)`.
3. **Synthesizer**: given ALL `(sub_i, tool_i, args_i)` tuples + any tool outputs +
   the original query → produces the **final answer**. The synthesizer is the
   "answer the question using the prior steps" role — its OUTPUT is what gets graded.

The four module specs (prompts) are the EVOLVABLE part. Initial specs live in
`src/prompts.py` (paper A.7) and are the starting point of every run.

## 2. Tool outputs — "execute if possible, else stay as close to real as possible"
- **tau-bench**: REAL execution against the vendored retail DB. The synthesizer sees
  real observations and DB effects.
- **toolbench**: REPLAY of the real recorded API responses recovered from the source
  DFSDT data (`Yhyu13/ToolBench_toolllama_G123_dfs`), attached to each instance from
  the `recorded_outputs.json` sidecar. A step with no recorded response is marked
  explicitly as unavailable (never a fake "success").
- **restbench**: typed mock responses derived from the REAL OAS schema (realistic
  example values), never "fake success + empty output".
- **bfcl**: the official paradigm is AST matching of the generated calls — there is NO
  execution. The synthesizer reasons over the chosen calls (no observations).

## 3. Reward = the OFFICIAL per-benchmark evaluation of the synthesizer's final output
Per benchmark:
- **restbench**: the official automatic **Correct-Path** metric — the gold endpoint
  sequence must be an ordered subsequence of the successfully executed endpoints
  (argument-agnostic).
- **bfcl**: real gorilla **AST checker** semantics (`src/eval_bfcl_ast.py`):
  function-name match + per-arg value match + exact call-count set-match (extra calls
  FAIL) + rejection of unexpected params + proper per-type checking (no loose
  str/int coercion). Negative controls fail.
- **tau-bench**: the official reward has TWO required halves — **DB-state equality AND
  required output-strings** (`r_outputs`, recovered from the source Tasks into the
  `required_outputs.json` sidecar; reward = DB-state-equal AND every required output
  string present in the answer).
- **toolbench**: the official ToolEval (GPT judge pass-rate over live RapidAPI) is not
  runnable offline. `src/eval_toolbench_judge.py` implements a **ToolEval-style LLM
  judge (offline approximation)**: the served model judges whether the synthesizer's
  final answer solves the query, given the query + recorded tool outputs. It is
  labeled as an approximation, not as "official".

The graded object is the SYNTHESIZER's final output (for bfcl/restbench that output is
the final tool-call list assembled from selector/caller; for taubench the executed
actions + response text; for toolbench the answer text). The synthesizer is naturally
heavy on QA tasks and light on pure call-matching tasks — by design, not a defect.

## 4. Blamer — LLM-driven, reads the trajectory (paper §4.1)
The Blamer LLM is the DECISION-MAKER:
- Input packet: the query, the FULL per-substep trajectory
  `(sub_i, tool_i, args_i, output_i/status)`, and the reward (pass/no-pass). It may
  also include the gold plan/answer (allowed — this is the training set) and the
  rule-extracted events as HINTS.
- The Blamer LLM judges which step(s) are wrong (or all-correct) and returns ONE module
  of {planner, selector, caller, synthesizer}. **The LLM's arg-max choice is taken
  directly** — no gold-derived rule overrides it.
- `rule_blame` exists ONLY as a fallback when the LLM output is unparseable.
- `diagnostics.py` rule-events are advisory HINTS only; they never decide the module.
  The synthesizer is a first-class blame target (it produces the graded answer).
- Scope follows the paper: blame operates on the single representative episode `e`
  (the lowest-reward episode of the mini-batch).

## 5. Mutator — Blamer gives module, Mutator writes the fix (paper §4.2)
`blame` returns the module; **`mutate` generates the prompt fix** for THAT module only,
from the trajectory + reward. The child differs from the parent in exactly one module.
The `use_trajectory` / `use_feedback` toggles implement the mutation-guidance ablation
(paper Table 3).

## 6. Accept + Diversity-Aware Selection (paper §4.3)
- A child enters the population iff it beats its parent (the gain is validated before
  admission).
- Diversity selection on **S_sel (validation)**: instance-wise winners
  `W(x)=argmax_Θ r_x(Θ)`; retain every Θ that wins ≥1 sel instance; weight
  `w(Θ) = fraction of sel wins`; sample the next parent ∝ `w(Θ)`.

## 7. Test-time deployment — SINGLE policy, not an ensemble
- At eval time the single `Θ*` = argmax average sel-reward is deployed (`best_policy`).
- The per-instance `route_episode` ensemble exists ONLY as an explicitly-named optional
  ablation (`evolve.test_ensemble`), never the headline number.

## 8. Budget — data-derived generations
- Generations per epoch = `ceil(n_train / batch_size)` (default 90/3 = 30); the default
  budget is **EPOCHS = 3** → 90 generations.
- Each epoch is a fresh shuffled pass over ALL of S_train (WITHOUT replacement within
  an epoch, so every train instance is used once per epoch; reshuffled each epoch).
  `evolve.epochs` is the exposed knob; `evolve.generations` exists only as an explicit
  budget-ablation override.

## 9. Logging — capture EVERYTHING needed to understand the optimization
A structured run log (JSONL, one record per generation) is written under the run's
output dir, including BOTH accepted and REJECTED generations:
- gen index, epoch, parent_id, mini-batch instance ids
- blamed module + the Blamer's full rationale (the LLM's parsed JSON)
- the mutated module's prompt BEFORE and AFTER (full text + a unified diff)
- child_id, accepted/rejected, parent vs child batch reward
- learning-curve points: train-batch reward, validation (S_sel) score, and a periodic
  held-out test (S_test) score at epoch boundaries; per-module cumulative mutation
  counts; token cost; population snapshot (ids + per-id sel win-counts/weights)
Also a final per-run summary (learning curves as arrays) so curves can be plotted
directly — this is the format behind `docs/replay.html`.

## 10. Module map
`src/policy/agent.py`, `src/policy/modules.py`,
`src/evolve/{loop,blame,diagnostics,mutate,select}.py`, `src/evaluators.py`,
`src/metrics.py`, `src/eval_bfcl.py`, `src/eval_bfcl_ast.py`,
`src/eval_toolbench_judge.py`, `src/envs/taubench_env.py`, `src/env/executor.py`,
`src/runlog.py`, `src/config.py`, `configs/base.yaml`, `run.py`.

## 11. Paper mechanism reference (for faithfulness checks)
- Blamer: diagnostics extracted from the trajectory; Blamer LLM emits module-wise blame
  scores; `π* = argmax_π b_π(e)` — the LLM's arg-max decides (NOT a gold rule).
- Mutator: `F = MutatorLLM(e, π*, D, Θ_p)`; `θ'_{π*} = EditPrompt(θ_{π*}, F)`; only π*
  changes.
- Diversity selection: instance-wise winners on S_sel; win-frequency weights;
  win-proportional parent sampling; **at inference return a single Θ\*** (argmax mean
  reward).
