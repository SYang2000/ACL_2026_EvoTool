# Data guide

This repo ships **curated demo subsets** derived from the official benchmarks — enough to run and inspect the full evolution loop offline and deterministically. They are **not** the official evaluation sets, and numbers on them will not match the paper (which evaluates on the full benchmarks). This page documents what is bundled, the unified instance schema, and how to rebuild every dataset from its official source.

## What ships vs. what doesn't

| Benchmark | Bundled demo subset | Official source | Build script |
|---|---|---|---|
| BFCL | `data/bfcl/samples.json` (150) | [ShishirPatil/gorilla](https://github.com/ShishirPatil/gorilla) (`berkeley-function-call-leaderboard`) | `scripts/data/build_bfcl.py` |
| RestBench | `data/restbench/samples.json` (150) | [Yifan-Song793/RestGPT](https://github.com/Yifan-Song793/RestGPT) | `scripts/data/build_restbench.py` |
| ToolBench | `data/toolbench/samples.json` (150) + `recorded_outputs.json` | [OpenBMB/ToolBench](https://github.com/OpenBMB/ToolBench) (via the processed DFSDT answer set [Yhyu13/ToolBench_toolllama_G123_dfs](https://huggingface.co/datasets/Yhyu13/ToolBench_toolllama_G123_dfs)) | `scripts/data/build_toolbench.py`, `scripts/data/build_toolbench_outputs.py` |
| tau-bench | **not bundled** — build locally | [sierra-research/tau-bench](https://github.com/sierra-research/tau-bench) | `scripts/data/build_taubench.py`, `scripts/data/build_taubench_outputs.py` |
| dummy / diverse | `data/dummy/samples.json` (6), `data/diverse/samples.json` (9) | synthetic toys for smoke tests | — |

Each 150-instance dataset is partitioned into disjoint splits by `src/benchmarks.split_instances` at load time: **90 train / 30 selection / 30 held-out test**, shuffled with `seed: 42` (`configs/base.yaml`). The held-out test split never overlaps train or selection.

`data/taubench` is not redistributed here; the code fully supports it, but you must build it yourself from a clone of the official tau-bench repo (one command, below). The tau-bench retail *environment* (users/orders/products DB the executor runs against) is vendored under `src/envs/taubench_data/`.

## Unified instance schema

Every benchmark is normalized into the same shape (`src/benchmarks.py`), so nothing downstream knows which benchmark it is running. Fields actually present in the bundled `samples.json` files:

```jsonc
{
  "id": "bfcl_1",                    // stable instance id (bfcl_N, rb_tmdb_N / rb_spotify_N, tb_N, tau_retail_N, d/dv toys)
  "query": "...",                    // the user task, verbatim from the source benchmark
  "available_tools": [               // the instance's tool index
    {"name": "...", "description": "...", "parameters": {"arg": "description"}}
  ],
  "gold_plan": [                     // ordered gold tool calls
    {"tool": "...", "args": {...}}
  ],
  "gold_answer": null,               // reference answer string (toys only; null for real benchmarks)
  "mock_outputs": {},                // {tool: output string} — toys only; empty for real benchmarks
  // per-benchmark extras:
  "gold_match": [...],               // bfcl only: the official possible-answer structure for AST matching
  "metric": "restbench_correct_path",// restbench only: metric selector
  "required_token": "alpha"          // diverse toy only: synthetic confirm-token dialect
}
```

At load time, `load_instances` additionally attaches `benchmark` (for metric dispatch) and merges two **sidecar files** (the source `samples.json` is never edited):

- `data/toolbench/recorded_outputs.json` — `{instance_id: [{"tool", "args", "output", "error"}, ...]}`, the real recorded ToolBench API responses aligned to the gold plan (built by `build_toolbench_outputs.py`); attached as `instance["recorded_outputs"]`.
- `data/taubench/required_outputs.json` — `{instance_id: [required strings]}`, the strings the agent's final answer must contain under the official tau-bench reward (built by `build_taubench_outputs.py`); attached as `instance["outputs"]`. If a sidecar is absent, the loader degrades explicitly (ToolBench steps are marked unavailable; the tau reward reduces to DB-state equality alone).

**Offline execution.** Tool calls are never sent to live APIs (`src/env/executor.py`): ToolBench *replays* the recorded real responses via per-tool FIFO queues — never fabricating content and never reporting a success the source did not record (recorded errors are replayed as errors, exhausted queues return an explicit `unavailable` status); RestBench uses a typed mock that synthesizes a realistically-shaped response from the real OAS endpoint shape; BFCL performs **no** execution at all (the official paradigm is AST matching of the generated calls); tau-bench executes for real against the vendored retail DB. The resulting `status` (success / error / unavailable) is itself a diagnostic signal the Blamer reads.

## Rebuilding from the official sources

The builders are deterministic converters from real source data (no fabrication); each one requires that every selected instance's **gold plan passes the benchmark's official success metric** before it is kept.

### BFCL

Clone [gorilla](https://github.com/ShishirPatil/gorilla) and point the builder at the `bfcl_eval/data` directory inside `berkeley-function-call-leaderboard` (ground truth defaults to its `possible_answer/` subdirectory):

```bash
python scripts/data/build_bfcl.py --bfcl-raw /path/to/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data \
    [--bfcl-gt /path/to/possible_answer]         # default: <bfcl-raw>/possible_answer
# or: export BFCL_RAW_DIR=... && python scripts/data/build_bfcl.py
```

Selects 150 genuinely multi-call instances (>= 2 gold calls) in source id order: 90 `parallel_multiple` + 40 `parallel` + 20 `live_parallel_multiple` from the `BFCL_v4_<category>.json` files, keeping only instances whose gold plan passes the official AST metric.

### RestBench

```bash
python scripts/data/build_restbench.py
```

No flags: it downloads the real instruction sets (`datasets/tmdb.json`, `datasets/spotify.json`) and the Spotify OpenAPI spec (`specs/spotify_oas.json`) straight from the RestGPT repo (cached under `/tmp/restgpt_src_cache`), reads the vendored TMDB endpoint universe from `scripts/data/tmdb_tools.json`, and emits 100 TMDB + 50 Spotify instances. Rebuilding from scratch reproduces the bundled `data/restbench/samples.json` byte-for-byte.

### ToolBench

```bash
python scripts/data/build_toolbench.py             # -> data/toolbench/samples.json
python scripts/data/build_toolbench_outputs.py     # -> data/toolbench/recorded_outputs.json (sidecar)
```

No flags: both download `toolllama_G123_dfs_eval.json` from the HF dataset [Yhyu13/ToolBench_toolllama_G123_dfs](https://huggingface.co/datasets/Yhyu13/ToolBench_toolllama_G123_dfs) — the official ToolBench DFSDT answer trees flattened to conversations — and require the `huggingface_hub` package. The first script selects 150 solved queries (deduplicated, every gold call resolvable in the instance's tool index); the second recovers the *real recorded API responses* for exactly those instances into the sidecar, which the offline `ReplayExecutor` replays at rollout time.

### tau-bench (must be built locally)

```bash
git clone https://github.com/sierra-research/tau-bench /path/to/tau-bench
python scripts/data/build_taubench.py --tau-repo /path/to/tau-bench            # -> data/taubench/samples.json
python scripts/data/build_taubench_outputs.py --tau-repo /path/to/tau-bench    # -> data/taubench/required_outputs.json
# (--tau-repo can also be supplied via the TAU_BENCH_REPO env var)
```

The first script parses the retail task literals (`tasks_test.py`, then `tasks_train.py`, then `tasks_dev.py`) into 150 instances over the 15 retail tools (tool schemas from `scripts/data/taubench_tools.json`), keeping only deduplicated tasks whose gold action sequence reaches a valid DB state under the official metric. The second recovers each task's required output strings (`outputs=[...]` in the upstream `Task` literals) into the sidecar used by the official success check. After both, `python run.py --config configs/evotool.yaml --benchmark taubench` works like any other benchmark.

## Verifying data quality

```bash
python scripts/verify_data.py                # default: restbench bfcl taubench toolbench
python scripts/verify_data.py bfcl toolbench # or a subset
```

(The default invocation reports an error row for `taubench` until you build it locally — expected, not a bug.)

The gate replays each instance's **gold plan** (as successful steps) through the benchmark's **official** success metric and reports the pass rate per benchmark — gold is correct by construction, so a low rate means malformed data (wrong tool names, wrong gold args, broken schema), and failing ids are printed. Caveats it inherits by design: with `answer=None` the tau-bench check covers DB-state equality only (the gold plan carries no natural-language answer for the `r_outputs` check), and ToolBench falls back to the offline call-subsequence heuristic since no judge LLM is attached.
