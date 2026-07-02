#!/usr/bin/env bash
# Launch a local OpenAI-compatible vLLM server on :8000 (paper backbone: Qwen3-8B).
# A single A100 (TP=1) is plenty for an 8B model.
set -euo pipefail

VENV_PY=${VENV_PY:-python}
MODEL=${MODEL:?set MODEL to a local path or HF id, e.g. Qwen/Qwen3-8B}
# Served name defaults to the basename of MODEL (e.g. Qwen/Qwen3-8B -> Qwen3-8B);
# it must match llm.model_name in the config.
SERVED_NAME=${SERVED_NAME:-$(basename "$MODEL")}
PORT=${PORT:-8000}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
# HF_HOME is honored only if the caller set it (optional passthrough).
if [ -n "${HF_HOME:-}" ]; then
  export HF_HOME
fi

# --enforce-eager skips torch.compile/CUDA-graph capture: no compile cache to
# write, faster startup, and plenty fast for our small sequential workload.
exec "$VENV_PY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "$SERVED_NAME" \
  --port "$PORT" \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192 \
  --enforce-eager \
  --trust-remote-code
