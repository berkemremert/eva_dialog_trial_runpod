#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/workspace/huggingface}"
export PORT="${PORT:-7860}"
export VLLM_PORT="${VLLM_PORT:-8000}"
export VLLM_URL="${VLLM_URL:-http://127.0.0.1:${VLLM_PORT}}"
export MODEL_ID="${MODEL_ID:-Qwen/Qwen3-Omni-30B-A3B-Instruct}"
VLLM_ENV="${VLLM_ENV:-/workspace/.venvs/qwen3-vllm-0.12.0}"

mkdir -p "$HF_HOME"
python -m pip install --upgrade -r requirements.txt

# vLLM and Gradio require incompatible Pydantic/Hugging Face Hub versions.
# Keep the GPU server isolated; its environment persists across Pod restarts.
if [[ ! -x "$VLLM_ENV/bin/python" ]]; then
  python -m venv "$VLLM_ENV"
fi
if ! "$VLLM_ENV/bin/python" -c \
  'import importlib.metadata as m; assert m.version("vllm") == "0.12.0"' \
  2>/dev/null; then
  "$VLLM_ENV/bin/python" -m pip install --upgrade pip
  "$VLLM_ENV/bin/python" -m pip install --no-cache-dir -r requirements-vllm.txt
fi

"$VLLM_ENV/bin/vllm" serve "$MODEL_ID" \
  --host 127.0.0.1 \
  --port "$VLLM_PORT" \
  --dtype bfloat16 \
  --tensor-parallel-size 2 \
  --distributed-executor-backend mp \
  --disable-custom-all-reduce \
  --enforce-eager \
  --max-model-len 8192 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.92 \
  --allowed-local-media-path / \
  --limit-mm-per-prompt '{"audio": 8}' \
  > /workspace/qwen3-vllm.log 2>&1 &
VLLM_PID=$!

cleanup() {
  kill "$VLLM_PID" 2>/dev/null || true
  wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Qwen vLLM başlatılıyor. İlk açılış birkaç dakika sürebilir..."
for attempt in $(seq 1 240); do
  if curl --silent --fail "$VLLM_URL/health" >/dev/null; then
    echo "Qwen hazır."
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "vLLM başlatılamadı. Son loglar:"
    tail -n 100 /workspace/qwen3-vllm.log
    exit 1
  fi
  if (( attempt % 6 == 0 )); then
    echo "Qwen hâlâ yükleniyor..."
  fi
  sleep 5
done

if ! curl --silent --fail "$VLLM_URL/health" >/dev/null; then
  echo "vLLM zaman aşımına uğradı. Son loglar:"
  tail -n 100 /workspace/qwen3-vllm.log
  exit 1
fi

python app.py
