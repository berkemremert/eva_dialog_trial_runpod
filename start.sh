#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/workspace/huggingface}"
export PORT="${PORT:-7860}"

mkdir -p "$HF_HOME"
python -m pip install --upgrade \
  --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
python -m pip install --upgrade -r requirements.txt
exec python app.py
