#!/usr/bin/env bash
# Set up SliceLens with the real backends on a Linux GPU server.
# Usage:  bash scripts/setup_server.sh
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
echo "==> Using $($PYTHON --version)"

# 1) Virtual environment.
if [ ! -d ".venv" ]; then
  echo "==> Creating .venv"
  "$PYTHON" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip

# 2) PyTorch.
#    If CUDA wheels from the default index don't match your driver, install the
#    right build first, e.g. for CUDA 12.1:
#      pip install torch --index-url https://download.pytorch.org/whl/cu121
if ! python -c "import torch" 2>/dev/null; then
  echo "==> Installing torch (default index; override for your CUDA if needed)"
  pip install torch
fi

# 3) SliceLens + real backends (Gemini + Qwen2.5-VL) + dev tools.
echo "==> Installing SliceLens with [gemini,qwen,dev] extras"
pip install -e ".[gemini,qwen,dev]"

# 4) Sanity checks.
echo "==> Running test suite"
python -m pytest -q

echo
echo "==> CUDA check"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

cat <<'EOF'

Done. Next:
  1) export GOOGLE_API_KEY=...                      # Gemini hypothesis generation
  2) python scripts/check_qwen.py                   # verify the VLM yes/no logits
  3) python examples/run_coco.py --gt ... --results ... --image-root ... --label bicycle

Faster VLM serving (optional, paper deployment):  pip install -e ".[vllm]"
EOF
