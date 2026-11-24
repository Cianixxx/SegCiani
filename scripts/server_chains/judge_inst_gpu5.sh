#!/bin/bash
# Giudice del run inst su GPU 5, in parallelo alla coda sintetica su GPU 2. Scrive nello stesso
# OUT del passo 9 della catena principale, che poi trovera' tutto in cache e finira' subito.
set -u
export PYTHONPATH=/workspace PYTHONIOENCODING=utf-8
D=outputs/slice_quality_inst
pip install --user -q scipy >/dev/null 2>&1
echo "$(date +%H:%M) == judge inst (GPU 5) =="
JUDGE_4BIT=1 NARR=$D/error_narratives.json DESC=$D/image_descriptions.json \
  DESC_TR=$D/train36/descriptions_train36_full.json OUT=$D/judge_descriptions_v1 \
  python -u slice_quality/examples/judge_descriptions.py || { echo "FAIL judge"; exit 1; }
echo "$(date +%H:%M) JUDGE_INST_GPU5_DONE"
