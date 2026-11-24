#!/bin/bash
# Coda della catena "unione": pattern sintetici (Qwen 4 bit: GPU 0 e' una 1080 Ti da 11 GB), giudice v9, giudice sintetico.
set -u
export PYTHONPATH=/workspace PYTHONIOENCODING=utf-8 SQ_CLAIM_COV=0.5 SQ_OUT=outputs/slice_quality_cov05b
D=$SQ_OUT
pip install --user -q scipy scikit-learn >/dev/null 2>&1
step() { echo "$(date +%H:%M) == $1 =="; }

step "5/7 describe_synth_gpu (Qwen 4 bit)"
QWEN_4BIT=1 OUT=$D/synth python -u slice_quality/examples/describe_synth_gpu.py || { echo "FAIL describe_synth"; exit 1; }

step "6/7 judge v9 (riusa v8 dove il testo e' identico)"
cp $D/judge_descriptions_v8.json $D/judge_descriptions_v9.json
JUDGE_4BIT=1 NARR=$D/error_narratives.json DESC=$D/image_descriptions.json \
  DESC_TR=$D/train36/descriptions_train36_full.json OUT=$D/judge_descriptions_v9 \
  python -u slice_quality/examples/judge_descriptions.py || { echo "FAIL judge"; exit 1; }

step "7/7 judge_synth"
JUDGE_4BIT=1 NARR=$D/synth/narratives_synth.json DESC=$D/synth/descriptions_synth.json \
  OUT=$D/synth/judge_synth python -u slice_quality/examples/judge_synth.py || { echo "FAIL judge_synth"; exit 1; }

echo "$(date +%H:%M) CHAIN_COV05B_UNION_DONE"
