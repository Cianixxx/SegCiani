#!/bin/bash
# Completa il run cov05b: train36, sintetiche, narrativa unita, candidati, giudici.
set -u
export PYTHONPATH=/workspace PYTHONIOENCODING=utf-8 SQ_CLAIM_COV=0.5 SQ_OUT=outputs/slice_quality_cov05b
D=$SQ_OUT
pip install --user -q scipy scikit-learn >/dev/null 2>&1
step() { echo "$(date +%H:%M) == $1 =="; }

step "1/7 describe_train36_gpu (Blocco 1 cov, DINOv2, Qwen, ipotesi + induzione dal run cov05b)"
QWEN_4BIT=0 python -u slice_quality/examples/describe_train36_gpu.py || { echo "FAIL train36"; exit 1; }

step "2/7 narrate_synth (CPU)"
OUT=$D/synth python -u slice_quality/examples/narrate_synth.py || { echo "FAIL narrate_synth"; exit 1; }

step "3/7 describe_synth_gpu"
QWEN_4BIT=0 OUT=$D/synth python -u slice_quality/examples/describe_synth_gpu.py || { echo "FAIL describe_synth"; exit 1; }

step "4/7 narrate_errors val + train36 (CPU)"
PKL=$D/result_tagged.pkl TRAIN36=$D/train36 OUT=$D/error_narratives \
  python -u slice_quality/examples/narrate_errors.py || { echo "FAIL narrate"; exit 1; }

step "5/7 explain_candidates (CPU)"
PKL=$D/result_tagged.pkl OUT=$D/candidate_clusters.json \
  python -u slice_quality/examples/explain_candidates.py || { echo "FAIL candidates"; exit 1; }

step "6/7 judge Gemma v7, 79 immagini (GPU)"
JUDGE_4BIT=1 NARR=$D/error_narratives.json DESC=$D/image_descriptions.json \
  DESC_TR=$D/train36/descriptions_train36_full.json OUT=$D/judge_descriptions_v7 \
  python -u slice_quality/examples/judge_descriptions.py || { echo "FAIL judge"; exit 1; }

step "7/7 judge_synth (GPU)"
JUDGE_4BIT=1 NARR=$D/synth/narratives_synth.json DESC=$D/synth/descriptions_synth.json \
  OUT=$D/synth/judge_synth python -u slice_quality/examples/judge_synth.py || { echo "FAIL judge_synth"; exit 1; }

echo "$(date +%H:%M) CHAIN_COV05B_ALL_DONE"
