#!/bin/bash
# Catena "claim per contenimento + GT inglobate come istanze" (1 + M), solo val.
set -u
export PYTHONPATH=/workspace PYTHONIOENCODING=utf-8
D=outputs/slice_quality_cov05b
mkdir -p $D
pip install --user -q scipy scikit-learn >/dev/null 2>&1
step() { echo "$(date +%H:%M) == $1 =="; }

step "0/4 retype_pickle (Blocco 1 con claim_cov=0.5, GT inglobate mantenute; CPU)"
SQ_CLAIM_COV=0.5 PKL=outputs/slice_quality/result_tagged.pkl OUT=$D/result_tagged.pkl \
  python -u slice_quality/examples/retype_pickle.py || { echo "FAIL retype"; exit 1; }

step "1/4 rescore_with_language (CPU)"
SQ_OUT=$D python -u slice_quality/examples/rescore_with_language.py || { echo "FAIL rescore"; exit 1; }

step "2/4 narrate_errors (CPU)"
PKL=$D/result_tagged.pkl OUT=$D/error_narratives TRAIN36=/nonexistent \
  python -u slice_quality/examples/narrate_errors.py || { echo "FAIL narrate"; exit 1; }

step "3/4 regen_gallery (CPU)"
SQ_OUT=$D python -u slice_quality/examples/regen_gallery.py || { echo "FAIL regen"; exit 1; }

step "4/4 judge Gemma v6 (GPU)"
JUDGE_4BIT=1 NARR=$D/error_narratives.json DESC=$D/image_descriptions.json DESC_TR= \
  OUT=$D/judge_descriptions_v6 python -u slice_quality/examples/judge_descriptions.py || { echo "FAIL judge"; exit 1; }

echo "$(date +%H:%M) CHAIN_COV05B_DONE"
