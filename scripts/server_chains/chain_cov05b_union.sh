#!/bin/bash
# Merge misurati contro l'unione delle GT fuse + aggettivo del contorno consapevole del boundary IoU.
set -u
export PYTHONPATH=/workspace PYTHONIOENCODING=utf-8 SQ_CLAIM_COV=0.5 SQ_OUT=outputs/slice_quality_cov05b
D=$SQ_OUT
pip install --user -q scipy scikit-learn >/dev/null 2>&1
step() { echo "$(date +%H:%M) == $1 =="; }

step "1/7 narrate_errors val + train36"
PKL=$D/result_tagged.pkl TRAIN36=$D/train36 OUT=$D/error_narratives \
  python -u slice_quality/examples/narrate_errors.py || { echo "FAIL narrate"; exit 1; }

step "2/7 regen_gallery (pattern val con i nuovi fatti)"
python -u slice_quality/examples/regen_gallery.py || { echo "FAIL regen"; exit 1; }

step "3/7 enrich_train36_details"
OUT=$D/train36 PKL=$D/result_tagged.pkl python -u slice_quality/examples/enrich_train36_details.py || { echo "FAIL enrich"; exit 1; }

step "4/7 narrate_synth"
OUT=$D/synth python -u slice_quality/examples/narrate_synth.py || { echo "FAIL narrate_synth"; exit 1; }

step "5/7 describe_synth_gpu"
QWEN_4BIT=0 OUT=$D/synth python -u slice_quality/examples/describe_synth_gpu.py || { echo "FAIL describe_synth"; exit 1; }

step "6/7 judge v9 (riusa v8 dove il testo e' identico)"
cp $D/judge_descriptions_v8.json $D/judge_descriptions_v9.json
JUDGE_4BIT=1 NARR=$D/error_narratives.json DESC=$D/image_descriptions.json \
  DESC_TR=$D/train36/descriptions_train36_full.json OUT=$D/judge_descriptions_v9 \
  python -u slice_quality/examples/judge_descriptions.py || { echo "FAIL judge"; exit 1; }

step "7/7 judge_synth"
JUDGE_4BIT=1 NARR=$D/synth/narratives_synth.json DESC=$D/synth/descriptions_synth.json \
  OUT=$D/synth/judge_synth python -u slice_quality/examples/judge_synth.py || { echo "FAIL judge_synth"; exit 1; }

echo "$(date +%H:%M) CHAIN_COV05B_UNION_DONE"
