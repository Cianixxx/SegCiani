#!/bin/bash
# Buchi nelle maschere per train36 e sintetiche (mask_facts), poi ri-giudizio dei soli testi cambiati.
set -u
export PYTHONPATH=/workspace PYTHONIOENCODING=utf-8 SQ_CLAIM_COV=0.5 SQ_OUT=outputs/slice_quality_cov05b
D=$SQ_OUT
pip install --user -q scipy scikit-learn >/dev/null 2>&1
step() { echo "$(date +%H:%M) == $1 =="; }

step "1/6 narrate_errors val + train36 (buchi su train36)"
PKL=$D/result_tagged.pkl TRAIN36=$D/train36 OUT=$D/error_narratives \
  python -u slice_quality/examples/narrate_errors.py || { echo "FAIL narrate"; exit 1; }

step "2/6 enrich_train36_details (righe di dettaglio con i buchi)"
OUT=$D/train36 PKL=$D/result_tagged.pkl python -u slice_quality/examples/enrich_train36_details.py || { echo "FAIL enrich"; exit 1; }

step "3/6 narrate_synth"
OUT=$D/synth python -u slice_quality/examples/narrate_synth.py || { echo "FAIL narrate_synth"; exit 1; }

step "4/6 describe_synth_gpu"
QWEN_4BIT=0 OUT=$D/synth python -u slice_quality/examples/describe_synth_gpu.py || { echo "FAIL describe_synth"; exit 1; }

step "5/6 judge v8 (riusa v7 dove il testo e' identico)"
cp $D/judge_descriptions_v7.json $D/judge_descriptions_v8.json
JUDGE_4BIT=1 NARR=$D/error_narratives.json DESC=$D/image_descriptions.json \
  DESC_TR=$D/train36/descriptions_train36_full.json OUT=$D/judge_descriptions_v8 \
  python -u slice_quality/examples/judge_descriptions.py || { echo "FAIL judge"; exit 1; }

step "6/6 judge_synth"
JUDGE_4BIT=1 NARR=$D/synth/narratives_synth.json DESC=$D/synth/descriptions_synth.json \
  OUT=$D/synth/judge_synth python -u slice_quality/examples/judge_synth.py || { echo "FAIL judge_synth"; exit 1; }

echo "$(date +%H:%M) CHAIN_COV05B_FIX_DONE"
