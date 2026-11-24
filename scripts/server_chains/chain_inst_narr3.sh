#!/bin/bash
# Narrativa v2 sul run inst (direzione+lato, regioni corrette, sconfinamenti, patch sotto soglia,
# verdetto globale, ordinamento) e ri-giudizio dei soli testi cambiati. GPU 0 (1080 Ti, 4 bit).
set -u
export PYTHONPATH=/workspace PYTHONIOENCODING=utf-8 SQ_CLAIM_COV=0.5 SQ_GT_INSTANCES=object SQ_OUT=outputs/slice_quality_inst
D=$SQ_OUT
pip install --user -q scipy scikit-learn >/dev/null 2>&1
step() { echo "$(date +%H:%M) == $1 =="; }

step "1/4 narrate_errors val + train36 (CPU)"
PKL=$D/result_tagged.pkl TRAIN36=$D/train36 OUT=$D/error_narratives \
  python -u slice_quality/examples/narrate_errors.py || { echo "FAIL narrate"; exit 1; }

step "2/4 narrate_synth (CPU)"
OUT=$D/synth python -u slice_quality/examples/narrate_synth.py || { echo "FAIL narrate_synth"; exit 1; }

used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "${used:-0}" -gt 2000 ]; then
  echo "$(date +%H:%M) GPU 0 occupata (${used} MiB): aspetto 20 minuti come concordato"
  sleep 1200
fi

step "3/4 judge v3 (riusa v2 dove il testo e' identico)"
cp $D/judge_descriptions_v2.json $D/judge_descriptions_v3.json
JUDGE_4BIT=1 NARR=$D/error_narratives.json DESC=$D/image_descriptions.json \
  DESC_TR=$D/train36/descriptions_train36_full.json OUT=$D/judge_descriptions_v3 \
  python -u slice_quality/examples/judge_descriptions.py || { echo "FAIL judge"; exit 1; }

step "4/4 judge_synth"
JUDGE_4BIT=1 NARR=$D/synth/narratives_synth.json DESC=$D/synth/descriptions_synth.json \
  OUT=$D/synth/judge_synth python -u slice_quality/examples/judge_synth.py || { echo "FAIL judge_synth"; exit 1; }

echo "$(date +%H:%M) CHAIN_INST_NARR3_DONE"
