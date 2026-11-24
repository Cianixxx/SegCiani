#!/bin/bash
# Plausibilita' della causa sul set sintetico (9 immagini con un pattern indotto).
# GPU 5 (24 GB): Gemma in bf16, niente 4 bit.
set -u
export PYTHONPATH=/workspace PYTHONIOENCODING=utf-8 SQ_OUT=outputs/slice_quality_inst
pip install --user -q scipy scikit-learn >/dev/null 2>&1
echo "$(date +%H:%M) == judge_cause_plausibility SYNTH (Gemma bf16) =="
SYNTH=1 SRC=datasets/cropped_synthetic_masks python -u slice_quality/examples/judge_cause_plausibility.py \
  || { echo "FAIL judge_cause_synth"; exit 1; }
echo "$(date +%H:%M) CHAIN_CAUSE_SYNTH_DONE"
