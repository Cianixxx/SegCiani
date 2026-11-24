#!/bin/bash
# Plausibilita' della causa su TUTTE le immagini che ne hanno una (599: 582 val + 17
# train36), giudicata con la sola narrativa -- fuori dalle 80 annotate non esiste un
# testo umano. Controllo su un campione di 150. GPU 3, Gemma 4 bit (20 s a giudizio
# contro i 55 del bf16 con offload: vedi chain_cause_synth).
set -u
export PYTHONPATH=/workspace PYTHONIOENCODING=utf-8 SQ_OUT=outputs/slice_quality_inst
pip install --user -q scipy scikit-learn >/dev/null 2>&1
echo "$(date +%H:%M) == judge_cause_plausibility ALL (Gemma 4bit, GPU 3) =="
ALL=1 JUDGE_4BIT=1 CONTROL_N=150 python -u slice_quality/examples/judge_cause_plausibility.py \
  || { echo "FAIL judge_cause_val"; exit 1; }
echo "$(date +%H:%M) CHAIN_CAUSE_VAL_DONE"
