#!/bin/bash
# Plausibilita' della CAUSA (slice/ipotesi SliceLens) contro due racconti dello stesso
# errore: annotazione umana e narrativa della pipeline. Gemma 4-bit, GPU 3 (TITAN X 12 GB).
set -u
export PYTHONPATH=/workspace PYTHONIOENCODING=utf-8 SQ_OUT=outputs/slice_quality_inst
pip install --user -q scipy scikit-learn >/dev/null 2>&1
echo "$(date +%H:%M) == judge_cause_plausibility (Gemma 4bit) =="
JUDGE_4BIT=1 python -u slice_quality/examples/judge_cause_plausibility.py || { echo "FAIL judge_cause"; exit 1; }
echo "$(date +%H:%M) CHAIN_CAUSE_DONE"
