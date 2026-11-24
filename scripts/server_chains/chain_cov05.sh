#!/bin/bash
# Catena sul run ritipato (claim_cov = 0.5), solo val. Dentro il container.
set -u
export PYTHONPATH=/workspace PYTHONIOENCODING=utf-8
D=outputs/slice_quality_cov05
pip install --user -q scipy scikit-learn >/dev/null 2>&1
step() { echo "$(date +%H:%M) == $1 =="; }

step "1/4 rescore_with_language (cluster + certificazione + SliceLens, CPU)"
SQ_OUT=$D python -u slice_quality/examples/rescore_with_language.py || { echo "FAIL rescore"; exit 1; }

step "2/4 narrate_errors (fatti di maschera + narrativa, CPU)"
PKL=$D/result_tagged.pkl OUT=$D/error_narratives TRAIN36=/nonexistent \
  python -u slice_quality/examples/narrate_errors.py || { echo "FAIL narrate"; exit 1; }

step "3/4 regen_gallery (pattern con righe di dettaglio, CPU)"
SQ_OUT=$D python -u slice_quality/examples/regen_gallery.py || { echo "FAIL regen"; exit 1; }

step "4/4 judge Gemma v5 (GPU)"
JUDGE_4BIT=1 NARR=$D/error_narratives.json DESC=$D/image_descriptions.json DESC_TR= \
  OUT=$D/judge_descriptions_v5 python -u slice_quality/examples/judge_descriptions.py || { echo "FAIL judge"; exit 1; }

echo "$(date +%H:%M) CHAIN_COV05_DONE"
