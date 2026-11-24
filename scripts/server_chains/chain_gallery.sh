#!/bin/bash
# Rigenera galleria + descrizioni dal pickle (CPU) e ri-giudica i soli testi dei
# pattern che cambiano: l'ordine delle righe "near candidate pattern" e' ora
# deterministico (prima seguiva l'iterazione di un set), quindi 315 descrizioni
# cambiano ordine e 11 di quelle giudicate con esse. GPU 6, Gemma 4 bit.
set -u
export PYTHONPATH=/workspace PYTHONIOENCODING=utf-8 SQ_OUT=outputs/slice_quality_inst
D=$SQ_OUT
pip install --user -q scipy scikit-learn >/dev/null 2>&1
step() { echo "$(date +%H:%M) == $1 =="; }

step "1/3 regen_gallery (CPU)"
cp $D/image_descriptions.json $D/image_descriptions.json.bak_order
python -u slice_quality/examples/regen_gallery.py || { echo "FAIL regen"; exit 1; }

step "2/3 judge_descriptions: rifa' solo i testi cambiati"
JUDGE_4BIT=1 NARR=$D/error_narratives.json DESC=$D/image_descriptions.json \
  DESC_TR=$D/train36/descriptions_train36_full.json OUT=$D/judge_descriptions_v3 \
  python -u slice_quality/examples/judge_descriptions.py || { echo "FAIL judge"; exit 1; }

step "3/3 rescore (score = somma dei criteri)"
SRC=$D/judge_descriptions_v3 OUT=$D/judge_descriptions_scored \
  python -u slice_quality/examples/rescore_judge.py || { echo "FAIL rescore"; exit 1; }

echo "$(date +%H:%M) CHAIN_GALLERY_DONE"
