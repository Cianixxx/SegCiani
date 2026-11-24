#!/bin/bash
# Ri-giudizio dei soli testi il cui RIFERIMENTO e' cambiato: sei annotazioni corrette
# il 2026-09-29 (nomi di classe che le maschere contraddicevano). ~31 giudizi, GPU 6 (Quadro 24 GB, nessun processo attivo; 4 bit per lasciare spazio)
# (1080 Ti, 11 GB) quindi Gemma in 4 bit. Tutto il resto viene riusato dalla cache.
set -u
export PYTHONPATH=/workspace PYTHONIOENCODING=utf-8 SQ_OUT=outputs/slice_quality_inst
D=$SQ_OUT
pip install --user -q scipy scikit-learn >/dev/null 2>&1
step() { echo "$(date +%H:%M) == $1 =="; }

step "0/4 invalida i giudizi dati all'annotazione vecchia"
ANN=/data/dataset/answer_gts.jsonl MAP=/data/dataset/voc_patch_matches.json \
  python -u slice_quality/examples/invalidate_judgements.py || { echo "FAIL invalidate"; exit 1; }

step "1/4 judge_descriptions (riusa tutto il resto)"
JUDGE_4BIT=1 NARR=$D/error_narratives.json DESC=$D/image_descriptions.json \
  DESC_TR=$D/train36/descriptions_train36_full.json OUT=$D/judge_descriptions_v3 \
  python -u slice_quality/examples/judge_descriptions.py || { echo "FAIL judge"; exit 1; }
SRC=$D/judge_descriptions_v3 OUT=$D/judge_descriptions_scored \
  python -u slice_quality/examples/rescore_judge.py || { echo "FAIL rescore"; exit 1; }

step "2/4 judge_synth"
JUDGE_4BIT=1 NARR=$D/synth/narratives_synth.json DESC=$D/synth/descriptions_synth.json \
  OUT=$D/synth/judge_synth python -u slice_quality/examples/judge_synth.py || { echo "FAIL judge_synth"; exit 1; }

step "3/4 plausibilita' della causa, lato annotazione (VOC)"
JUDGE_4BIT=1 python -u slice_quality/examples/judge_cause_plausibility.py || { echo "FAIL cause"; exit 1; }

step "4/4 plausibilita' della causa, sintetiche"
SYNTH=1 SRC=datasets/cropped_synthetic_masks JUDGE_4BIT=1 \
  python -u slice_quality/examples/judge_cause_plausibility.py || { echo "FAIL cause synth"; exit 1; }

echo "$(date +%H:%M) CHAIN_REJUDGE_DONE"
