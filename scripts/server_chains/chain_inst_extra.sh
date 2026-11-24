#!/bin/bash
# Aggiunge alla run inst le ipotesi generiche nuove certificate dopo la riverifica
# sull'intera popolazione (ottobre 2026): stessa regola delle 38 di luglio. Poi
# rifa' a cascata cio' che dipende dall'insieme delle situazioni certificate --
# train36 e set sintetico (Qwen), giudici sui testi cambiati e plausibilita' della
# causa (Gemma 4 bit, dalla cache tutto il resto), galleria. Le narrative non
# dipendono dalle situazioni e non si toccano.
# Lanciata in un container con Qwen (/models/qwen) e Gemma (/models/gemma) montati.
# Backup della run PRIMA, lato host: outputs/slice_quality_inst_bak_20261006.
set -u
export PYTHONPATH=/workspace PYTHONIOENCODING=utf-8 SQ_CLAIM_COV=0.5 SQ_GT_INSTANCES=object \
       SQ_OUT=outputs/slice_quality_inst
D=$SQ_OUT
pip install --user -q scipy scikit-learn >/dev/null 2>&1
step() { echo "$(date +%H:%M) == $1 =="; }

step "1/10 rescore_with_language con le ipotesi nuove (CPU)"
LANG_DIRS=/data/outputs:/data/outputs_extra python -u slice_quality/examples/rescore_with_language.py \
  || { echo "FAIL rescore"; exit 1; }
# controllo: rispetto alla base senza ipotesi nuove; la base deve riprodurre il backup
EXTRA=$D OFFICIAL=outputs/slice_quality_inst_bak_20261006 OUT=$D/extra_merge_check \
  python -u slice_quality/examples/compare_extra_hypotheses.py > $D/extra_merge_check.log 2>&1 \
  || { echo "FAIL check"; tail -5 $D/extra_merge_check.log; exit 1; }
grep -aE "controllo|certificate:|immagini con" $D/extra_merge_check.log

step "2/10 describe_train36_gpu (Qwen bf16): applica anche le ipotesi nuove"
QWEN_4BIT=0 python -u slice_quality/examples/describe_train36_gpu.py || { echo "FAIL train36"; exit 1; }
OUT=$D/train36 PKL=$D/result_tagged.pkl python -u slice_quality/examples/enrich_train36_details.py \
  || { echo "FAIL enrich"; exit 1; }

step "3/10 describe_synth_gpu (Qwen bf16)"
QWEN_4BIT=0 OUT=$D/synth python -u slice_quality/examples/describe_synth_gpu.py || { echo "FAIL synth"; exit 1; }

step "4/10 regen_gallery (CPU): descrizioni nell'ordine deterministico"
python -u slice_quality/examples/regen_gallery.py || { echo "FAIL regen"; exit 1; }

step "5/10 judge_descriptions: solo i testi cambiati"
JUDGE_4BIT=1 NARR=$D/error_narratives.json DESC=$D/image_descriptions.json \
  DESC_TR=$D/train36/descriptions_train36_full.json OUT=$D/judge_descriptions_v3 \
  python -u slice_quality/examples/judge_descriptions.py || { echo "FAIL judge"; exit 1; }
SRC=$D/judge_descriptions_v3 OUT=$D/judge_descriptions_scored \
  python -u slice_quality/examples/rescore_judge.py || { echo "FAIL rescore_judge"; exit 1; }

step "6/10 judge_synth: solo i testi cambiati"
JUDGE_4BIT=1 NARR=$D/synth/narratives_synth.json DESC=$D/synth/descriptions_synth.json \
  OUT=$D/synth/judge_synth python -u slice_quality/examples/judge_synth.py || { echo "FAIL judge_synth"; exit 1; }

step "7/10 plausibilita' della causa, immagini annotate"
JUDGE_4BIT=1 python -u slice_quality/examples/judge_cause_plausibility.py || { echo "FAIL cause"; exit 1; }

step "8/10 plausibilita' della causa, tutte le immagini con una causa"
ALL=1 JUDGE_4BIT=1 CONTROL_N=150 python -u slice_quality/examples/judge_cause_plausibility.py \
  || { echo "FAIL cause all"; exit 1; }

step "9/10 plausibilita' della causa, sintetiche"
SYNTH=1 SRC=datasets/cropped_synthetic_masks JUDGE_4BIT=1 \
  python -u slice_quality/examples/judge_cause_plausibility.py || { echo "FAIL cause synth"; exit 1; }

step "10/10 regen_gallery con i voti nuovi (CPU)"
python -u slice_quality/examples/regen_gallery.py || { echo "FAIL regen2"; exit 1; }

echo "$(date +%H:%M) CHAIN_INST_EXTRA_DONE"
