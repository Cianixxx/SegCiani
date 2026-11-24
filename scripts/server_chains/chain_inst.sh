#!/bin/bash
# Run "inst": claim per contenimento + istanze GT = oggetti VOC (SegmentationObject).
set -u
export PYTHONPATH=/workspace PYTHONIOENCODING=utf-8 SQ_CLAIM_COV=0.5 SQ_GT_INSTANCES=object SQ_OUT=outputs/slice_quality_inst
D=$SQ_OUT
mkdir -p $D
pip install --user -q scipy scikit-learn >/dev/null 2>&1
step() { echo "$(date +%H:%M) == $1 =="; }

step "0/10 retype_instances (Blocco 1 a oggetti; DINOv2 + Qwen solo sulle istanze nuove)"
PKL=outputs/slice_quality/result_tagged.pkl OUT=$D/result_tagged.pkl QWEN_4BIT=0 \
  python -u slice_quality/examples/retype_instances.py || { echo "FAIL retype"; exit 1; }

step "1/10 rescore_with_language (CPU)"
python -u slice_quality/examples/rescore_with_language.py || { echo "FAIL rescore"; exit 1; }

step "2/10 describe_train36_gpu"
QWEN_4BIT=0 python -u slice_quality/examples/describe_train36_gpu.py || { echo "FAIL train36"; exit 1; }

step "3/10 narrate_errors val + train36 (CPU)"
PKL=$D/result_tagged.pkl TRAIN36=$D/train36 OUT=$D/error_narratives \
  python -u slice_quality/examples/narrate_errors.py || { echo "FAIL narrate"; exit 1; }

step "4/10 regen_gallery (CPU)"
python -u slice_quality/examples/regen_gallery.py || { echo "FAIL regen"; exit 1; }

step "5/10 enrich_train36_details (CPU)"
OUT=$D/train36 PKL=$D/result_tagged.pkl python -u slice_quality/examples/enrich_train36_details.py || { echo "FAIL enrich"; exit 1; }

step "6/10 narrate_synth (CPU, componenti: nessuna maschera di oggetto)"
OUT=$D/synth python -u slice_quality/examples/narrate_synth.py || { echo "FAIL narrate_synth"; exit 1; }

step "7/10 describe_synth_gpu"
QWEN_4BIT=0 OUT=$D/synth python -u slice_quality/examples/describe_synth_gpu.py || { echo "FAIL describe_synth"; exit 1; }

step "8/10 explain_candidates (CPU)"
PKL=$D/result_tagged.pkl OUT=$D/candidate_clusters.json python -u slice_quality/examples/explain_candidates.py || { echo "FAIL candidates"; exit 1; }

step "9/10 judge Gemma (79 immagini)"
JUDGE_4BIT=1 NARR=$D/error_narratives.json DESC=$D/image_descriptions.json \
  DESC_TR=$D/train36/descriptions_train36_full.json OUT=$D/judge_descriptions_v1 \
  python -u slice_quality/examples/judge_descriptions.py || { echo "FAIL judge"; exit 1; }

step "10/10 judge_synth"
JUDGE_4BIT=1 NARR=$D/synth/narratives_synth.json DESC=$D/synth/descriptions_synth.json \
  OUT=$D/synth/judge_synth python -u slice_quality/examples/judge_synth.py || { echo "FAIL judge_synth"; exit 1; }

echo "$(date +%H:%M) CHAIN_INST_DONE"
