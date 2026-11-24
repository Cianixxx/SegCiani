#!/bin/bash
# Relaxed cluster gates on copies of the official VOC run (Daniele, 2026-10-22: "prova a rilassare qualche vincolo
# nella selezione dei gruppi cluster di VOC"). CPU only; the official run is not touched.
#   A  moderate: attribute shared at p <= 0.10 (was 0.05), coherence >= 0.5 (was 0.6)
#   B  no attribute gate: a group need not share a VLM attribute beyond chance (coherence as in the thesis)
# Each copy: rescore (same language groundings as the official run), images' IoU per slice, split-half under the
# same gates; then one report beside the official run.
#   setsid nohup bash outputs/chain_relaxed_cluster.sh > outputs/relaxed_cluster.log 2>&1 < /dev/null &
set -u
cd ~/storage/SLICELENS
D=outputs/slice_quality_inst
MOUNTS="-v $HOME/storage/SLICELENS:/workspace -v $HOME/storage/pascal_vlm:/data"
ENV="-e HOME=/tmp -e PYTHONIOENCODING=utf-8 -e PYTHONPATH=/workspace"
PIP="pip install --user -q scipy scikit-learn >/dev/null 2>&1"
IMG=segciani:dev
say() { echo "$(date '+%m-%d %H:%M') $*"; }
variant() {   # $1 name, rest: -e overrides
  local V=$1; shift
  local O=${D}_relax$V
  mkdir -p $O
  cp $D/result_tagged.pkl $D/cluster_names.json $O/ 2>/dev/null
  say "[$V] rescore"
  docker run --rm -u $(id -u):$(id -g) $ENV "$@" -e SQ_OUT=$O -e LANG_DIRS=/data/outputs_rank:/data/outputs_extra_rank \
    -e LANG_TAGS=":extra" -e VOC_DATA=/data/voc2012/VOCdevkit/VOC2012 -e GALLERY=0 $MOUNTS -w /workspace $IMG \
    bash -c "$PIP; python -u slice_quality/examples/rescore_with_language.py" > $O/rescore.log 2>&1
  grep -q "DONE rescore" $O/rescore.log || { say "[$V] rescore failed"; tail -4 $O/rescore.log; return 1; }
  grep -E "language slices total|images with SliceLens" $O/rescore.log
  say "[$V] images' IoU per slice"
  docker run --rm -u $(id -u):$(id -g) $ENV -e SQ_OUT=$O $MOUNTS -w /workspace $IMG bash -c \
    "$PIP; python -u slice_quality/examples/slice_image_iou.py" > $O/slice_image_iou.log 2>&1
  say "[$V] split-half under the same gates"
  docker run --rm -u $(id -u):$(id -g) $ENV "$@" -e SQ_OUT=$O $MOUNTS -w /workspace $IMG bash -c \
    "$PIP; python -u slice_quality/examples/split_half.py" > $O/split_half.log 2>&1
  grep "OVERALL" $O/split_half.log | head -2
  say "[$V] done"; }
say "== relaxed cluster gates on copies of the VOC run =="
( variant A -e SQ_PURITY_ALPHA=0.10 -e SQ_COH_MIN=0.5 ) &
PA=$!
( variant B -e SQ_PURITY_OFF=1 ) &
PB=$!
wait $PA $PB
docker run --rm -u $(id -u):$(id -g) $ENV -e RUNS="official=$D A=${D}_relaxA B=${D}_relaxB" -e OUT=outputs/relaxed_cluster_report \
  $MOUNTS -w /workspace $IMG bash -c "$PIP; python -u slice_quality/examples/relaxed_report.py" > outputs/relaxed_report.log 2>&1
cat outputs/relaxed_cluster_report.txt
say "CHAIN_RELAXED_DONE"
