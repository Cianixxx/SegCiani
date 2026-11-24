#!/bin/bash
# Second verdict (AUC of the VLM confidence, image-block bootstrap; slope_permutation.py PHASE=rank), CPU only:
# p-values on the saved groundings, copies of the grounding files with the new verdict
# (<dir>_perm), rescores of COPIES of the inst and SBD pickles, comparison. The thesis runs and
# the original grounding directories are not touched.
#   setsid nohup bash outputs/chain_slope_rank.sh > outputs/slope_rank.log 2>&1 < /dev/null &
set -u
cd ~/storage/SLICELENS
MOUNTS="-v $HOME/storage/SLICELENS:/workspace -v $HOME/storage/pascal_vlm:/data"
ENV="-e HOME=/tmp -e PYTHONIOENCODING=utf-8 -e PYTHONPATH=/workspace"
PIP="pip install --user -q scipy scikit-learn >/dev/null 2>&1"
step() { echo "$(date '+%m-%d %H:%M') == $1 =="; }
cpu() { docker run --rm -u $(id -u):$(id -g) $ENV $MOUNTS -w /workspace "$@" segciani:dev; }
rescore() {   # $1 out dir, $2 source pickle, $3 LANG_DIRS, $4 LANG_TAGS, $5 VOC_DATA
  mkdir -p $1; cp $2 $1/result_tagged.pkl
  docker run --rm -u $(id -u):$(id -g) $ENV -e SQ_OUT=$1 -e LANG_DIRS=$3 -e LANG_TAGS=$4 -e VOC_DATA=$5 $MOUNTS \
    -w /workspace segciani:dev bash -c "$PIP; python -u slice_quality/examples/rescore_with_language.py" > $1/rescore.log 2>&1
  grep -q "DONE rescore" $1/rescore.log || { echo "FAIL rescore $1"; tail -5 $1/rescore.log; exit 1; }
  grep -E "language slices total|images with SliceLens" $1/rescore.log
}

step "1. AUC verdict with an image-block bootstrap, checks, rank copies of the groundings"
docker run --rm -u $(id -u):$(id -g) $ENV -e PHASE=rank $MOUNTS -w /workspace segciani:dev \
  bash -c "$PIP; python -u slice_quality/examples/slope_permutation.py" > outputs/slope_rank_test.log 2>&1
grep -q SLOPE_RANK_DONE outputs/slope_rank_test.log || { echo "FAIL test"; tail -20 outputs/slope_rank_test.log; exit 1; }
cat outputs/slope_permutation/report_rank.txt

step "2. rescore a copy of the inst run with the new verdict (CPU)"
rescore outputs/slice_quality_inst_rank outputs/slice_quality_inst/result_tagged.pkl \
  /data/outputs_rank:/data/outputs_extra_rank ":extra" /data/voc2012/VOCdevkit/VOC2012

step "3. rescore a copy of the SBD run with the new verdict (CPU)"
rescore outputs/slice_quality_sbd_rank outputs/slice_quality_sbd/result_tagged.pkl \
  /data/outputs_sbd_rank "" /data/sbd/VOCdevkit/VOC2012

step "4. compare with the thesis runs"
docker run --rm -u $(id -u):$(id -g) $ENV -e PHASE=compare -e VERDICT=rank $MOUNTS -w /workspace segciani:dev \
  python -u slice_quality/examples/slope_permutation.py > outputs/slope_rank_compare.log 2>&1
grep -q SLOPE_PERM_COMPARE_DONE outputs/slope_rank_compare.log || { echo "FAIL compare"; tail -20 outputs/slope_rank_compare.log; exit 1; }
tail -40 outputs/slope_permutation/report_rank.txt
echo "$(date '+%m-%d %H:%M') CHAIN_SLOPE_RANK_DONE"
