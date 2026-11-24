#!/bin/bash
# Size-aware verdict for the slope-trend test (slice_quality/examples/slope_permutation.py), CPU only:
# p-values on the saved groundings, copies of the grounding files with the new verdict
# (<dir>_perm), rescores of COPIES of the inst and SBD pickles, comparison. The thesis runs and
# the original grounding directories are not touched.
#   setsid nohup bash outputs/chain_slope_perm.sh > outputs/slope_perm.log 2>&1 < /dev/null &
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

step "1. p-values with a null at the population's size, checks, perm copies of the groundings"
docker run --rm -u $(id -u):$(id -g) $ENV -e PHASE=test $MOUNTS -w /workspace segciani:dev \
  python -u slice_quality/examples/slope_permutation.py > outputs/slope_perm_test.log 2>&1
grep -q SLOPE_PERM_TEST_DONE outputs/slope_perm_test.log || { echo "FAIL test"; tail -20 outputs/slope_perm_test.log; exit 1; }
cat outputs/slope_permutation/report.txt

step "2. rescore a copy of the inst run with the new verdict (CPU)"
rescore outputs/slice_quality_inst_perm outputs/slice_quality_inst/result_tagged.pkl \
  /data/outputs_perm:/data/outputs_extra_perm ":extra" /data/voc2012/VOCdevkit/VOC2012

step "3. rescore a copy of the SBD run with the new verdict (CPU)"
rescore outputs/slice_quality_sbd_perm outputs/slice_quality_sbd/result_tagged.pkl \
  /data/outputs_sbd_perm "" /data/sbd/VOCdevkit/VOC2012

step "4. compare with the thesis runs"
docker run --rm -u $(id -u):$(id -g) $ENV -e PHASE=compare $MOUNTS -w /workspace segciani:dev \
  python -u slice_quality/examples/slope_permutation.py > outputs/slope_perm_compare.log 2>&1
grep -q SLOPE_PERM_COMPARE_DONE outputs/slope_perm_compare.log || { echo "FAIL compare"; tail -20 outputs/slope_perm_compare.log; exit 1; }
tail -40 outputs/slope_permutation/report.txt
echo "$(date '+%m-%d %H:%M') CHAIN_SLOPE_PERM_DONE"
