#!/bin/bash
# Before the geometric join (3 November): do we know how the three slope runs were rescored? Each one is rescored
# again on a scratch copy of its pickle, by id as before, with the settings we believe it had; the certified slices
# and the image descriptions must come out identical to the run as it is. CPU only, writes only in outputs/geo_repro.
#   setsid nohup bash outputs/geo_repro_check.sh > outputs/geo_repro_check.log 2>&1 < /dev/null &
set -u
cd ~/storage/SLICELENS
PV=$HOME/storage/pascal_vlm
MOUNTS="-v $HOME/storage/SLICELENS:/workspace -v $PV:/data"
ENV0="-e HOME=/tmp -e PYTHONIOENCODING=utf-8 -e PYTHONPATH=/workspace"
ENVC="$ENV0 -e SQ_CLAIM_COV=0.5 -e SQ_GT_INSTANCES=object"
PIP="pip install --user -q scipy scikit-learn >/dev/null 2>&1"
IMG=segciani:dev
R=outputs/geo_repro
mkdir -p $R
say() { echo "$(date '+%m-%d %H:%M') $*"; }
repro() {   # $1 tag, $2 run, $3 env, $4 LANG_DIRS, $5 LANG_TAGS, $6 VOC_DATA
  local O=$R/$1
  mkdir -p $O; cp $2/result_tagged.pkl $O/
  [ -f $2/cluster_names.json ] && cp $2/cluster_names.json $O/
  docker run --rm -u $(id -u):$(id -g) $3 $MOUNTS -e SQ_OUT=$O -e LANG_DIRS=$4 -e LANG_TAGS=$5 -e VOC_DATA=$6 -e GALLERY=0 \
    -w /workspace $IMG bash -c "$PIP; python -u slice_quality/examples/rescore_with_language.py" > $O/rescore.log 2>&1
  grep -q "DONE rescore" $O/rescore.log || { say "$1: rescore failed"; tail -3 $O/rescore.log; return; }
  for f in slices_index.json image_descriptions.json; do
    cmp -s $2/$f $O/$f && say "$1: $f identical" || say "$1: $f DIFFERS"
  done
}
say "== reproduction of the slope runs, join by id =="
repro inst_slope outputs/slice_quality_inst_slope "$ENV0" /data/outputs:/data/outputs_extra ":extra" /data/voc2012/VOCdevkit/VOC2012 &
repro sbd_slope outputs/slice_quality_sbd_slope "$ENV0" /data/outputs_sbd "" /data/sbd/VOCdevkit/VOC2012 &
repro ea_slope outputs/slice_quality_ea "$ENVC" /data/outputs_ea "" /data/voc2012_ea/VOCdevkit/VOC2012 &
repro ea_slope0 outputs/slice_quality_ea "$ENV0" /data/outputs_ea "" /data/voc2012_ea/VOCdevkit/VOC2012 &
repro ea_auc outputs/slice_quality_ea_auc "$ENVC" /data/outputs_ea_rank_m10 "" /data/voc2012_ea/VOCdevkit/VOC2012 &
wait
say "GEO_REPRO_DONE"
