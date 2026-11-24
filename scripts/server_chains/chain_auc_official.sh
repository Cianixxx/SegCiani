#!/bin/bash
# The rank test (AUC of the VLM confidence, image-block bootstrap, BH) becomes the certification of the
# language branch in the thesis runs (Daniele, 2026-10-21: "fai 7", option "copia già pronta"). The inst and
# SBD runs are rescored with the AUC verdict on the saved groundings (/data/outputs_rank,
# /data/outputs_extra_rank, /data/outputs_sbd_rank, written by chain_slope_rank.sh), after a full copy of
# each run as it is now (*_slope, the slope rule). The certified slices must come out identical to the rank
# copies of chain_slope_rank.sh. Then the inst gallery is regenerated. CPU only.
#   setsid nohup bash outputs/chain_auc_official.sh > outputs/auc_official.log 2>&1 < /dev/null &
set -u
cd ~/storage/SLICELENS
MOUNTS="-v $HOME/storage/SLICELENS:/workspace -v $HOME/storage/pascal_vlm:/data"
ENV="-e HOME=/tmp -e PYTHONIOENCODING=utf-8 -e PYTHONPATH=/workspace"
PIP="pip install --user -q scipy scikit-learn >/dev/null 2>&1"
step() { echo "$(date '+%m-%d %H:%M') == $1 =="; }
rescore() {   # $1 run dir, $2 LANG_DIRS, $3 LANG_TAGS, $4 VOC_DATA
  docker run --rm -u $(id -u):$(id -g) $ENV -e SQ_OUT=$1 -e LANG_DIRS=$2 -e LANG_TAGS=$3 -e VOC_DATA=$4 -e GALLERY=0 $MOUNTS \
    -w /workspace segciani:dev bash -c "$PIP; python -u slice_quality/examples/rescore_with_language.py" \
    > $1/rescore_auc.log 2>&1
  grep -q "DONE rescore" $1/rescore_auc.log || { echo "FAIL rescore $1"; tail -5 $1/rescore_auc.log; exit 1; }
  grep -E "language slices total|images with SliceLens" $1/rescore_auc.log
}
same() {      # $1 run, $2 rank copy: the certified slices must be the same
  python3 -I - $1/slices_index.json $2/slices_index.json <<'PY'
import json, sys
a, b = ({s["slice_id"] for s in json.load(open(f))["slices"] if s.get("in_descriptions")} for f in sys.argv[1:3])
print("  certified slices:", len(a), "| identical to the rank copy" if a == b else "| DIFFERENT: %s" % sorted(a ^ b))
sys.exit(0 if a == b else 1)
PY
}

step "1. full copies of the runs as they are now (slope rule)"
for R in outputs/slice_quality_inst outputs/slice_quality_sbd; do
  [ -d ${R}_slope ] || cp -a $R ${R}_slope
  du -sh ${R}_slope
done

step "2. inst rescored in place with the AUC verdict"
rescore outputs/slice_quality_inst /data/outputs_rank:/data/outputs_extra_rank ":extra" /data/voc2012/VOCdevkit/VOC2012
same outputs/slice_quality_inst outputs/slice_quality_inst_rank || { echo "CHAIN_AUC_OFFICIAL_FAIL"; exit 1; }

step "3. inst gallery (cause scores refreshed later, after the judges)"
docker run --rm -u $(id -u):$(id -g) $ENV -e SQ_OUT=outputs/slice_quality_inst $MOUNTS -w /workspace segciani:dev \
  bash -c "$PIP; python -u slice_quality/examples/regen_gallery.py" > outputs/slice_quality_inst/regen_gallery_auc.log 2>&1
grep -q "gallery regenerated" outputs/slice_quality_inst/regen_gallery_auc.log && tail -3 outputs/slice_quality_inst/regen_gallery_auc.log \
  || { echo "WARN gallery"; tail -5 outputs/slice_quality_inst/regen_gallery_auc.log; }

step "4. SBD rescored in place with the AUC verdict"
rescore outputs/slice_quality_sbd /data/outputs_sbd_rank "" /data/sbd/VOCdevkit/VOC2012
same outputs/slice_quality_sbd outputs/slice_quality_sbd_rank || { echo "CHAIN_AUC_OFFICIAL_FAIL"; exit 1; }
echo "$(date '+%m-%d %H:%M') CHAIN_AUC_OFFICIAL_DONE"
