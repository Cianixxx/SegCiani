#!/bin/bash
# Regenerate the galleries with the short cluster names (slice_quality/naming.py, gallery.py):
# the thesis run (inst), val-EA and SBD, each with the settings it was built with, plus the share
# copies. CPU only; the pickles and the texts are not changed, only the pages.
#   setsid nohup bash outputs/chain_gallery_names.sh > outputs/gallery_names.log 2>&1 < /dev/null &
set -u
cd ~/storage/SLICELENS
MOUNTS="-v $HOME/storage/SLICELENS:/workspace -v $HOME/storage/pascal_vlm:/data -v $DATASETS:/ds"
ENV="-e HOME=/tmp -e PYTHONIOENCODING=utf-8 -e PYTHONPATH=/workspace"
PIP="pip install --user -q scipy scikit-learn >/dev/null 2>&1"
step() { echo "$(date '+%m-%d %H:%M') == $1 =="; }
regen() {     # $1 run dir, then extra docker args
  local L=$1; shift
  docker run --rm -u $(id -u):$(id -g) $ENV -e SQ_OUT=$L "$@" $MOUNTS -w /workspace segciani:dev \
    bash -c "$PIP; python -u slice_quality/examples/regen_gallery.py" > $L/regen_gallery_names.log 2>&1
  grep -q "gallery regenerated" $L/regen_gallery_names.log || { echo "FAIL gallery $L"; tail -5 $L/regen_gallery_names.log; return 1; }
  tail -2 $L/regen_gallery_names.log
}
share() {     # $1 run dir, $2 src, $3 dst
  docker run --rm -u $(id -u):$(id -g) $ENV -e SQ_OUT=$1 -e GALLERY_SRC=$2 -e GALLERY_DST=$3 $MOUNTS -w /workspace \
    segciani:dev python -u docs/make_share_gallery.py | tail -2
}

step "1. thesis run (inst)"
regen outputs/slice_quality_inst && share outputs/slice_quality_inst gallery_full_tagged.html gallery_share.html

step "2. val-EA"
L=outputs/slice_quality_ea
regen $L -e SQ_CLAIM_COV=0.5 -e SQ_GT_INSTANCES=object -e VOC_DATA=/data/voc2012_ea/VOCdevkit/VOC2012 \
  -e ORIG_DATA=/data/voc2012/VOCdevkit/VOC2012 -e IMAGE_NOTES=$L/gallery_notes.json -e MAX_IMAGES=244 \
  -e GALLERY_OUT=$L/gallery_val_ea_full.html \
  -e "GALLERY_TITLE=VOC2012 val-EA: 691 immagini con il materiale del soggetto modificato (Pascal-EA)" \
  && share $L gallery_val_ea_full.html gallery_val_ea_share.html

step "3. SBD"
L=outputs/slice_quality_sbd
regen $L -e VOC_DATA=/data/sbd/VOCdevkit/VOC2012 -e GALLERY_OUT=$L/gallery_sbd_full.html -e MAX_IMAGES=150 \
  -e "GALLERY_TITLE=SBD: 9118 immagini non annotate da VOC 2012" \
  && share $L gallery_sbd_full.html gallery_sbd_share.html
grep -c 'class="sname"' outputs/slice_quality_inst/gallery_share.html outputs/slice_quality_ea/gallery_val_ea_share.html \
  outputs/slice_quality_sbd/gallery_sbd_share.html
echo "$(date '+%m-%d %H:%M') GALLERY_NAMES_DONE"
