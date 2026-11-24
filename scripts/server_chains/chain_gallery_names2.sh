#!/bin/bash
# After chain_gallery_names.sh: the default-named galleries of the val-EA and SBD runs
# (gallery_full_tagged.html), regenerated with the cluster names too. CPU only.
set -u
cd ~/storage/SLICELENS
until grep -q GALLERY_NAMES_DONE outputs/gallery_names.log 2>/dev/null; do sleep 30; done
MOUNTS="-v $HOME/storage/SLICELENS:/workspace -v $HOME/storage/pascal_vlm:/data -v $DATASETS:/ds"
ENV="-e HOME=/tmp -e PYTHONIOENCODING=utf-8 -e PYTHONPATH=/workspace"
PIP="pip install --user -q scipy scikit-learn >/dev/null 2>&1"
L=outputs/slice_quality_ea
docker run --rm -u $(id -u):$(id -g) $ENV -e SQ_OUT=$L -e SQ_CLAIM_COV=0.5 -e SQ_GT_INSTANCES=object \
  -e VOC_DATA=/data/voc2012_ea/VOCdevkit/VOC2012 \
  -e "GALLERY_TITLE=VOC2012 val-EA: 691 immagini con il materiale del soggetto modificato (Pascal-EA)" \
  $MOUNTS -w /workspace segciani:dev \
  bash -c "$PIP; python -u slice_quality/examples/regen_gallery.py" > $L/regen_gallery_default_names.log 2>&1
tail -1 $L/regen_gallery_default_names.log
L=outputs/slice_quality_sbd
docker run --rm -u $(id -u):$(id -g) $ENV -e SQ_OUT=$L -e VOC_DATA=/data/sbd/VOCdevkit/VOC2012 \
  -e "GALLERY_TITLE=SBD: 9118 immagini non annotate da VOC 2012" $MOUNTS -w /workspace segciani:dev \
  bash -c "$PIP; python -u slice_quality/examples/regen_gallery.py" > $L/regen_gallery_default_names.log 2>&1
tail -1 $L/regen_gallery_default_names.log
grep -c 'class="sname"' outputs/slice_quality_ea/gallery_full_tagged.html outputs/slice_quality_sbd/gallery_full_tagged.html
echo "$(date '+%m-%d %H:%M') GALLERY_NAMES2_DONE"
