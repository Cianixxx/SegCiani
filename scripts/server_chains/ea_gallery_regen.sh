#!/bin/bash
# val-EA gallery, representative (CPU only): the 144 victims, then 100 edited images drawn
# at random among those the edit did not change; under its own name, plus the share copy.
set -u
cd ~/storage/SLICELENS
L=outputs/slice_quality_ea
MOUNTS="-v $HOME/storage/SLICELENS:/workspace -v $HOME/storage/pascal_vlm:/data -v $DATASETS:/ds"
ENV="-e HOME=/tmp -e PYTHONIOENCODING=utf-8 -e PYTHONPATH=/workspace -e SQ_CLAIM_COV=0.5 -e SQ_GT_INSTANCES=object"
docker run --rm -u $(id -u):$(id -g) $ENV -e N_UNCHANGED=100 $MOUNTS -w /workspace segciani:dev \
  python -u slice_quality/examples/ea_gallery_notes.py || { echo "FAIL notes"; exit 1; }
docker run --rm --name ea_gallery -u $(id -u):$(id -g) $ENV -e SQ_OUT=$L -e VOC_DATA=/data/voc2012_ea/VOCdevkit/VOC2012 \
  -e ORIG_DATA=/data/voc2012/VOCdevkit/VOC2012 -e IMAGE_NOTES=$L/gallery_notes.json -e MAX_IMAGES=244 \
  -e GALLERY_OUT=$L/gallery_val_ea_full.html \
  -e GALLERY_TITLE="VOC2012 val-EA: 691 immagini con il materiale del soggetto modificato (Pascal-EA)" \
  $MOUNTS -w /workspace segciani:dev bash -c "pip install --user -q scipy scikit-learn >/dev/null 2>&1; \
  python -u slice_quality/examples/regen_gallery.py" > $L/regen_gallery.log 2>&1
grep -q "gallery regenerated" $L/regen_gallery.log || { echo "FAIL gallery"; tail -5 $L/regen_gallery.log; exit 1; }
docker run --rm -u $(id -u):$(id -g) $ENV -e SQ_OUT=$L -e GALLERY_SRC=gallery_val_ea_full.html \
  -e GALLERY_DST=gallery_val_ea_share.html $MOUNTS -w /workspace segciani:dev \
  python -u docs/make_share_gallery.py || { echo "FAIL share"; exit 1; }
ls -la $L/gallery_val_ea_*
echo EA_GALLERY_REGEN_DONE
