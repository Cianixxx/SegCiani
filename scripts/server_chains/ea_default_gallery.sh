#!/bin/bash
# the run's default gallery (as rescore writes it: no edit notes, no original panels),
# under its usual name; the val-EA gallery lives in gallery_val_ea_full.html
set -u
cd ~/storage/SLICELENS
L=outputs/slice_quality_ea
mv $L/gallery_full_tagged.html $L/gallery_val_ea_full.html
mv $L/gallery_share.html $L/gallery_val_ea_share.html
MOUNTS="-v $HOME/storage/SLICELENS:/workspace -v $HOME/storage/pascal_vlm:/data -v $DATASETS:/ds"
docker run --rm --name ea_gallery_default -u $(id -u):$(id -g) -e HOME=/tmp -e PYTHONIOENCODING=utf-8 -e PYTHONPATH=/workspace \
  -e SQ_CLAIM_COV=0.5 -e SQ_GT_INSTANCES=object -e SQ_OUT=$L -e VOC_DATA=/data/voc2012_ea/VOCdevkit/VOC2012 \
  $MOUNTS -w /workspace segciani:dev bash -c "pip install --user -q scipy scikit-learn >/dev/null 2>&1; \
  python -u slice_quality/examples/regen_gallery.py" > $L/regen_gallery_default.log 2>&1
tail -3 $L/regen_gallery_default.log
ls -la $L/*.html
echo EA_DEFAULT_GALLERY_DONE
