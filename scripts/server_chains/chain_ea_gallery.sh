#!/bin/bash
# val-EA gallery (CPU only, no GPU): per-image narratives on the edited images, the
# notes on what each edit did, then the gallery with the edited images first.
#   nohup bash outputs/chain_ea_gallery.sh > outputs/slice_quality_ea/gallery_chain.log 2>&1 &
set -u
cd ~/storage/SLICELENS
L=outputs/slice_quality_ea
MOUNTS="-v $HOME/storage/SLICELENS:/workspace -v $HOME/storage/pascal_vlm:/data -v $DATASETS:/ds"
ENV="-e HOME=/tmp -e PYTHONIOENCODING=utf-8 -e PYTHONPATH=/workspace -e SQ_CLAIM_COV=0.5 -e SQ_GT_INSTANCES=object"
EA=/data/voc2012_ea/VOCdevkit/VOC2012
step() { echo "$(date '+%m-%d %H:%M') == $1 =="; }
[ -f $L/result_tagged_prenarr.pkl ] || cp $L/result_tagged.pkl $L/result_tagged_prenarr.pkl
cp $L/planted_eval.txt $L/planted_eval_prenarr.txt

step "1/4 narrate_errors on val-EA (CPU)"
docker run --rm --name ea_narr -u $(id -u):$(id -g) $ENV -e PKL=$L/result_tagged.pkl -e TRAIN36=/nonexistent \
  -e OUT=$L/error_narratives -e VOC_DATA=$EA -e CROPS=/workspace/voc_crops_full_ea $MOUNTS -w /workspace \
  segciani:dev bash -c "pip install --user -q scipy scikit-learn >/dev/null 2>&1; \
  python -u slice_quality/examples/narrate_errors.py" > $L/narrate.log 2>&1
grep -q NARRATE_DONE $L/narrate.log || { echo "FAIL narrate"; tail -5 $L/narrate.log; exit 1; }
grep -E "^val:|mask facts|narratives:" $L/narrate.log

step "2/4 notes on the edits (CPU)"
docker run --rm -u $(id -u):$(id -g) $ENV $MOUNTS -w /workspace segciani:dev \
  python -u slice_quality/examples/ea_gallery_notes.py || { echo "FAIL notes"; exit 1; }

step "3/4 gallery (CPU)"
docker run --rm --name ea_gallery -u $(id -u):$(id -g) $ENV -e SQ_OUT=$L -e VOC_DATA=$EA \
  -e ORIG_DATA=/data/voc2012/VOCdevkit/VOC2012 -e IMAGE_NOTES=$L/gallery_notes.json -e MAX_IMAGES=150 \
  -e GALLERY_OUT=$L/gallery_val_ea_full.html -e GALLERY_TITLE="VOC2012 val-EA: 691 immagini con il materiale del soggetto modificato (Pascal-EA)" \
  $MOUNTS -w /workspace segciani:dev bash -c "pip install --user -q scipy scikit-learn >/dev/null 2>&1; \
  python -u slice_quality/examples/regen_gallery.py" > $L/regen_gallery.log 2>&1
grep -q "gallery regenerated" $L/regen_gallery.log || { echo "FAIL gallery"; tail -5 $L/regen_gallery.log; exit 1; }
tail -4 $L/regen_gallery.log

step "4/4 planted-cause numbers unchanged? (CPU)"
docker run --rm -u $(id -u):$(id -g) $ENV $MOUNTS -w /workspace segciani:dev bash -c \
  "pip install --user -q scipy scikit-learn >/dev/null 2>&1; python -u slice_quality/examples/ea_planted_eval.py" > $L/planted_eval_rerun.log 2>&1
diff $L/planted_eval_prenarr.txt $L/planted_eval.txt && echo "planted_eval: identical"
ls -la $L/gallery_val_ea_full.html
echo "$(date '+%m-%d %H:%M') CHAIN_EA_GALLERY_DONE"
