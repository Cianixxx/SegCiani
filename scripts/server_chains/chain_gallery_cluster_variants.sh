#!/bin/bash
# The VOC gallery with the three variants of the cluster gates (Daniele, 2026-10-23: "aggiungi le 3 varianti di
# cluster all'html di voc, specificando per ogni immagine a quale slice cluster appartiene e a quale variante").
# The official run's gallery, rebuilt with a note on every image in a certified cluster group of any variant
# (ufficiale / A / B, see chain_relaxed_cluster.sh): the 549 such images are all shown. CPU only; the official
# pages and files are not touched (the texts regen_gallery.py rebuilds go to their own directory).
#   setsid nohup bash outputs/chain_gallery_cluster_variants.sh > outputs/gallery_cluster_variants.log 2>&1 < /dev/null &
set -u
cd ~/storage/SLICELENS
D=outputs/slice_quality_inst
V=${D}_clustervariants
mkdir -p $V
MOUNTS="-v $HOME/storage/SLICELENS:/workspace -v $HOME/storage/pascal_vlm:/data -v $DATASETS:/ds"
ENV="-e HOME=/tmp -e PYTHONIOENCODING=utf-8 -e PYTHONPATH=/workspace -e SQ_CLAIM_COV=0.5 -e SQ_GT_INSTANCES=object"
PIP="pip install --user -q scipy scikit-learn >/dev/null 2>&1"
IMG=segciani:dev
say() { echo "$(date '+%m-%d %H:%M') $*"; }

say "notes: which cluster groups each image falls in, under which variant"
docker run --rm -u $(id -u):$(id -g) $ENV $MOUNTS -w /workspace $IMG bash -c \
  "$PIP; python -u slice_quality/examples/cluster_variants_notes.py" > $V/notes.log 2>&1
grep -q CLUSTER_VARIANTS_NOTES_DONE $V/notes.log || { say "notes failed"; tail -5 $V/notes.log; exit 1; }
head -n 1 $V/notes.log
N=$(python3 -c "import json; print(len(json.load(open('$V/cluster_variants_notes.json'))['order']))")

say "gallery: the official run, $N images in the by-image view"
docker run --rm -u $(id -u):$(id -g) $ENV -e SQ_OUT=$V -e PKL=$D/result_tagged.pkl -e NARR=$D/error_narratives.json \
  -e CAUSE_VAL=$D/judge_cause_val.json -e CAUSE_ANN=$D/judge_cause.json -e CLAIMS=$D/judge_claims.json \
  -e CLAIMS_MANUAL=$D/judge_claims_manual.json -e IMAGE_NOTES=$V/cluster_variants_notes.json -e MAX_IMAGES=$N \
  -e GALLERY_TITLE="VOC2012 val · varianti dei gruppi cluster" -e GALLERY_OUT=$V/gallery_voc_varianti_cluster.html \
  $MOUNTS -w /workspace $IMG bash -c "$PIP; python -u slice_quality/examples/regen_gallery.py" > $V/regen_gallery.log 2>&1
grep -q "gallery regenerated" $V/regen_gallery.log || { say "gallery failed"; tail -5 $V/regen_gallery.log; exit 1; }
cmp -s $V/image_descriptions.json $D/image_descriptions.json && say "descriptions identical to the official run's" \
  || say "WARNING: rebuilt descriptions differ from the official run's"

say "share copy (pictures re-encoded smaller)"
docker run --rm -u $(id -u):$(id -g) $ENV -e SQ_OUT=$V -e GALLERY_SRC=gallery_voc_varianti_cluster.html \
  -e GALLERY_DST=gallery_voc_varianti_cluster_share.html $MOUNTS -w /workspace $IMG \
  python -u docs/make_share_gallery.py | tail -2
ls -la $V/*.html
say "CHAIN_GALLERY_CLUSTER_VARIANTS_DONE"
