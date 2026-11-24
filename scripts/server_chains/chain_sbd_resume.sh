#!/bin/bash
# SBD run resumed on GPU 5 + 6 after GPU 5 was stopped by mistake on 10-11 18:10 (Daniele:
# keep GPU 5). Same steps as chain_sbd.sh; GPU6's worker starts once the tag container
# still running there (first pass) has finished. Every step skips the work on disk.
#   setsid nohup bash outputs/chain_sbd_resume.sh > outputs/slice_quality_sbd/chain_resume.log 2>&1 &
set -u
cd ~/storage/SLICELENS
L=outputs/slice_quality_sbd
O=$HOME/storage/pascal_vlm/outputs_sbd
MOUNTS="-v $HOME/storage/SLICELENS:/workspace -v $HOME/storage/pascal_vlm:/data -v $DATASETS:/ds"
ENV="-e HOME=/tmp -e PYTHONIOENCODING=utf-8 -e PYTHONPATH=/workspace"
SQ="-e SQ_CLAIM_COV=0.5 -e SQ_GT_INSTANCES=object"
PIP="pip install --user -q scipy scikit-learn >/dev/null 2>&1"
SBD=/data/sbd/VOCdevkit/VOC2012
CLASSES="aeroplane bicycle bird boat bottle bus car cat chair cow diningtable dog horse motorbike person pottedplant sheep sofa train tvmonitor"
step() { echo "$(date '+%m-%d %H:%M') == $1 =="; }
cpu() { docker run --rm -u $(id -u):$(id -g) $ENV "$@"; }
all_done() {      # $1 = _cc or _full
  for c in $CLASSES; do
    [ -f $O/voc_paper_$c$1.json ] || [ -f $O/.none$1_$c ] || return 1
  done
}
rescore() {
  cpu -e SQ_OUT=$L -e LANG_DIRS=/data/outputs_sbd -e VOC_DATA=$SBD $MOUNTS -w /workspace segciani:dev \
    bash -c "$PIP; python -u slice_quality/examples/rescore_with_language.py" > $L/rescore_$1.log 2>&1
  grep -q "DONE rescore" $L/rescore_$1.log || { echo "FAIL rescore $1"; tail -5 $L/rescore_$1.log; exit 1; }
  grep -E "language slices total|images with SliceLens|corroborated" $L/rescore_$1.log
}

step "1. workers: GPU5 now, GPU6 once its running tag container is done"
N=$(grep -m1 -o "tag queue: [0-9]*" $L/tag_gpu6.log | grep -o "[0-9]*$"); N=${N:-113}
cp $L/tag_gpu5.log $L/tag_gpu5_pass1.log
bash outputs/chain_sbd_worker.sh 5 > $L/worker_gpu5b.log 2>&1 &
W5=$!
( while docker ps --format '{{.Names}}' | grep -qx sbd_tag_gpu6; do sleep 60; done
  cp $L/tag_gpu6.log $L/tag_gpu6_pass1.log
  bash outputs/chain_sbd_worker.sh 6 ) > $L/worker_gpu6b.log 2>&1 &
W6=$!
until [ "$(ls $L/tags | grep -c '^chunk_.*json$')" -ge "$N" ]; do sleep 120; done
step "   all $N tag chunks on disk"

step "2. tags onto the instances (CPU)"
cpu -e STEP=merge $MOUNTS -w /workspace segciani:dev bash -c "$PIP; python -u slice_quality/examples/build_sbd.py" \
  > $L/merge.log 2>&1
grep -q "BUILD_SBD_DONE merge" $L/merge.log || { echo "FAIL merge"; tail -5 $L/merge.log; exit 1; }
tail -2 $L/merge.log

step "3. waiting for the sample phase of every class"
until all_done _cc; do sleep 120; done

step "4. rescore #1 (CPU): which hypotheses are validated"
cp $L/result_tagged.pkl $L/result_tagged_merged.pkl
rescore 1
touch $L/.phase_full_ready

step "5. full-population re-grounding of the validated hypotheses (workers)"
until all_done _full; do sleep 120; done

step "6. rescore #2 (CPU): final SBD run"
cp $L/result_tagged_merged.pkl $L/result_tagged.pkl
rescore 2
wait $W5 $W6

step "7. per-image narratives (CPU)"
cpu $SQ -e PKL=$L/result_tagged.pkl -e TRAIN36=/nonexistent -e OUT=$L/error_narratives -e VOC_DATA=$SBD \
  -e CROPS=/workspace/voc_crops_sbd $MOUNTS -w /workspace segciani:dev \
  bash -c "$PIP; python -u slice_quality/examples/narrate_errors.py" > $L/narrate.log 2>&1
grep -q NARRATE_DONE $L/narrate.log || { echo "FAIL narrate"; tail -5 $L/narrate.log; exit 1; }

step "8. gallery under its own name, and the shareable copy (CPU)"
cpu -e SQ_OUT=$L -e VOC_DATA=$SBD -e GALLERY_OUT=$L/gallery_sbd_full.html -e MAX_IMAGES=150 \
  -e GALLERY_TITLE="SBD: 9118 immagini non annotate da VOC 2012" \
  $MOUNTS -w /workspace segciani:dev bash -c "$PIP; python -u slice_quality/examples/regen_gallery.py" \
  > $L/regen_gallery.log 2>&1
grep -q "gallery regenerated" $L/regen_gallery.log || { echo "FAIL gallery"; tail -5 $L/regen_gallery.log; exit 1; }
cpu -e SQ_OUT=$L -e GALLERY_SRC=gallery_sbd_full.html -e GALLERY_DST=gallery_sbd_share.html $MOUNTS -w /workspace \
  segciani:dev python -u docs/make_share_gallery.py || echo "share copy failed (gallery_sbd_full.html is there)"
ls -la $L/gallery_sbd_*
echo "$(date '+%m-%d %H:%M') CHAIN_SBD_DONE"
