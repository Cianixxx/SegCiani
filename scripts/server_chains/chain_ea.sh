#!/bin/bash
# val-EA: the whole pipeline (inst configuration) on VOC val with the 690 Pascal-EA
# material edits in place of their originals -- the planted-cause test.
#   GPU6 now: cluster-branch embeddings + tags, then language-branch worker;
#   GPU1 from START1 (UTC, "HH:MM"): a second language-branch worker;
#   CPU: rescore after the sample phase (selects the validated hypotheses), and the
#   final rescore after the full-population re-grounding.
#   START1=21:50 nohup bash outputs/chain_ea.sh > outputs/slice_quality_ea/chain.log 2>&1 &
set -u
cd ~/storage/SLICELENS
L=outputs/slice_quality_ea
mkdir -p $L
O=$HOME/storage/pascal_vlm/outputs_ea
MOUNTS="-v $HOME/storage/SLICELENS:/workspace -v $HOME/storage/pascal_vlm:/data \
  -v $DATASETS:/ds -v $STORAGE/models/qwen2.5-vl-7b:/models/qwen"
ENV="-e HOME=/tmp -e PYTHONIOENCODING=utf-8 -e PYTHONPATH=/workspace"
CLASSES="aeroplane bicycle bird boat bottle bus car cat chair cow diningtable dog horse motorbike person pottedplant sheep sofa train tvmonitor"
step() { echo "$(date '+%m-%d %H:%M') == $1 =="; }
rescore() {
  docker run --rm -u $(id -u):$(id -g) $ENV -e SQ_OUT=$L -e LANG_DIRS=/data/outputs_ea \
    -e VOC_DATA=/data/voc2012_ea/VOCdevkit/VOC2012 $MOUNTS -w /workspace segciani:dev \
    bash -c "pip install --user -q scipy scikit-learn >/dev/null 2>&1; \
    python -u slice_quality/examples/rescore_with_language.py" > $L/rescore_$1.log 2>&1
  grep -q "DONE rescore" $L/rescore_$1.log || { echo "FAIL rescore $1"; tail -5 $L/rescore_$1.log; exit 1; }
  grep -E "language slices total|images with SliceLens|corroborated" $L/rescore_$1.log
}
all_done() {      # $1 = _cc or _full
  for c in $CLASSES; do
    [ -f $O/voc_paper_$c$1.json ] || [ -f $O/.none$1_$c ] || return 1
  done
}

step "GPU1 worker scheduled from ${START1:-never} UTC"
if [ -n "${START1:-}" ]; then
  ( until [ "$(date -u +%H:%M)" = "$START1" ]; do sleep 30; done
    until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)" -lt 1000 ]; do
      echo "$(date '+%H:%M') GPU1 busy, waiting"; sleep 300; done
    bash outputs/chain_ea_worker.sh 1 > $L/worker_gpu1.log 2>&1 ) &
fi

step "1. val-EA layout + language substrate (CPU)"
until grep -q BUILD_VOC_EA_DONE $HOME/storage/pascal_vlm/voc2012_ea/build.log 2>/dev/null; do sleep 60; done
grep -E "layout|substrate|untouched|edited images" $HOME/storage/pascal_vlm/voc2012_ea/build.log

step "2. cluster branch: BLOCK 1 (inst config) + DINOv2 + Qwen tags on the edited images (GPU6)"
docker run --rm --name ea_retype --gpus device=6 -u $(id -u):$(id -g) $ENV -e SQ_CLAIM_COV=0.5 -e SQ_GT_INSTANCES=object \
  $MOUNTS -w /workspace segciani:dev bash -c "pip install --user -q scipy scikit-learn >/dev/null 2>&1; \
  python -u slice_quality/examples/retype_ea.py" > $L/retype.log 2>&1
grep -q RETYPE_EA_DONE $L/retype.log || { echo "FAIL retype"; tail -5 $L/retype.log; exit 1; }
grep -E "reused|DINOv2|Qwen|PICKLE" $L/retype.log

step "3. language branch, sample phase (GPU6 worker; GPU1 joins at START1)"
bash outputs/chain_ea_worker.sh 6 > $L/worker_gpu6.log 2>&1 &
W6=$!
until all_done _cc; do sleep 120; done
step "   all classes grounded on the sample"

step "4. rescore #1 (CPU): which hypotheses are validated"
cp $L/result_tagged.pkl $L/result_tagged_retyped.pkl
rescore 1
touch $L/.phase_full_ready

step "5. full-population re-grounding (workers pick it up)"
until all_done _full; do sleep 120; done

step "6. rescore #2 (CPU): final val-EA run"
cp $L/result_tagged_retyped.pkl $L/result_tagged.pkl
rescore 2
wait $W6
echo "$(date '+%m-%d %H:%M') CHAIN_EA_DONE"
