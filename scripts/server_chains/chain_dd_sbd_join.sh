#!/bin/bash
# Adds a worker on GPU <gpu> to chain_dd_sbd.sh once nobody else uses that GPU (Daniele, 2026-10-19:
# "anche la gpu2 puoi usare" while user gu held it). It waits; when the GPU is free it runs the same
# queued steps as the chain's workers (generation if classes are left, then the 120-instance sample),
# every container under outputs/gpu_guard.sh. It gives up when the sample phase is over first.
#   setsid nohup bash outputs/chain_dd_sbd_join.sh <gpu> > outputs/slice_quality_sbd/dd_worker_join_gpu<gpu>.log 2>&1 < /dev/null &
set -u
GPU=$1
cd ~/storage/SLICELENS
L=outputs/slice_quality_sbd
P=$HOME/storage/pascal_vlm
CLASSES="bicycle bird boat bottle bus car cat diningtable dog horse motorbike pottedplant sheep sofa train tvmonitor"
MOUNTS="-v $HOME/storage/SLICELENS:/workspace -v $P:/data -v $DATASETS:/ds"
MODELS="-v $STORAGE/models/qwen2.5-vl-7b:/models/qwen -v $STORAGE/models/gemma-4-12B-it:/models/gemma"
ENV="-e HOME=/tmp -e PYTHONIOENCODING=utf-8 -e PYTHONPATH=/workspace -e SQ_QWEN_DEVICE_MAP=cuda:0"
PIP="pip install --user -q scipy scikit-learn >/dev/null 2>&1"
ME=$(id -u)
UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU")
gen_done() { for c in $CLASSES; do [ -f $P/outputs_dd_sbd/dd_$c.txt ] || return 1; done; }
cc_done() { for c in $CLASSES; do [ -f $P/outputs_dd_sbd_ground/voc_paper_${c}_cc.json ] || [ -d $P/outputs_dd_sbd_ground/.claim_cc_$c ] || return 1; done; }
others() {
  for pid in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v u="$UUID" '$1==u {print $2}'); do
    uid=$(ps -o uid= -p "$pid" 2>/dev/null | tr -d ' ')
    [ -n "$uid" ] && [ "$uid" != "$ME" ] && echo "$pid"
  done
}
run() {
  local NAME=$1 LOG=$2
  shift 2
  if [ -n "$(others)" ]; then echo "GPU $GPU used by another user before $NAME: worker ends"; exit 0; fi
  setsid nohup bash outputs/gpu_guard.sh $GPU $NAME > $L/guard_$NAME.log 2>&1 < /dev/null &
  CLASSES="$CLASSES" docker run --rm --name $NAME --gpus device=$GPU -u $(id -u):$(id -g) $ENV -e CLASSES $MOUNTS $MODELS \
    -w /workspace "$@" > $LOG 2>&1
  sleep 3
  if grep -q GUARD_STOPPED $L/guard_$NAME.log; then echo "guard stopped $NAME: worker ends, nothing restarted"; exit 0; fi
  tail -n 1 $LOG
}

echo "$(date '+%m-%d %H:%M') [gpu$GPU] waiting until no other user is on GPU $GPU"
until [ -z "$(others)" ]; do
  cc_done && { echo "$(date '+%m-%d %H:%M') sample phase already claimed before GPU $GPU was free: nothing to do"; exit 0; }
  sleep 60
done
echo "$(date '+%m-%d %H:%M') [gpu$GPU] GPU free: joining"
if ! gen_done; then
  run dd_sbd_gen_gpu$GPU $L/dd_gen_gpu$GPU.log -e OUT_DIR=/data/outputs_dd_sbd -e SUBSTRATE=/data/sbd/substrate_cc.pkl \
    segciani:dev bash -c "$PIP; python -u examples/gen_datadriven.py"
fi
for c in $CLASSES; do until [ -f $P/outputs_dd_sbd/dd_$c.txt ]; do sleep 60; done; done
cc_done && { echo "sample phase already claimed: nothing to do"; exit 0; }
run dd_sbd_ground_gpu$GPU $L/dd_ground_gpu$GPU.log -e OUT_DIR=/data/outputs_dd_sbd_ground \
  -e QUERIES_TEMPLATE=/data/outputs_dd_sbd/dd_{cls}.txt -e SUBSTRATE=/data/sbd/substrate_cc.pkl -e OLD_SUBSTRATE= \
  -e EDITED= -e CACHE_DIRS= -e SAMPLING=srs -e DATASET=SBD -e PHASE=sample \
  segciani:dev bash -c "$PIP; python -u examples/ground_ea_hypotheses.py"
echo "$(date '+%m-%d %H:%M') [gpu$GPU] worker done"
