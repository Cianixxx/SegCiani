#!/bin/bash
# GPU 6 joins the queue of chain_night_pending.sh (Daniele, 2026-10-22: "ho a disposizione gpu 2 e 6 ora"): the
# same shared class claims, under the guard, waiting while another user holds the GPU. When every class is
# done (the other GPUs may still be on their last one) the preview of the rank verdict is written again.
#   setsid nohup bash outputs/pend_gpu6.sh > outputs/pend_gpu6.log 2>&1 < /dev/null &
set -u
cd ~/storage/SLICELENS
PD=outputs/slice_quality_pending
L=$PD/logs
P=$HOME/storage/pascal_vlm
RG=$P/outputs_pending
G=6
MOUNTS="-v $HOME/storage/SLICELENS:/workspace -v $P:/data -v $DATASETS:/ds"
MODELS="-v $STORAGE/models/qwen2.5-vl-7b:/models/qwen"
ENV="-e HOME=/tmp -e PYTHONIOENCODING=utf-8 -e PYTHONPATH=/workspace -e SQ_CLAIM_COV=0.5 -e SQ_GT_INSTANCES=object \
  -e SQ_QWEN_DEVICE_MAP=cuda:0 -e SQ_OUT=outputs/slice_quality_inst"
PIP="pip install --user -q scipy scikit-learn >/dev/null 2>&1"
IMG=segciani:dev
CLASSES=$(cat $PD/classes.txt)
say() { echo "$(date '+%m-%d %H:%M') $*"; }
others() {
  local uuid me pid uid
  uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $1); me=$(id -u)
  for pid in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v u="$uuid" '$1==u {print $2}'); do
    uid=$(ps -o uid= -p "$pid" 2>/dev/null | tr -d ' '); [ -n "$uid" ] && [ "$uid" != "$me" ] && echo $pid
  done; }
for c in $CLASSES; do
  [ -f $RG/.done_$c ] && continue
  mkdir $RG/.claim_$c 2>/dev/null || continue
  while true; do
    if [ -n "$(others $G)" ]; then say "[gpu$G] busy with another user: waiting"; sleep 300; continue; fi
    name=pend_${c}_gpu$G
    say "[gpu$G] grounding the pending hypotheses of $c on every object"
    setsid nohup bash outputs/gpu_guard.sh $G $name > $L/guard_$name.log 2>&1 < /dev/null &
    docker run --rm --name $name --gpus device=$G -u $(id -u):$(id -g) $ENV $MOUNTS $MODELS -w /workspace $IMG bash -c \
      "$PIP; python -u examples/reground_validated.py --slices-index $PD/pending_index.json \
       --substrate-cache /data/voc2012/substrate_cc.pkl --vlm-backend qwen --vlm-model /models/qwen \
       --out-dir /data/outputs_pending --classes $c" > $L/reground_$c.log 2>&1
    sleep 3
    if grep -q GUARD_STOPPED $L/guard_$name.log; then say "[gpu$G] guard stopped $c: will retry"; sleep 300; continue; fi
    grep -q "Traceback" $L/reground_$c.log && { say "[gpu$G] $c failed"; tail -3 $L/reground_$c.log; rmdir $RG/.claim_$c; break 2; }
    touch $RG/.done_$c; break
  done
done
say "[gpu$G] queue done"
for i in $(seq 1 480); do                     # wait for the other GPUs' last classes
  ALL=1; for c in $CLASSES; do [ -f $RG/.done_$c ] || ALL=0; done
  [ $ALL = 1 ] && break
  sleep 30
done
say "[cpu] preview again: the rank verdict on the pending groundings (all classes done: $ALL)"
docker run --rm -u $(id -u):$(id -g) $ENV $MOUNTS -w /workspace $IMG bash -c \
  "$PIP; RUNS_DIR=/data/outputs_pending STAGE=full OUT=$PD/dd_auc_full python -u slice_quality/examples/dd_auc.py" \
  > $L/pending_auc.log 2>&1
head -1 $L/pending_auc.log
say "PEND_GPU6_DONE"
