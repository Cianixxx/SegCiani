#!/bin/bash
# GPU 1 rejoins the re-grounding of the slope-only situations on model 2 (chain_slicelens_cmp.sh), after its
# guard stopped it at 16:33 for another user's short job. Same shared class queue as GPU 2; under the guard.
# At the end, when no class is left, the disagreement report is written again with every class in.
#   setsid nohup bash outputs/cmp_rg_gpu1.sh > outputs/cmp_rg_gpu1.log 2>&1 < /dev/null &
set -u
cd ~/storage/SLICELENS
D=outputs/slice_quality_inst
C=outputs/slice_quality_slcmp
L=$C/logs
P=$HOME/storage/pascal_vlm
RG=$P/outputs_model2_slopeonly
MOUNTS="-v $HOME/storage/SLICELENS:/workspace -v $P:/data -v $DATASETS:/ds"
MODELS="-v $STORAGE/models/qwen2.5-vl-7b:/models/qwen"
ENV="-e HOME=/tmp -e PYTHONIOENCODING=utf-8 -e PYTHONPATH=/workspace -e SQ_CLAIM_COV=0.5 -e SQ_GT_INSTANCES=object \
  -e SQ_QWEN_DEVICE_MAP=cuda:0 -e SQ_OUT=$D"
PIP="pip install --user -q scipy scikit-learn >/dev/null 2>&1"
IMG=segciani:dev
RG_CLASSES="person diningtable tvmonitor dog boat bus cat motorbike"
say() { echo "$(date '+%m-%d %H:%M') $*"; }
others() {
  local uuid me pid uid
  uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $1); me=$(id -u)
  for pid in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v u="$uuid" '$1==u {print $2}'); do
    uid=$(ps -o uid= -p "$pid" 2>/dev/null | tr -d ' '); [ -n "$uid" ] && [ "$uid" != "$me" ] && echo $pid
  done; }
for c in $RG_CLASSES; do
  [ -f $RG/.done_$c ] && continue
  mkdir $RG/.claim_$c 2>/dev/null || continue
  if [ -n "$(others 1)" ]; then say "GPU 1 used by another user: stop"; rmdir $RG/.claim_$c; break; fi
  name=cmp_rg_${c}_gpu1b
  say "[gpu1] re-grounding slope-only situations of $c on model 2"
  setsid nohup bash outputs/gpu_guard.sh 1 $name > $L/guard_$name.log 2>&1 < /dev/null &
  docker run --rm --name $name --gpus device=1 -u $(id -u):$(id -g) $ENV $MOUNTS $MODELS -w /workspace $IMG bash -c \
    "$PIP; python -u examples/reground_validated.py --slices-index $C/slices_index_slopeonly.json \
     --substrate-cache $D/model2/substrate.pkl --vlm-backend qwen --vlm-model /models/qwen \
     --out-dir /data/outputs_model2_slopeonly --classes $c" > $L/reground_$c.log 2>&1
  sleep 3
  if grep -q GUARD_STOPPED $L/guard_$name.log; then say "guard stopped $name"; rmdir $RG/.claim_$c; break; fi
  grep -q "Traceback" $L/reground_$c.log && { say "re-grounding $c failed"; tail -3 $L/reground_$c.log; break; }
  touch $RG/.done_$c
done
# when every class is done (the other GPU may still be on its last one), the report again
for i in $(seq 1 240); do
  ALL=1; for c in $RG_CLASSES; do [ -f $RG/.done_$c ] || ALL=0; done
  [ $ALL = 1 ] && break
  docker ps --format '{{.Names}}' | grep -q "^cmp_rg_" || break
  sleep 30
done
docker run --rm -u $(id -u):$(id -g) $ENV $MOUNTS -w /workspace $IMG bash -c \
  "$PIP; python -u slice_quality/examples/disagreement_report.py" > $L/disagreement.log 2>&1
grep -E "^[A-Z ]+:|precision|model 2" $L/disagreement.log | head -12
say "CMP_RG_GPU1_DONE (all classes: $ALL)"
