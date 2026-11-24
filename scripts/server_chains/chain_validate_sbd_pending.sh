#!/bin/bash
# 27 October (Daniele: validate SBD with the 162 hypotheses, "Sì, su GPU 2"). The copy outputs/slice_quality_sbd_pending
# (156 situations + 13 clusters, 45.2%) through the checks that SBD can take; nothing official changes.
#   GPU 2  precision of the shipped phrases on the copy (Qwen), then the plausibility of the blamed situation on every
#          image with a cause (Gemma 4-bit; SBD has no human notes, so narrative only, control on a seeded sample)
#   The second segmenter has no predictions on SBD: that check is VOC only.
# Every container runs as the user (-u $(id -u):$(id -g)) under gpu_guard.sh; the queue waits while another user holds GPU 2.
#   setsid nohup bash outputs/chain_validate_sbd_pending.sh > outputs/validate_sbd_pending.log 2>&1 < /dev/null &
set -u
cd ~/storage/SLICELENS
D=outputs/slice_quality_sbd
P=${D}_pending
L=$P/logs
PV=$HOME/storage/pascal_vlm
G=2
mkdir -p $L
MOUNTS="-v $HOME/storage/SLICELENS:/workspace -v $PV:/data -v $DATASETS:/ds"
MODELS="-v $STORAGE/models/qwen2.5-vl-7b:/models/qwen -v $STORAGE/models/gemma-4-12B-it:/models/gemma"
ENV="-e HOME=/tmp -e PYTHONIOENCODING=utf-8 -e PYTHONPATH=/workspace -e SQ_QWEN_DEVICE_MAP=cuda:0 -e SQ_OUT=$P"
PIP="pip install --user -q scipy scikit-learn >/dev/null 2>&1"
IMG=segciani:dev
DEADLINE=$(( $(date +%s) + 30 * 3600 ))
say() { echo "$(date '+%m-%d %H:%M') $*"; }
others() {
  local uuid me pid uid
  uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $1); me=$(id -u)
  for pid in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v u="$uuid" '$1==u {print $2}'); do
    uid=$(ps -o uid= -p "$pid" 2>/dev/null | tr -d ' '); [ -n "$uid" ] && [ "$uid" != "$me" ] && echo $pid
  done; }
wait_gpu() {
  local said=0
  while [ -n "$(others $1)" ]; do
    [ $(date +%s) -ge $DEADLINE ] && return 1
    [ $said = 0 ] && { say "[gpu$1] busy with another user: waiting"; said=1; }
    sleep 300
  done
  return 0; }
guarded() {   # $1 gpu, $2 name, $3 log, rest: docker args; 2 = stopped by the guard
  local g=$1 name=$2 log=$3; shift 3
  setsid nohup bash outputs/gpu_guard.sh $g $name > $L/guard_$name.log 2>&1 < /dev/null &
  docker run --rm --name $name --gpus device=$g -u $(id -u):$(id -g) $ENV $MOUNTS $MODELS -w /workspace "$@" > $log 2>&1
  sleep 3
  grep -q GUARD_STOPPED $L/guard_$name.log && return 2
  return 0; }
retry() {     # $1 gpu, $2 name, $3 log, rest: docker args -- wait for the GPU, rerun after a guard stop
  local g=$1 rc
  while true; do
    wait_gpu $g || { say "[gpu$g] deadline: $2 not run"; return 1; }
    guarded "$@"; rc=$?
    [ $rc = 2 ] && { say "[gpu$g] guard stopped $2: will retry"; sleep 300; continue; }
    grep -q Traceback $3 && { say "[gpu$g] $2 failed"; tail -3 $3; return 1; }
    return 0
  done; }

say "== validating SBD with the 162 hypotheses (copy $P) =="
cp -n $D/error_narratives.json $P/ 2>/dev/null
say "[gpu$G] precision of the shipped phrases on the copy"
retry $G val_sbd_prec_gpu$G $L/precision.log $IMG bash -c "$PIP; python -u slice_quality/examples/eval_description_precision.py" \
  && tail -n 4 $L/precision.log
say "[gpu$G] plausibility of the causes, every image with one (narrative only)"
retry $G val_sbd_cause_gpu$G $L/cause_val.log -e JUDGE_4BIT=1 -e ALL=1 -e CONTROL_N=150 $IMG bash -c \
  "$PIP; python -u slice_quality/examples/judge_cause_plausibility.py" && tail -n 4 $L/cause_val.log
say "CHAIN_VALIDATE_SBD_PENDING_DONE"
