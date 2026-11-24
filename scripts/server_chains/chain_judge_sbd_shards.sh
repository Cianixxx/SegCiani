#!/bin/bash
# 27 October (Daniele: "hai disponibili gpu1 e2"). The plausibility of the SBD 162 causes ran on GPU 2 alone at
# 18.6 s per judgement, 30 h for 5834. This splits the same job list over GPU 1 and GPU 2 (SHARD=0/2, 1/2): the
# single-GPU container is stopped (its verdicts stay on disk and are reused), shard 0 starts on GPU 1 once the
# precision there ends, and a last run without SHARD merges the shard files and writes the report, no model loaded.
# Every container runs as the user (-u $(id -u):$(id -g)) under gpu_guard.sh; a guard stop waits and resumes.
#   setsid nohup bash outputs/chain_judge_sbd_shards.sh > outputs/judge_sbd_shards.log 2>&1 < /dev/null &
set -u
cd ~/storage/SLICELENS
P=outputs/slice_quality_sbd_pending
L=$P/logs
PV=$HOME/storage/pascal_vlm
mkdir -p $L
MOUNTS="-v $HOME/storage/SLICELENS:/workspace -v $PV:/data -v $DATASETS:/ds"
MODELS="-v $STORAGE/models/qwen2.5-vl-7b:/models/qwen -v $STORAGE/models/gemma-4-12B-it:/models/gemma"
ENV="-e HOME=/tmp -e PYTHONIOENCODING=utf-8 -e PYTHONPATH=/workspace -e SQ_OUT=$P -e JUDGE_4BIT=1 -e ALL=1 -e CONTROL_N=150"
PIP="pip install --user -q scipy scikit-learn >/dev/null 2>&1"
IMG=segciani:dev
RUN="python -u slice_quality/examples/judge_cause_plausibility.py"
DEADLINE=$(( $(date +%s) + 40 * 3600 ))
say() { echo "$(date '+%m-%d %H:%M') $*"; }
running() { docker ps --format '{{.Names}}' | grep -qx "$1"; }
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
  docker run --rm --name $name --gpus device=$g -u $(id -u):$(id -g) $ENV $MOUNTS $MODELS -w /workspace "$@" >> $log 2>&1
  sleep 3
  grep -q GUARD_STOPPED $L/guard_$name.log && return 2
  return 0; }
retry() {     # $1 gpu, $2 name, $3 log, rest: docker args -- wait for the GPU, rerun after a guard stop
  local g=$1 rc
  while true; do
    wait_gpu $g || { say "[gpu$g] deadline: $2 not run"; return 1; }
    guarded "$@"; rc=$?
    [ $rc = 2 ] && { say "[gpu$g] guard stopped $2: will resume"; sleep 300; continue; }
    grep -q Traceback $3 && { say "[gpu$g] $2 failed"; tail -3 $3; return 1; }
    grep -q JUDGE_CAUSE_DONE $3 || { say "[gpu$g] $2 ended without finishing"; return 1; }
    return 0
  done; }

say "== SBD 162 plausibility over GPU 1 + GPU 2 =="
(
  running val_sbd_cause_gpu2 && { say "[gpu2] stopping the single-GPU run (verdicts on disk are reused)"; docker stop val_sbd_cause_gpu2 >/dev/null; }
  while running val_sbd_cause_gpu2; do sleep 10; done
  say "[gpu2] shard 1/2"
  retry 2 val_sbd_judge_s1 $L/cause_val_shard1.log -e SHARD=1/2 $IMG bash -c "$PIP; $RUN" && say "[gpu2] shard 1/2 done"
) &
(
  while running val_sbd_prec_gpu1; do sleep 30; done
  say "[gpu1] precision over; shard 0/2"
  retry 1 val_sbd_judge_s0 $L/cause_val_shard0.log -e SHARD=0/2 $IMG bash -c "$PIP; $RUN" && say "[gpu1] shard 0/2 done"
) &
wait
if grep -q JUDGE_CAUSE_DONE $L/cause_val_shard0.log && grep -q JUDGE_CAUSE_DONE $L/cause_val_shard1.log; then
  say "merging the shards and writing the report"
  docker run --rm --name val_sbd_judge_merge -u $(id -u):$(id -g) $ENV $MOUNTS -w /workspace $IMG bash -c "$PIP; $RUN" \
    > $L/cause_val_merge.log 2>&1
  tail -n 25 $L/cause_val_merge.log
else
  say "a shard did not finish: no merge (rerun this script, it resumes)"
fi
say "CHAIN_JUDGE_SBD_SHARDS_DONE"
