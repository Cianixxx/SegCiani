#!/bin/bash
# GPU 6 worker for outputs/chain_sbd_pending.sh (Daniele, 2026-10-25 22:52: "da ora in poi usa gpu 2 e 6"). It takes
# the place of sbd_pending_gpu1.sh (GPU 1 released at that moment, its unfinished block put back in the queue) and of
# the chain's GPU 5 worker (stopped: GPU 5 is no longer ours). Person blocks from the same queue (ground_split_class.py,
# mkdir claims): same groundings, only the scheduling changes. Under outputs/gpu_guard.sh; a GPU held by another user
# is waited for, a block stopped by the guard is released (only this worker's, by its CLAIM_OWNER file) and retried.
# The classes left (bottle) and the CPU tail are done by outputs/sbd_pending_rescue.sh, not here.
#   setsid nohup bash outputs/sbd_pending_gpu6.sh > outputs/sbd_pending_gpu6.log 2>&1 < /dev/null &
set -u
cd ~/storage/SLICELENS
G=6
L=outputs/slice_quality_sbd_pending
LOGS=$L/logs
P=$HOME/storage/pascal_vlm
O=$P/outputs_sbd_pending
MOUNTS="-v $HOME/storage/SLICELENS:/workspace -v $P:/data -v $DATASETS:/ds"
QWEN="-v $STORAGE/models/qwen2.5-vl-7b:/models/qwen"
ENV="-e HOME=/tmp -e PYTHONIOENCODING=utf-8 -e PYTHONPATH=/workspace -e SQ_QWEN_DEVICE_MAP=cuda:0"
LANG="-e OUT_DIR=/data/outputs_sbd_pending -e FULL_HYPS=/data/full_pending_sbd.json -e SUBSTRATE=/data/sbd/substrate_cc.pkl \
  -e OLD_SUBSTRATE= -e EDITED= -e CACHE_DIRS= -e SAMPLING=srs -e DATASET=SBD -e PHASE=full"
PIP="pip install --user -q scipy scikit-learn >/dev/null 2>&1"
IMG=segciani:dev
DEADLINE=$(( $(date +%s) + 4 * 24 * 3600 ))
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
  [ $said = 1 ] && say "[gpu$1] free again"
  return 0; }
say "== SBD pending: worker on GPU $G (person blocks) =="
while true; do
  n=$(ls $O/split/ 2>/dev/null | grep -c "^person_.*json$")
  free=0
  for b in $(seq -f %03g 0 23); do [ -f $O/split/person_$b.json ] || [ -d $O/split/.claim_person_$b ] || free=$((free + 1)); done
  [ $free = 0 ] && { say "[gpu$G] no free block left ($n of 24 done)"; break; }
  wait_gpu $G || { say "[gpu$G] deadline"; break; }
  name=sbdp_person_gpu${G}_$(date +%d%H%M)
  say "[gpu$G] person blocks ($name; $n of 24 done, $free free)"
  setsid nohup bash outputs/gpu_guard.sh $G $name > $LOGS/guard_$name.log 2>&1 < /dev/null &
  docker run --rm --name $name --gpus device=$G -u $(id -u):$(id -g) $ENV $LANG -e CLS=person -e NBLOCKS=24 -e CLAIM_OWNER=$name \
    $MOUNTS $QWEN -w /workspace $IMG bash -c "$PIP; PHASE=ground python -u examples/ground_split_class.py" > $LOGS/$name.log 2>&1
  sleep 3
  grep -E "block [0-9]+/24\]" $LOGS/$name.log | tail -3
  if grep -q GUARD_STOPPED $LOGS/guard_$name.log || grep -q Traceback $LOGS/$name.log; then
    grep -q Traceback $LOGS/$name.log && { say "[gpu$G] $name failed"; tail -3 $LOGS/$name.log; }
    for f in $(grep -l "^$name\$" $O/split/.owner_person_* 2>/dev/null); do
      b=${f##*_}
      [ -f $O/split/person_$b.json ] || { rmdir $O/split/.claim_person_$b 2>/dev/null; say "[gpu$G] block $b released"; }
      rm -f $f
    done
    grep -q Traceback $LOGS/$name.log && break
    say "[gpu$G] guard stopped $name: retrying"; sleep 300; continue
  fi
  break
done
say "[gpu$G] done"
say "SBD_PENDING_GPU6_DONE"
