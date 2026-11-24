#!/bin/bash
# GPU 5 was taken by another user on 25 Oct at 13:34 (deintrona's VLA server) with bottle at 11/20 hypotheses; the
# chain's GPU 5 worker waits for it (deadline 28 Oct 16:00) and its class queue was released. Daniele has GPU 1 and 2
# (2026-10-25: "ho a disposizione gpu 1 e 2 ora"), then GPU 2 and 6 (22:52: "da ora in poi usa gpu 2 e 6"; GPU 5's
# worker stopped, GPU 1 handed to outputs/sbd_pending_gpu6.sh). So:
#   1. when no person block is left to claim and the person container of a GPU in $GPUS has ended, that GPU runs the
#      class queue (ground_ea_hypotheses.py: mkdir claims, so it never collides with GPU 5 if that comes back;
#      groundings_bottle.json resumes the 11 hypotheses already done). Same groundings, only the scheduling changes.
#      It goes on until every class is on disk: a class claimed by GPU 5 comes back to the queue when the guard stops
#      GPU 5 again (the first version, 25 Oct 15:56, left at 17:46 on such a claim and waited for nothing).
#   2. when every class and the 24 person blocks are on disk and the chain is still only waiting for GPU 5 (no
#      container of its own running), the chain is stopped and its CPU tail (sbd_pending_finish.sh) is run here,
#      appended to outputs/sbd_pending.log. If GPU 5 comes back first, the chain ends by itself: its tail is run
#      here only if it stopped before it (a class still running here) and sbd_pending_gpu1.sh did not run it.
#   setsid nohup bash outputs/sbd_pending_rescue.sh > outputs/sbd_pending_rescue.log 2>&1 < /dev/null &
set -u
cd ~/storage/SLICELENS
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
CLASSES=$(cat $L/classes.txt)
GPUS="2 6"                                          # Daniele, 25 Oct 22:52: "da ora in poi usa gpu 2 e 6"
DEADLINE=$(( $(date +%s) + 4 * 24 * 3600 ))
say() { echo "$(date '+%m-%d %H:%M') $*"; }
others() {
  local uuid me pid uid
  uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $1); me=$(id -u)
  for pid in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v u="$uuid" '$1==u {print $2}'); do
    uid=$(ps -o uid= -p "$pid" 2>/dev/null | tr -d ' '); [ -n "$uid" ] && [ "$uid" != "$me" ] && echo $pid
  done; }
open_classes() { for c in $CLASSES; do [ -f $O/voc_paper_${c}_full.json ] || [ -f $O/.none_full_$c ] || [ -d $O/.claim_full_$c ] || echo $c; done; }
missing_classes() { for c in $CLASSES; do [ -f $O/voc_paper_${c}_full.json ] || [ -f $O/.none_full_$c ] || echo $c; done; }
free_blocks() { local b n=0; for b in $(seq -f %03g 0 23); do [ -f $O/split/person_$b.json ] || [ -d $O/split/.claim_person_$b ] || n=$((n + 1)); done; echo $n; }
done_blocks() { ls $O/split/ 2>/dev/null | grep -c "^person_.*json$"; }
running() { docker ps --format '{{.Names}}' | grep -E "^$1"; }

started() { grep -oP "^=== \S+ \[\K[a-z]+(?=\])" $1; }   # the classes a class-queue container has begun
keep_claims() {   # $1 container, $2 its log. The chain's GPU 5 worker, when its guard stops it, releases EVERY unfinished
  local i=0 c     # class claim, also one held here (25 Oct 17:48 it released its own while this script had moved on):
  while [ $i -lt 60 ] && [ -z "$(running $1)" ]; do sleep 5; i=$((i + 1)); done   # put back, while the container runs
  while [ -n "$(running $1)" ]; do
    for c in $(started $2); do
      [ -f $O/voc_paper_${c}_full.json ] || [ -d $O/.claim_full_$c ] || { mkdir $O/.claim_full_$c && say "[keep] claim of $c put back"; }
    done
    sleep 10
  done; }

say "== SBD pending: the classes left by GPU 5, on GPU $(echo $GPUS | sed 's/ / or /') when their person blocks run out =="
while true; do                                       # 1. the class queue, until every class is on disk
  [ -z "$(missing_classes)" ] && { say "every class is on disk"; break; }
  [ $(date +%s) -ge $DEADLINE ] && { say "deadline"; exit 1; }
  if [ -z "$(open_classes)" ] && [ -z "$(running sbdp_cls_)" ]; then   # claimed, but no class container anywhere
    sleep 60
    if [ -z "$(running sbdp_cls_)" ]; then
      for c in $(missing_classes); do [ -d $O/.claim_full_$c ] && rmdir $O/.claim_full_$c && say "stale claim of $c released"; done
    fi
  fi
  G=
  if [ -n "$(open_classes)" ] && [ "$(free_blocks)" = 0 ]; then
    for g in $GPUS; do [ -z "$(running sbdp_.*_gpu$g)" ] && [ -z "$(others $g)" ] && { G=$g; break; }; done
  fi
  [ -z "$G" ] && { sleep 300; continue; }
  name=sbdp_cls_gpu${G}_$(date +%d%H%M)
  say "[gpu$G] class queue ($name): $(open_classes | tr '\n' ' ')"
  setsid nohup bash outputs/gpu_guard.sh $G $name > $LOGS/guard_$name.log 2>&1 < /dev/null &
  keep_claims $name $LOGS/$name.log &
  docker run --rm --name $name --gpus device=$G -u $(id -u):$(id -g) $ENV $LANG -e CLASSES="$CLASSES" \
    $MOUNTS $QWEN -w /workspace $IMG bash -c "$PIP; python -u examples/ground_ea_hypotheses.py" > $LOGS/$name.log 2>&1
  sleep 3; wait                                      # keep_claims ends with the container
  grep -E "^=== done" $LOGS/$name.log | tail -3
  if grep -q GUARD_STOPPED $LOGS/guard_$name.log || grep -q Traceback $LOGS/$name.log; then
    grep -q Traceback $LOGS/$name.log && { say "[gpu$G] $name failed"; tail -3 $LOGS/$name.log; exit 1; }
    for c in $(started $LOGS/$name.log); do          # only the classes this container started
      [ -f $O/voc_paper_${c}_full.json ] || { rmdir $O/.claim_full_$c 2>/dev/null; say "[gpu$G] class $c released"; }
    done
    say "[gpu$G] guard stopped $name: retrying"; sleep 300; continue
  fi
done
while [ -n "$(missing_classes)" ] || [ "$(done_blocks)" != 24 ]; do   # 2. everything on disk
  [ $(date +%s) -ge $DEADLINE ] && { say "deadline"; exit 1; }
  sleep 300
done
say "every class and the 24 person blocks are on disk"
waiting5() {   # the chain's GPU 5 worker is waiting for another user, the chain has not reached its CPU tail
  ! grep -q "\[cpu\]" outputs/sbd_pending.log && grep "\[gpu5\]" outputs/sbd_pending.log | tail -1 | grep -q "busy with another user: waiting" &&
    [ -z "$(running sbdp_.*_gpu5)" ] && [ -z "$(running sbdp_.*_gpu2)" ]; }
finish() {
  bash outputs/sbd_pending_finish.sh >> outputs/sbd_pending.log 2>&1
  tail -4 outputs/sbd_pending.log; }
while pgrep -f chain_sbd_pending.sh > /dev/null; do
  if waiting5; then
    sleep 60                                         # look again before stopping anything
    waiting5 && pgrep -f chain_sbd_pending.sh > /dev/null || continue
    say "the chain is only waiting for GPU 5: stopping it, its CPU tail runs here" | tee -a outputs/sbd_pending.log
    pkill -f chain_sbd_pending.sh; sleep 5
    finish; say "SBD_PENDING_RESCUE_DONE"; exit 0
  fi
  sleep 300
done
while pgrep -f sbd_pending_gpu1.sh > /dev/null; do sleep 60; done   # it may be running the CPU tail itself
if grep -q CHAIN_SBD_PENDING_DONE outputs/sbd_pending.log; then
  say "the CPU tail has already run: nothing to do"
elif ! grep -q "\[cpu\]" outputs/sbd_pending.log; then
  say "the chain stopped before its CPU tail (a class or block was still running here): running it" | tee -a outputs/sbd_pending.log
  finish
else
  say "the CPU tail started and failed: see outputs/sbd_pending.log"
fi
say "SBD_PENDING_RESCUE_DONE"
