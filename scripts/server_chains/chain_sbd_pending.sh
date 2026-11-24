#!/bin/bash
# SBD: the hypotheses the rank test finds significant on the 120-object sample but the slope rule had kept from the
# whole class (162, the limit declared for SBD in §18.5), grounded on every object of their class (Daniele,
# 2026-10-23: "lancia le 162 frasi di SBD su gpu 2 4 5"; Qwen in bf16 runs on 2 and 5 only, the 1080 Ti on 4 would
# need 4-bit and give confidences unlike every other SBD grounding). ~324k groundings: about three days on two GPUs.
#   GPU 2, 5   every class but person (shared queue of ground_ea_hypotheses.py, heaviest first), then person in 24
#              blocks shared by both (ground_split_class.py: same groundings, only the scheduling changes).
#              Every container under outputs/gpu_guard.sh; a GPU held by another user is waited for, a class or
#              block stopped by the guard is released and retried.
#   CPU        person assembled from its blocks; the rank verdict on the whole classes (dd_auc.py, BH within these);
#              a COPY of the SBD run rescored with them beside the official groundings (the official run is not
#              touched), and its summary against the official run.
#   setsid nohup bash outputs/chain_sbd_pending.sh > outputs/sbd_pending.log 2>&1 < /dev/null &
set -u
cd ~/storage/SLICELENS
L=outputs/slice_quality_sbd_pending
LOGS=$L/logs
P=$HOME/storage/pascal_vlm
O=$P/outputs_sbd_pending
mkdir -p $LOGS $O
MOUNTS="-v $HOME/storage/SLICELENS:/workspace -v $P:/data -v $DATASETS:/ds"
QWEN="-v $STORAGE/models/qwen2.5-vl-7b:/models/qwen"
ENV="-e HOME=/tmp -e PYTHONIOENCODING=utf-8 -e PYTHONPATH=/workspace -e SQ_QWEN_DEVICE_MAP=cuda:0"
LANG="-e OUT_DIR=/data/outputs_sbd_pending -e FULL_HYPS=/data/full_pending_sbd.json -e SUBSTRATE=/data/sbd/substrate_cc.pkl \
  -e OLD_SUBSTRATE= -e EDITED= -e CACHE_DIRS= -e SAMPLING=srs -e DATASET=SBD -e PHASE=full"
PIP="pip install --user -q scipy scikit-learn >/dev/null 2>&1"
IMG=segciani:dev
DEADLINE=$(( $(date +%s) + 5 * 24 * 3600 ))
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
guarded() {   # $1 gpu, $2 name, $3 log, rest: docker args; 2 = stopped by the guard
  local g=$1 name=$2 log=$3; shift 3
  setsid nohup bash outputs/gpu_guard.sh $g $name > $LOGS/guard_$name.log 2>&1 < /dev/null &
  docker run --rm --name $name --gpus device=$g -u $(id -u):$(id -g) $ENV $LANG $MOUNTS $QWEN -w /workspace "$@" > $log 2>&1
  sleep 3
  grep -q GUARD_STOPPED $LOGS/guard_$name.log && return 2
  grep -q Traceback $log && return 1
  return 0; }

say "== SBD: the 162 hypotheses kept from the whole class by the slope rule, on every object of their class =="
python3 -I - <<'PY' || { say "set-up failed"; exit 1; }
import json, collections
rows = json.load(open("outputs/slope_permutation/results_rank.json"))["results"]["SBD"]
full = {(x["cls"], x["query"]) for x in rows if x["stage"] == "full"}
pend = sorted({(x["cls"], x["query"]) for x in rows if x["stage"] == "sample" and x["new_sys"]} - full)
by = collections.defaultdict(list)
for c, q in pend:
    by[c].append(q)
json.dump(dict(by), open("$HOME/storage/pascal_vlm/full_pending_sbd.json", "w"), indent=1)
size = {}                                    # only to order the queue: the whole class, else the sample's file
for c in by:
    for st in ("full", "cc"):
        try:
            size[c] = json.load(open("$HOME/storage/pascal_vlm/outputs_sbd/voc_paper_%s_%s.json" % (c, st)))["num_instances"]
            break
        except OSError:
            pass
    size.setdefault(c, 0)
order = sorted((c for c in by if c != "person"), key=lambda c: -len(by[c]) * size[c])
open("outputs/slice_quality_sbd_pending/classes.txt", "w").write(" ".join(order))
tot = sum(len(by[c]) * size[c] for c in by)
print("hypotheses %d in %d classes; groundings %d (person %d)" % (len(pend), len(by), tot, len(by.get("person", [])) * size.get("person", 0)))
PY
CLASSES=$(cat $L/classes.txt)
mkdir -p $O/.claim_full_person               # person is done in blocks: the class queue must skip it

worker() {   # $1 gpu
  local g=$1 rc name c b
  while true; do                             # 1. the classes
    wait_gpu $g || { say "[gpu$g] deadline"; return 1; }
    name=sbdp_cls_gpu${g}_$(date +%d%H%M)
    say "[gpu$g] class queue ($name)"
    CLASSES="$CLASSES" guarded $g $name $LOGS/$name.log -e CLASSES $IMG bash -c "$PIP; python -u examples/ground_ea_hypotheses.py"
    rc=$?
    if [ $rc = 2 ]; then
      say "[gpu$g] guard stopped $name: its unfinished class is released, retrying"
      for c in $CLASSES; do [ -f $O/voc_paper_${c}_full.json ] || [ -f $O/.none_full_$c ] || rmdir $O/.claim_full_$c 2>/dev/null; done
      sleep 300; continue
    fi
    [ $rc = 1 ] && { say "[gpu$g] class queue failed"; tail -3 $LOGS/$name.log; return 1; }
    break
  done
  say "[gpu$g] classes done, person blocks"
  while true; do                             # 2. person, in blocks
    wait_gpu $g || { say "[gpu$g] deadline"; return 1; }
    name=sbdp_person_gpu${g}_$(date +%d%H%M)
    guarded $g $name $LOGS/$name.log -e CLS=person -e NBLOCKS=24 $IMG bash -c \
      "$PIP; PHASE=ground python -u examples/ground_split_class.py"
    rc=$?
    if [ $rc = 2 ]; then
      say "[gpu$g] guard stopped $name: its unfinished block is released, retrying"
      for b in $(seq -f %03g 0 23); do [ -f $O/split/person_$b.json ] || rmdir $O/split/.claim_person_$b 2>/dev/null; done
      sleep 300; continue
    fi
    [ $rc = 1 ] && { say "[gpu$g] person blocks failed"; tail -3 $LOGS/$name.log; return 1; }
    break
  done
  say "[gpu$g] done"; }

worker 2 &
W2=$!
sleep 60
worker 5 &
W5=$!
wait $W2 $W5
for c in $CLASSES; do [ -f $O/voc_paper_${c}_full.json ] || [ -f $O/.none_full_$c ] || { say "class $c missing: stop"; exit 1; }; done
n=$(ls $O/split/ 2>/dev/null | grep -c "^person_.*json$")
[ "$n" -eq 24 ] || { say "person blocks $n/24: stop"; exit 1; }

say "[cpu] person: blocks merged, its file written from the cache"
docker run --rm -u $(id -u):$(id -g) $ENV $LANG -e CLS=person -e NBLOCKS=24 $MOUNTS -w /workspace $IMG \
  bash -c "$PIP; PHASE=merge python -u examples/ground_split_class.py" > $LOGS/person_merge.log 2>&1
grep -q SPLIT_MERGE_DONE $LOGS/person_merge.log || { say "person merge failed"; tail -5 $LOGS/person_merge.log; exit 1; }
rmdir $O/.claim_full_person
docker run --rm -u $(id -u):$(id -g) $ENV $LANG -e CLASSES=person $MOUNTS -w /workspace $IMG \
  bash -c "$PIP; python -u examples/ground_ea_hypotheses.py" > $LOGS/person_assemble.log 2>&1
[ -f $O/voc_paper_person_full.json ] || { say "person assembly failed"; tail -5 $LOGS/person_assemble.log; exit 1; }

say "[cpu] the rank verdict on the whole classes, then a COPY of the SBD run rescored with them"
docker run --rm -u $(id -u):$(id -g) $ENV -e RUNS_DIR=/data/outputs_sbd_pending -e STAGE=full -e OUT=$L/dd_auc_full \
  -e RANK_COPY=/data/outputs_sbd_pending_rank $MOUNTS -w /workspace $IMG \
  bash -c "$PIP; python -u slice_quality/examples/dd_auc.py" > $LOGS/dd_auc.log 2>&1
head -n 1 $LOGS/dd_auc.log
cp outputs/slice_quality_sbd/result_tagged.pkl $L/
[ -f outputs/slice_quality_sbd/cluster_names.json ] && cp outputs/slice_quality_sbd/cluster_names.json $L/
docker run --rm -u $(id -u):$(id -g) -e HOME=/tmp -e PYTHONIOENCODING=utf-8 -e PYTHONPATH=/workspace -e SQ_OUT=$L \
  -e LANG_DIRS=/data/outputs_sbd_rank:/data/outputs_sbd_pending_rank -e LANG_TAGS=":pend" \
  -e VOC_DATA=/data/sbd/VOCdevkit/VOC2012 -e GALLERY=0 $MOUNTS -w /workspace $IMG \
  bash -c "$PIP; python -u slice_quality/examples/rescore_with_language.py" > $LOGS/rescore.log 2>&1
grep -q "DONE rescore" $LOGS/rescore.log || { say "rescore failed"; tail -5 $LOGS/rescore.log; exit 1; }
docker run --rm -u $(id -u):$(id -g) -e HOME=/tmp -e PYTHONPATH=/workspace $MOUNTS -w /workspace $IMG bash -c "$PIP; python -c \"
import sys; sys.path.insert(0, 'docs')
from _revision_numbers import load_inst_summary as L
for d in ('outputs/slice_quality_sbd', '$L'):
    s = L(d); print(d, 'situations', s['slices']['language'], 'clusters', s['slices']['cluster'],
                    'errors in a group %.1f%%' % (100 * s['cov_union'] / s['n_err']), 'images with a situation', s['situational_images'])
\"" 2>&1 | tail -2
say "CHAIN_SBD_PENDING_DONE"
