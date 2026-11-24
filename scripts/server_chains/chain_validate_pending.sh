#!/bin/bash
# Night of 22-23 October (Daniele: "valida le 67 situazioni stanotte su gpu 2 4 6"). The situations that the pending
# hypotheses add, on the copy outputs/slice_quality_inst_pending, through the checks of the official run:
#   GPU 2  precision of the shipped phrases on the copy (Qwen), then the model-2 queue
#   GPU 6  the new hypotheses grounded on the SECOND segmenter's objects (/data/outputs_model2_pending), class queue
#   GPU 4  (GTX 1080 Ti, Gemma 4-bit) plausibility of the blamed situation: annotated images, then every image;
#          the official verdicts are copied in, so only causes that changed are judged again
#   CPU    the report: new against official on every check
# Queues wait while another user holds a GPU and retry a class the guard stopped; nothing official changes.
#   setsid nohup bash outputs/chain_validate_pending.sh > outputs/validate_pending.log 2>&1 < /dev/null &
set -u
cd ~/storage/SLICELENS
D=outputs/slice_quality_inst
P=${D}_pending
L=$P/logs
PV=$HOME/storage/pascal_vlm
RG=$PV/outputs_model2_pending
mkdir -p $L $RG
MOUNTS="-v $HOME/storage/SLICELENS:/workspace -v $PV:/data -v $DATASETS:/ds"
MODELS="-v $STORAGE/models/qwen2.5-vl-7b:/models/qwen -v $STORAGE/models/gemma-4-12B-it:/models/gemma"
ENV="-e HOME=/tmp -e PYTHONIOENCODING=utf-8 -e PYTHONPATH=/workspace -e SQ_CLAIM_COV=0.5 -e SQ_GT_INSTANCES=object \
  -e SQ_QWEN_DEVICE_MAP=cuda:0 -e SQ_OUT=$P"
PIP="pip install --user -q scipy scikit-learn >/dev/null 2>&1"
IMG=segciani:dev
DEADLINE=$(( $(date +%s) + 20 * 3600 ))
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

say "== validating the situations the pending hypotheses add (copy $P) =="
# ---- set-up: the new situations' index; the judge's inputs and the official verdicts to resume from
python3 - <<'PY' || { say "set-up failed"; exit 1; }
import json, re, collections
cls_of = lambda sid: re.sub(r"_(extra|dd|pend)$", "", sid.rsplit("_", 1)[0])
def sits(d):
    return {(cls_of(s["slice_id"]), s["hypothesis"]) for s in json.load(open(d + "/slices_index.json"))["slices"]
            if s["source"] == "language" and s.get("in_descriptions")}
new = sorted(sits("outputs/slice_quality_inst_pending") - sits("outputs/slice_quality_inst"))
by = collections.defaultdict(list)
for c, q in new:
    by[c].append(q)
sl = [{"slice_id": "%s_%d" % (c, k), "source": "language", "in_descriptions": True, "hypothesis": q}
      for c, qs in by.items() for k, q in enumerate(qs)]
json.dump({"slices": sl}, open("outputs/slice_quality_inst_pending/new_index.json", "w"), indent=1)
size = {}
for c in by:   # model 2's objects of the class: heaviest classes first, so two GPUs finish together
    try:
        size[c] = json.load(open("$HOME/storage/pascal_vlm/outputs_model2_inst/voc_paper_%s_full.json" % c))["num_instances"]
    except Exception:
        size[c] = 300
order = sorted(by, key=lambda c: -len(by[c]) * size[c])
open("outputs/slice_quality_inst_pending/new_classes.txt", "w").write(" ".join(order))
print("new situations:", len(new), "in", len(by), "classes; model-2 evals ~", sum(len(by[c]) * size[c] for c in by))
print("order:", " ".join("%s(%d)" % (c, len(by[c]) * size[c]) for c in order))
PY
CLASSES=$(cat $P/new_classes.txt)
cp $D/error_narratives.json $D/typed_per_image.json $P/ 2>/dev/null
cp -r $D/train36 $P/ 2>/dev/null
for f in judge_cause judge_cause_val; do for e in json txt; do cp $D/$f.$e $P/$f.$e 2>/dev/null; done; done

m2_queue() {   # $1 gpu: the new hypotheses on model 2's objects, shared class queue
  local g=$1 c
  for c in $CLASSES; do
    [ -f $RG/.done_$c ] && continue
    mkdir $RG/.claim_$c 2>/dev/null || continue
    say "[gpu$g] model 2: the new situations of $c"
    retry $g val_m2_${c}_gpu$g $L/m2_$c.log $IMG bash -c "$PIP; python -u examples/reground_validated.py \
      --slices-index $P/new_index.json --substrate-cache $D/model2/substrate.pkl --vlm-backend qwen \
      --vlm-model /models/qwen --out-dir /data/outputs_model2_pending --classes $c" || { rmdir $RG/.claim_$c; return 1; }
    touch $RG/.done_$c
  done; }

( say "[gpu2] precision of the shipped phrases on the copy"
  retry 2 val_prec_gpu2 $L/precision.log $IMG bash -c "$PIP; python -u slice_quality/examples/eval_description_precision.py" \
    && tail -n 3 $L/precision.log
  m2_queue 2
  say "[gpu2] queue done" ) &
Q2=$!
( m2_queue 6; say "[gpu6] queue done" ) &
Q6=$!
( say "[gpu4] plausibility of the causes, annotated images"
  retry 4 val_cause_gpu4 $L/cause.log -e JUDGE_4BIT=1 $IMG bash -c "$PIP; python -u slice_quality/examples/judge_cause_plausibility.py" \
    && tail -n 3 $L/cause.log
  say "[gpu4] plausibility of the causes, every image with one"
  retry 4 val_causeval_gpu4 $L/cause_val.log -e JUDGE_4BIT=1 -e ALL=1 -e CONTROL_N=150 $IMG bash -c \
    "$PIP; python -u slice_quality/examples/judge_cause_plausibility.py" && tail -n 3 $L/cause_val.log
  say "[gpu4] queue done" ) &
Q4=$!
wait $Q2 $Q6 $Q4
ALL=1; for c in $CLASSES; do [ -f $RG/.done_$c ] || ALL=0; done
say "[cpu] rank test on model 2's groundings of the new situations (model 2 complete: $ALL)"
docker run --rm -u $(id -u):$(id -g) $ENV $MOUNTS -e RUNS_DIR=/data/outputs_model2_pending -e STAGE=full \
  -e OUT=$P/model2_new_dd_auc_full -w /workspace $IMG bash -c "$PIP; python -u slice_quality/examples/dd_auc.py" \
  > $L/model2_auc.log 2>&1
head -n 1 $L/model2_auc.log
say "[cpu] report"
docker run --rm -u $(id -u):$(id -g) $ENV $MOUNTS -w /workspace $IMG bash -c \
  "$PIP; python -u slice_quality/examples/pending_validation_report.py" > $L/report.log 2>&1
cat $P/validation_report.txt
say "CHAIN_VALIDATE_PENDING_DONE"
