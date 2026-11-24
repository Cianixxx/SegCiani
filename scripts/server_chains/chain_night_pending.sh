#!/bin/bash
# Night of 21-22 October (Daniele: "fai il punto 3 su gpu 1 e 2, fai anche 1 e 2 su gpu0").
#   GPU 1, 2  point 3: the hypotheses the rank test finds significant on the 120-object sample of VOC but that
#             were never grounded on the whole class (the declared limit), grounded now on every object of
#             their class -> /data/outputs_pending. The queue WAITS while another user holds the GPU, and
#             retries a class if its guard stops it. Nothing in the official run changes: at the end only a
#             preview of the rank verdict on these groundings (CPU).
#   GPU 0     point 1: the "SliceLens + rank test" description (the official run's situations, SliceLens style)
#             judged by the facts judge and by the rubric judge, as the slope version was.
#   CPU       point 1: Pascal-EA rescored with the rank test (val-EA groundings), victims by branch;
#             point 2: every SliceLens rule (first and final version of the paper) vs the AUC test.
#   setsid nohup bash outputs/chain_night_pending.sh > outputs/night_pending.log 2>&1 < /dev/null &
set -u
cd ~/storage/SLICELENS
D=outputs/slice_quality_inst
PD=outputs/slice_quality_pending
CA=outputs/slice_quality_slcmp_auc
EA=outputs/slice_quality_ea_auc
L=$PD/logs
P=$HOME/storage/pascal_vlm
RG=$P/outputs_pending
mkdir -p $L $RG $CA/synth $EA
MOUNTS="-v $HOME/storage/SLICELENS:/workspace -v $P:/data -v $DATASETS:/ds"
MODELS="-v $STORAGE/models/qwen2.5-vl-7b:/models/qwen -v $STORAGE/models/gemma-4-12B-it:/models/gemma"
ENV="-e HOME=/tmp -e PYTHONIOENCODING=utf-8 -e PYTHONPATH=/workspace -e SQ_CLAIM_COV=0.5 -e SQ_GT_INSTANCES=object \
  -e SQ_QWEN_DEVICE_MAP=cuda:0 -e SQ_OUT=$D"
PIP="pip install --user -q scipy scikit-learn >/dev/null 2>&1"
IMG=segciani:dev
DEADLINE=$(( $(date +%s) + 30 * 3600 ))      # the waiting queue gives up after 30 hours
say() { echo "$(date '+%m-%d %H:%M') $*"; }
others() {
  local uuid me pid uid
  uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $1); me=$(id -u)
  for pid in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v u="$uuid" '$1==u {print $2}'); do
    uid=$(ps -o uid= -p "$pid" 2>/dev/null | tr -d ' '); [ -n "$uid" ] && [ "$uid" != "$me" ] && echo $pid
  done; }
guarded() {   # $1 gpu, $2 name, $3 log, rest: docker args; returns 2 when the guard stopped it
  local g=$1 name=$2 log=$3; shift 3
  setsid nohup bash outputs/gpu_guard.sh $g $name > $L/guard_$name.log 2>&1 < /dev/null &
  docker run --rm --name $name --gpus device=$g -u $(id -u):$(id -g) $ENV $MOUNTS $MODELS -w /workspace "$@" > $log 2>&1
  sleep 3
  grep -q GUARD_STOPPED $L/guard_$name.log && return 2
  return 0; }
wait_gpu() {  # $1 gpu: until no other user is on it (or the deadline)
  local said=0
  while [ -n "$(others $1)" ]; do
    [ $(date +%s) -ge $DEADLINE ] && return 1
    [ $said = 0 ] && { say "[gpu$1] busy with another user: waiting"; said=1; }
    sleep 300
  done
  [ $said = 1 ] && say "[gpu$1] free"
  return 0; }
cpu_run() { docker run --rm -u $(id -u):$(id -g) $ENV $MOUNTS -w /workspace "$@"; }

say "== night: pending hypotheses on whole classes, SliceLens + rank test, every SliceLens rule =="
# ---- set-up: the pending list (as in _revision_numbers.load_official_validations) and the judges' inputs
python3 -I - <<'PY' || { say "set-up failed"; exit 1; }
import json, collections
rows = json.load(open("outputs/slope_permutation/results_rank.json"))["results"]["VOC val"]
full = {(x["cls"], x["query"]) for x in rows if x["stage"] == "full"}
pend = sorted({(x["cls"], x["query"]) for x in rows if x["stage"] == "sample" and x["new_sys"]} - full)
by = collections.defaultdict(list)
for c, q in pend:
    by[c].append(q)
sl = [{"slice_id": f"{c}_{k}", "source": "language", "in_descriptions": True, "hypothesis": q}
      for c, qs in by.items() for k, q in enumerate(qs)]
json.dump({"slices": sl}, open("outputs/slice_quality_pending/pending_index.json", "w"), indent=1)
sizes = {}
for c in by:
    for d in ("$HOME/storage/pascal_vlm/outputs_rank", "$HOME/storage/pascal_vlm/outputs"):
        for st in ("full", "cc"):
            try:
                sizes[c] = json.load(open(f"{d}/voc_paper_{c}_{st}.json"))["num_instances"]
                break
            except Exception:
                pass
        if c in sizes:
            break
order = sorted(by, key=lambda c: -len(by[c]) * sizes.get(c, 0))
open("outputs/slice_quality_pending/classes.txt", "w").write(" ".join(order))
tot = sum(len(by[c]) * sizes.get(c, 0) for c in by)
print(f"pending hypotheses: {len(pend)} over {len(by)} classes; groundings ~{tot} "
      f"(~{tot * 1.5 / 3600 / 2:.1f} h on two GPUs at 1.5 s each)")
for c in order:
    print(f"  {c:12s} {len(by[c]):3d} x {sizes.get(c, 0):5d}")
F = {}
for k in (0, 1):
    F.update(json.load(open(f"outputs/slice_quality_inst/judge_claims_facts_shard{k}.json", encoding="utf-8"))["facts"])
json.dump({"facts": F, "complete": True}, open("outputs/slice_quality_slcmp_auc/judge_claims_facts_shard0.json", "w",
          encoding="utf-8"), indent=1, ensure_ascii=False)
PY
CLASSES=$(cat $PD/classes.txt)
cp $D/error_narratives.json $CA/error_narratives.json
echo '{}' > $CA/synth/narratives_synth.json; echo '{}' > $CA/synth/descriptions_synth.json; echo '{}' > $CA/empty.json

# ---- GPU 1 and 2: point 3, a shared class queue that waits for the GPU and retries after a guard stop
pend_queue() {
  local g=$1 c rc
  for c in $CLASSES; do
    [ -f $RG/.done_$c ] && continue
    mkdir $RG/.claim_$c 2>/dev/null || continue
    while true; do
      wait_gpu $g || { rmdir $RG/.claim_$c; say "[gpu$g] deadline: queue ends"; return 0; }
      say "[gpu$g] grounding the pending hypotheses of $c on every object"
      guarded $g pend_${c}_gpu$g $L/reground_$c.log $IMG bash -c "$PIP; python -u examples/reground_validated.py \
        --slices-index $PD/pending_index.json --substrate-cache /data/voc2012/substrate_cc.pkl --vlm-backend qwen \
        --vlm-model /models/qwen --out-dir /data/outputs_pending --classes $c"; rc=$?
      if [ $rc = 2 ]; then say "[gpu$g] guard stopped $c: will retry"; sleep 300; continue; fi
      grep -q "Traceback" $L/reground_$c.log && { say "[gpu$g] $c failed"; tail -3 $L/reground_$c.log; rmdir $RG/.claim_$c; return 1; }
      touch $RG/.done_$c; break
    done
  done; }
( pend_queue 1; say "[gpu1] queue done" ) &
Q1=$!
( pend_queue 2; say "[gpu2] queue done" ) &
Q2=$!

# ---- GPU 0: point 1, the "SliceLens + rank test" description, both judges
( say "[cpu] SliceLens-style text from the official run (rank test)"
  cpu_run -e SLOPE_RUN=$D -e OUT=$CA $IMG bash -c "$PIP; python -u slice_quality/examples/slicelens_style_texts.py" \
    > $L/texts_auc.log 2>&1 || { say "texts failed"; exit 0; }
  head -1 $L/texts_auc.log
  wait_gpu 0 || exit 0
  say "[gpu0] facts judge on the SliceLens + rank test text"
  guarded 0 night_claims_auc_gpu0 $L/claims_auc.log -e SQ_OUT=$CA -e KINDS=pattern -e OLD_DUP=0 -e JUDGE_4BIT=1 \
    -e SHARD=0 -e NSHARDS=1 $IMG bash -c "$PIP; python -u slice_quality/examples/judge_claims.py" || { say "[gpu0] stopped"; exit 0; }
  cpu_run -e SQ_OUT=$CA -e KINDS=pattern -e OLD_DUP=0 -e NSHARDS=1 -e MERGE=1 $IMG bash -c \
    "$PIP; python -u slice_quality/examples/judge_claims.py" > $L/claims_auc_merge.log 2>&1
  wait_gpu 0 || exit 0
  say "[gpu0] rubric judge on the same text"
  guarded 0 night_rubric_auc_gpu0 $L/rubric_auc.log -e JUDGE_4BIT=1 -e NARR=$CA/empty.json -e DESC=$CA/image_descriptions.json \
    -e DESC_TR=/nonexistent -e OUT=$CA/judge_descriptions $IMG bash -c "$PIP; python -u slice_quality/examples/judge_descriptions.py" \
    || { say "[gpu0] stopped"; exit 0; }
  grep -E "pattern \(score" $L/rubric_auc.log | head -2
  say "[gpu0] queue done" ) &
Q0=$!

# ---- CPU: point 1 (Pascal-EA with the rank test) and point 2 (every SliceLens rule)
( say "[cpu] Pascal-EA: rank verdict on the val-EA groundings"
  for st in cc full; do
    cpu_run -e RUNS_DIR=/data/outputs_ea -e STAGE=$st -e OUT=$EA/dd_auc_$st -e RANK_COPY=/data/outputs_ea_rank $IMG bash -c \
      "$PIP; python -u slice_quality/examples/dd_auc.py" > $L/ea_auc_$st.log 2>&1
    head -1 $L/ea_auc_$st.log
  done
  for f in result_tagged.pkl voc_full_val_tagged.json cluster_names.json planted_eval.json ea_summary.json typed_per_image.json; do
    [ -f outputs/slice_quality_ea/$f ] && cp outputs/slice_quality_ea/$f $EA/
  done
  cpu_run -e SQ_OUT=$EA -e LANG_DIRS=/data/outputs_ea_rank -e VOC_DATA=/data/voc2012_ea/VOCdevkit/VOC2012 -e GALLERY=0 $IMG bash -c \
    "$PIP; python -u slice_quality/examples/rescore_with_language.py" > $L/ea_rescore.log 2>&1
  grep -q "DONE rescore" $L/ea_rescore.log || { say "[cpu] val-EA rescore failed"; tail -4 $L/ea_rescore.log; }
  cpu_run -e EA_RUN=$EA $IMG bash -c "$PIP; python -u slice_quality/examples/ea_branch_coverage.py" > $L/ea_branch_auc.log 2>&1
  head -4 $L/ea_branch_auc.log
  say "[cpu] every SliceLens rule vs AUC (simulation)"
  cpu_run -e WORKERS=24 -e OUT=outputs/slope_rules_sim $IMG bash -c "$PIP; python -u slice_quality/examples/slope_rules_sim.py" \
    > $L/rules_sim.log 2>&1
  grep -q SLOPE_RULES_SIM_DONE $L/rules_sim.log && grep -A3 "REAL LABELS" $L/rules_sim.log || { say "[cpu] simulation failed"; tail -3 $L/rules_sim.log; }
  say "[cpu] done" ) &
QC=$!

wait $Q0 $QC
say "NIGHT_PART_GPU0_CPU_DONE"
wait $Q1 $Q2
ALL=1
for c in $CLASSES; do [ -f $RG/.done_$c ] || ALL=0; done
say "[cpu] preview: the rank verdict on the pending groundings (all classes done: $ALL)"
cpu_run -e RUNS_DIR=/data/outputs_pending -e STAGE=full -e OUT=$PD/dd_auc_full $IMG bash -c \
  "$PIP; python -u slice_quality/examples/dd_auc.py" > $L/pending_auc.log 2>&1
head -1 $L/pending_auc.log
say "CHAIN_NIGHT_PENDING_DONE"
