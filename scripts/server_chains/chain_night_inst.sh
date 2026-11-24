#!/bin/bash
# The validations of the thesis re-done on the official inst run (Daniele, 2026-10-21: "fai 1 2 3 7 ... su gpu
# 1-2-6"), after chain_auc_official.sh made the AUC verdict official. One queue per GPU, every container under
# outputs/gpu_guard.sh: if another user takes a GPU, that queue ends and nothing is restarted.
#   GPU 1: second segmenter (DeepLabV3) typed like inst -> its objects (substrate.pkl), embeddings, tagging,
#          cluster branch; then re-grounding of the official situations on its objects (shared class queue)
#   GPU 2: (after the facts judge) precision of the shipped phrases; then the re-grounding queue
#   GPU 6: (after the facts judge) plausibility of the causes on the 80 annotated images, then on every image
#          with a cause; then the corruption experiment at two doses; then the re-grounding queue
#   CPU:   split-half replication at once; at the end the AUC verdict on model 2's groundings, its two-branch
#          result, the gallery with the new cause scores.
#   setsid nohup bash outputs/chain_night_inst.sh > outputs/night_inst.log 2>&1 < /dev/null &
set -u
cd ~/storage/SLICELENS
D=outputs/slice_quality_inst
L=$D/night
P=$HOME/storage/pascal_vlm
RG=$P/outputs_model2_inst
mkdir -p $L $RG
MOUNTS="-v $HOME/storage/SLICELENS:/workspace -v $P:/data -v $DATASETS:/ds"
MODELS="-v $STORAGE/models/qwen2.5-vl-7b:/models/qwen -v $STORAGE/models/gemma-4-12B-it:/models/gemma"
ENV="-e HOME=/tmp -e PYTHONIOENCODING=utf-8 -e PYTHONPATH=/workspace -e SQ_CLAIM_COV=0.5 -e SQ_GT_INSTANCES=object \
  -e SQ_QWEN_DEVICE_MAP=cuda:0 -e SQ_OUT=$D"
PIP="pip install --user -q scipy scikit-learn >/dev/null 2>&1"
IMG=segciani:dev
RG_CLASSES="person chair car dog cat bird pottedplant bottle diningtable sofa boat horse sheep cow tvmonitor bus motorbike bicycle train aeroplane"
say() { echo "$(date '+%m-%d %H:%M') $*"; }
others() {    # processes of OTHER users on GPU $1
  local uuid me pid uid
  uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $1); me=$(id -u)
  for pid in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v u="$uuid" '$1==u {print $2}'); do
    uid=$(ps -o uid= -p "$pid" 2>/dev/null | tr -d ' '); [ -n "$uid" ] && [ "$uid" != "$me" ] && echo $pid
  done; }
gpu_run() {   # $1 gpu, $2 container name, $3 log, rest: docker args after the mounts (image, command)
  local g=$1 name=$2 log=$3; shift 3
  if [ -n "$(others $g)" ]; then say "GPU $g used by another user: $name not started, queue ends"; return 1; fi
  setsid nohup bash outputs/gpu_guard.sh $g $name > $L/guard_$name.log 2>&1 < /dev/null &
  docker run --rm --name $name --gpus device=$g -u $(id -u):$(id -g) $ENV $MOUNTS $MODELS -w /workspace "$@" > $log 2>&1
  sleep 3
  if grep -q GUARD_STOPPED $L/guard_$name.log; then say "guard stopped $name, queue ends"; return 1; fi
  return 0; }
cpu_run() { docker run --rm -u $(id -u):$(id -g) $ENV $MOUNTS -w /workspace "$@"; }
wait_auc() { until grep -q "CHAIN_AUC_OFFICIAL_DONE" outputs/auc_official.log; do
  grep -q "CHAIN_AUC_OFFICIAL_FAIL\|FAIL rescore" outputs/auc_official.log && { say "AUC switch failed: queue ends"; return 1; }
  sleep 60; done; }
wait_free() { while docker ps --format '{{.Names}}' | grep -qx "$1"; do sleep 60; done; }
reground_queue() {   # $1 gpu: claims classes of the shared queue
  local c
  until [ -f $D/model2/substrate.pkl ]; do
    grep -q "Traceback" $L/model2_cluster.log 2>/dev/null && { say "model 2 Block 1 failed: no re-grounding"; return 1; }
    sleep 60; done
  for c in $RG_CLASSES; do
    [ -f $RG/.done_$c ] && continue
    mkdir $RG/.claim_$c 2>/dev/null || continue
    say "[gpu$1] re-grounding $c on model 2"
    gpu_run $1 night_rg_${c}_gpu$1 $L/reground_$c.log $IMG bash -c "$PIP; python -u examples/reground_validated.py \
      --slices-index $D/slices_index.json --substrate-cache $D/model2/substrate.pkl --vlm-backend qwen \
      --vlm-model /models/qwen --out-dir /data/outputs_model2_inst --classes $c" || return 1
    grep -q "Traceback" $L/reground_$c.log && { say "re-grounding $c failed"; tail -3 $L/reground_$c.log; return 1; }
    touch $RG/.done_$c
  done; }

say "== night chain on the official inst run =="
wait_auc || exit 0
grep -E "certified slices" outputs/auc_official.log

# ---- CPU: split-half replication of the cluster branch ---------------------------------------------
( say "[cpu] split-half"
  cpu_run $IMG bash -c "$PIP; python -u slice_quality/examples/split_half.py" > $L/split_half.log 2>&1
  grep -q SPLIT_HALF_DONE $L/split_half.log && grep "OVERALL" $L/split_half.log || { say "split-half failed"; tail -3 $L/split_half.log; }
) &
CPU_PID=$!

# ---- GPU 1 -----------------------------------------------------------------------------------------
( say "[gpu1] second segmenter, cluster branch"
  gpu_run 1 night_model2_gpu1 $L/model2_cluster.log -e OUT_TAG=model2 $IMG bash -c \
    "$PIP; python -u slice_quality/examples/run_second_model_cluster.py" || exit 0
  grep -E "KEPT|substrate|errors," $L/model2_cluster.log | head -5
  reground_queue 1
  say "[gpu1] queue done" ) &
Q1=$!

# ---- GPU 2 -----------------------------------------------------------------------------------------
( wait_free judgeclaims_gpu2
  say "[gpu2] precision of the shipped phrases"
  gpu_run 2 night_prec_gpu2 $L/precision.log $IMG bash -c \
    "$PIP; python -u slice_quality/examples/eval_description_precision.py" || exit 0
  tail -n 6 $L/precision.log
  reground_queue 2
  say "[gpu2] queue done" ) &
Q2=$!

# ---- GPU 6 -----------------------------------------------------------------------------------------
( wait_free judgeclaims_gpu6
  for f in judge_cause judge_cause_val; do
    for e in json txt; do [ -f $D/${f}_slope.$e ] || cp $D/$f.$e $D/${f}_slope.$e; done
  done
  say "[gpu6] plausibility of the causes, annotated images"
  gpu_run 6 night_cause_gpu6 $L/cause.log -e JUDGE_4BIT=1 $IMG bash -c \
    "$PIP; python -u slice_quality/examples/judge_cause_plausibility.py" || exit 0
  tail -n 4 $L/cause.log
  say "[gpu6] plausibility of the causes, every image with one"
  gpu_run 6 night_causeval_gpu6 $L/cause_val.log -e JUDGE_4BIT=1 -e ALL=1 -e CONTROL_N=150 $IMG bash -c \
    "$PIP; python -u slice_quality/examples/judge_cause_plausibility.py" || exit 0
  tail -n 4 $L/cause_val.log
  for pct in 25 50; do
    say "[gpu6] corruption experiment, small chairs at the ${pct}th percentile"
    gpu_run 6 night_corr${pct}_gpu6 $L/corruption_p$pct.log -e TAG_MODE=verify -e CORRUPTION_TAG_HOSTS=1 \
      -e RULE_A_AREA_PCT=$pct -e CORRUPTION_TAG=_p$pct $IMG bash -c \
      "$PIP; python -u slice_quality/examples/corruption_experiment.py" || exit 0
    grep -A3 "=== RULE A" $L/corruption_p$pct.log; grep -A2 "=== RULE B" $L/corruption_p$pct.log
  done
  reground_queue 6
  say "[gpu6] queue done" ) &
Q6=$!

wait $CPU_PID $Q1 $Q2 $Q6

# ---- CPU: the second segmenter's two-branch result under the AUC verdict -----------------------------
ALLDONE=1
for c in $RG_CLASSES; do [ -f $RG/.done_$c ] || ALLDONE=0; done
if [ $ALLDONE = 1 ] && [ -f $D/model2/result_tagged.pkl ]; then
  say "[cpu] AUC verdict on model 2's groundings, then its two-branch result"
  cpu_run -e RUNS_DIR=/data/outputs_model2_inst -e STAGE=full -e OUT=$D/model2/dd_auc_full \
    -e RANK_COPY=/data/outputs_model2_inst_rank $IMG bash -c "$PIP; python -u slice_quality/examples/dd_auc.py" \
    > $L/model2_auc.log 2>&1
  head -n 1 $L/model2_auc.log
  cpu_run -e OUT_TAG=model2 -e LANG_DIR=/data/outputs_model2_inst_rank $IMG bash -c \
    "$PIP; python -u slice_quality/examples/run_second_model_language.py" > $L/model2_language.log 2>&1
  tail -n 8 $L/model2_language.log
else
  say "[cpu] model 2 incomplete (re-grounding or cluster stage missing): two-branch result skipped"
fi

say "[cpu] gallery with the new cause scores"
cpu_run $IMG bash -c "$PIP; python -u slice_quality/examples/regen_gallery.py" > $L/regen_gallery.log 2>&1
tail -n 2 $L/regen_gallery.log
say "CHAIN_NIGHT_INST_DONE"
