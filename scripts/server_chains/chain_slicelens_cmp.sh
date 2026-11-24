#!/bin/bash
# SliceLens against the thesis pipeline, and the slope rule against the AUC test (Daniele, 2026-10-21: "sì,
# lancia tutto su gpu 1 2, se ti serve usa anche gpu 0"). Every GPU container under outputs/gpu_guard.sh.
#   GPU 1: facts judge, shard 0 (SliceLens-style text vs our narrative) -> precision of the slope run's
#          phrases -> re-grounding of the slope-only situations on model 2 (shared queue)
#   GPU 2: facts judge, shard 1 -> the re-grounding queue
#   GPU 0: (GTX 1080 Ti) rubric judge on the same two texts; if it cannot run there, GPU 1 does it at the end
#   CPU:   simulation slope vs AUC on planted effects; Pascal-EA victims by branch; at the end the merge of
#          the facts judge and the disagreement report
#   setsid nohup bash outputs/chain_slicelens_cmp.sh > outputs/slicelens_cmp.log 2>&1 < /dev/null &
set -u
cd ~/storage/SLICELENS
D=outputs/slice_quality_inst
C=outputs/slice_quality_slcmp
L=$C/logs
P=$HOME/storage/pascal_vlm
RG=$P/outputs_model2_slopeonly
mkdir -p $L $C/synth $RG
MOUNTS="-v $HOME/storage/SLICELENS:/workspace -v $P:/data -v $DATASETS:/ds"
MODELS="-v $STORAGE/models/qwen2.5-vl-7b:/models/qwen -v $STORAGE/models/gemma-4-12B-it:/models/gemma"
ENV="-e HOME=/tmp -e PYTHONIOENCODING=utf-8 -e PYTHONPATH=/workspace -e SQ_CLAIM_COV=0.5 -e SQ_GT_INSTANCES=object \
  -e SQ_QWEN_DEVICE_MAP=cuda:0 -e SQ_OUT=$D"
PIP="pip install --user -q scipy scikit-learn >/dev/null 2>&1"
IMG=segciani:dev
say() { echo "$(date '+%m-%d %H:%M') $*"; }
others() {
  local uuid me pid uid
  uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $1); me=$(id -u)
  for pid in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v u="$uuid" '$1==u {print $2}'); do
    uid=$(ps -o uid= -p "$pid" 2>/dev/null | tr -d ' '); [ -n "$uid" ] && [ "$uid" != "$me" ] && echo $pid
  done; }
gpu_run() {   # $1 gpu, $2 container name, $3 log, rest: docker args after the mounts (env, image, command)
  local g=$1 name=$2 log=$3; shift 3
  if [ -n "$(others $g)" ]; then say "GPU $g used by another user: $name not started, queue ends"; return 1; fi
  setsid nohup bash outputs/gpu_guard.sh $g $name > $L/guard_$name.log 2>&1 < /dev/null &
  docker run --rm --name $name --gpus device=$g -u $(id -u):$(id -g) $ENV $MOUNTS $MODELS -w /workspace "$@" > $log 2>&1
  sleep 3
  if grep -q GUARD_STOPPED $L/guard_$name.log; then say "guard stopped $name, queue ends"; return 1; fi
  return 0; }
cpu_run() { docker run --rm -u $(id -u):$(id -g) $ENV $MOUNTS -w /workspace "$@"; }
wait_free() { while docker ps --format '{{.Names}}' | grep -qx "$1"; do sleep 60; done; }
RG_CLASSES="person diningtable tvmonitor dog boat bus cat motorbike"
reground_queue() {   # $1 gpu: the slope-only situations on model 2's objects (shared class queue)
  local c
  for c in $RG_CLASSES; do
    [ -f $RG/.done_$c ] && continue
    mkdir $RG/.claim_$c 2>/dev/null || continue
    say "[gpu$1] re-grounding slope-only situations of $c on model 2"
    gpu_run $1 cmp_rg_${c}_gpu$1 $L/reground_$c.log $IMG bash -c "$PIP; python -u examples/reground_validated.py \
      --slices-index $C/slices_index_slopeonly.json --substrate-cache $D/model2/substrate.pkl --vlm-backend qwen \
      --vlm-model /models/qwen --out-dir /data/outputs_model2_slopeonly --classes $c" || return 1
    grep -q "Traceback" $L/reground_$c.log && { say "re-grounding $c failed"; tail -3 $L/reground_$c.log; return 1; }
    touch $RG/.done_$c
  done; }
JUDGE_ENV="-e SQ_OUT=$C -e KINDS=narrative,pattern -e OLD_DUP=0 -e JUDGE_4BIT=1 -e NSHARDS=2"
RUBRIC_ENV="-e JUDGE_4BIT=1 -e NARR=$D/error_narratives.json -e DESC=$C/image_descriptions.json \
  -e DESC_TR=/nonexistent -e OUT=$C/judge_descriptions"

say "== SliceLens comparison and slope vs AUC =="
# ---- set-up: our narrative beside the SliceLens-style text; the facts of the human notes reused as they are
cp $D/error_narratives.json $C/error_narratives.json
echo '{}' > $C/synth/narratives_synth.json; echo '{}' > $C/synth/descriptions_synth.json
cpu_run $IMG python -c "
import json
F = {}
for k in (0, 1):
    F.update(json.load(open('$D/judge_claims_facts_shard%d.json' % k, encoding='utf-8'))['facts'])
for k in (0, 1):
    json.dump({'facts': F, 'complete': True}, open('$C/judge_claims_facts_shard%d.json' % k, 'w', encoding='utf-8'),
              indent=1, ensure_ascii=False)
A = json.load(open('$D/slices_index.json', encoding='utf-8'))['slices']
S = json.load(open('outputs/slice_quality_inst_slope/slices_index.json', encoding='utf-8'))['slices']
key = lambda s: (s['slice_id'].rsplit('_', 1)[0].removesuffix('_extra').removesuffix('_dd'), s['hypothesis'])
lang = lambda xs: [s for s in xs if s['source'] == 'language' and s.get('in_descriptions')]
ka = {key(s) for s in lang(A)}
only = [s for s in lang(S) if key(s) not in ka]
json.dump({'slices': only}, open('$C/slices_index_slopeonly.json', 'w'), indent=1)
print('facts', len(F), '| slope-only situations', len(only), sorted({key(s)[0] for s in only}))
" || { say "set-up failed"; exit 1; }

# ---- CPU, at once
( say "[cpu] simulation slope vs AUC"
  cpu_run -e WORKERS=24 -e OUT=outputs/slope_vs_auc_sim $IMG bash -c "$PIP; python -u slice_quality/examples/slope_vs_auc_sim.py" \
    > $L/sim.log 2>&1
  grep -q SLOPE_VS_AUC_SIM_DONE $L/sim.log && say "[cpu] simulation done" || { say "simulation failed"; tail -3 $L/sim.log; }
  cpu_run $IMG bash -c "$PIP; python -u slice_quality/examples/ea_branch_coverage.py" > $L/ea_branch.log 2>&1
  cat $L/ea_branch.log | head -5 ) &
CPU_PID=$!

# ---- GPU 0: rubric judge
( say "[gpu0] rubric judge"
  gpu_run 0 cmp_rubric_gpu0 $L/rubric.log $RUBRIC_ENV $IMG bash -c "$PIP; python -u slice_quality/examples/judge_descriptions.py"
  if grep -q "Traceback" $L/rubric.log || ! [ -s $C/judge_descriptions.json ]; then
    say "[gpu0] rubric judge did not run on the 1080 Ti: GPU 1 will do it"; touch $C/.rubric_on_gpu1
  else say "[gpu0] rubric judge done"; fi ) &
Q0=$!

# ---- GPU 1
( say "[gpu1] facts judge, shard 0"
  gpu_run 1 cmp_claims_gpu1 $L/claims0.log -e SHARD=0 $JUDGE_ENV $IMG bash -c \
    "$PIP; python -u slice_quality/examples/judge_claims.py" || exit 0
  say "[gpu1] precision of the slope run's phrases"
  gpu_run 1 cmp_prec_gpu1 $L/precision_slope.log -e SQ_OUT=outputs/slice_quality_inst_slope $IMG bash -c \
    "$PIP; python -u slice_quality/examples/eval_description_precision.py" || exit 0
  tail -n 4 $L/precision_slope.log
  reground_queue 1
  wait_free cmp_rubric_gpu0
  if [ -f $C/.rubric_on_gpu1 ]; then
    say "[gpu1] rubric judge"
    gpu_run 1 cmp_rubric_gpu1 $L/rubric.log $RUBRIC_ENV $IMG bash -c "$PIP; python -u slice_quality/examples/judge_descriptions.py"
  fi
  say "[gpu1] queue done" ) &
Q1=$!

# ---- GPU 2
( say "[gpu2] facts judge, shard 1"
  gpu_run 2 cmp_claims_gpu2 $L/claims1.log -e SHARD=1 $JUDGE_ENV $IMG bash -c \
    "$PIP; python -u slice_quality/examples/judge_claims.py" || exit 0
  reground_queue 2
  say "[gpu2] queue done" ) &
Q2=$!

wait $Q1 $Q2
say "[cpu] facts judge, merge"
cpu_run -e MERGE=1 $JUDGE_ENV $IMG bash -c "$PIP; python -u slice_quality/examples/judge_claims.py" > $L/claims_merge.log 2>&1
grep -iE "narrative|pattern" $L/claims_merge.log | head -12
ALLDONE=1
for c in $RG_CLASSES; do [ -f $RG/.done_$c ] || ALLDONE=0; done
[ $ALLDONE = 1 ] || say "[cpu] model 2 re-grounding incomplete: the report shows what is there"
cpu_run $IMG bash -c "$PIP; python -u slice_quality/examples/disagreement_report.py" > $L/disagreement.log 2>&1
grep -E "^[A-Z ]+:|precision|model 2" $L/disagreement.log | head -12
wait $CPU_PID $Q0
say "CHAIN_SLICELENS_CMP_DONE"
