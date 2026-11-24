#!/bin/bash
# SliceLens against the pipeline on GESD, and the paired tests of the comparison (Daniele, 2026-10-23: "fai i test
# statistici ... voglio anche un confronto diretto sul dataset GESD", GPUs "2 4 5": Qwen in bf16 runs on 2 and 5 only,
# the 1080 Ti on 4 would need 4-bit and give confidences unlike the other groundings).
#   CPU  slicelens_cmp_stats.py (VOC, SBD, Pascal-EA, judges, simulation, disagreement)
#   CPU  gesd_cmp.py MODE=prep: the hypotheses SliceLens sends to the whole class but never grounded there, and the
#        rank-test copy of the whole-class groundings; the rescore of a GESD copy with the rank test
#   GPU  2 and 5 (guarded, waiting while another user holds them): those hypotheses on the whole class
#   CPU  gesd_cmp.py MODE=slope, the rescore of a GESD copy with the slope rule, gesd_cmp.py MODE=report
#   setsid nohup bash outputs/chain_gesd_cmp.sh > outputs/gesd_cmp.log 2>&1 < /dev/null &
set -u
cd ~/storage/SLICELENS
L=outputs/slice_quality_gesd
G=$HOME/storage/pascal_vlm/gesd
LOGS=$L/cmp_logs
mkdir -p $LOGS $G/outputs_gesd_slopeonly
CLASSES="airplane apple banana bed book carrot clock keyboard orange suitcase surfboard teddy_bear tie umbrella vase"
MOUNTS="-v $HOME/storage/SLICELENS:/workspace -v $G:/data -v $HOME/datasets/private:/priv:ro \
  -v $HOME/datasets/private/coco/val2017_gesd:/coco_img:ro -v $STORAGE/models/qwen2.5-vl-7b:/models/qwen"
MOUNTS_INST="-v $HOME/storage/SLICELENS:/workspace -v $HOME/storage/pascal_vlm:/data"
ENV="-e HOME=/tmp -e PYTHONIOENCODING=utf-8 -e PYTHONPATH=/workspace -e SQ_QWEN_DEVICE_MAP=cuda:0"
PIP="pip install --user -q scipy scikit-learn >/dev/null 2>&1"
IMG=segciani:dev
DEADLINE=$(( $(date +%s) + 20 * 3600 ))
say() { echo "$(date '+%m-%d %H:%M') $*"; }
cpu() { docker run --rm -u $(id -u):$(id -g) $ENV "$@"; }
others() {
  local uuid me pid uid
  uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $1); me=$(id -u)
  for pid in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v u="$uuid" '$1==u {print $2}'); do
    uid=$(ps -o uid= -p "$pid" 2>/dev/null | tr -d ' '); [ -n "$uid" ] && [ "$uid" != "$me" ] && echo $pid
  done; }
rescore() {   # $1 copy dir, $2 LANG_DIRS
  mkdir -p $1
  cp $L/result_tagged_merged.pkl $1/result_tagged.pkl
  [ -f $L/cluster_names.json ] && cp $L/cluster_names.json $1/
  cpu -e SQ_OUT=$1 -e LANG_DIRS=$2 -e CLASSES="$CLASSES" -e GALLERY=0 $MOUNTS -w /workspace $IMG \
    bash -c "$PIP; python -u slice_quality/examples/rescore_with_language.py" > $1/rescore.log 2>&1
  grep -q "DONE rescore" $1/rescore.log || { say "rescore $1 failed"; tail -5 $1/rescore.log; return 1; }
  grep -E "language slices total|images with SliceLens" $1/rescore.log; }

say "== SliceLens against the pipeline: paired tests, and GESD =="
( cpu $MOUNTS_INST -w /workspace $IMG bash -c "$PIP; python -u slice_quality/examples/slicelens_cmp_stats.py" \
    > $LOGS/stats.log 2>&1
  grep -q CMP_STATS_DONE $LOGS/stats.log && say "[cpu] paired tests done" || { say "[cpu] paired tests failed"; tail -5 $LOGS/stats.log; } ) &
QS=$!

say "[cpu] GESD: what SliceLens would ground on the whole class; the rank-test copy"
cpu -e MODE=prep $MOUNTS -w /workspace $IMG bash -c "$PIP; python -u slice_quality/examples/gesd_cmp.py" > $LOGS/prep.log 2>&1
grep -q GESD_CMP_PREP_DONE $LOGS/prep.log || { say "prep failed"; tail -5 $LOGS/prep.log; exit 1; }
cat $LOGS/prep.log | grep -v "^$" | tail -3
TODO=$(python3 -c "import json; print(' '.join(c for c, v in json.load(open('$G/full_slopeonly.json')).items() if v))")
( say "[cpu] rescore of a GESD copy with the rank test"; rescore outputs/slice_quality_gesd_cmp_auc /data/outputs_gesd_cmp_auc ) &
QA=$!

worker() {   # $1 gpu: classes from the shared queue of ground_ea_hypotheses.py, waiting for the GPU, retrying a stop
  local g=$1 name
  while true; do
    [ $(date +%s) -ge $DEADLINE ] && { say "[gpu$g] deadline"; return 1; }
    if [ -n "$(others $g)" ]; then sleep 300; continue; fi
    name=gesd_cmp_gpu${g}_$(date +%H%M)
    setsid nohup bash outputs/gpu_guard.sh $g $name > $LOGS/guard_$name.log 2>&1 < /dev/null &
    CLASSES="$TODO" docker run --rm --name $name --gpus device=$g -u $(id -u):$(id -g) $ENV -e OUT_DIR=/data/outputs_gesd_slopeonly \
      -e FULL_HYPS=/data/full_slopeonly.json -e SUBSTRATE=/data/substrate.pkl -e OLD_SUBSTRATE= -e EDITED= -e CACHE_DIRS= \
      -e SAMPLING=srs -e DATASET=GESD -e CLASSES -e PHASE=full $MOUNTS -w /workspace $IMG \
      bash -c "$PIP; python -u examples/ground_ea_hypotheses.py" > $LOGS/ground_$name.log 2>&1
    sleep 3
    if grep -q GUARD_STOPPED $LOGS/guard_$name.log; then
      say "[gpu$g] guard stopped $name: the claimed class is released, retrying"
      for c in $TODO; do [ -f $G/outputs_gesd_slopeonly/voc_paper_${c}_full.json ] || rmdir $G/outputs_gesd_slopeonly/.claim_full_$c 2>/dev/null; done
      sleep 300; continue
    fi
    grep -q Traceback $LOGS/ground_$name.log && { say "[gpu$g] failed"; tail -3 $LOGS/ground_$name.log; return 1; }
    say "[gpu$g] queue done"; return 0
  done; }
say "[gpu] the hypotheses SliceLens sends to the whole class, classes: $TODO"
worker 2 &
W2=$!
sleep 30
worker 5 &
W5=$!
wait $W2 $W5 $QA
for c in $TODO; do [ -f $G/outputs_gesd_slopeonly/voc_paper_${c}_full.json ] || { say "class $c not grounded: stop"; exit 1; }; done

say "[cpu] the slope-rule copy, its rescore, the report"
cpu -e MODE=slope $MOUNTS -w /workspace $IMG bash -c "$PIP; python -u slice_quality/examples/gesd_cmp.py" > $LOGS/slope.log 2>&1
grep -q GESD_CMP_SLOPE_DONE $LOGS/slope.log || { say "slope copy failed"; tail -5 $LOGS/slope.log; exit 1; }
rescore outputs/slice_quality_gesd_cmp_slope /data/outputs_gesd_cmp_slope || exit 1
cpu -e MODE=report $MOUNTS -w /workspace $IMG bash -c "$PIP; python -u slice_quality/examples/gesd_cmp.py" > $LOGS/report.log 2>&1
grep -q GESD_CMP_DONE $LOGS/report.log && cat $L/gesd_cmp.txt || { say "report failed"; tail -5 $LOGS/report.log; }
wait $QS
cat outputs/slicelens_cmp_stats.txt 2>/dev/null
say "CHAIN_GESD_CMP_DONE"
