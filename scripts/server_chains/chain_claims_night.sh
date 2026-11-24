#!/bin/bash
# The facts score (judge_claims.py, Gemma 4-bit) re-run on the CURRENT narratives (v6) of the inst run: the
# Gemma figure in the thesis was measured on the previous text for 53 images (Daniele, 2026-10-21: "fai 2").
# The narratives do not depend on the language verdict, so this runs beside chain_auc_official.sh. One shard
# per GPU, every container under outputs/gpu_guard.sh (a stopped container ends that shard; nothing is
# restarted), then the merge on CPU. The previous results are kept as *_v2gemma.
#   setsid nohup bash outputs/chain_claims_night.sh "2 6" > outputs/claims_night.log 2>&1 < /dev/null &
set -u
GPUS=($1)
N=${#GPUS[@]}
cd ~/storage/SLICELENS
D=outputs/slice_quality_inst
MOUNTS="-v $HOME/storage/SLICELENS:/workspace -v $HOME/storage/pascal_vlm:/data \
  -v $STORAGE/models/gemma-4-12B-it:/models/gemma"
ENV="-e HOME=/tmp -e PYTHONIOENCODING=utf-8 -e PYTHONPATH=/workspace -e SQ_OUT=$D -e NSHARDS=$N -e JUDGE_4BIT=1"
step() { echo "$(date '+%m-%d %H:%M') == $1 =="; }

step "0. keep the previous facts score"
for f in judge_claims.json judge_claims.txt; do
  [ -f $D/${f%.*}_v2gemma.${f##*.} ] || cp $D/$f $D/${f%.*}_v2gemma.${f##*.}
done

step "1. shards on GPU ${GPUS[*]}"
PIDS=""
for k in $(seq 0 $((N - 1))); do
  g=${GPUS[$k]}
  (
    UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $g); ME=$(id -u)
    for pid in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v u="$UUID" '$1==u {print $2}'); do
      uid=$(ps -o uid= -p "$pid" 2>/dev/null | tr -d ' ')
      [ -n "$uid" ] && [ "$uid" != "$ME" ] && { echo "GPU $g used by another user: shard $k not started"; exit 0; }
    done
    NAME=judgeclaims_gpu$g
    setsid nohup bash outputs/gpu_guard.sh $g $NAME > $D/guard_$NAME.log 2>&1 < /dev/null &
    docker run --rm --name $NAME --gpus device=$g -u $(id -u):$(id -g) $ENV -e SHARD=$k $MOUNTS -w /workspace segciani:dev \
      bash -c "pip install --user -q scipy >/dev/null 2>&1; python -u slice_quality/examples/judge_claims.py" \
      > $D/judge_claims_shard$k.log 2>&1
    grep -q GUARD_STOPPED $D/guard_$NAME.log && echo "guard stopped $NAME"
  ) &
  PIDS="$PIDS $!"
  sleep 20
done
wait $PIDS
for k in $(seq 0 $((N - 1))); do
  grep -q "CLAIMS_SHARD_DONE $k" $D/judge_claims_shard$k.log \
    || { echo "FAIL shard $k"; tail -5 $D/judge_claims_shard$k.log; echo "CHAIN_CLAIMS_NIGHT_STOPPED"; exit 0; }
done

step "2. merge and report (CPU)"
docker run --rm -u $(id -u):$(id -g) $ENV -e MERGE=1 $MOUNTS -w /workspace segciani:dev \
  bash -c "pip install --user -q scipy >/dev/null 2>&1; python -u slice_quality/examples/judge_claims.py" \
  > $D/judge_claims_merge.log 2>&1 || { echo "FAIL merge"; tail -5 $D/judge_claims_merge.log; exit 1; }
tail -n 15 $D/judge_claims.txt
echo "$(date '+%m-%d %H:%M') CHAIN_CLAIMS_NIGHT_DONE"
