#!/bin/bash
# Voto per fatti (judge_claims.py): due worker in parallelo, uno per GPU, poi
# l'unione e il rapporto su CPU. Lanciatore lato host.
#   GPUS="1 6" bash outputs/run_judge_claims.sh
set -u
GPUS=(${GPUS:-1 6})
N=${#GPUS[@]}
cd ~/storage/SLICELENS
step() { echo "$(date +%H:%M) == $1 =="; }
MOUNTS="-v $HOME/storage/SLICELENS:/workspace -v $HOME/storage/pascal_vlm:/data \
  -v $STORAGE/models/gemma-4-12B-it:/models/gemma"
ENV="-e HOME=/tmp -e PYTHONIOENCODING=utf-8 -e PYTHONPATH=/workspace -e SQ_OUT=outputs/slice_quality_inst \
  -e NSHARDS=$N -e JUDGE_4BIT=1"

step "worker su GPU ${GPUS[*]}"
for k in $(seq 0 $((N - 1))); do
  g=${GPUS[$k]}
  docker run -d --rm --name judgeclaims_gpu$g --gpus device="$g" -u $(id -u):$(id -g) $ENV -e SHARD=$k \
    $MOUNTS -w /workspace segciani:dev \
    bash -c "pip install --user -q scipy >/dev/null 2>&1; python -u slice_quality/examples/judge_claims.py \
      > outputs/slice_quality_inst/judge_claims_shard$k.log 2>&1"
done
sleep 30
until [ -z "$(docker ps -q --filter name=judgeclaims_gpu)" ]; do sleep 60; done
for k in $(seq 0 $((N - 1))); do
  grep -q "CLAIMS_SHARD_DONE $k" outputs/slice_quality_inst/judge_claims_shard$k.log \
    || { echo "FAIL shard $k"; tail -5 outputs/slice_quality_inst/judge_claims_shard$k.log; exit 1; }
done
echo "--- GPU:"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed -n "$((${GPUS[0]} + 1))p;$((${GPUS[1]} + 1))p"

step "unione e rapporto (CPU)"
docker run --rm -u $(id -u):$(id -g) $ENV -e MERGE=1 $MOUNTS -w /workspace segciani:dev \
  bash -c "pip install --user -q scipy >/dev/null 2>&1; python -u slice_quality/examples/judge_claims.py" \
  || { echo "FAIL merge"; exit 1; }
echo "$(date +%H:%M) JUDGE_CLAIMS_DONE"
