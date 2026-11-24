#!/bin/bash
# SBD run, phase 1 (data): layout (CPU) -> segmenter predictions (GPU5) -> language
# substrate (CPU) in parallel with BLOCK 1 + DINOv2 of the cluster branch (GPU6).
#   setsid nohup bash outputs/chain_sbd_prep.sh > outputs/slice_quality_sbd/prep_chain.log 2>&1 &
set -u
cd ~/storage/SLICELENS
L=outputs/slice_quality_sbd
mkdir -p $L
MOUNTS="-v $HOME/storage/SLICELENS:/workspace -v $HOME/storage/pascal_vlm:/data \
  -v $DATASETS:/ds -v $STORAGE/models/qwen2.5-vl-7b:/models/qwen"
ENV="-e HOME=/tmp -e PYTHONIOENCODING=utf-8 -e PYTHONPATH=/workspace"
PIP="pip install --user -q scipy scikit-learn >/dev/null 2>&1"
step() { echo "$(date '+%m-%d %H:%M') == $1 =="; }
ok() { grep -q "BUILD_SBD_DONE $1" $L/$2 || { echo "FAIL $1"; tail -5 $L/$2; exit 1; }; }

step "1. layout SBD -> /data/sbd (CPU)"
docker run --rm --name sbd_layout -u $(id -u):$(id -g) $ENV -e STEP=layout $MOUNTS -w /workspace segciani:dev \
  bash -c "$PIP; python -u slice_quality/examples/build_sbd.py" > $L/layout.log 2>&1
ok layout layout.log; tail -2 $L/layout.log

step "2. segmenter predictions (GPU5)"
docker run --rm --name sbd_predict --gpus device=5 -u $(id -u):$(id -g) $ENV -e STEP=predict $MOUNTS -w /workspace \
  segciani:dev bash -c "$PIP; python -u slice_quality/examples/build_sbd.py" > $L/predict.log 2>&1
ok predict predict.log; tail -1 $L/predict.log

step "3. language substrate (CPU) || BLOCK 1 inst config + DINOv2 (GPU6)"
docker run -d --rm --name sbd_substrate -u $(id -u):$(id -g) $ENV -e STEP=substrate $MOUNTS -w /workspace \
  segciani:dev bash -c "$PIP; python -u slice_quality/examples/build_sbd.py > $L/substrate.log 2>&1"
until [ -z "$(docker ps -q --filter name=sbd_bench)" ]; do sleep 30; done      # GPU6 free of the benchmark
docker run --rm --name sbd_prep --gpus device=6 -u $(id -u):$(id -g) $ENV -e STEP=prep -e SQ_CLAIM_COV=0.5 \
  -e SQ_GT_INSTANCES=object $MOUNTS -w /workspace segciani:dev \
  bash -c "$PIP; python -u slice_quality/examples/build_sbd.py" > $L/prep.log 2>&1
ok prep prep.log; tail -1 $L/prep.log
until [ -z "$(docker ps -q --filter name=sbd_substrate)" ]; do sleep 30; done
ok substrate substrate.log; tail -1 $L/substrate.log
echo "$(date '+%m-%d %H:%M') CHAIN_SBD_PREP_DONE"
