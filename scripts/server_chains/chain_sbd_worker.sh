#!/bin/bash
# SBD run, one GPU worker, the thesis protocol: Qwen verify tags of the cluster branch,
# then the language branch on the 120-instance sample of every class, then -- once
# chain_sbd.sh has rescored and written the trigger -- the full-population
# re-grounding of the validated hypotheses. Every step claims work from a shared queue.
#   bash outputs/chain_sbd_worker.sh <gpu>
# After a crash: remove the .claim* dirs that have no output, then relaunch.
set -u
GPU=$1
cd ~/storage/SLICELENS
L=outputs/slice_quality_sbd
MOUNTS="-v $HOME/storage/SLICELENS:/workspace -v $HOME/storage/pascal_vlm:/data \
  -v $DATASETS:/ds -v $STORAGE/models/qwen2.5-vl-7b:/models/qwen"
ENV="-e HOME=/tmp -e PYTHONIOENCODING=utf-8 -e PYTHONPATH=/workspace -e SQ_QWEN_DEVICE_MAP=cuda:0"
LANG="-e OUT_DIR=/data/outputs_sbd -e SUBSTRATE=/data/sbd/substrate_cc.pkl -e OLD_SUBSTRATE= -e EDITED= \
  -e CACHE_DIRS= -e SAMPLING=srs -e DATASET=SBD -e SLICES_INDEX=$L/slices_index.json"
PIP="pip install --user -q scipy scikit-learn >/dev/null 2>&1"
step() { echo "$(date '+%m-%d %H:%M') [gpu$GPU] == $1 =="; }

until grep -q CHAIN_SBD_PREP_DONE $L/prep_chain.log 2>/dev/null; do sleep 60; done
step "cluster branch: Qwen verify tags of the errors (queue)"
docker run --rm --name sbd_tag_gpu$GPU --gpus device=$GPU -u $(id -u):$(id -g) $ENV -e STEP=tag $MOUNTS \
  -w /workspace segciani:dev bash -c "$PIP; python -u slice_quality/examples/build_sbd.py" \
  > $L/tag_gpu$GPU.log 2>&1
tail -2 $L/tag_gpu$GPU.log

step "language branch, 120-instance sample, all classes (queue)"
docker run --rm --name sbd_lang_gpu$GPU --gpus device=$GPU -u $(id -u):$(id -g) $ENV $LANG -e PHASE=sample $MOUNTS \
  -w /workspace segciani:dev bash -c "$PIP; python -u examples/ground_ea_hypotheses.py" \
  > $L/lang_sample_gpu$GPU.log 2>&1
tail -2 $L/lang_sample_gpu$GPU.log

step "waiting for the rescore that selects the validated hypotheses"
until [ -f $L/.phase_full_ready ]; do sleep 120; done
step "language branch, full-population re-grounding of the validated hypotheses (queue)"
docker run --rm --name sbd_full_gpu$GPU --gpus device=$GPU -u $(id -u):$(id -g) $ENV $LANG -e PHASE=full $MOUNTS \
  -w /workspace segciani:dev bash -c "$PIP; python -u examples/ground_ea_hypotheses.py" \
  > $L/lang_full_gpu$GPU.log 2>&1
tail -2 $L/lang_full_gpu$GPU.log
step "worker done"
