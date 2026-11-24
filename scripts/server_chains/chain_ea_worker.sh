#!/bin/bash
# val-EA language-branch worker on ONE GPU (host-level launcher). Claims classes from
# the shared queue in /data/outputs_ea: first the 120-instance sample phase, then --
# once chain_ea.sh has rescored and written the trigger -- the full-population
# re-grounding of the validated hypotheses.
#   bash outputs/chain_ea_worker.sh <gpu>
set -u
GPU=$1
cd ~/storage/SLICELENS
L=outputs/slice_quality_ea
MOUNTS="-v $HOME/storage/SLICELENS:/workspace -v $HOME/storage/pascal_vlm:/data \
  -v $DATASETS:/ds -v $STORAGE/models/qwen2.5-vl-7b:/models/qwen"
ENV="-e HOME=/tmp -e PYTHONIOENCODING=utf-8 -e PYTHONPATH=/workspace"
step() { echo "$(date '+%m-%d %H:%M') [gpu$GPU] == $1 =="; }

until grep -q BUILD_VOC_EA_DONE $HOME/storage/pascal_vlm/voc2012_ea/build.log 2>/dev/null; do sleep 60; done
step "language branch, 120-instance sample, all classes (queue)"
docker run --rm --name ea_lang_gpu$GPU --gpus device=$GPU -u $(id -u):$(id -g) $ENV -e PHASE=sample $MOUNTS \
  -w /workspace segciani:dev bash -c "pip install --user -q scipy scikit-learn >/dev/null 2>&1; \
  python -u examples/ground_ea_hypotheses.py" > $L/lang_sample_gpu$GPU.log 2>&1
tail -2 $L/lang_sample_gpu$GPU.log

step "waiting for the rescore that selects the validated hypotheses"
until [ -f $L/.phase_full_ready ]; do sleep 120; done
step "language branch, full-population re-grounding of the validated hypotheses (queue)"
docker run --rm --name ea_full_gpu$GPU --gpus device=$GPU -u $(id -u):$(id -g) $ENV -e PHASE=full $MOUNTS \
  -w /workspace segciani:dev bash -c "pip install --user -q scipy scikit-learn >/dev/null 2>&1; \
  python -u examples/ground_ea_hypotheses.py" > $L/lang_full_gpu$GPU.log 2>&1
tail -2 $L/lang_full_gpu$GPU.log
step "worker done"
