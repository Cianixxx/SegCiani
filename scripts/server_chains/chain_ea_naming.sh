#!/bin/bash
# Per-instance naming on ALL val-EA victims: groundings on GPU 1 and GPU 6 (Daniele, 2026-10-15:
# "vai con 1 e 6, se qualcuno la occupa liberale subito"), each under its own guard
# (outputs/gpu_guard.sh), sharing a class queue; then the CPU measure. If a guard stops a job,
# nothing is restarted.
#   setsid nohup bash outputs/chain_ea_naming.sh > outputs/ea_naming.log 2>&1 < /dev/null &
set -u
cd ~/storage/SLICELENS
MOUNTS="-v $HOME/storage/SLICELENS:/workspace -v $HOME/storage/pascal_vlm:/data \
  -v $STORAGE/models/qwen2.5-vl-7b:/models/qwen -v $DATASETS:/ds"
ENV="-e HOME=/tmp -e PYTHONIOENCODING=utf-8 -e PYTHONPATH=/workspace -e SQ_QWEN_DEVICE_MAP=cuda:0"
O=$HOME/storage/pascal_vlm/outputs_ea_naming
step() { echo "$(date '+%m-%d %H:%M') == $1 =="; }

step "1. groundings of the 'auc' hypotheses on the victims and survivors (GPU 1 + GPU 6, guarded)"
for GPU in 1 6; do
  NAME=ea_naming_gpu$GPU
  setsid nohup bash outputs/gpu_guard.sh $GPU $NAME > outputs/gpu_guard_$GPU.log 2>&1 < /dev/null &
  docker run --rm --name $NAME --gpus device=$GPU -u $(id -u):$(id -g) $ENV -e OUT_DIR=/data/outputs_ea_naming $MOUNTS \
    -w /workspace segciani:dev bash -c "pip install --user -q scipy scikit-learn >/dev/null 2>&1; \
    python -u examples/ground_ea_naming.py" > outputs/slice_quality_ea/ea_naming_gpu$GPU.log 2>&1 &
  sleep 30
done
wait
for GPU in 1 6; do echo "gpu$GPU: $(tail -1 outputs/slice_quality_ea/ea_naming_gpu$GPU.log) | guard: $(tail -1 outputs/gpu_guard_$GPU.log)"; done
left=$(python3 -I -c "
import json, os
p = json.load(open('$O/plan_classes.json'))
print(' '.join(c for c in p if not os.path.exists('$O/.done_' + c)))")
if [ -n "$left" ]; then
  echo "classes not done: $left -- nothing restarted"
  echo "$(date '+%m-%d %H:%M') CHAIN_EA_NAMING_STOPPED"
  exit 0
fi

step "2. the measure, on CPU, with the new groundings"
docker run --rm -u $(id -u):$(id -g) $ENV -e EXTRA=/data/outputs_ea_naming $MOUNTS -w /workspace segciani:dev \
  python -u slice_quality/examples/per_instance_naming_ea.py > outputs/slice_quality_ea/per_instance_naming_all.log 2>&1
sed -n '/ALL VICTIMS/,/examples/p' outputs/slice_quality_ea/per_instance_naming_all.log
echo "$(date '+%m-%d %H:%M') CHAIN_EA_NAMING_DONE"
