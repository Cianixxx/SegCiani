#!/bin/bash
# Data-driven hypotheses (SliceLens Sec. 3.1 part 2) for the 16 VOC classes without them,
# on VOC val: one worker per GPU (2 and 6, authorized by Daniele 2026-10-14), shared queue.
#   setsid nohup bash outputs/chain_dd_gen.sh > outputs/dd_gen.log 2>&1 &
set -u
cd ~/storage/SLICELENS
MOUNTS="-v $HOME/storage/SLICELENS:/workspace -v $HOME/storage/pascal_vlm:/data \
  -v $STORAGE/models/qwen2.5-vl-7b:/models/qwen -v $STORAGE/models/gemma-4-12B-it:/models/gemma"
ENV="-e HOME=/tmp -e PYTHONIOENCODING=utf-8 -e PYTHONPATH=/workspace -e SQ_QWEN_DEVICE_MAP=cuda:0 -e OUT_DIR=/data/outputs_dd"
for g in 2 6; do
  docker run --rm --name dd_gen_gpu$g --gpus device=$g -u $(id -u):$(id -g) $ENV $MOUNTS -w /workspace segciani:dev \
    bash -c "pip install --user -q scipy scikit-learn >/dev/null 2>&1; python -u examples/gen_datadriven.py" \
    > outputs/dd_gen_gpu$g.log 2>&1 &
  sleep 20
done
wait
grep -h "=== done\|still missing" outputs/dd_gen_gpu*.log
echo "$(date '+%m-%d %H:%M') CHAIN_DD_GEN_DONE"
