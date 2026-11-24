#!/bin/bash
set -u
cd ~/storage/SLICELENS
PD=outputs/slice_quality_pending; L=$PD/logs; RG=$HOME/storage/pascal_vlm/outputs_pending; name=pend_aeroplane_gpu6
setsid nohup bash outputs/gpu_guard.sh 6 $name > $L/guard_$name.log 2>&1 < /dev/null &
docker run --rm --name $name --gpus device=6 -u $(id -u):$(id -g) -e HOME=/tmp -e PYTHONIOENCODING=utf-8 -e PYTHONPATH=/workspace \
  -e SQ_CLAIM_COV=0.5 -e SQ_GT_INSTANCES=object -e SQ_QWEN_DEVICE_MAP=cuda:0 \
  -v $HOME/storage/SLICELENS:/workspace -v $HOME/storage/pascal_vlm:/data -v $DATASETS:/ds \
  -v $STORAGE/models/qwen2.5-vl-7b:/models/qwen -w /workspace segciani:dev bash -c \
  "pip install --user -q scipy scikit-learn >/dev/null 2>&1; python -u examples/reground_validated.py --slices-index $PD/pending_index.json \
   --substrate-cache /data/voc2012/substrate_cc.pkl --vlm-backend qwen --vlm-model /models/qwen --out-dir /data/outputs_pending --classes aeroplane" \
  > $L/reground_aeroplane.log 2>&1
sleep 3
if grep -q GUARD_STOPPED $L/guard_$name.log; then echo "$(date +%H:%M) guard stopped aeroplane on gpu6"; exit 1; fi
grep -q Traceback $L/reground_aeroplane.log && { echo "aeroplane failed"; tail -3 $L/reground_aeroplane.log; exit 1; }
touch $RG/.done_aeroplane; echo "$(date +%H:%M) AEROPLANE_DONE"
