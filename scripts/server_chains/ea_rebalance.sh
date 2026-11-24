#!/bin/bash
# val-EA full-population re-grounding: GPU6 must be free by 09:55 UTC, so the long
# class (person, ~2 h) goes to GPU1 and GPU6 takes the rest; at STOP6 GPU6 is stopped
# whatever it is doing (its groundings are saved per hypothesis) and GPU1 finishes.
#   nohup bash outputs/ea_rebalance.sh > outputs/slice_quality_ea/rebalance.log 2>&1 &
set -u
cd ~/storage/SLICELENS
O=$HOME/storage/pascal_vlm/outputs_ea
L=outputs/slice_quality_ea
STOP6=${STOP6:-09:50}
MOUNTS="-v $HOME/storage/SLICELENS:/workspace -v $HOME/storage/pascal_vlm:/data \
  -v $STORAGE/models/qwen2.5-vl-7b:/models/qwen"
ENV="-e HOME=/tmp -e PYTHONIOENCODING=utf-8 -e PYTHONPATH=/workspace -e PHASE=full"
say() { echo "$(date -u '+%H:%M') $1"; }
unfinished() { for d in $O/.claim_full_*; do c=${d##*.claim_full_}; \
  [ -f $O/voc_paper_${c}_full.json ] || [ -f $O/.none_full_$c ] || echo $c; done; }
gpu1_worker() {   # $1 = name suffix, $2 = classes
  docker run --rm --name ea_full_gpu1$1 --gpus device=1 -u $(id -u):$(id -g) $ENV -e CLASSES="$2" $MOUNTS \
    -w /workspace segciani:dev bash -c "pip install --user -q scipy scikit-learn >/dev/null 2>&1; \
    python -u examples/ground_ea_hypotheses.py" >> $L/lang_full_gpu1$1.log 2>&1
}

mkdir $O/.claim_full_person 2>/dev/null && say "person reserved for GPU1"
until [ -f $O/voc_paper_cow_full.json ]; do sleep 15; done
sleep 10
say "cow done: stopping the GPU1 queue worker"
docker stop ea_full_gpu1 >/dev/null
took=$(grep -oE "^=== [0-9:]+ \[[a-z]+\]" $L/lang_full_gpu1.log | tail -1 | grep -oE "\[[a-z]+\]" | tr -d "[]")
if [ -n "$took" ] && [ "$took" != cow ] && [ ! -f $O/voc_paper_${took}_full.json ]; then
  rmdir $O/.claim_full_$took && say "released $took (GPU1 had just claimed it)"
fi
rmdir $O/.claim_full_person && say "person: launching on GPU1"
( gpu1_worker b person; say "person done on GPU1"
  gpu1_worker c "aeroplane bicycle bird boat bottle bus car cat chair cow diningtable dog horse motorbike person pottedplant sheep sofa train tvmonitor"
  say "GPU1 leftover worker done" ) &

until [ "$(date -u +%H:%M)" \> "$STOP6" ] || [ -z "$(docker ps -q --filter name=ea_full_gpu6)" ]; do sleep 30; done
if [ -n "$(docker ps -q --filter name=ea_full_gpu6)" ]; then
  say "STOP6 reached: stopping GPU6"
  docker stop ea_full_gpu6 >/dev/null
  for c in $(unfinished); do
    [ "$c" = person ] && continue
    rmdir $O/.claim_full_$c && say "released $c for GPU1"
  done
else
  say "GPU6 worker finished before $STOP6"
fi
wait
say "REBALANCE_DONE; unfinished: $(unfinished | tr '\n' ' ')"
