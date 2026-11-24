#!/bin/bash
# SBD full phase: the person class split in 24 blocks over GPU5 and GPU6 (Daniele, 10-13).
# person is reserved (.claim_full_person) so the class workers skip it; each GPU takes
# person blocks as soon as its class worker is done; then the blocks are merged into
# groundings_person.json, the reservation is released, and ground_ea_hypotheses.py writes
# voc_paper_person_full.json from the cache on CPU (no Qwen mounted: a miss would fail).
#   setsid nohup bash outputs/sbd_person_split.sh > outputs/slice_quality_sbd/person_split.log 2>&1 &
set -u
cd ~/storage/SLICELENS
L=outputs/slice_quality_sbd
O=$HOME/storage/pascal_vlm/outputs_sbd
MOUNTS="-v $HOME/storage/SLICELENS:/workspace -v $HOME/storage/pascal_vlm:/data -v $DATASETS:/ds"
QWEN="-v $STORAGE/models/qwen2.5-vl-7b:/models/qwen"
ENV="-e HOME=/tmp -e PYTHONIOENCODING=utf-8 -e PYTHONPATH=/workspace -e SQ_QWEN_DEVICE_MAP=cuda:0"
LANG="-e OUT_DIR=/data/outputs_sbd -e SUBSTRATE=/data/sbd/substrate_cc.pkl -e OLD_SUBSTRATE= -e EDITED= \
  -e CACHE_DIRS= -e SAMPLING=srs -e DATASET=SBD -e SLICES_INDEX=$L/slices_index.json -e CLS=person"
PIP="pip install --user -q scipy scikit-learn >/dev/null 2>&1"
step() { echo "$(date '+%m-%d %H:%M') == $1 =="; }
split_on() {      # $1 = gpu: wait for its class worker, then take person blocks
  local g=$1
  while docker ps --format '{{.Names}}' | grep -qx sbd_full_gpu$g; do sleep 120; done
  step "[gpu$g] class worker done, person blocks"
  docker run --rm --name sbd_split_gpu$g --gpus device=$g -u $(id -u):$(id -g) $ENV $LANG -e PHASE=ground \
    $MOUNTS $QWEN -w /workspace segciani:dev bash -c "$PIP; python -u examples/ground_split_class.py" \
    > $L/split_person_gpu$g.log 2>&1
  step "[gpu$g] $(tail -1 $L/split_person_gpu$g.log)"
}

step "person reserved: $(ls -d $O/.claim_full_person 2>/dev/null)"
split_on 5 &
split_on 6 &
wait
n=$(ls $O/split/ 2>/dev/null | grep -c "^person_.*json$")
step "person blocks on disk: $n/24"
[ "$n" -eq 24 ] || { echo "FAIL split: blocks missing"; exit 1; }

step "merge the blocks into groundings_person.json (CPU)"
docker run --rm -u $(id -u):$(id -g) $ENV $LANG -e PHASE=merge $MOUNTS -w /workspace segciani:dev \
  bash -c "$PIP; python -u examples/ground_split_class.py" > $L/split_person_merge.log 2>&1
grep -q SPLIT_MERGE_DONE $L/split_person_merge.log || { echo "FAIL merge"; tail -5 $L/split_person_merge.log; exit 1; }
tail -2 $L/split_person_merge.log

step "voc_paper_person_full.json from the cache (CPU, no Qwen)"
rmdir $O/.claim_full_person
docker run --rm -u $(id -u):$(id -g) $ENV $LANG -e PHASE=full -e CLASSES=person $MOUNTS -w /workspace segciani:dev \
  bash -c "$PIP; python -u examples/ground_ea_hypotheses.py" > $L/split_person_assemble.log 2>&1
[ -f $O/voc_paper_person_full.json ] || { echo "FAIL assemble"; tail -5 $L/split_person_assemble.log; exit 1; }
grep -E "=== done person" $L/split_person_assemble.log
echo "$(date '+%m-%d %H:%M') PERSON_SPLIT_DONE"
