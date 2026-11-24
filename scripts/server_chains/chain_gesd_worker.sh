#!/bin/bash
# GESD run, one GPU worker, the thesis protocol (as chain_sbd_worker.sh): [lead only] Gemma's
# knowledge-driven hypotheses of the 15 classes -> Qwen verify tags of the errors (queue) ->
# data-driven hypotheses (queue) -> [after chain_gesd.sh has written the hypothesis files]
# the language branch on the 120-instance sample of every class (queue). The full-population
# re-grounding is NOT here: it is asked for after the sample results.
# Every container runs under outputs/gpu_guard.sh: if another user starts on this GPU the guard
# stops the container and this worker ends -- nothing is restarted.
#   bash outputs/chain_gesd_worker.sh <gpu> [lead]
set -u
GPU=$1
LEAD=${2:-}
cd ~/storage/SLICELENS
L=outputs/slice_quality_gesd
MOUNTS="-v $HOME/storage/SLICELENS:/workspace -v $HOME/storage/pascal_vlm/gesd:/data \
  -v $HOME/datasets/private:/priv:ro -v $HOME/datasets/private/coco/val2017_gesd:/coco_img:ro \
  -v $STORAGE/models/qwen2.5-vl-7b:/models/qwen -v $STORAGE/models/gemma-4-12B-it:/models/gemma"
ENV="-e HOME=/tmp -e PYTHONIOENCODING=utf-8 -e PYTHONPATH=/workspace -e SQ_QWEN_DEVICE_MAP=cuda:0"
CLASSES="airplane apple banana bed book carrot clock keyboard orange suitcase surfboard teddy_bear tie umbrella vase"
LANG="-e OUT_DIR=/data/outputs_gesd -e SUBSTRATE=/data/substrate.pkl -e OLD_SUBSTRATE= -e EDITED= \
  -e CACHE_DIRS= -e SAMPLING=srs -e DATASET=GESD -e CLASSES"
PIP="pip install --user -q scipy scikit-learn >/dev/null 2>&1"
ME=$(id -u)
UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU")
step() { echo "$(date '+%m-%d %H:%M') [gpu$GPU] == $1 =="; }
others() {      # processes of other users on this GPU
  for pid in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v u="$UUID" '$1==u {print $2}'); do
    uid=$(ps -o uid= -p "$pid" 2>/dev/null | tr -d ' ')
    [ -n "$uid" ] && [ "$uid" != "$ME" ] && echo "$pid"
  done
}
guarded() {     # $1 container name, $2 log, rest: image + command
  local NAME=$1 LOG=$2
  shift 2
  if [ -n "$(others)" ]; then echo "GPU $GPU used by another user before $NAME: worker ends"; exit 0; fi
  setsid nohup bash outputs/gpu_guard.sh $GPU $NAME > $L/guard_$NAME.log 2>&1 < /dev/null &
  CLASSES="$CLASSES" docker run --rm --name $NAME --gpus device=$GPU -u $(id -u):$(id -g) $ENV $MOUNTS -w /workspace "$@" > $LOG 2>&1
  sleep 3
  if grep -q GUARD_STOPPED $L/guard_$NAME.log; then echo "guard stopped $NAME: worker ends, nothing restarted"; exit 0; fi
  tail -2 $LOG
}

until grep -q "BUILD_GESD_DONE.*prep" $L/prep.log 2>/dev/null; do sleep 60; done
if [ -n "$LEAD" ] && [ ! -f $L/.knowledge_done ]; then
  step "knowledge-driven hypotheses, Gemma, 15 classes"
  guarded gesd_know_gpu$GPU $L/knowledge_gpu$GPU.log segciani:dev bash -c "$PIP; mkdir -p /data/knowledge; \
    python -u examples/_archive/gen_hypotheses_batch.py --llm-model /models/gemma --load-in-4bit \
    --out-dir /data/knowledge --classes $CLASSES"
  grep -q HYPOTHESES_BATCH_DONE $L/knowledge_gpu$GPU.log && touch $L/.knowledge_done
fi

step "cluster branch: Qwen verify tags of the errors (queue)"
guarded gesd_tag_gpu$GPU $L/tag_gpu$GPU.log -e STEP=tag segciani:dev \
  bash -c "$PIP; python -u slice_quality/examples/build_gesd.py"

step "data-driven hypotheses (queue)"
guarded gesd_dd_gpu$GPU $L/dd_gpu$GPU.log -e OUT_DIR=/data/outputs_dd -e SUBSTRATE=/data/substrate.pkl -e CLASSES \
  segciani:dev bash -c "$PIP; python -u examples/gen_datadriven.py"

step "waiting for the hypothesis files (chain_gesd.sh)"
until [ -f $L/.hyp_ready ]; do sleep 60; done
step "language branch, 120-instance sample, all classes (queue)"
guarded gesd_lang_gpu$GPU $L/lang_sample_gpu$GPU.log $LANG -e PHASE=sample segciani:dev \
  bash -c "$PIP; python -u examples/ground_ea_hypotheses.py"
step "worker done"
