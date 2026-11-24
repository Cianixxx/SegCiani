#!/bin/bash
# GESD point B, full-population re-grounding (Daniele, 2026-10-19: "Tesi + AUC + frasi-nome"), GPUs 1 2 6:
#   pass A  /data/full_protocol.json (certified after the sample + AUC verdict) -> /data/outputs_gesd
#   pass B  /data/full_naming.json (phrases naming an expert situation, to MEASURE only) -> /data/outputs_gesd_naming
# One worker per GPU (every container under outputs/gpu_guard.sh; a stopped container ends that worker,
# nothing is restarted), then rescore #2 on pass A only, then the point B analysis (STAGE=full).
#   setsid nohup bash outputs/chain_gesd_full.sh "<gpus>" > outputs/slice_quality_gesd/chain_full.log 2>&1 < /dev/null &
set -u
GPUS=$1
cd ~/storage/SLICELENS
L=outputs/slice_quality_gesd
G=$HOME/storage/pascal_vlm/gesd
CLASSES="airplane apple banana bed book carrot clock keyboard orange suitcase surfboard teddy_bear tie umbrella vase"
MOUNTS="-v $HOME/storage/SLICELENS:/workspace -v $G:/data -v $HOME/datasets/private:/priv:ro \
  -v $HOME/datasets/private/coco/val2017_gesd:/coco_img:ro -v $STORAGE/models/qwen2.5-vl-7b:/models/qwen"
ENV="-e HOME=/tmp -e PYTHONIOENCODING=utf-8 -e PYTHONPATH=/workspace -e SQ_QWEN_DEVICE_MAP=cuda:0"
PIP="pip install --user -q scipy scikit-learn >/dev/null 2>&1"
step() { echo "$(date '+%m-%d %H:%M') == $1 =="; }
cpu() { docker run --rm -u $(id -u):$(id -g) $ENV "$@"; }
done_dir() { for c in $CLASSES; do [ -f $1/voc_paper_${c}_full.json ] || [ -f $1/.none_full_$c ] || return 1; done; }
alive() { for p in $PIDS; do kill -0 $p 2>/dev/null && return 0; done; return 1; }

step "1. workers on $GPUS: pass A (protocol) then pass B (naming phrases, measure only)"
PIDS=""
for g in $GPUS; do
  bash -c "
    set -u; GPU=$g; ME=\$(id -u); UUID=\$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i \$GPU)
    others() { for pid in \$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v u=\"\$UUID\" '\$1==u {print \$2}'); do
      uid=\$(ps -o uid= -p \"\$pid\" 2>/dev/null | tr -d ' '); [ -n \"\$uid\" ] && [ \"\$uid\" != \"\$ME\" ] && echo \$pid; done; }
    for pass in A B; do
      if [ \$pass = A ]; then OD=/data/outputs_gesd; FH=/data/full_protocol.json; else OD=/data/outputs_gesd_naming; FH=/data/full_naming.json; fi
      NAME=gesd_full\${pass}_gpu\$GPU
      if [ -n \"\$(others)\" ]; then echo \"GPU \$GPU used by another user before \$NAME: worker ends\"; exit 0; fi
      setsid nohup bash outputs/gpu_guard.sh \$GPU \$NAME > $L/guard_\$NAME.log 2>&1 < /dev/null &
      CLASSES=\"$CLASSES\" docker run --rm --name \$NAME --gpus device=\$GPU -u $(id -u):$(id -g) $ENV -e OUT_DIR=\$OD -e FULL_HYPS=\$FH \
        -e SUBSTRATE=/data/substrate.pkl -e OLD_SUBSTRATE= -e EDITED= -e CACHE_DIRS= -e SAMPLING=srs -e DATASET=GESD \
        -e CLASSES -e PHASE=full $MOUNTS -w /workspace segciani:dev \
        bash -c \"$PIP; python -u examples/ground_ea_hypotheses.py\" > $L/lang_full\${pass}_gpu\$GPU.log 2>&1
      sleep 3
      if grep -q GUARD_STOPPED $L/guard_\$NAME.log; then echo \"guard stopped \$NAME: worker ends, nothing restarted\"; exit 0; fi
      tail -n 1 $L/lang_full\${pass}_gpu\$GPU.log
    done
    echo \"\$(date '+%m-%d %H:%M') [gpu\$GPU] worker done\"" > $L/worker_full_gpu$g.log 2>&1 &
  PIDS="$PIDS $!"
  sleep 20
done

step "2. waiting for both passes"
until done_dir $G/outputs_gesd && done_dir $G/outputs_gesd_naming; do
  alive || { echo "workers ended before the end -- nothing restarted"; cat $L/worker_full_gpu*.log; echo "$(date '+%m-%d %H:%M') CHAIN_GESD_FULL_STOPPED"; exit 0; }
  sleep 120
done
wait

step "3. rescore #2 (CPU), pass A only: the final GESD run"
cp $L/result_tagged_merged.pkl $L/result_tagged.pkl
cpu -e SQ_OUT=$L -e LANG_DIRS=/data/outputs_gesd -e CLASSES="$CLASSES" -e GALLERY=0 $MOUNTS -w /workspace \
  segciani:dev bash -c "$PIP; python -u slice_quality/examples/rescore_with_language.py" > $L/rescore_2.log 2>&1
grep -q "DONE rescore" $L/rescore_2.log || { echo "FAIL rescore 2"; tail -5 $L/rescore_2.log; exit 1; }
grep -E "language slices total|images with SliceLens|corroborated" $L/rescore_2.log

step "4. point B analysis on the full stage (CPU)"
cpu -e STAGE=full $MOUNTS -w /workspace segciani:dev \
  bash -c "$PIP; python -u slice_quality/examples/gesd_pointB.py" > $L/pointB_full.log 2>&1
tail -n 3 $L/pointB_full.log
echo "$(date '+%m-%d %H:%M') CHAIN_GESD_FULL_DONE"
