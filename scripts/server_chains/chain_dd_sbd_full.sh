#!/bin/bash
# SBD data-driven hypotheses, whole classes (Daniele, 2026-10-20: "Certificata + AUC su 1,2,5"): the one
# certified after the sample + the ones significant under the rank test on the sample, re-grounded on every
# instance of their class (examples/ground_ea_hypotheses.py PHASE=full, FULL_HYPS), then the rescore of the
# COPY outputs/slice_quality_sbd_dd, the comparison and the rank test on whole classes (CPU).
# One worker per GPU, every container under outputs/gpu_guard.sh; nothing is restarted.
#   setsid nohup bash outputs/chain_dd_sbd_full.sh "<gpus>" > outputs/slice_quality_sbd/dd_full_chain.log 2>&1 < /dev/null &
set -u
GPUS=$1
cd ~/storage/SLICELENS
L=outputs/slice_quality_sbd
DDL=outputs/slice_quality_sbd_dd
P=$HOME/storage/pascal_vlm
CLASSES="bicycle bird boat bottle bus car cat diningtable dog horse motorbike pottedplant sheep sofa train tvmonitor"
MOUNTS="-v $HOME/storage/SLICELENS:/workspace -v $P:/data -v $DATASETS:/ds"
QWEN="-v $STORAGE/models/qwen2.5-vl-7b:/models/qwen"
ENV="-e HOME=/tmp -e PYTHONIOENCODING=utf-8 -e PYTHONPATH=/workspace"
PIP="pip install --user -q scipy scikit-learn >/dev/null 2>&1"
step() { echo "$(date '+%m-%d %H:%M') == $1 =="; }
cpu() { docker run --rm -u $(id -u):$(id -g) $ENV "$@"; }
full_done() { for c in $CLASSES; do [ -f $P/outputs_dd_sbd_ground/voc_paper_${c}_full.json ] || [ -f $P/outputs_dd_sbd_ground/.none_full_$c ] || return 1; done; }
alive() { for p in $PIDS; do kill -0 $p 2>/dev/null && return 0; done
  docker ps --format "{{.Names}}" | grep -q "^dd_sbd_full_" && return 0; return 1; }

step "0. the list: certified after the sample + rank test on the sample"
python3 -I - "$DDL/slices_index.json" "$P/outputs_dd_sbd_ground/dd_auc_cc.json" "$P/full_dd_sbd.json" <<'PY'
import json, sys
idx, auc, out = sys.argv[1:4]
lst = {}
for s in json.load(open(idx))["slices"]:
    if s.get("in_descriptions") and "_dd_" in s["slice_id"] and s.get("hypothesis"):
        lst.setdefault(s["slice_id"].split("_dd_")[0], []).append(s["hypothesis"])
for r in json.load(open(auc)):
    if r["auc_sys"] and r["query"] not in lst.get(r["cls"], []):
        lst.setdefault(r["cls"], []).append(r["query"])
json.dump(lst, open(out, "w"), indent=1)
for c, q in sorted(lst.items()):
    print("  %-12s %d: %s" % (c, len(q), q))
print("total", sum(map(len, lst.values())))
PY

step "1. workers on $GPUS (queue)"
PIDS=""
for g in $GPUS; do
  bash -c "
    set -u; GPU=$g; ME=\$(id -u); UUID=\$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i \$GPU)
    others() { for pid in \$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v u=\"\$UUID\" '\$1==u {print \$2}'); do
      uid=\$(ps -o uid= -p \"\$pid\" 2>/dev/null | tr -d ' '); [ -n \"\$uid\" ] && [ \"\$uid\" != \"\$ME\" ] && echo \$pid; done; }
    NAME=dd_sbd_full_gpu\$GPU
    if [ -n \"\$(others)\" ]; then echo \"GPU \$GPU used by another user before \$NAME: worker ends\"; exit 0; fi
    setsid nohup bash outputs/gpu_guard.sh \$GPU \$NAME > $L/guard_\$NAME.log 2>&1 < /dev/null &
    CLASSES=\"$CLASSES\" docker run --rm --name \$NAME --gpus device=\$GPU -u $(id -u):$(id -g) $ENV -e SQ_QWEN_DEVICE_MAP=cuda:0 \
      -e CLASSES -e OUT_DIR=/data/outputs_dd_sbd_ground -e FULL_HYPS=/data/full_dd_sbd.json -e SUBSTRATE=/data/sbd/substrate_cc.pkl \
      -e OLD_SUBSTRATE= -e EDITED= -e CACHE_DIRS= -e SAMPLING=srs -e DATASET=SBD -e PHASE=full $MOUNTS $QWEN -w /workspace \
      segciani:dev bash -c \"$PIP; python -u examples/ground_ea_hypotheses.py\" > $L/dd_full_gpu\$GPU.log 2>&1
    sleep 3
    if grep -q GUARD_STOPPED $L/guard_\$NAME.log; then echo \"guard stopped \$NAME: worker ends, nothing restarted\"; exit 0; fi
    tail -n 1 $L/dd_full_gpu\$GPU.log
    echo \"\$(date '+%m-%d %H:%M') [gpu\$GPU] worker done\"" > $L/dd_full_worker_gpu$g.log 2>&1 &
  PIDS="$PIDS $!"
  sleep 20
done

step "2. waiting for every class"
until full_done; do
  alive || { echo "workers ended before the end -- nothing restarted"; cat $L/dd_full_worker_gpu*.log; echo "$(date '+%m-%d %H:%M') CHAIN_DD_SBD_FULL_STOPPED"; exit 0; }
  sleep 120
done
wait

step "3. rescore of the COPY with the re-grounded data-driven hypotheses (CPU), comparison, rank test"
cp $DDL/dd_hypotheses_report.txt $DDL/dd_hypotheses_report_sample.txt
cp $L/result_tagged.pkl $DDL/result_tagged.pkl
cpu -e SQ_OUT=$DDL -e LANG_DIRS=/data/outputs_sbd:/data/outputs_dd_sbd_ground -e LANG_TAGS=":dd" -e GALLERY=0 $MOUNTS \
  -w /workspace segciani:dev bash -c "$PIP; python -u slice_quality/examples/rescore_with_language.py" > $DDL/rescore_full.log 2>&1
grep -q "DONE rescore" $DDL/rescore_full.log || { echo "FAIL rescore"; tail -5 $DDL/rescore_full.log; exit 1; }
grep -E "language slices total|images with SliceLens" $DDL/rescore_full.log
cpu -e BASE=outputs/slice_quality_sbd_dd_base -e DD=$DDL -e RUNS_DIR=/data/outputs_dd_sbd_ground -e DATASET=SBD $MOUNTS \
  -w /workspace segciani:dev python -u slice_quality/examples/compare_dd_hypotheses.py > $DDL/compare_full.log 2>&1
tail -n 12 $DDL/compare_full.log
cpu -e RUNS_DIR=/data/outputs_dd_sbd_ground -e STAGE=full $MOUNTS -w /workspace segciani:dev \
  bash -c "$PIP; python -u slice_quality/examples/dd_auc.py" > $DDL/dd_auc_full.log 2>&1
head -n 30 $DDL/dd_auc_full.log
echo "$(date '+%m-%d %H:%M') CHAIN_DD_SBD_FULL_DONE"
