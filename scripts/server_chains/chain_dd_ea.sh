#!/bin/bash
# SliceLens's data-driven hypotheses on val-EA (Daniele, 2026-10-20: "testale su gpu 1 e 2"), as on SBD
# (chain_dd_sbd.sh): all 20 classes -- the 4 pilot classes too, whose June hypotheses came from the
# original images --, generated from val-EA's own instances (the 691 Pascal-EA material edits in place of
# their originals), grounded on the val-EA run's 120-instance sample (the same coupled draw), two rescores
# of COPIES of the val-EA pickle without and with them, the slope rule and the rank test side by side,
# the planted-cause evaluation on both copies, and whether any data-driven phrase or caption names the
# material. Stops there: the re-grounding on whole classes is asked for after these results.
# One worker per GPU, every container under outputs/gpu_guard.sh (a stopped container ends that worker;
# nothing is restarted).
#   setsid nohup bash outputs/chain_dd_ea.sh "1 2" > outputs/slice_quality_ea/dd_chain.log 2>&1 < /dev/null &
set -u
GPUS=$1
cd ~/storage/SLICELENS
L=outputs/slice_quality_ea
P=$HOME/storage/pascal_vlm
CLASSES="aeroplane bicycle bird boat bottle bus car cat chair cow diningtable dog horse motorbike person pottedplant sheep sofa train tvmonitor"
MOUNTS="-v $HOME/storage/SLICELENS:/workspace -v $P:/data -v $DATASETS:/ds"
MODELS="-v $STORAGE/models/qwen2.5-vl-7b:/models/qwen -v $STORAGE/models/gemma-4-12B-it:/models/gemma"
ENV="-e HOME=/tmp -e PYTHONIOENCODING=utf-8 -e PYTHONPATH=/workspace"
PIP="pip install --user -q scipy scikit-learn >/dev/null 2>&1"
step() { echo "$(date '+%m-%d %H:%M') == $1 =="; }
cpu() { docker run --rm -u $(id -u):$(id -g) $ENV "$@"; }
gen_done() { for c in $CLASSES; do [ -f $P/outputs_dd_ea/dd_$c.txt ] || return 1; done; }
cc_done() { for c in $CLASSES; do [ -f $P/outputs_dd_ea_ground/voc_paper_${c}_cc.json ] || return 1; done; }
alive() {      # a worker of this chain, or a container of one added later, still running
  for p in $PIDS; do kill -0 $p 2>/dev/null && return 0; done
  docker ps --format "{{.Names}}" | grep -q "^dd_ea_" && return 0; return 1; }
stopped() { echo "workers ended before $1 -- nothing restarted"; cat $L/dd_worker_gpu*.log; echo "$(date '+%m-%d %H:%M') CHAIN_DD_EA_STOPPED"; exit 0; }
rescore() {   # $1 out dir, $2 LANG_DIRS, $3 LANG_TAGS
  mkdir -p $1; cp $L/result_tagged.pkl $1/result_tagged.pkl
  cpu -e SQ_OUT=$1 -e LANG_DIRS=$2 -e LANG_TAGS=$3 -e GALLERY=0 -e VOC_DATA=/data/voc2012_ea/VOCdevkit/VOC2012 \
    $MOUNTS -w /workspace segciani:dev bash -c "$PIP; python -u slice_quality/examples/rescore_with_language.py" > $1/rescore.log 2>&1
  grep -q "DONE rescore" $1/rescore.log || { echo "FAIL rescore $1"; tail -5 $1/rescore.log; exit 1; }
  grep -E "language slices total|images with SliceLens" $1/rescore.log
}

step "1. workers on $GPUS: generation (queue), then the 120-instance sample (queue)"
mkdir -p $P/outputs_dd_ea $P/outputs_dd_ea_ground
PIDS=""
for g in $GPUS; do
  bash -c "
    set -u; GPU=$g; ME=\$(id -u); UUID=\$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i \$GPU)
    others() { for pid in \$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v u=\"\$UUID\" '\$1==u {print \$2}'); do
      uid=\$(ps -o uid= -p \"\$pid\" 2>/dev/null | tr -d ' '); [ -n \"\$uid\" ] && [ \"\$uid\" != \"\$ME\" ] && echo \$pid; done; }
    run() { NAME=\$1; LOG=\$2; shift 2
      if [ -n \"\$(others)\" ]; then echo \"GPU \$GPU used by another user before \$NAME: worker ends\"; exit 0; fi
      setsid nohup bash outputs/gpu_guard.sh \$GPU \$NAME > $L/guard_\$NAME.log 2>&1 < /dev/null &
      CLASSES=\"$CLASSES\" docker run --rm --name \$NAME --gpus device=\$GPU -u $(id -u):$(id -g) $ENV -e SQ_QWEN_DEVICE_MAP=cuda:0 -e CLASSES \
        $MOUNTS $MODELS -w /workspace \"\$@\" > \$LOG 2>&1
      sleep 3
      if grep -q GUARD_STOPPED $L/guard_\$NAME.log; then echo \"guard stopped \$NAME: worker ends, nothing restarted\"; exit 0; fi
      tail -n 1 \$LOG; }
    run dd_ea_gen_gpu\$GPU $L/dd_gen_gpu\$GPU.log -e OUT_DIR=/data/outputs_dd_ea -e SUBSTRATE=/data/voc2012_ea/substrate_cc.pkl \
      segciani:dev bash -c \"$PIP; python -u examples/gen_datadriven.py\"
    for c in $CLASSES; do until [ -f $P/outputs_dd_ea/dd_\$c.txt ]; do sleep 60; done; done
    run dd_ea_ground_gpu\$GPU $L/dd_ground_gpu\$GPU.log -e OUT_DIR=/data/outputs_dd_ea_ground \
      -e QUERIES_TEMPLATE=/data/outputs_dd_ea/dd_{cls}.txt -e DATASET=val-EA -e PHASE=sample \
      segciani:dev bash -c \"$PIP; python -u examples/ground_ea_hypotheses.py\"
    echo \"\$(date '+%m-%d %H:%M') [gpu\$GPU] worker done\"" > $L/dd_worker_gpu$g.log 2>&1 &
  PIDS="$PIDS $!"
  sleep 20
done

step "2. waiting for the generation"
until gen_done; do alive || stopped "the generation"; sleep 120; done
empty=$(grep -L "^[^#]" $P/outputs_dd_ea/dd_*.txt)
[ -n "$empty" ] && echo "WARNING data-driven gave no hypothesis for: $empty"
for f in $P/outputs_dd_ea/dd_*.txt; do echo "  $(basename $f .txt): $(grep -vc '^#' $f)"; done
step "3. waiting for the sample phase"
until cc_done; do alive || stopped "the sample phase"; sleep 120; done
wait

step "4. rescores of COPIES of the val-EA pickle, without and with the data-driven hypotheses (CPU)"
rescore outputs/slice_quality_ea_dd_base /data/outputs_ea ""
rescore outputs/slice_quality_ea_dd /data/outputs_ea:/data/outputs_dd_ea_ground ":dd"
cpu -e BASE=outputs/slice_quality_ea_dd_base -e DD=outputs/slice_quality_ea_dd -e RUNS_DIR=/data/outputs_dd_ea_ground \
  -e DATASET=val-EA $MOUNTS -w /workspace segciani:dev python -u slice_quality/examples/compare_dd_hypotheses.py \
  > outputs/slice_quality_ea_dd/compare_sample.log 2>&1
tail -n 25 outputs/slice_quality_ea_dd/compare_sample.log

step "5. the rank test beside the slope rule on the data-driven groundings (CPU)"
cpu -e RUNS_DIR=/data/outputs_dd_ea_ground -e STAGE=cc $MOUNTS -w /workspace segciani:dev \
  bash -c "$PIP; python -u slice_quality/examples/dd_auc.py" > outputs/slice_quality_ea_dd/dd_auc_cc.log 2>&1
head -n 40 outputs/slice_quality_ea_dd/dd_auc_cc.log

step "6. planted cause on both copies (CPU): victims inside certified slices"
for R in outputs/slice_quality_ea_dd_base outputs/slice_quality_ea_dd; do
  cpu -e EA_RUN=$R $MOUNTS -w /workspace segciani:dev bash -c \
    "$PIP; python -u slice_quality/examples/ea_planted_eval.py" > $R/planted_eval.log 2>&1
  echo "  $R"; grep -iE "victim|certified" $R/planted_eval.txt | head -6
done
diff -q $L/planted_eval.txt outputs/slice_quality_ea_dd_base/planted_eval.txt && echo "  base copy reproduces the val-EA planted_eval"

step "7. does any data-driven phrase, or any caption behind it, name the material?"
python3 -I - $P/outputs_dd_ea <<'PY'
import json, re, sys
from pathlib import Path
W = re.compile(r"\b(toy|toys|statue|model|stuffed|figurine|sculpture|fake|artificial|paper|cardboard|origami|metal|metallic|plastic|wood|wooden|replica|miniature|craft)\b", re.I)
d = Path(sys.argv[1]); nh = nc = hc = cc = 0
for f in sorted(d.glob("dd_*.json")):
    r = json.load(open(f, encoding="utf-8"))
    hyps = [h["hypothesis"] for h in r.get("condensed", [])]
    caps = [c for a in r.get("attributes", []) for c in a.get("captions", [])]
    hm = [h for h in hyps if W.search(h)]; cm = [c for c in caps if W.search(c)]
    nh += len(hyps); hc += len(hm); nc += len(caps); cc += len(cm)
    print("  %-12s seeds %-8s phrases with a material word %d/%d, captions %d/%d %s" % (
        r["label"], r.get("seeds_from"), len(hm), len(hyps), len(cm), len(caps), hm))
print("  total: phrases %d/%d, captions %d/%d" % (hc, nh, cc, nc))
PY
echo "$(date '+%m-%d %H:%M') CHAIN_DD_EA_SAMPLE_DONE"
