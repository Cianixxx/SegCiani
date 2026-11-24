#!/bin/bash
# GESD run (point B), the thesis protocol up to the sample phase: GPU workers (chain_gesd_worker.sh,
# each container under outputs/gpu_guard.sh) + the CPU steps here. Stops after rescore #1: the
# full-population re-grounding is asked for after the sample results.
#   setsid nohup bash outputs/chain_gesd.sh "<gpus>" > outputs/slice_quality_gesd/chain.log 2>&1 < /dev/null &
set -u
GPUS=$1
FIRST=lead                       # "nolead" as $2: the knowledge-driven files are made apart
[ "${2:-}" = nolead ] && FIRST=
cd ~/storage/SLICELENS
L=outputs/slice_quality_gesd
G=$HOME/storage/pascal_vlm/gesd
MOUNTS="-v $HOME/storage/SLICELENS:/workspace -v $G:/data -v $HOME/datasets/private:/priv:ro \
  -v $HOME/datasets/private/coco/val2017_gesd:/coco_img:ro"
ENV="-e HOME=/tmp -e PYTHONIOENCODING=utf-8 -e PYTHONPATH=/workspace"
PIP="pip install --user -q scipy scikit-learn >/dev/null 2>&1"
CLASSES="airplane apple banana bed book carrot clock keyboard orange suitcase surfboard teddy_bear tie umbrella vase"
step() { echo "$(date '+%m-%d %H:%M') == $1 =="; }
cpu() { docker run --rm -u $(id -u):$(id -g) $ENV "$@"; }
alive() { for p in $PIDS; do kill -0 $p 2>/dev/null && return 0; done; return 1; }
stopped() { echo "workers ended before $1 -- nothing restarted"; grep -h "worker ends\|GUARD_STOPPED" $L/worker_gpu*.log $L/guard_*.log 2>/dev/null; echo "$(date '+%m-%d %H:%M') CHAIN_GESD_STOPPED"; exit 0; }
dd_done() { for c in $CLASSES; do [ -f $G/outputs_dd/dd_$c.txt ] || return 1; done; }
cc_done() { for c in $CLASSES; do [ -f $G/outputs_gesd/voc_paper_${c}_cc.json ] || return 1; done; }

step "1. GPU workers on $GPUS (first = lead: Gemma knowledge hypotheses)"
PIDS=""
LEAD=$FIRST
for g in $GPUS; do
  bash outputs/chain_gesd_worker.sh $g $LEAD > $L/worker_gpu$g.log 2>&1 &
  PIDS="$PIDS $!"
  LEAD=
  sleep 20
done

step "2. waiting for the knowledge-driven and data-driven hypotheses"
until [ -f $L/.knowledge_done ] && dd_done; do alive || stopped "the hypotheses"; sleep 60; done
empty=$(grep -L "^[^#]" $G/outputs_dd/dd_*.txt)
if [ -n "$empty" ]; then echo "FAIL data-driven gave no hypothesis for: $empty"; grep -h " instances, " $L/dd_gpu*.log; exit 1; fi
step "3. hypothesis files: 7 canonical + Gemma knowledge-driven + data-driven (the script adds the 8 extra)"
python3 -I - "$G" $CLASSES <<'EOF'
import sys
g, classes = sys.argv[1], sys.argv[2:]
for c in classes:
    kn = [l.strip() for l in open(f"{g}/knowledge/combined_{c}.txt", encoding="utf-8") if l.strip() and not l.startswith("#")]
    dd = [l.strip() for l in open(f"{g}/outputs_dd/dd_{c}.txt", encoding="utf-8") if l.strip() and not l.startswith("#")]
    seen, out = set(), []
    for q in kn + dd:
        if q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)
    n_dd = len(out) - len({q.lower() for q in kn})
    open(f"{g}/combined_{c}.txt", "w", encoding="utf-8").write(
        f"# {c}: knowledge-driven (7 canonical + {len(kn) - 7} Gemma, batch) + data-driven ({n_dd})\n" + "\n".join(out) + "\n")
    print(f"  {c}: {len(kn)} knowledge-driven + {n_dd} data-driven", flush=True)
EOF
cpu -e DRY=1 -e OUT_DIR=/data/outputs_gesd -e SUBSTRATE=/data/substrate.pkl -e OLD_SUBSTRATE= -e EDITED= -e CACHE_DIRS= \
  -e SAMPLING=srs -e DATASET=GESD -e CLASSES="$CLASSES" -e PHASE=sample $MOUNTS -w /workspace segciani:dev \
  bash -c "$PIP; python -u examples/ground_ea_hypotheses.py" 2>&1 | grep "DRY"
touch $L/.hyp_ready

step "4. tags onto the instances (CPU), once every tag chunk is there"
until cpu -e STEP=merge $MOUNTS -w /workspace segciani:dev bash -c "$PIP; python -u slice_quality/examples/build_gesd.py" \
  > $L/merge.log 2>&1 && grep -q "BUILD_GESD_DONE merge" $L/merge.log; do alive || stopped "the tags"; sleep 120; done
tail -2 $L/merge.log

step "5. waiting for the sample phase of every class"
until cc_done; do alive || stopped "the sample phase"; sleep 120; done

step "6. rescore #1 (CPU): which hypotheses are validated, both branches"
cp $L/result_tagged.pkl $L/result_tagged_merged.pkl
cpu -e SQ_OUT=$L -e LANG_DIRS=/data/outputs_gesd -e CLASSES="$CLASSES" -e GALLERY=0 $MOUNTS -w /workspace \
  segciani:dev bash -c "$PIP; python -u slice_quality/examples/rescore_with_language.py" > $L/rescore_1.log 2>&1
grep -q "DONE rescore" $L/rescore_1.log || { echo "FAIL rescore 1"; tail -5 $L/rescore_1.log; exit 1; }
grep -E "language slices total|images with SliceLens|corroborated" $L/rescore_1.log
wait
echo "$(date '+%m-%d %H:%M') CHAIN_GESD_SAMPLE_DONE"
