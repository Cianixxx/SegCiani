#!/bin/bash
# Riverifica sull'INTERA popolazione delle sole ipotesi nuove certificate sul
# campione (come reground_validated a luglio), poi rescore + confronto su CPU.
#   GPU=0 QWEN4=1 bash outputs/run_extra_reground.sh
# Non tocca /data/outputs ne' le run esistenti: scrive voc_paper_<cls>_full.json
# in /data/outputs_extra (che il rescore preferisce ai _cc della stessa cartella)
# e il rescore in outputs/slice_quality_inst_extra_full.
set -u
GPU="${GPU:?specificare GPU=N}"
Q4=""; [ "${QWEN4:-0}" = "1" ] && Q4="--load-in-4bit"
cd ~/storage/SLICELENS
step() { echo "$(date +%H:%M) == $1 =="; }
SRC=outputs/slice_quality_inst_extra/slices_index.json
IDX=outputs/slice_quality_inst_extra/extra_validated_index.json
RUN="docker run --rm -u $(id -u):$(id -g) -e HOME=/tmp -e PYTHONIOENCODING=utf-8 -e PYTHONPATH=/workspace \
  -v $HOME/storage/SLICELENS:/workspace -v $HOME/storage/pascal_vlm:/data \
  -w /workspace segciani:dev"

step "indice filtrato: solo le ipotesi nuove certificate sul campione"
python3 - "$SRC" "$IDX" <<'PY'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
sl = json.load(open(src, encoding="utf-8"))["slices"]
keep = []
for s in sl:
    if s.get("source") == "language" and s.get("in_descriptions") and "_extra_" in s["slice_id"]:
        t = dict(s)
        t["slice_id"] = s["slice_id"].replace("_extra_", "_")   # reground legge la classe da <cls>_<n>
        keep.append(t)
json.dump({"slices": keep}, open(dst, "w", encoding="utf-8"), indent=1)
for t in keep:
    print(f"  {t['slice_id']:20s} {t['hypothesis']}")
print(f"{len(keep)} ipotesi da riverificare")
PY

step "riverifica su GPU $GPU ${Q4:-bf16}"
docker run --rm --gpus device="$GPU" --name extrareground_gpu$GPU -u $(id -u):$(id -g) -e HOME=/tmp \
  -e PYTHONIOENCODING=utf-8 -e PYTHONPATH=/workspace \
  -v $HOME/storage/SLICELENS:/workspace -v $HOME/storage/pascal_vlm:/data \
  -v $STORAGE/models/qwen2.5-vl-7b:/models/qwen -w /workspace segciani:dev \
  bash -c "pip install --user -q scipy >/dev/null 2>&1; python -u examples/reground_validated.py \
    --slices-index $IDX --substrate-cache /data/voc2012/substrate_cc.pkl \
    --vlm-backend qwen --vlm-model /models/qwen $Q4 --out-dir /data/outputs_extra" \
  || { echo "FAIL reground"; exit 1; }
echo "--- GPU $GPU:"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed -n "$((GPU + 1))p"

step "rescore inst CON le nuove ipotesi riverificate (CPU)"
E=outputs/slice_quality_inst_extra_full; mkdir -p $E
cp outputs/slice_quality_inst/result_tagged.pkl $E/
$RUN bash -c "pip install --user -q scipy scikit-learn >/dev/null 2>&1; \
  SQ_CLAIM_COV=0.5 SQ_GT_INSTANCES=object SQ_OUT=$E LANG_DIRS=/data/outputs:/data/outputs_extra \
  python -u slice_quality/examples/rescore_with_language.py" > $E/rescore.log 2>&1 \
  || { echo "FAIL rescore"; tail -5 $E/rescore.log; exit 1; }

step "confronto con la base"
$RUN bash -c "EXTRA=$E python -u slice_quality/examples/compare_extra_hypotheses.py" || { echo "FAIL compare"; exit 1; }
echo "$(date +%H:%M) EXTRA_REGROUND_DONE"
