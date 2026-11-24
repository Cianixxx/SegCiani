#!/bin/bash
# Data-driven hypotheses on VOC val, phase 2 (after chain_dd_gen.sh): ground them on the July
# 120-instance sample (GPU2 + GPU6), rescore a COPY of the inst pickle with and without them
# (CPU), compare; then re-ground the validated ones on their whole class and rescore again.
#   setsid nohup bash outputs/chain_dd_voc.sh > outputs/dd_voc.log 2>&1 &
set -u
cd ~/storage/SLICELENS
MOUNTS="-v $HOME/storage/SLICELENS:/workspace -v $HOME/storage/pascal_vlm:/data"
QWEN="-v $STORAGE/models/qwen2.5-vl-7b:/models/qwen"
ENV="-e HOME=/tmp -e PYTHONIOENCODING=utf-8 -e PYTHONPATH=/workspace"
PIP="pip install --user -q scipy scikit-learn >/dev/null 2>&1"
O=$HOME/storage/pascal_vlm/outputs_dd_ground
step() { echo "$(date '+%m-%d %H:%M') == $1 =="; }
rescore() {   # $1 out dir, $2 LANG_DIRS, $3 LANG_TAGS
  mkdir -p $1; cp outputs/slice_quality_inst/result_tagged.pkl $1/result_tagged.pkl
  docker run --rm -u $(id -u):$(id -g) $ENV -e SQ_OUT=$1 -e LANG_DIRS=$2 -e LANG_TAGS=$3 $MOUNTS -w /workspace segciani:dev \
    bash -c "$PIP; python -u slice_quality/examples/rescore_with_language.py" > $1/rescore.log 2>&1
  grep -q "DONE rescore" $1/rescore.log || { echo "FAIL rescore $1"; tail -5 $1/rescore.log; exit 1; }
  grep -E "language slices total|images with SliceLens" $1/rescore.log
}

until grep -q CHAIN_DD_GEN_DONE outputs/dd_gen.log 2>/dev/null; do sleep 60; done
step "1. ground the data-driven hypotheses on the 120-instance sample (GPU2 + GPU6)"
mkdir -p $O
for g in 2 6; do
  docker run --rm --name dd_ground_gpu$g --gpus device=$g -u $(id -u):$(id -g) $ENV -e SQ_QWEN_DEVICE_MAP=cuda:0 \
    -e OUT_DIR=/data/outputs_dd_ground -e QUERIES_TEMPLATE=/data/outputs_dd/dd_{cls}.txt $MOUNTS $QWEN \
    -w /workspace segciani:dev bash -c "$PIP; python -u examples/ground_extra_hypotheses.py" \
    > outputs/dd_ground_gpu$g.log 2>&1 &
  sleep 30
done
wait
grep -h "still missing" outputs/dd_ground_gpu*.log

step "2. rescore base (= inst) and with the data-driven hypotheses (CPU)"
rescore outputs/slice_quality_inst_dd_base /data/outputs:/data/outputs_extra ":extra"
rescore outputs/slice_quality_inst_dd /data/outputs:/data/outputs_extra:/data/outputs_dd_ground ":extra:dd"
docker run --rm -u $(id -u):$(id -g) $ENV $MOUNTS -w /workspace segciani:dev \
  python -u slice_quality/examples/compare_dd_hypotheses.py > outputs/slice_quality_inst_dd/compare_sample.log 2>&1
cp outputs/slice_quality_inst_dd/dd_hypotheses_report.txt outputs/slice_quality_inst_dd/dd_hypotheses_report_sample.txt
cat outputs/slice_quality_inst_dd/dd_hypotheses_report.txt

step "3. re-ground the validated data-driven hypotheses on their whole class (GPU2)"
python3 - <<PY
import json
d = json.load(open("outputs/slice_quality_inst_dd/slices_index.json"))
keep = [s for s in d["slices"] if s.get("in_descriptions") and "_dd_" in s["slice_id"]]
json.dump({"slices": keep}, open("outputs/slice_quality_inst_dd/dd_validated_index.json", "w"))
print("validated data-driven slices to re-ground:", len(keep))
PY
docker run --rm --name dd_full_gpu2 --gpus device=2 -u $(id -u):$(id -g) $ENV -e SQ_QWEN_DEVICE_MAP=cuda:0 $MOUNTS $QWEN \
  -w /workspace segciani:dev bash -c "$PIP; python -u examples/reground_validated.py \
  --slices-index outputs/slice_quality_inst_dd/dd_validated_index.json --substrate-cache /data/voc2012/substrate_cc.pkl \
  --vlm-backend qwen --vlm-model /models/qwen --out-dir /data/outputs_dd_ground" > outputs/dd_full_gpu2.log 2>&1
tail -3 outputs/dd_full_gpu2.log

step "4. rescore with the re-grounded data-driven hypotheses (CPU) and compare"
rescore outputs/slice_quality_inst_dd /data/outputs:/data/outputs_extra:/data/outputs_dd_ground ":extra:dd"
docker run --rm -u $(id -u):$(id -g) $ENV $MOUNTS -w /workspace segciani:dev \
  python -u slice_quality/examples/compare_dd_hypotheses.py > outputs/slice_quality_inst_dd/compare_full.log 2>&1
cat outputs/slice_quality_inst_dd/dd_hypotheses_report.txt
echo "$(date '+%m-%d %H:%M') CHAIN_DD_VOC_DONE"
