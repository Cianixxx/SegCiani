#!/bin/bash
# Step 4 of chain_geo_join.sh run early (3 November, 13:25 UTC): every GPU step but the SBD ones is done, so the
# merges, model 2, the gallery and the numbers are computed now; the chain repeats them at its end (same inputs).
set -u
cd ~/storage/SLICELENS
PV=$HOME/storage/pascal_vlm
O=outputs
D=$O/slice_quality_inst
C=$O/slice_quality_slcmp
CA=$O/slice_quality_slcmp_auc
SB=$O/slice_quality_sbd
L=$O/geo_join_logs
mkdir -p $L
MOUNTS="-v $HOME/storage/SLICELENS:/workspace -v $PV:/data -v $DATASETS:/ds"
MODELS="-v $STORAGE/models/qwen2.5-vl-7b:/models/qwen -v $STORAGE/models/gemma-4-12B-it:/models/gemma"
ENV0="-e HOME=/tmp -e PYTHONIOENCODING=utf-8 -e PYTHONPATH=/workspace"
ENVC="$ENV0 -e SQ_CLAIM_COV=0.5 -e SQ_GT_INSTANCES=object"
PIP="pip install --user -q scipy scikit-learn >/dev/null 2>&1"
IMG=segciani:dev
VOC=/data/voc2012/VOCdevkit/VOC2012
SBD=/data/sbd/VOCdevkit/VOC2012
EAV=/data/voc2012_ea/VOCdevkit/VOC2012
GPUS="5 6"
DEADLINE=$(( $(date +%s) + 48 * 3600 ))      # a queue waiting for a busy GPU gives up after 48 hours
say() { echo "$(date '+%m-%d %H:%M') $*"; }
fail() { say "CHAIN_GEO_JOIN_FAIL: $1"; exit 1; }
cpu0() { docker run --rm -u $(id -u):$(id -g) $ENV0 $MOUNTS -w /workspace "$@"; }
cpuc() { docker run --rm -u $(id -u):$(id -g) $ENVC $MOUNTS -w /workspace "$@"; }
others() {
  local uuid me pid uid
  uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $1); me=$(id -u)
  for pid in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v u="$uuid" '$1==u {print $2}'); do
    uid=$(ps -o uid= -p "$pid" 2>/dev/null | tr -d ' '); [ -n "$uid" ] && [ "$uid" != "$me" ] && echo $pid
  done; }
wait_gpu() {  # $1 gpu: until no other user is on it (or the deadline)
  local said=0
  while [ -n "$(others $1)" ]; do
    [ $(date +%s) -ge $DEADLINE ] && return 1
    [ $said = 0 ] && { say "[gpu$1] busy with another user: waiting"; said=1; }
    sleep 300
  done
  [ $said = 1 ] && say "[gpu$1] free"
  return 0; }
summary() { cpuc $IMG bash -c "$PIP; python -c \"
import sys; sys.path.insert(0, 'docs')
from _revision_numbers import load_inst_summary as L
for d in sys.argv[1:]:
    s = L(d); print('  %-44s situations %d, clusters %d, errors in a group %.1f%%, images with a situation %d' % (
        d, s['slices']['language'], s['slices']['cluster'], 100 * s['cov_union'] / s['n_err'], s['situational_images']))
\" $*" 2>&1 | tail -n $#; }

# ---- 4. merges and the rest (CPU)
say "4. merges, model 2, gallery, numbers"
merge_cause() {   # $1 run, $2 settings, $3 tag
  docker run --rm -u $(id -u):$(id -g) $2 $MOUNTS -e SQ_OUT=$1 -e JUDGE_4BIT=1 -e ALL=1 -e CONTROL_N=150 -w /workspace $IMG bash -c \
    "$PIP; python -u slice_quality/examples/judge_cause_plausibility.py" > $L/merge_$3.log 2>&1
  grep -q JUDGE_CAUSE_DONE $L/merge_$3.log && tail -n 12 $1/judge_cause_val.txt | head -4 || say "  merge $3 FAILED"; }
[ -f $L/.done_voc_cv0 ] && [ -f $L/.done_voc_cv1 ] && merge_cause $D "$ENVC" voc_cv
if [ -f $L/.done_cmp_cl0 ] && [ -f $L/.done_cmp_cl1 ]; then
  cpuc -e SQ_OUT=$C -e KINDS=narrative,pattern -e OLD_DUP=0 -e NSHARDS=2 -e MERGE=1 $IMG bash -c \
    "$PIP; python -u slice_quality/examples/judge_claims.py" > $L/merge_cmp_cl.log 2>&1
  grep -q CLAIMS_MERGE_DONE $L/merge_cmp_cl.log || say "  facts judge merge (slope text) FAILED"
fi
if [ -f $L/.done_cmpa_cl ]; then
  cpuc -e SQ_OUT=$CA -e KINDS=pattern -e OLD_DUP=0 -e NSHARDS=1 -e MERGE=1 $IMG bash -c \
    "$PIP; python -u slice_quality/examples/judge_claims.py" > $L/merge_cmpa_cl.log 2>&1
  grep -q CLAIMS_MERGE_DONE $L/merge_cmpa_cl.log || say "  facts judge merge (rank-test text) FAILED"
fi
# model 2: the groundings of the situations model 1 now ships, its rank verdict, its language branch
if [ -f $L/.done_m2 ]; then
  M=$PV/outputs_model2_inst_geo
  mkdir -p $M
  python3 -I - $PV/outputs_model2_inst111 $PV/outputs_model2_dd $PV/outputs_model2_geo $M <<'PY' || fail "model 2 merge"
import glob, json, os, re, sys
srcs, out = sys.argv[1:-1], sys.argv[-1]
cls_of = lambda sid: re.sub(r"_(extra|dd|pend)$", "", sid.rsplit("_", 1)[0])
ship = {(cls_of(s["slice_id"]), s["hypothesis"]) for s in json.load(open("outputs/slice_quality_inst/slices_index.json"))["slices"]
        if s["source"] == "language" and s.get("in_descriptions")}
cls = sorted({os.path.basename(f)[10:-10] for d in srcs for f in glob.glob(d + "/voc_paper_*_full.json")})
n = 0
for c in cls:
    parts = [json.load(open(p)) for p in (d + "/voc_paper_%s_full.json" % c for d in srcs) if os.path.exists(p)]
    r = parts[0]
    assert all(x["num_instances"] == r["num_instances"] for x in parts), c
    seen, cands = set(), []
    for x in parts:
        for cd in x["candidates"]:
            if len(cd["scored"]) == x["num_instances"] and cd["query"] not in seen and (c, cd["query"]) in ship:
                seen.add(cd["query"])
                cands.append(cd)
    if not cands:
        continue
    r = dict(r, candidates=cands, systematic_error_slices=sorted({q for x in parts for q in x["systematic_error_slices"]
                                                                  if (c, q) in ship}))
    json.dump(r, open(out + "/voc_paper_%s_full.json" % c, "w"))
    n += len(cands)
print("  model 2: groundings of the %d shipped situations merged: %d hypotheses" % (len(ship), n))
PY
  cpuc -e RUNS_DIR=/data/outputs_model2_inst_geo -e STAGE=full -e OUT=$D/model2/dd_auc_full \
    -e RANK_COPY=/data/outputs_model2_inst_geo_rank $IMG bash -c "$PIP; python -u slice_quality/examples/dd_auc.py" > $L/model2_auc.log 2>&1
  grep -q DD_AUC_DONE $L/model2_auc.log && head -n 1 $L/model2_auc.log || say "  model 2 rank verdict FAILED"
  cpuc -e SQ_OUT=$D -e OUT_TAG=model2 -e LANG_DIR=/data/outputs_model2_inst_geo_rank $IMG bash -c \
    "$PIP; python -u slice_quality/examples/run_second_model_language.py" > $L/model2_language.log 2>&1
  grep -q STAGE3_DONE $L/model2_language.log && grep -E "language slices total|KEPT|images with a certified" $L/model2_language.log \
    || say "  model 2 language branch FAILED"
fi
cpuc -e SQ_OUT=$D $IMG bash -c "$PIP; python -u slice_quality/examples/regen_gallery.py" > $L/regen_gallery.log 2>&1
grep -q "gallery regenerated" $L/regen_gallery.log && cpuc -e SQ_OUT=$D -e GALLERY_SRC=gallery_full_tagged.html \
  -e GALLERY_DST=gallery_share.html $IMG python -u docs/make_share_gallery.py | tail -1 || say "  gallery FAILED"
cpu0 $IMG bash -c "$PIP; python -u slice_quality/examples/doc_numbers_sbd_ea.py" > $L/doc_numbers.log 2>&1
grep -q DOC_NUMBERS_DONE $L/doc_numbers.log || say "  sbd/ea summary FAILED"
cpuc $IMG bash -c "$PIP; python -u slice_quality/examples/slicelens_cmp_stats.py" > $L/cmp_stats.log 2>&1
grep -q CMP_STATS_DONE $L/cmp_stats.log && grep -A 8 "ERRORS INSIDE A CERTIFIED GROUP" $L/cmp_stats.log || say "  paired tests FAILED"
for f in eval_precision judge_cause judge_cause_val; do
  [ -f $D/$f.txt ] && say "  VOC $f: $(grep -m1 -iE 'precision|media|mean' $D/$f.txt | cut -c1-120)"
done
summary $O/slice_quality_inst_idjoin $D $O/slice_quality_sbd_idjoin $SB
say "EARLY_STEP4_DONE"
