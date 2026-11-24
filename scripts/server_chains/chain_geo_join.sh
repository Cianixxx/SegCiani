#!/bin/bash
# The geometric join made official (Daniele, 3 November 2026: "sì, rifai tutto con l'abbinamento geometrico, usa gpu 5
# e 6"). The VLM read the objects of the first typing of Block 1 (substrate_cc.pkl); the final typing numbers its
# objects differently, so the join by id of the official rescores paired part of the groundings with other objects
# (ground_id_check.py: VOC 3,775 of 6,682 ids consistent, SBD 23,476 of 32,127, val-EA 3,611 of 6,184). Every run that
# joins groundings is rescored IN PLACE with LINK_SUBSTRATE (rescore_with_language.py): a substrate object goes to the
# final error of the same image and class with the largest box IoU >= 0.5; wrong or correct is the final typing's
# word; the rank-test verdicts stay as they are. The second segmenter's groundings were made on its own substrate and
# all 3,463 ids agree: only the new situations are grounded on it. Then every check that reads the slices again.
#   0  CPU  full copies of every run touched, as it is now: <run>_idjoin
#   1  CPU  the rescores, in parallel (inst must equal the validated copy slice_quality_inst_geo)
#   2  CPU  the inputs of the GPU steps: the new situations for model 2, the SliceLens-style texts
#   3  GPU  5 and 6, a shared queue under outputs/gpu_guard.sh: the new situations on model 2's objects, precision of the
#           shipped phrases (VOC, SBD), plausibility of the causes (VOC annotated; VOC and SBD every image, in two
#           and three shards), the facts and rubric judges on the SliceLens-style texts. A judge reuses every verdict whose
#           two texts did not change. If another user takes the GPU, the step is stopped and retried when it is free.
#      CPU  meanwhile: slice-image IoU, failure probe, relaxed cluster variants, Pascal-EA victims
#   4  CPU  the merges, model 2's language branch, the gallery, the numbers of the thesis, the paired tests
# Every container runs as the user (-u $(id -u):$(id -g)). Resumable: a step with its marker in $L is skipped.
#   setsid nohup bash outputs/chain_geo_join.sh >> outputs/geo_join.log 2>&1 < /dev/null &
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

say "== the geometric join, every run (GPU $GPUS) =="

# ---- 0. full copies of every run the rescores or the checks rewrite
RUNS="inst inst_slope inst_relaxA inst_relaxB sbd sbd_slope ea ea_auc slcmp slcmp_auc"
if [ ! -f $L/.backup_done ]; then
  for r in $RUNS; do
    [ -d $O/slice_quality_${r}_idjoin ] || cp -a $O/slice_quality_$r $O/slice_quality_${r}_idjoin || fail "copy $r"
  done
  for f in slicelens_cmp_stats relaxed_cluster_report; do for e in json txt; do
    [ -f $O/$f.$e ] && [ ! -f $O/${f}_idjoin.$e ] && cp $O/$f.$e $O/${f}_idjoin.$e
  done; done
  touch $L/.backup_done
fi
say "0. copies as they were: outputs/slice_quality_<run>_idjoin"

# ---- 1. the rescores, in parallel. Each run with the settings and groundings of its last rescore
#         (inst: chain_dd_official_v3.sh; slope copies: chain_auc_official.sh, reproduced by geo_repro_check.sh;
#         SBD: chain_official_sbd162.sh; val-EA: chain_rule_ea_stats.sh)
rescore() {   # $1 run, $2 settings, $3 LANG_DIRS, $4 LANG_TAGS, $5 VOC_DATA, $6 substrate
  local R=$O/slice_quality_$1
  [ -f $L/.rescore_$1 ] && return 0
  docker run --rm -u $(id -u):$(id -g) $2 $MOUNTS -e SQ_OUT=$R -e LANG_DIRS=$3 -e LANG_TAGS=$4 -e VOC_DATA=$5 \
    -e LINK_SUBSTRATE=$6 -e GALLERY=0 -w /workspace $IMG bash -c "$PIP; python -u slice_quality/examples/rescore_with_language.py" \
    > $R/rescore_geo.log 2>&1
  grep -q "DONE rescore" $R/rescore_geo.log || { say "  $1: rescore FAILED"; tail -4 $R/rescore_geo.log; return 1; }
  say "  $1: $(grep -E 'geometric join' $R/rescore_geo.log | cut -c1-110)"
  touch $L/.rescore_$1; }
LANG_NEW=/data/outputs_rank_m10:/data/outputs_extra_rank_m10:/data/outputs_pending_rank_m10:/data/outputs_dd_voc_rank
say "1. rescores"
rescore inst "$ENV0" $LANG_NEW ":extra:pend:dd" $VOC /data/voc2012/substrate_cc.pkl &
rescore inst_relaxA "$ENV0 -e SQ_PURITY_ALPHA=0.10 -e SQ_COH_MIN=0.5" $LANG_NEW ":extra:pend:dd" $VOC /data/voc2012/substrate_cc.pkl &
rescore inst_relaxB "$ENV0 -e SQ_PURITY_OFF=1" $LANG_NEW ":extra:pend:dd" $VOC /data/voc2012/substrate_cc.pkl &
rescore inst_slope "$ENV0" /data/outputs:/data/outputs_extra ":extra" $VOC /data/voc2012/substrate_cc.pkl &
rescore sbd "$ENV0" /data/outputs_sbd_rank:/data/outputs_sbd_pending_rank ":pend" $SBD /data/sbd/substrate_cc.pkl &
rescore sbd_slope "$ENV0" /data/outputs_sbd "" $SBD /data/sbd/substrate_cc.pkl &
rescore ea "$ENVC" /data/outputs_ea "" $EAV /data/voc2012_ea/substrate_cc.pkl &
rescore ea_auc "$ENVC" /data/outputs_ea_rank_m10 "" $EAV /data/voc2012_ea/substrate_cc.pkl &
wait
for r in inst inst_relaxA inst_relaxB inst_slope sbd sbd_slope ea ea_auc; do [ -f $L/.rescore_$r ] || fail "rescore $r"; done
for f in slices_index.json image_descriptions.json; do
  cmp -s $D/$f $O/slice_quality_inst_geo/$f && echo "  inst: $f identical to the validated copy slice_quality_inst_geo" \
    || fail "inst: $f differs from the validated copy"
done
summary $O/slice_quality_inst_idjoin $D $O/slice_quality_inst_slope_idjoin $O/slice_quality_inst_slope \
  $O/slice_quality_inst_relaxA $O/slice_quality_inst_relaxB $O/slice_quality_sbd_idjoin $SB \
  $O/slice_quality_sbd_slope_idjoin $O/slice_quality_sbd_slope $O/slice_quality_ea_idjoin $O/slice_quality_ea \
  $O/slice_quality_ea_auc_idjoin $O/slice_quality_ea_auc

# ---- 2. the inputs of the GPU steps
python3 -I - <<'PY' || fail "new situations"
import glob, json, re
P = "$HOME/storage/pascal_vlm"
L = "outputs/geo_join_logs"
cls_of = lambda sid: re.sub(r"_(extra|dd|pend)$", "", sid.rsplit("_", 1)[0])
def sits(d):
    return {(cls_of(s["slice_id"]), s["hypothesis"]) for s in json.load(open(d + "/slices_index.json"))["slices"]
            if s["source"] == "language" and s.get("in_descriptions")}
new, old = sits("outputs/slice_quality_inst"), sits("outputs/slice_quality_inst_idjoin")
open(L + "/situations_voc.txt", "w").write("".join("+ %s | %s\n" % x for x in sorted(new - old))
                                         + "".join("- %s | %s\n" % x for x in sorted(old - new)))
have = set()                                     # whole-class groundings on model 2's objects
for d in ("outputs_model2_inst111", "outputs_model2_dd"):
    for f in glob.glob(P + "/" + d + "/voc_paper_*_full.json"):
        r = json.load(open(f))
        have |= {(r["label"], c["query"]) for c in r["candidates"] if len(c.get("scored", [])) == r["num_instances"]}
todo = sorted(new - have)
json.dump({"slices": [{"slice_id": "%s_%d" % (c, k), "source": "language", "in_descriptions": True, "hypothesis": q}
                      for k, (c, q) in enumerate(todo)]}, open(L + "/m2_new_index.json", "w"), indent=1)
open(L + "/m2_classes.txt", "w").write(",".join(sorted({c for c, _ in todo})))
print("  VOC situations %d -> %d (+%d -%d); to ground on model 2: %d" % (len(old), len(new), len(new - old),
                                                                        len(old - new), len(todo)))
PY
sed 's/^/    /' $L/situations_voc.txt
[ -s $L/m2_classes.txt ] || touch $L/.done_m2
if [ ! -f $L/.texts_done ]; then
  cpuc -e SLOPE_RUN=$O/slice_quality_inst_slope -e OUT=$C $IMG bash -c "$PIP; python -u slice_quality/examples/slicelens_style_texts.py" \
    > $L/texts_slope.log 2>&1 || fail "SliceLens-style texts (slope)"
  cpuc -e SLOPE_RUN=$D -e OUT=$CA $IMG bash -c "$PIP; python -u slice_quality/examples/slicelens_style_texts.py" \
    > $L/texts_auc.log 2>&1 || fail "SliceLens-style texts (rank test)"
  [ -f $CA/empty.json ] || echo '{}' > $CA/empty.json
  touch $L/.texts_done
fi
say "2. texts: $(head -1 $L/texts_slope.log) | $(head -1 $L/texts_auc.log)"

# ---- 3. GPU 5 and 6: one shared queue
run_job() {   # $1 gpu, $2 job. 0 done, 1 failed, 2 stopped by the guard
  local g=$1 j=$2 name=geo_${2}_gpu$1 log=$L/$2.log mark
  setsid nohup bash $O/gpu_guard.sh $g $name > $L/guard_$name.log 2>&1 < /dev/null &
  dock() { docker run --rm --name $name --gpus device=$g -u $(id -u):$(id -g) $MOUNTS $MODELS -w /workspace "$@" > $log 2>&1; }
  local Q="-e SQ_QWEN_DEVICE_MAP=cuda:0"
  local CV="-e JUDGE_4BIT=1 -e ALL=1 -e CONTROL_N=150"
  local JC="-e KINDS=narrative,pattern -e OLD_DUP=0 -e JUDGE_4BIT=1 -e NSHARDS=2"
  case $j in
    m2)        mark=REGROUND_DONE
               dock $ENVC $Q $IMG bash -c "$PIP; python -u examples/reground_validated.py --slices-index $L/m2_new_index.json \
                 --substrate-cache $D/model2/substrate.pkl --vlm-backend qwen --vlm-model /models/qwen \
                 --out-dir /data/outputs_model2_geo --classes $(cat $L/m2_classes.txt)";;
    voc_prec)  mark=PRECISION_EVAL_DONE
               dock $ENVC $Q -e SQ_OUT=$D -e PREC_MAX_EVALS=3000 $IMG bash -c "$PIP; python -u slice_quality/examples/eval_description_precision.py";;
    sbd_prec)  mark=PRECISION_EVAL_DONE
               dock $ENV0 $Q -e SQ_OUT=$SB -e PREC_MAX_EVALS=4500 $IMG bash -c "$PIP; python -u slice_quality/examples/eval_description_precision.py";;
    voc_cause) mark=JUDGE_CAUSE_DONE
               dock $ENVC -e SQ_OUT=$D -e JUDGE_4BIT=1 $IMG bash -c "$PIP; python -u slice_quality/examples/judge_cause_plausibility.py";;
    voc_cv0|voc_cv1) mark=JUDGE_CAUSE_DONE
               dock $ENVC -e SQ_OUT=$D $CV -e SHARD=${j: -1}/2 $IMG bash -c "$PIP; python -u slice_quality/examples/judge_cause_plausibility.py";;
    sbd_cv0|sbd_cv1|sbd_cv2) mark=JUDGE_CAUSE_DONE
               dock $ENV0 -e SQ_OUT=$SB $CV -e SHARD=${j: -1}/3 $IMG bash -c "$PIP; python -u slice_quality/examples/judge_cause_plausibility.py";;
    cmp_cl0|cmp_cl1) mark=CLAIMS_SHARD_DONE
               dock $ENVC -e SQ_OUT=$C $JC -e SHARD=${j: -1} $IMG bash -c "$PIP; python -u slice_quality/examples/judge_claims.py";;
    cmpa_cl)   mark=CLAIMS_SHARD_DONE
               dock $ENVC -e SQ_OUT=$CA -e KINDS=pattern -e OLD_DUP=0 -e JUDGE_4BIT=1 -e SHARD=0 -e NSHARDS=1 $IMG bash -c \
                 "$PIP; python -u slice_quality/examples/judge_claims.py";;
    cmp_rub)   mark=JUDGE_DONE
               dock $ENVC -e SQ_OUT=$D -e JUDGE_4BIT=1 -e NARR=$D/error_narratives.json -e DESC=$C/image_descriptions.json \
                 -e DESC_TR=/nonexistent -e OUT=$C/judge_descriptions $IMG bash -c "$PIP; python -u slice_quality/examples/judge_descriptions.py";;
    cmpa_rub)  mark=JUDGE_DONE
               dock $ENVC -e SQ_OUT=$D -e JUDGE_4BIT=1 -e NARR=$CA/empty.json -e DESC=$CA/image_descriptions.json \
                 -e DESC_TR=/nonexistent -e OUT=$CA/judge_descriptions $IMG bash -c "$PIP; python -u slice_quality/examples/judge_descriptions.py";;
  esac
  sleep 3
  grep -q GUARD_STOPPED $L/guard_$name.log && return 2
  grep -q "$mark" $log && return 0
  tail -n 3 $log; return 1; }
JOBS="m2 voc_prec voc_cause voc_cv0 voc_cv1 cmp_cl0 cmp_cl1 cmpa_cl cmp_rub cmpa_rub sbd_prec sbd_cv0 sbd_cv1 sbd_cv2"
queue() {     # $1 gpu: takes the next job nobody has taken
  local g=$1 j rc
  for j in $JOBS; do
    [ -f $L/.done_$j ] && continue
    mkdir $L/.claim_$j 2>/dev/null || continue
    while true; do
      wait_gpu $g || { rmdir $L/.claim_$j; say "[gpu$g] deadline: queue ends"; return 1; }
      say "[gpu$g] $j"
      run_job $g $j; rc=$?
      if [ $rc = 2 ]; then say "[gpu$g] the guard stopped $j (another user on the GPU): retried when it is free"; sleep 300; continue; fi
      if [ $rc = 1 ]; then say "[gpu$g] $j FAILED, the queue goes on"; touch $L/.failed_$j; break; fi
      touch $L/.done_$j; say "[gpu$g] $j done"; break
    done
  done
  say "[gpu$g] queue over"; }
for j in $JOBS; do [ -f $L/.done_$j ] || { rmdir $L/.claim_$j 2>/dev/null; rm -f $L/.failed_$j; }; done
say "3. GPU queue: $JOBS"
QP=""
for g in $GPUS; do queue $g & QP="$QP $!"; done

# ---- CPU meanwhile
( say "[cpu] slice-image IoU, failure probe, relaxed variants, Pascal-EA victims"
  cpuc -e SQ_OUT=$D $IMG bash -c "$PIP; python -u slice_quality/examples/slice_image_iou.py" > $L/slice_image_iou.log 2>&1
  grep -q SLICE_IMAGE_IOU_DONE $L/slice_image_iou.log || say "[cpu] slice-image IoU FAILED"
  for V in relaxA relaxB; do
    cpuc -e SQ_OUT=${D}_$V $IMG bash -c "$PIP; python -u slice_quality/examples/slice_image_iou.py" > $L/slice_image_iou_$V.log 2>&1
  done
  cpuc -e RUNS="official=$D A=${D}_relaxA B=${D}_relaxB" -e OUT=$O/relaxed_cluster_report $IMG bash -c \
    "$PIP; python -u slice_quality/examples/relaxed_report.py" > $L/relaxed_report.log 2>&1
  grep -q RELAXED_REPORT_DONE $L/relaxed_report.log && grep -E "certified cluster slices" $O/relaxed_cluster_report.txt \
    || say "[cpu] relaxed report FAILED"
  cpuc -e RUN=$D $IMG bash -c "$PIP; python -u slice_quality/examples/failure_probe.py" > $L/failure_probe.log 2>&1
  grep -q FAILURE_PROBE_DONE $L/failure_probe.log || say "[cpu] failure probe FAILED"
  # Pascal-EA: the victims (recall drop on the edited images) are the same; which certified slice holds them is not.
  # ea_planted_eval.py on the slope run beside VOC val's slope run, as on 8 October; its victims go to the rank run.
  cpuc -e EA_RUN=$O/slice_quality_ea -e INST_RUN=$O/slice_quality_inst_slope $IMG bash -c \
    "$PIP; python -u slice_quality/examples/ea_planted_eval.py" > $L/ea_planted.log 2>&1
  grep -q "written outputs/slice_quality_ea/planted_eval.json" $L/ea_planted.log || say "[cpu] Pascal-EA planted eval FAILED"
  cp $O/slice_quality_ea/planted_eval.json $O/slice_quality_ea/planted_eval.txt $O/slice_quality_ea_auc/
  for R in ea ea_auc; do
    cpuc -e EA_RUN=$O/slice_quality_$R $IMG bash -c "$PIP; python -u slice_quality/examples/ea_branch_coverage.py" > $L/ea_branch_$R.log 2>&1
    grep -q EA_BRANCH_DONE $L/ea_branch_$R.log && say "[cpu] $R: $(head -1 $L/ea_branch_$R.log)" || say "[cpu] $R branch coverage FAILED"
  done
  say "[cpu] done" ) &
QC=$!
wait $QP $QC

# ---- 4. merges and the rest (CPU)
say "4. merges, model 2, gallery, numbers"
for j in $JOBS; do [ -f $L/.done_$j ] || say "  WARNING: $j not done: what depends on it keeps the copy's file"; done
merge_cause() {   # $1 run, $2 settings, $3 tag
  docker run --rm -u $(id -u):$(id -g) $2 $MOUNTS -e SQ_OUT=$1 -e JUDGE_4BIT=1 -e ALL=1 -e CONTROL_N=150 -w /workspace $IMG bash -c \
    "$PIP; python -u slice_quality/examples/judge_cause_plausibility.py" > $L/merge_$3.log 2>&1
  grep -q JUDGE_CAUSE_DONE $L/merge_$3.log && tail -n 12 $1/judge_cause_val.txt | head -4 || say "  merge $3 FAILED"; }
[ -f $L/.done_voc_cv0 ] && [ -f $L/.done_voc_cv1 ] && merge_cause $D "$ENVC" voc_cv
[ -f $L/.done_sbd_cv0 ] && [ -f $L/.done_sbd_cv1 ] && [ -f $L/.done_sbd_cv2 ] && merge_cause $SB "$ENV0" sbd_cv
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
say "CHAIN_GEO_JOIN_DONE"
