#!/bin/bash
# The data-driven hypotheses of VOC (SliceLens's own generator, 96 over 16 classes, grounded on the 120-object sample
# on 14 October) put into the official run (Daniele, 2026-10-24: "inserisci le frasi data driven nel run ufficiale",
# "lancialo su gpu6, ma se viene utilizzata gpu sei interrompi che non ho la prenotazione").
# The protocol of the official run: the rank test on the sample (BH within the 96, as the extra and the pending
# hypotheses had their own family) sends 12 to the whole class; the rank test there certifies; then the checks of the
# official run on a COPY, and the official run rescored in place as on 23 October (chain_official_111.sh).
#   GPU 6 ONLY, one step at a time, every container under outputs/gpu_guard.sh. GPU 6 is not reserved: if another
#   user is on it before a step, or the guard stops a step, the chain ENDS -- it never waits and never retries.
#   1  GPU  the 12 hypotheses on every object of their class                         -> /data/outputs_dd_voc_full
#   2  CPU  rank verdict on the whole classes (BH within the 12)                      -> /data/outputs_dd_voc_rank
#           the copy outputs/slice_quality_inst_withdd rescored with them beside the official groundings
#   3  GPU  the checks on the copy: precision of the shipped phrases, the new situations on the second segmenter's
#           objects, the plausibility of the blamed situation (Gemma 4-bit, as for the 67 of 23 October)
#   4  CPU  the report, new against official
#   5  CPU  official: backup outputs/slice_quality_inst_auc111, rescore in place (must equal the copy), validations
#           moved in, second segmenter merged, slice-image IoU, gallery, cluster variants
#   setsid nohup bash outputs/chain_dd_official.sh > outputs/dd_official.log 2>&1 < /dev/null &
set -u
cd ~/storage/SLICELENS
G=6
D=outputs/slice_quality_inst
C=${D}_withdd
B=${D}_auc111
L=$C/logs
PV=$HOME/storage/pascal_vlm
mkdir -p $L
MOUNTS="-v $HOME/storage/SLICELENS:/workspace -v $PV:/data -v $DATASETS:/ds"
MODELS="-v $STORAGE/models/qwen2.5-vl-7b:/models/qwen -v $STORAGE/models/gemma-4-12B-it:/models/gemma"
ENV0="-e HOME=/tmp -e PYTHONIOENCODING=utf-8 -e PYTHONPATH=/workspace"
ENV="$ENV0 -e SQ_CLAIM_COV=0.5 -e SQ_GT_INSTANCES=object -e SQ_QWEN_DEVICE_MAP=cuda:0"
PIP="pip install --user -q scipy scikit-learn >/dev/null 2>&1"
IMG=segciani:dev
LANG_OLD=/data/outputs_rank:/data/outputs_extra_rank:/data/outputs_pending_rank
LANG_NEW=$LANG_OLD:/data/outputs_dd_voc_rank
TAGS_NEW=":extra:pend:dd"
say() { echo "$(date '+%m-%d %H:%M') $*"; }
fail() { say "CHAIN_DD_OFFICIAL_FAIL: $1"; exit 1; }
others() {
  local uuid me pid uid
  uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $1); me=$(id -u)
  for pid in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v u="$uuid" '$1==u {print $2}'); do
    uid=$(ps -o uid= -p "$pid" 2>/dev/null | tr -d ' '); [ -n "$uid" ] && [ "$uid" != "$me" ] && echo $pid
  done; }
gpu() {       # $1 name, $2 log, rest: docker args. Ends the chain if GPU 6 is taken (before or during the step).
  local name=$1 log=$2; shift 2
  [ -n "$(others $G)" ] && { say "GPU $G in use by another user before $name: chain ends (no reservation)"; say "CHAIN_DD_OFFICIAL_STOPPED"; exit 0; }
  say "[gpu$G] $name"
  setsid nohup bash outputs/gpu_guard.sh $G $name > $L/guard_$name.log 2>&1 < /dev/null &
  docker run --rm --name $name --gpus device=$G -u $(id -u):$(id -g) $ENV $MOUNTS $MODELS -w /workspace "$@" > $log 2>&1
  sleep 3
  if grep -q GUARD_STOPPED $L/guard_$name.log; then
    say "the guard stopped $name: another user took GPU $G. Chain ends, nothing restarted"; say "CHAIN_DD_OFFICIAL_STOPPED"; exit 0
  fi
  grep -q Traceback $log && { tail -3 $log; fail "$name"; }
  return 0; }
cpu() { docker run --rm -u $(id -u):$(id -g) $ENV $MOUNTS -w /workspace "$@"; }
cpu0() { docker run --rm -u $(id -u):$(id -g) $ENV0 $MOUNTS -w /workspace "$@"; }   # the rescores: as every earlier rescore
summary() { cpu $IMG bash -c "$PIP; python -c \"
import sys; sys.path.insert(0, 'docs')
from _revision_numbers import load_inst_summary as L
for d in sys.argv[1:]:
    s = L(d); print('  %-40s situations %d, clusters %d, errors in a group %.1f%%, images with a situation %d' % (
        d, s['slices']['language'], s['slices']['cluster'], 100 * s['cov_union'] / s['n_err'], s['situational_images']))
\" $*" 2>&1 | tail -n $#; }

say "== VOC data-driven hypotheses into the official run (GPU $G, stops if another user takes it) =="
# ---- 0. the 12 the rank test sends to the whole class (verdict of 24 October on the 14 October sample groundings)
python3 -I - <<'PY' || fail "set-up"
import json, collections
rows = [r for r in json.load(open("outputs/slice_quality_inst_dd/dd_auc_cc.json")) if r["auc_sys"]]
by = collections.defaultdict(list)
for r in rows:
    by[r["cls"]].append(r["query"])
sl = [{"slice_id": "%s_%d" % (c, k), "source": "language", "in_descriptions": True, "hypothesis": q}
      for c, qs in by.items() for k, q in enumerate(qs)]
json.dump({"slices": sl}, open("outputs/slice_quality_inst_withdd/dd_index.json", "w"), indent=1)
print("  %d hypotheses to the whole class: %s" % (len(rows), ", ".join("%s %d" % (c, len(q)) for c, q in sorted(by.items()))))
PY

# ---- 1. GPU: on every object of their class
if [ ! -f $L/.full_done ]; then
  gpu ddoff_full_gpu$G $L/full.log $IMG bash -c "$PIP; python -u examples/reground_validated.py \
    --slices-index $C/dd_index.json --substrate-cache /data/voc2012/substrate_cc.pkl --vlm-backend qwen \
    --vlm-model /models/qwen --out-dir /data/outputs_dd_voc_full"
  grep -q REGROUND_DONE $L/full.log || fail "whole-class grounding"
  touch $L/.full_done
fi

# ---- 2. CPU: the rank verdict on the whole classes, the copy rescored
cpu -e RUNS_DIR=/data/outputs_dd_voc_full -e STAGE=full -e OUT=$C/dd_auc_full -e RANK_COPY=/data/outputs_dd_voc_rank \
  $IMG bash -c "$PIP; python -u slice_quality/examples/dd_auc.py" > $L/dd_auc_full.log 2>&1
grep -q DD_AUC_DONE $L/dd_auc_full.log || fail "rank verdict"
grep -v "^  " $L/dd_auc_full.log | head -1; grep " AUC slope" $L/dd_auc_full.log
cp $D/result_tagged.pkl $D/cluster_names.json $C/
cpu0 -e SQ_OUT=$C -e LANG_DIRS=$LANG_NEW -e LANG_TAGS=$TAGS_NEW -e VOC_DATA=/data/voc2012/VOCdevkit/VOC2012 -e GALLERY=0 \
  $IMG bash -c "$PIP; python -u slice_quality/examples/rescore_with_language.py" > $L/rescore.log 2>&1
grep -q "DONE rescore" $L/rescore.log || { tail -5 $L/rescore.log; fail "rescore of the copy"; }
summary $D $C
NEW=$(python3 -I - <<'PY'
import json, re
cls_of = lambda sid: re.sub(r"_(extra|dd|pend)$", "", sid.rsplit("_", 1)[0])
def sits(d):
    return {(cls_of(s["slice_id"]), s["hypothesis"]) for s in json.load(open(d + "/slices_index.json"))["slices"]
            if s["source"] == "language" and s.get("in_descriptions")}
new = sorted(sits("outputs/slice_quality_inst_withdd") - sits("outputs/slice_quality_inst"))
lost = sorted(sits("outputs/slice_quality_inst") - sits("outputs/slice_quality_inst_withdd"))
json.dump({"slices": [{"slice_id": "%s_%d" % (c, k), "source": "language", "in_descriptions": True, "hypothesis": q}
                      for k, (c, q) in enumerate(new)]},
          open("outputs/slice_quality_inst_withdd/new_index.json", "w"), indent=1)
open("outputs/slice_quality_inst_withdd/new_situations.txt", "w").write(
    "".join("+ %s | %s\n" % x for x in new) + "".join("- %s | %s\n" % x for x in lost))
print(len(new))
PY
)
sed 's/^/  /' $C/new_situations.txt
[ "${NEW:-0}" -gt 0 ] || { say "no new situation: the official run stays as it is"; say "CHAIN_DD_OFFICIAL_DONE"; exit 0; }

# ---- 3. GPU: the checks of the official run on the copy (resumable: a step already done is skipped)
cp $D/error_narratives.json $D/typed_per_image.json $C/ 2>/dev/null
[ -d $C/train36 ] || cp -r $D/train36 $C/ 2>/dev/null
for f in judge_cause judge_cause_val; do for e in json txt; do [ -f $C/$f.$e ] || cp $D/$f.$e $C/$f.$e 2>/dev/null; done; done
if [ ! -f $L/.precision_done ]; then
  gpu ddoff_prec_gpu$G $L/precision.log -e SQ_OUT=$C $IMG bash -c "$PIP; python -u slice_quality/examples/eval_description_precision.py"
  tail -n 3 $L/precision.log; touch $L/.precision_done
fi
if [ ! -f $L/.model2_done ]; then
  CL=$(python3 -c "import json; print(','.join(sorted({s['slice_id'].rsplit('_', 1)[0] for s in json.load(open('$C/new_index.json'))['slices']})))")
  gpu ddoff_m2_gpu$G $L/model2.log $IMG bash -c "$PIP; python -u examples/reground_validated.py \
    --slices-index $C/new_index.json --substrate-cache $D/model2/substrate.pkl --vlm-backend qwen \
    --vlm-model /models/qwen --out-dir /data/outputs_model2_dd --classes $CL"
  grep -q REGROUND_DONE $L/model2.log || fail "second segmenter"
  touch $L/.model2_done
fi
if [ ! -f $L/.cause_done ]; then
  gpu ddoff_cause_gpu$G $L/cause.log -e SQ_OUT=$C -e JUDGE_4BIT=1 $IMG bash -c \
    "$PIP; python -u slice_quality/examples/judge_cause_plausibility.py"
  tail -n 3 $L/cause.log
  gpu ddoff_causeval_gpu$G $L/cause_val.log -e SQ_OUT=$C -e JUDGE_4BIT=1 -e ALL=1 -e CONTROL_N=150 $IMG bash -c \
    "$PIP; python -u slice_quality/examples/judge_cause_plausibility.py"
  tail -n 3 $L/cause_val.log; touch $L/.cause_done
fi

# ---- 4. CPU: the report
cpu -e RUNS_DIR=/data/outputs_model2_dd -e STAGE=full -e OUT=$C/model2_new_dd_auc_full $IMG bash -c \
  "$PIP; python -u slice_quality/examples/dd_auc.py" > $L/model2_auc.log 2>&1
head -n 1 $L/model2_auc.log
cpu -e OFFICIAL=$D -e COPY=$C -e M2_NEW=/data/outputs_model2_dd -e M2_OFF=/data/outputs_model2_inst111 $IMG bash -c \
  "$PIP; python -u slice_quality/examples/pending_validation_report.py" > $L/report.log 2>&1
cat $C/validation_report.txt || fail "report"

# ---- 5. CPU: official
say "== the official run, with the data-driven situations =="
[ -d $B ] || cp -a $D $B
cpu0 -e SQ_OUT=$D -e LANG_DIRS=$LANG_NEW -e LANG_TAGS=$TAGS_NEW -e VOC_DATA=/data/voc2012/VOCdevkit/VOC2012 -e GALLERY=0 \
  $IMG bash -c "$PIP; python -u slice_quality/examples/rescore_with_language.py" > $D/rescore_dd.log 2>&1
grep -q "DONE rescore" $D/rescore_dd.log || { tail -5 $D/rescore_dd.log; fail "official rescore"; }
for f in slices_index.json image_descriptions.json; do
  cmp -s $D/$f $C/$f && echo "  $f identical to the validated copy" || fail "$f differs from the validated copy"
done
for f in eval_precision judge_cause judge_cause_val; do for e in json txt; do cp $C/$f.$e $D/$f.$e; done; done
cp $C/validation_report.txt $C/validation_report.json $D/
mkdir -p $D/night
cp $L/precision.log $D/night/precision_dd.log; cp $L/cause.log $D/night/cause_dd.log; cp $L/cause_val.log $D/night/cause_val_dd.log
M=$PV/outputs_model2_inst_dd
mkdir -p $M
python3 -I - $PV/outputs_model2_inst111 $PV/outputs_model2_dd $M <<'PY' || fail "model 2 merge"
import glob, json, os, sys
a, b, out = sys.argv[1:4]
cls = sorted({os.path.basename(f)[10:-10] for d in (a, b) for f in glob.glob(d + "/voc_paper_*_full.json")})
n = 0
for c in cls:
    parts = [json.load(open(p)) for p in (a + "/voc_paper_%s_full.json" % c, b + "/voc_paper_%s_full.json" % c)
             if os.path.exists(p)]
    r = parts[0]
    assert all(x["num_instances"] == r["num_instances"] for x in parts), c
    seen, cands = set(), []
    for x in parts:
        for cd in x["candidates"]:
            if len(cd["scored"]) == x["num_instances"] and cd["query"] not in seen:
                seen.add(cd["query"])
                cands.append(cd)
    r = dict(r, candidates=cands, systematic_error_slices=sorted({q for x in parts for q in x["systematic_error_slices"]}))
    json.dump(r, open(out + "/voc_paper_%s_full.json" % c, "w"))
    n += len(cands)
print("  model 2 groundings merged: %d classes, %d hypotheses" % (len(cls), n))
PY
cpu -e RUNS_DIR=/data/outputs_model2_inst_dd -e STAGE=full -e OUT=$D/model2/dd_auc_full \
  -e RANK_COPY=/data/outputs_model2_inst_dd_rank $IMG bash -c "$PIP; python -u slice_quality/examples/dd_auc.py" \
  > $D/night/model2_auc_dd.log 2>&1
head -n 1 $D/night/model2_auc_dd.log
cpu -e SQ_OUT=$D -e OUT_TAG=model2 -e LANG_DIR=/data/outputs_model2_inst_dd_rank $IMG bash -c \
  "$PIP; python -u slice_quality/examples/run_second_model_language.py" > $D/night/model2_language_dd.log 2>&1
grep -q STAGE3_DONE $D/night/model2_language_dd.log || { tail -5 $D/night/model2_language_dd.log; fail "model 2 language"; }
grep -E "language slices total|KEPT|images with a certified" $D/night/model2_language_dd.log
cpu -e SQ_OUT=$D $IMG bash -c "$PIP; python -u slice_quality/examples/slice_image_iou.py" > $D/slice_image_iou.log 2>&1
grep -E "^(language|cluster|all)" $D/slice_image_iou.txt | head -6
cpu -e SQ_OUT=$D $IMG bash -c "$PIP; python -u slice_quality/examples/regen_gallery.py" > $D/regen_gallery_dd.log 2>&1
grep -q "gallery regenerated" $D/regen_gallery_dd.log || { tail -5 $D/regen_gallery_dd.log; fail gallery; }
cpu -e SQ_OUT=$D -e GALLERY_SRC=gallery_full_tagged.html -e GALLERY_DST=gallery_share.html $IMG \
  python -u docs/make_share_gallery.py | tail -2
for V in A B; do
  O=${D}_relax$V
  case $V in A) X="-e SQ_PURITY_ALPHA=0.10 -e SQ_COH_MIN=0.5";; B) X="-e SQ_PURITY_OFF=1";; esac
  cp $D/result_tagged.pkl $O/
  cpu0 $X -e SQ_OUT=$O -e LANG_DIRS=$LANG_NEW -e LANG_TAGS=$TAGS_NEW -e VOC_DATA=/data/voc2012/VOCdevkit/VOC2012 \
    -e GALLERY=0 $IMG bash -c "$PIP; python -u slice_quality/examples/rescore_with_language.py" > $O/rescore_dd.log 2>&1
  grep -q "DONE rescore" $O/rescore_dd.log || { tail -5 $O/rescore_dd.log; fail "rescore $V"; }
  cpu -e SQ_OUT=$O $IMG bash -c "$PIP; python -u slice_quality/examples/slice_image_iou.py" > $O/slice_image_iou.log 2>&1
done
cpu -e RUNS="official=$D A=${D}_relaxA B=${D}_relaxB" -e OUT=outputs/relaxed_cluster_report $IMG bash -c \
  "$PIP; python -u slice_quality/examples/relaxed_report.py" > outputs/relaxed_report.log 2>&1
grep -E "certified cluster slices" outputs/relaxed_cluster_report.txt
cpu $IMG bash -c "$PIP; python -u slice_quality/examples/relaxed_gallery.py" > outputs/relaxed_gallery.log 2>&1 || echo "  WARN relaxed gallery"
bash outputs/chain_gallery_cluster_variants.sh | tail -4
summary $B $D ${D}_relaxA ${D}_relaxB
say "CHAIN_DD_OFFICIAL_DONE"
