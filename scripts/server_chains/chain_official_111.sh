#!/bin/bash
# The hypotheses that the rank test certifies on the 120-object sample but the slope rule had kept from the
# whole-class grounding, made official (Daniele, 2026-10-23: "ok rendile ufficiali e aggiorna documento e slide").
# They were grounded on whole classes on 22 October (chain_night_pending.sh -> /data/outputs_pending, rank verdict
# in /data/outputs_pending_rank by pending_rescore.sh) and validated on the copy outputs/slice_quality_inst_pending
# in the night of 22-23 October (chain_validate_pending.sh). CPU only:
#   1. full copy of the official run as it is now: outputs/slice_quality_inst_auc44
#   2. the official run rescored in place with the pending groundings beside the official ones; it must certify
#      exactly what the validated copy certifies
#   3. the validations already measured on that copy (precision, both cause judges) moved in
#   4. second segmenter: its groundings of the official and of the new situations in one directory, the rank test
#      over all of them (BH over the whole set, as on model 1), the two-branch result
#   5. the IoU of each slice's class on its images; the gallery and its share copy; the gallery of the cluster
#      variants, whose copies are rescored with the same groundings first
#   setsid nohup bash outputs/chain_official_111.sh > outputs/official_111.log 2>&1 < /dev/null &
set -u
cd ~/storage/SLICELENS
D=outputs/slice_quality_inst
P=${D}_pending
B=${D}_auc44
PV=$HOME/storage/pascal_vlm
MOUNTS="-v $HOME/storage/SLICELENS:/workspace -v $PV:/data -v $DATASETS:/ds"
ENV0="-e HOME=/tmp -e PYTHONIOENCODING=utf-8 -e PYTHONPATH=/workspace"
ENV="-e HOME=/tmp -e PYTHONIOENCODING=utf-8 -e PYTHONPATH=/workspace -e SQ_CLAIM_COV=0.5 -e SQ_GT_INSTANCES=object"
PIP="pip install --user -q scipy scikit-learn >/dev/null 2>&1"
IMG=segciani:dev
LANG_DIRS=/data/outputs_rank:/data/outputs_extra_rank:/data/outputs_pending_rank
step() { echo "$(date '+%m-%d %H:%M') == $1 =="; }
cpu() { docker run --rm -u $(id -u):$(id -g) $ENV $MOUNTS -w /workspace "$@"; }
cpu0() { docker run --rm -u $(id -u):$(id -g) $ENV0 $MOUNTS -w /workspace "$@"; }   # the rescores: as every earlier rescore
fail() { echo "CHAIN_OFFICIAL_111_FAIL: $1"; exit 1; }

step "1. full copy of the official run (44 situations)"
[ -d $B ] || cp -a $D $B
du -sh $B

step "2. official run rescored with the pending groundings"
cpu0 -e SQ_OUT=$D -e LANG_DIRS=$LANG_DIRS -e LANG_TAGS=":extra:pend" -e VOC_DATA=/data/voc2012/VOCdevkit/VOC2012 \
  -e GALLERY=0 $IMG bash -c "$PIP; python -u slice_quality/examples/rescore_with_language.py" > $D/rescore_auc111.log 2>&1
grep -q "DONE rescore" $D/rescore_auc111.log || { tail -5 $D/rescore_auc111.log; fail rescore; }
grep -E "language slices total|images with SliceLens" $D/rescore_auc111.log
for f in slices_index.json image_descriptions.json; do
  cmp -s $D/$f $P/$f && echo "  $f identical to the validated copy" || fail "$f differs from the validated copy"
done

step "3. validations measured on the copy"
for f in eval_precision judge_cause judge_cause_val; do for e in json txt; do cp $P/$f.$e $D/$f.$e; done; done
cp $P/validation_report.txt $P/validation_report.json $D/
cp $P/logs/precision.log $D/night/precision_111.log; cp $P/logs/cause.log $D/night/cause_111.log
cp $P/logs/cause_val.log $D/night/cause_val_111.log

step "4. second segmenter: official + new groundings, rank test over the whole set, two-branch result"
M=$PV/outputs_model2_inst111
mkdir -p $M
python3 -I - $PV/outputs_model2_inst $PV/outputs_model2_pending $M <<'PY' || fail "model 2 merge"
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
cpu -e RUNS_DIR=/data/outputs_model2_inst111 -e STAGE=full -e OUT=$D/model2/dd_auc_full \
  -e RANK_COPY=/data/outputs_model2_inst111_rank $IMG bash -c "$PIP; python -u slice_quality/examples/dd_auc.py" \
  > $D/night/model2_auc_111.log 2>&1
head -n 1 $D/night/model2_auc_111.log
cpu -e SQ_OUT=$D -e OUT_TAG=model2 -e LANG_DIR=/data/outputs_model2_inst111_rank $IMG bash -c \
  "$PIP; python -u slice_quality/examples/run_second_model_language.py" > $D/night/model2_language_111.log 2>&1
grep -q STAGE3_DONE $D/night/model2_language_111.log || { tail -5 $D/night/model2_language_111.log; fail "model 2 language"; }
grep -E "language slices total|KEPT|images with a certified" $D/night/model2_language_111.log

step "5a. IoU of each slice's class on its images"
cpu -e SQ_OUT=$D $IMG bash -c "$PIP; python -u slice_quality/examples/slice_image_iou.py" > $D/slice_image_iou.log 2>&1
grep -E "^(language|cluster|all)" $D/slice_image_iou.txt | head -6

step "5b. official gallery and share copy"
cpu -e SQ_OUT=$D $IMG bash -c "$PIP; python -u slice_quality/examples/regen_gallery.py" > $D/regen_gallery_111.log 2>&1
grep -q "gallery regenerated" $D/regen_gallery_111.log || { tail -5 $D/regen_gallery_111.log; fail gallery; }
cpu -e SQ_OUT=$D -e GALLERY_SRC=gallery_full_tagged.html -e GALLERY_DST=gallery_share.html $IMG \
  python -u docs/make_share_gallery.py | tail -2

step "5c. cluster variants rescored with the same groundings, their report, their two pages"
for V in A B; do
  O=${D}_relax$V
  case $V in A) X="-e SQ_PURITY_ALPHA=0.10 -e SQ_COH_MIN=0.5";; B) X="-e SQ_PURITY_OFF=1";; esac
  cp $D/result_tagged.pkl $O/
  cpu0 $X -e SQ_OUT=$O -e LANG_DIRS=$LANG_DIRS -e LANG_TAGS=":extra:pend" -e VOC_DATA=/data/voc2012/VOCdevkit/VOC2012 \
    -e GALLERY=0 $IMG bash -c "$PIP; python -u slice_quality/examples/rescore_with_language.py" > $O/rescore_111.log 2>&1
  grep -q "DONE rescore" $O/rescore_111.log || { tail -5 $O/rescore_111.log; fail "rescore $V"; }
  cpu -e SQ_OUT=$O $IMG bash -c "$PIP; python -u slice_quality/examples/slice_image_iou.py" > $O/slice_image_iou.log 2>&1
done
cpu -e RUNS="official=$D A=${D}_relaxA B=${D}_relaxB" -e OUT=outputs/relaxed_cluster_report $IMG bash -c \
  "$PIP; python -u slice_quality/examples/relaxed_report.py" > outputs/relaxed_report.log 2>&1
grep -E "certified cluster slices" outputs/relaxed_cluster_report.txt
cpu $IMG bash -c "$PIP; python -u slice_quality/examples/relaxed_gallery.py" > outputs/relaxed_gallery.log 2>&1 || echo "  WARN relaxed gallery"
bash outputs/chain_gallery_cluster_variants.sh | tail -4
echo "$(date '+%m-%d %H:%M') CHAIN_OFFICIAL_111_DONE"
