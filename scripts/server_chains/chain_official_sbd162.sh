#!/bin/bash
# SBD with the 162 hypotheses made official (Daniele, 28 October 2026: "sì, rendilo ufficiale e aggiorna la sezione SBD").
# The 162 SBD hypotheses that the rank test certifies on the 120-object sample but the slope rule had kept from the
# whole-class grounding were grounded on whole classes (chain_sbd_pending.sh, 23-27 October, rank verdict in
# /data/outputs_sbd_pending_rank) and validated on the copy outputs/slice_quality_sbd_pending (precision of the phrases,
# plausibility of the causes on every image: chain_validate_sbd_*.sh, chain_judge_sbd_shards.sh, 27-28 October).
# CPU only, as chain_official_111.sh did for VOC:
#   1. full copy of the official SBD run as it is now: outputs/slice_quality_sbd_rank43
#   2. the official run rescored in place with the pending groundings beside the official ones; it must certify
#      exactly what the validated copy certifies
#   3. the validations measured on the copy moved in
#   4. the numbers of the thesis: sbd_summary.json (doc_numbers_sbd_ea.py) and the paired comparison with SliceLens
#      (slicelens_cmp_stats.py; the version before is kept as outputs/slicelens_cmp_stats_sbd43.*)
# Every container runs as the user (-u $(id -u):$(id -g)).
#   setsid nohup bash outputs/chain_official_sbd162.sh > outputs/official_sbd162.log 2>&1 < /dev/null &
set -u
cd ~/storage/SLICELENS
D=outputs/slice_quality_sbd
P=${D}_pending
B=${D}_rank43
N=$D/night_sbd162
PV=$HOME/storage/pascal_vlm
MOUNTS="-v $HOME/storage/SLICELENS:/workspace -v $PV:/data -v $DATASETS:/ds"
ENV0="-e HOME=/tmp -e PYTHONIOENCODING=utf-8 -e PYTHONPATH=/workspace"
PIP="pip install --user -q scipy scikit-learn >/dev/null 2>&1"
IMG=segciani:dev
step() { echo "$(date '+%m-%d %H:%M') == $1 =="; }
cpu0() { docker run --rm -u $(id -u):$(id -g) $ENV0 $MOUNTS -w /workspace "$@"; }   # the rescores: as every earlier rescore
fail() { echo "CHAIN_OFFICIAL_SBD162_FAIL: $1"; exit 1; }

step "1. full copy of the official SBD run (43 situations)"
[ -d $B ] || cp -a $D $B || fail "copy"
du -sh $B

step "2. official run rescored with the pending groundings"
cpu0 -e SQ_OUT=$D -e LANG_DIRS=/data/outputs_sbd_rank:/data/outputs_sbd_pending_rank -e LANG_TAGS=":pend" \
  -e VOC_DATA=/data/sbd/VOCdevkit/VOC2012 -e GALLERY=0 $IMG \
  bash -c "$PIP; python -u slice_quality/examples/rescore_with_language.py" > $D/rescore_auc162.log 2>&1
grep -q "DONE rescore" $D/rescore_auc162.log || { tail -5 $D/rescore_auc162.log; fail "rescore"; }
grep -E "language slices total|images with SliceLens" $D/rescore_auc162.log
for f in slices_index.json image_descriptions.json; do
  cmp -s $D/$f $P/$f && echo "  $f identical to the validated copy" || fail "$f differs from the validated copy"
done

step "3. validations measured on the copy"
mkdir -p $N
for f in eval_precision judge_cause_val dd_auc_full; do for e in json txt; do cp $P/$f.$e $D/$f.$e; done; done
cp $P/judge_cause_val_shard*of2.* $N/
cp $P/logs/*.log $N/ 2>/dev/null

step "4. the numbers of the thesis"
cpu0 $IMG bash -c "$PIP; python -u slice_quality/examples/doc_numbers_sbd_ea.py" > $N/doc_numbers.log 2>&1 || fail "doc_numbers"
tail -n 3 $N/doc_numbers.log
[ -f outputs/slicelens_cmp_stats_sbd43.json ] || {
  cp outputs/slicelens_cmp_stats.json outputs/slicelens_cmp_stats_sbd43.json
  cp outputs/slicelens_cmp_stats.txt outputs/slicelens_cmp_stats_sbd43.txt; }
cpu0 $IMG bash -c "$PIP; python -u slice_quality/examples/slicelens_cmp_stats.py" > $N/cmp_stats.log 2>&1 || fail "cmp_stats"
grep -A 8 "ERRORS INSIDE A CERTIFIED GROUP" $N/cmp_stats.log
echo "CHAIN_OFFICIAL_SBD162_DONE"
