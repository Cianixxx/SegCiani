#!/bin/bash
# The 10-correct-objects rule (Daniele, 2026-10-24: "almeno 10 corretti") carried to the last run that still used the
# rank test without it, Pascal-EA (val-EA groundings, slice_quality_ea_auc: sheep has 7 correct objects), and the
# paired tests of the comparison with SliceLens re-run on the official VOC run of 25 October (99 situations). CPU only.
#   1. rank_min_correct.py on /data/outputs_ea_rank (one family) -> /data/outputs_ea_rank_m10
#   2. the copy kept as outputs/slice_quality_ea_auc_nomin; slice_quality_ea_auc rescored in place with the _m10
#      groundings; victims by branch again (ea_branch_coverage.py)
#   3. slicelens_cmp_stats.py (the 23 October results kept as outputs/slicelens_cmp_stats_111.*)
#   setsid nohup bash outputs/chain_rule_ea_stats.sh > outputs/rule_ea_stats.log 2>&1 < /dev/null &
set -u
cd ~/storage/SLICELENS
P=$HOME/storage/pascal_vlm
EA=outputs/slice_quality_ea_auc
MOUNTS="-v $HOME/storage/SLICELENS:/workspace -v $P:/data"
ENV="-e HOME=/tmp -e PYTHONIOENCODING=utf-8 -e PYTHONPATH=/workspace -e SQ_CLAIM_COV=0.5 -e SQ_GT_INSTANCES=object"
PIP="pip install --user -q scipy scikit-learn >/dev/null 2>&1"
IMG=segciani:dev
say() { echo "$(date '+%m-%d %H:%M') $*"; }
cpu() { docker run --rm -u $(id -u):$(id -g) $ENV $MOUNTS -w /workspace "$@"; }

say "== the 10-correct rule on Pascal-EA; the paired tests on the official run of 25 October =="
cpu -e FAMILY=/data/outputs_ea_rank -e SUFFIX=_m10 -e MIN_CORRECT=10 -e OUT=/workspace/$EA/rank_min_correct $IMG bash -c \
  "$PIP; python -u slice_quality/examples/rank_min_correct.py" > outputs/rule_ea_min.log 2>&1
grep -q RANK_MIN_DONE outputs/rule_ea_min.log || { say "rank_min_correct failed"; tail -5 outputs/rule_ea_min.log; say "CHAIN_RULE_EA_FAIL"; exit 1; }
grep -v RANK_MIN_DONE outputs/rule_ea_min.log | grep -v "^$" | tail -20

[ -d ${EA}_nomin ] || cp -r $EA ${EA}_nomin
cpu -e SQ_OUT=$EA -e LANG_DIRS=/data/outputs_ea_rank_m10 -e VOC_DATA=/data/voc2012_ea/VOCdevkit/VOC2012 -e GALLERY=0 $IMG bash -c \
  "$PIP; python -u slice_quality/examples/rescore_with_language.py" > $EA/rescore_m10.log 2>&1
grep -q "DONE rescore" $EA/rescore_m10.log || { say "val-EA rescore failed"; tail -5 $EA/rescore_m10.log; say "CHAIN_RULE_EA_FAIL"; exit 1; }
grep -E "language slices total" $EA/rescore_m10.log
cpu -e EA_RUN=$EA $IMG bash -c "$PIP; python -u slice_quality/examples/ea_branch_coverage.py" > $EA/branch_m10.log 2>&1
head -6 $EA/branch_m10.log
say "  before the rule:"; head -4 ${EA}_nomin/branch_coverage.txt

for x in json txt; do [ -f outputs/slicelens_cmp_stats_111.$x ] || cp outputs/slicelens_cmp_stats.$x outputs/slicelens_cmp_stats_111.$x; done
cpu $IMG bash -c "$PIP; python -u slice_quality/examples/slicelens_cmp_stats.py" > outputs/rule_cmp_stats.log 2>&1
grep -q CMP_STATS_DONE outputs/rule_cmp_stats.log || { say "paired tests failed"; tail -5 outputs/rule_cmp_stats.log; say "CHAIN_RULE_EA_FAIL"; exit 1; }
cat outputs/slicelens_cmp_stats.txt
say "CHAIN_RULE_EA_DONE"
