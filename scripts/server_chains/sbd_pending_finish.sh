#!/bin/bash
# The CPU tail of outputs/chain_sbd_pending.sh, unchanged, for when the chain stopped at "person blocks n/24" because
# a block was still running on the extra GPU 1 worker (sbd_pending_gpu1.sh) when its own two workers ended.
# Run by sbd_pending_gpu1.sh; output appended to outputs/sbd_pending.log.
set -u
cd ~/storage/SLICELENS
L=outputs/slice_quality_sbd_pending
LOGS=$L/logs
P=$HOME/storage/pascal_vlm
O=$P/outputs_sbd_pending
MOUNTS="-v $HOME/storage/SLICELENS:/workspace -v $P:/data -v $DATASETS:/ds"
ENV="-e HOME=/tmp -e PYTHONIOENCODING=utf-8 -e PYTHONPATH=/workspace -e SQ_QWEN_DEVICE_MAP=cuda:0"
LANG="-e OUT_DIR=/data/outputs_sbd_pending -e FULL_HYPS=/data/full_pending_sbd.json -e SUBSTRATE=/data/sbd/substrate_cc.pkl \
  -e OLD_SUBSTRATE= -e EDITED= -e CACHE_DIRS= -e SAMPLING=srs -e DATASET=SBD -e PHASE=full"
PIP="pip install --user -q scipy scikit-learn >/dev/null 2>&1"
IMG=segciani:dev
say() { echo "$(date '+%m-%d %H:%M') $*"; }
CLASSES=$(cat $L/classes.txt)
for c in $CLASSES; do [ -f $O/voc_paper_${c}_full.json ] || [ -f $O/.none_full_$c ] || { say "class $c missing: stop"; exit 1; }; done
n=$(ls $O/split/ 2>/dev/null | grep -c "^person_.*json$")
[ "$n" -eq 24 ] || { say "person blocks $n/24: stop"; exit 1; }
say "[cpu] person: blocks merged, its file written from the cache"
docker run --rm -u $(id -u):$(id -g) $ENV $LANG -e CLS=person -e NBLOCKS=24 $MOUNTS -w /workspace $IMG \
  bash -c "$PIP; PHASE=merge python -u examples/ground_split_class.py" > $LOGS/person_merge.log 2>&1
grep -q SPLIT_MERGE_DONE $LOGS/person_merge.log || { say "person merge failed"; tail -5 $LOGS/person_merge.log; exit 1; }
rmdir $O/.claim_full_person
docker run --rm -u $(id -u):$(id -g) $ENV $LANG -e CLASSES=person $MOUNTS -w /workspace $IMG \
  bash -c "$PIP; python -u examples/ground_ea_hypotheses.py" > $LOGS/person_assemble.log 2>&1
[ -f $O/voc_paper_person_full.json ] || { say "person assembly failed"; tail -5 $LOGS/person_assemble.log; exit 1; }
say "[cpu] the rank verdict on the whole classes, then a COPY of the SBD run rescored with them"
docker run --rm -u $(id -u):$(id -g) $ENV -e RUNS_DIR=/data/outputs_sbd_pending -e STAGE=full -e OUT=$L/dd_auc_full \
  -e RANK_COPY=/data/outputs_sbd_pending_rank $MOUNTS -w /workspace $IMG \
  bash -c "$PIP; python -u slice_quality/examples/dd_auc.py" > $LOGS/dd_auc.log 2>&1
head -n 1 $LOGS/dd_auc.log
cp outputs/slice_quality_sbd/result_tagged.pkl $L/
[ -f outputs/slice_quality_sbd/cluster_names.json ] && cp outputs/slice_quality_sbd/cluster_names.json $L/
docker run --rm -u $(id -u):$(id -g) -e HOME=/tmp -e PYTHONIOENCODING=utf-8 -e PYTHONPATH=/workspace -e SQ_OUT=$L \
  -e LANG_DIRS=/data/outputs_sbd_rank:/data/outputs_sbd_pending_rank -e LANG_TAGS=":pend" \
  -e VOC_DATA=/data/sbd/VOCdevkit/VOC2012 -e GALLERY=0 $MOUNTS -w /workspace $IMG \
  bash -c "$PIP; python -u slice_quality/examples/rescore_with_language.py" > $LOGS/rescore.log 2>&1
grep -q "DONE rescore" $LOGS/rescore.log || { say "rescore failed"; tail -5 $LOGS/rescore.log; exit 1; }
docker run --rm -u $(id -u):$(id -g) -e HOME=/tmp -e PYTHONPATH=/workspace $MOUNTS -w /workspace $IMG bash -c "$PIP; python -c \"
import sys; sys.path.insert(0, 'docs')
from _revision_numbers import load_inst_summary as L
for d in ('outputs/slice_quality_sbd', '$L'):
    s = L(d); print(d, 'situations', s['slices']['language'], 'clusters', s['slices']['cluster'],
                    'errors in a group %.1f%%' % (100 * s['cov_union'] / s['n_err']), 'images with a situation', s['situational_images'])
\"" 2>&1 | tail -2
say "CHAIN_SBD_PENDING_DONE"
