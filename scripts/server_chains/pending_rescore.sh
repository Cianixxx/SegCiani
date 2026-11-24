#!/bin/bash
# The official run plus the pending hypotheses grounded on whole classes, on a COPY (outputs/slice_quality_inst_pending):
# the rank verdict written into copies of the pending groundings, then the rescore with them beside the official
# groundings. CPU only; the official run is not touched.
set -u
cd ~/storage/SLICELENS
D=outputs/slice_quality_inst; O=${D}_pending
MOUNTS="-v $HOME/storage/SLICELENS:/workspace -v $HOME/storage/pascal_vlm:/data"
ENV="-e HOME=/tmp -e PYTHONIOENCODING=utf-8 -e PYTHONPATH=/workspace"
PIP="pip install --user -q scipy scikit-learn >/dev/null 2>&1"
mkdir -p $O; cp $D/result_tagged.pkl $D/cluster_names.json $O/
docker run --rm -u $(id -u):$(id -g) $ENV -e RUNS_DIR=/data/outputs_pending -e STAGE=full -e OUT=$O/dd_auc_full \
  -e RANK_COPY=/data/outputs_pending_rank $MOUNTS -w /workspace segciani:dev bash -c "$PIP; python -u slice_quality/examples/dd_auc.py" \
  > $O/dd_auc.log 2>&1
head -1 $O/dd_auc.log
docker run --rm -u $(id -u):$(id -g) $ENV -e SQ_OUT=$O -e LANG_DIRS=/data/outputs_rank:/data/outputs_extra_rank:/data/outputs_pending_rank \
  -e LANG_TAGS=":extra:pend" -e VOC_DATA=/data/voc2012/VOCdevkit/VOC2012 -e GALLERY=0 $MOUNTS -w /workspace segciani:dev \
  bash -c "$PIP; python -u slice_quality/examples/rescore_with_language.py" > $O/rescore.log 2>&1
grep -E "language slices total|images with SliceLens|DONE" $O/rescore.log
docker run --rm -u $(id -u):$(id -g) $ENV $MOUNTS -w /workspace segciani:dev bash -c "$PIP; python -c \"
import sys, json; sys.path.insert(0, 'docs')
from _revision_numbers import load_inst_summary as L
for d in ('$D', '$O'):
    s = L(d); print(d, 'language', s['slices']['language'], 'cluster', s['slices']['cluster'],
                    'cov %.1f%%' % (100 * s['cov_union'] / s['n_err']), 'lang cov %.1f%%' % (100 * s['cov_language'] / s['n_err']),
                    'images with a situation', s['situational_images'])
\"" 2>&1 | tail -3
echo PENDING_RESCORE_DONE
