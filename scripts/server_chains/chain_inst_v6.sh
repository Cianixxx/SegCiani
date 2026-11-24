#!/bin/bash
# Narrative v6 (corners, smaller defect, ragged outline place, two classes, small patches,
# holes of correct regions), tried on a COPY of the inst run (v5 untouched). CPU only.
set -u
cd ~/storage/SLICELENS
D=outputs/slice_quality_inst
N=outputs/slice_quality_inst_v6
mkdir -p $N
cp $D/result_tagged.pkl $N/
cp -r $D/train36 $N/
MOUNTS="-v $HOME/storage/SLICELENS:/workspace -v $HOME/storage/pascal_vlm:/data"
ENV="-e HOME=/tmp -e PYTHONIOENCODING=utf-8 -e PYTHONPATH=/workspace -e SQ_CLAIM_COV=0.5 -e SQ_GT_INSTANCES=object"
PIP="pip install --user -q scipy scikit-learn >/dev/null 2>&1"
docker run --rm -u $(id -u):$(id -g) $ENV -e PKL=$N/result_tagged.pkl -e TRAIN36=$N/train36 -e OUT=$N/error_narratives \
  $MOUNTS -w /workspace segciani:dev bash -c "$PIP; python -u slice_quality/examples/narrate_errors.py" \
  > $N/narrate.log 2>&1
grep -q NARRATE_DONE $N/narrate.log || { echo "FAIL narrate"; tail -5 $N/narrate.log; exit 1; }
docker run --rm -u $(id -u):$(id -g) $ENV -e OUT=$N/synth -e STAGE=$N/synth/_stage \
  $MOUNTS -w /workspace segciani:dev bash -c "$PIP; python -u slice_quality/examples/narrate_synth.py" \
  > $N/narrate_synth.log 2>&1 || { echo "FAIL narrate_synth"; tail -5 $N/narrate_synth.log; exit 1; }
echo V6_DONE
