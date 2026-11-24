#!/bin/bash
# Narrative v5 on the inst run: "which class took the place of which" (relabelled_as /
# over_gt), already generated on the copy outputs/slice_quality_inst_relabel. The v4
# narratives and the pickles they were measured with stay beside them as *_v4. CPU only.
set -u
cd ~/storage/SLICELENS
D=outputs/slice_quality_inst
N=outputs/slice_quality_inst_relabel
for f in error_narratives.json error_narratives.txt train36/error_narratives_train36.json \
         train36/error_narratives_train36.txt synth/narratives_synth.json synth/narratives_synth.txt; do
  [ -f $D/${f%.*}_v4.${f##*.} ] || cp $D/$f $D/${f%.*}_v4.${f##*.}
  cp $N/$f $D/$f
done
for f in result_tagged.pkl train36/instances_train36.pkl synth/instances_synth.pkl; do
  [ -f $D/${f%.*}_v4narr.pkl ] || cp $D/$f $D/${f%.*}_v4narr.pkl
  cp $N/$f $D/$f
done
md5sum $D/error_narratives.json $N/error_narratives.json
MOUNTS="-v $HOME/storage/SLICELENS:/workspace -v $HOME/storage/pascal_vlm:/data"
docker run --rm -u $(id -u):$(id -g) -e HOME=/tmp -e PYTHONIOENCODING=utf-8 -e PYTHONPATH=/workspace -e SQ_OUT=$D \
  $MOUNTS -w /workspace segciani:dev bash -c "pip install --user -q scipy scikit-learn >/dev/null 2>&1; \
  python -u slice_quality/examples/regen_gallery.py" > $D/regen_gallery_v5.log 2>&1
grep -q "gallery regenerated" $D/regen_gallery_v5.log || { echo "FAIL gallery"; tail -5 $D/regen_gallery_v5.log; exit 1; }
tail -4 $D/regen_gallery_v5.log
echo NARR5_DONE
