#!/bin/bash
# Run SliceLens (Qwen grounding + slope-trend verify) for VOC classes, producing
# voc_paper_<class>.json (candidates + is_systematic + top_instances). Uses the
# pre-generated queries combined_<class>.txt. Host-level launcher (calls docker).
#
# Usage:  GPU=4 bash run_slicelens_classes.sh aeroplane chair
set -e
GPU="${GPU:-4}"
cd ~/storage/SLICELENS
for CLS in "$@"; do
  echo "=== SliceLens grounding+verify: $CLS (GPU $GPU) ==="
  docker run --rm --gpus device="$GPU" -u $(id -u):$(id -g) -e HOME=/tmp -e PYTHONIOENCODING=utf-8 \
    -v $HOME/storage/SLICELENS:/workspace \
    -v $HOME/storage/pascal_vlm:/data \
    -v $STORAGE/models/qwen2.5-vl-7b:/models/qwen \
    -w /workspace segciani:dev \
    python examples/run_voc.py \
      --dataset-root /data/voc2012/VOCdevkit/VOC2012 \
      --crops-dir /data/voc2012/voc_crops \
      --names-file /data/voc2012/VOCdevkit/VOC2012/ImageSets/Segmentation/trainval.txt \
      --vlm-backend qwen --vlm-model /models/qwen --grounding bbox \
      --max-instances 120 --label "$CLS" \
      --queries-file /data/combined_"$CLS".txt \
      --out /data/outputs/voc_paper_"$CLS".json
  echo "=== done $CLS -> /data/outputs/voc_paper_$CLS.json ==="
done
echo "SLICELENS_CLASSES_DONE"
