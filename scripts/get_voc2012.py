"""Download the full Pascal-VOC2012 segmentation set (~2 GB) via torchvision.

Extracts to <root>/VOCdevkit/VOC2012/{JPEGImages, SegmentationClass,
ImageSets/Segmentation/trainval.txt} -- the ~2,913 trainval images with GT,
to scale the SliceLens discovery beyond the 80 annotated samples.
"""

import sys

import torchvision

root = sys.argv[1] if len(sys.argv) > 1 else "/data/voc2012"
print(f"Downloading VOC2012 trainval segmentation to {root} ...", flush=True)
ds = torchvision.datasets.VOCSegmentation(
    root=root, year="2012", image_set="trainval", download=True)
print(f"VOC2012 READY: {len(ds)} images", flush=True)
