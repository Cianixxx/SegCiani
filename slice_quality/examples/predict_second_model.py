"""Stage 1a: run a STRONGER segmenter over VOC val and store its masks.

Everything measured so far is true of ONE model (LR-ASPP MobileNetV3). That makes
every discovered situation ambiguous: is it a property of the task, or of that
particular network? A second, much stronger model answers it — the situations that
survive are intrinsic, the ones that vanish were weaknesses of the first model.

No training: torchvision ships DeepLabV3-ResNet101 trained on the COCO subset that
uses the 21 VOC classes, with the SAME class indices as VOC_CLASSES, so its argmax
map drops straight into the existing pipeline.

Output format matches what adapters/voc_seg_samples.py expects: one PNG per image
named mask_pr_<stem>.png whose (grey) value is the class index, at ORIGINAL image
resolution — the loader does its own resize/center-crop to 520, so writing full
resolution keeps this comparable with the mobilenet masks.

Run (GPU):
  PYTHONPATH=/workspace python slice_quality/examples/predict_second_model.py
Env: PRED_DIR (default deeplab_pred), BATCH (default 8), LIMIT (0 = all).
"""
import os
import time

import numpy as np
import torch
from PIL import Image
from torchvision.models.segmentation import (DeepLabV3_ResNet101_Weights,
                                             deeplabv3_resnet101)

DATA = "/data/voc2012/VOCdevkit/VOC2012"
PRED_DIR = os.path.join(DATA, os.environ.get("PRED_DIR", "deeplab_pred"))
BATCH = int(os.environ.get("BATCH", "8"))
LIMIT = int(os.environ.get("LIMIT", "0"))


def main():
    names = [l.strip() for l in
             open(f"{DATA}/ImageSets/Segmentation/val.txt", encoding="utf-8") if l.strip()]
    if LIMIT:
        names = names[:LIMIT]
    os.makedirs(PRED_DIR, exist_ok=True)
    print(f"{len(names)} images -> {PRED_DIR}", flush=True)

    # Same gotcha DINOv2Extractor already works around: this image's torch+cuDNN
    # raises CUDNN_STATUS_NOT_INITIALIZED on conv2d under driver 520. Native
    # convolutions are slower but correct, and inference here is not the bottleneck.
    torch.backends.cudnn.enabled = os.environ.get("USE_CUDNN") == "1"

    weights = DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
    model = deeplabv3_resnet101(weights=weights).eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    tf = weights.transforms()
    print("model on", dev, "| classes:", len(weights.meta["categories"]), flush=True)

    t0 = time.time()
    done = 0
    for i in range(0, len(names), BATCH):
        chunk = names[i:i + BATCH]
        imgs, sizes = [], []
        for stem in chunk:
            im = Image.open(f"{DATA}/JPEGImages/{stem}.jpg").convert("RGB")
            sizes.append(im.size)                     # (W, H)
            imgs.append(tf(im))
        # images differ in size: pad to a common canvas, then crop each result back
        h = max(t.shape[1] for t in imgs)
        w = max(t.shape[2] for t in imgs)
        batch = torch.zeros(len(imgs), 3, h, w)
        for j, t in enumerate(imgs):
            batch[j, :, :t.shape[1], :t.shape[2]] = t
        with torch.no_grad():
            out = model(batch.to(dev))["out"].argmax(1).cpu().numpy().astype(np.uint8)
        for j, stem in enumerate(chunk):
            th, tw = imgs[j].shape[1], imgs[j].shape[2]
            pred = out[j, :th, :tw]
            # back to the original image size, nearest so class ids stay ids
            pred_img = Image.fromarray(pred, mode="L").resize(sizes[j], Image.NEAREST)
            pred_img.save(os.path.join(PRED_DIR, f"mask_pr_{stem}.png"))
        done += len(chunk)
        if done % 200 < BATCH:
            el = time.time() - t0
            print(f"  {done}/{len(names)}  ({el:.0f}s, {el / max(1, done):.2f}s/img)",
                  flush=True)
    print(f"PREDICT_DONE {done} masks in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
