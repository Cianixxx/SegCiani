"""Generate LR-ASPP MobileNetV3 predictions for the full VOC2012 set.

Reproduces the identified model under test (torchvision lraspp_mobilenet_v3_large,
short-side->520 + center-crop, ImageNet norm) on every trainval image, saving
mask_pr_<name>.png (RGB, R=G=B=class) into <voc-root>/mobilenet_pred -- the exact
format the voc_seg adapter consumes. Lets the SliceLens discovery scale beyond
the 80 annotated samples to per-class, paper-style statistical power.
"""

import argparse
import os
import time

import numpy as np
import torch
import torchvision.models.segmentation as seg
from PIL import Image
from torchvision import transforms as T

# Workaround for CUDNN_STATUS_NOT_INITIALIZED seen on this GPU: skip cuDNN and use
# native conv (lraspp is tiny, so the slowdown is negligible).
torch.backends.cudnn.enabled = False

S = 520
_NORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def prep(img):
    w, h = img.size
    r = S / min(w, h)
    img = img.resize((round(w * r), round(h * r)))
    w, h = img.size
    l, t = (w - S) // 2, (h - S) // 2
    return _NORM(T.ToTensor()(img.crop((l, t, l + S, t + S))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voc-root", default="/data/voc2012/VOCdevkit/VOC2012")
    ap.add_argument("--split", default="trainval")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    img_dir = os.path.join(args.voc_root, "JPEGImages")
    split_file = os.path.join(args.voc_root, "ImageSets", "Segmentation", args.split + ".txt")
    names = [ln.strip() for ln in open(split_file) if ln.strip()]
    if args.limit:
        names = names[: args.limit]
    pred_dir = os.path.join(args.voc_root, "mobilenet_pred")
    os.makedirs(pred_dir, exist_ok=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = seg.lraspp_mobilenet_v3_large(weights="DEFAULT").eval().to(dev)
    print(f"{len(names)} images -> {pred_dir} | device={dev}", flush=True)

    t0 = time.time()
    with torch.no_grad():
        for i, n in enumerate(names):
            out_p = os.path.join(pred_dir, f"mask_pr_{n}.png")
            if os.path.exists(out_p):
                continue
            img = Image.open(os.path.join(img_dir, n + ".jpg")).convert("RGB")
            logits = model(prep(img).unsqueeze(0).to(dev))["out"][0]
            pred = logits.argmax(0).cpu().numpy().astype(np.uint8)  # (520, 520)
            Image.fromarray(np.stack([pred] * 3, -1)).save(out_p)
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(names)}  ({time.time() - t0:.0f}s)", flush=True)
    print(f"DONE: {len(names)} predictions in {time.time() - t0:.0f}s -> {pred_dir}",
          flush=True)


if __name__ == "__main__":
    main()
