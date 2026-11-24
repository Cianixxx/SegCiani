"""Identify which model produced the provided mask_pr_* predictions.

Reproduces predictions for a sample of the annotated images with the two
torchvision MobileNet segmentation models (the 520-resolution + VOC-21-class
output points there) under two preprocessings, and reports pixel agreement vs
the existing mask_pr_*. >~0.95 = match -> that model is the generator, and we
can scale to full VOC2012 with it.
"""

import os

import numpy as np
import torch
import torchvision.models.segmentation as seg
from PIL import Image
from torchvision import transforms as T

DATA = os.environ.get("DATA", "/data/dataset")
IMG = os.path.join(DATA, "JPEGImages")
PRED = os.path.join(DATA, "mobilenet_pred")
S = 520
dev = "cuda" if torch.cuda.is_available() else "cpu"
norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def prep_square(img):
    return norm(T.ToTensor()(img.resize((S, S))))


def prep_crop(img):
    w, h = img.size
    r = S / min(w, h)
    img = img.resize((round(w * r), round(h * r)))
    w, h = img.size
    l, t = (w - S) // 2, (h - S) // 2
    return norm(T.ToTensor()(img.crop((l, t, l + S, t + S))))


def main() -> None:
    names = sorted({f[len("mask_pr_"):-4].replace("_rgb", "")
                    for f in os.listdir(PRED) if f.endswith(".png")})
    names = [n for n in names if os.path.exists(os.path.join(IMG, n + ".jpg"))][:12]
    print(f"{len(names)} sample images, device={dev}", flush=True)
    refs = {n: np.array(Image.open(os.path.join(PRED, f"mask_pr_{n}.png")).convert("RGB"))[:, :, 0]
            for n in names}

    models = {
        "deeplabv3_mobilenet_v3_large": lambda: seg.deeplabv3_mobilenet_v3_large(weights="DEFAULT"),
        "lraspp_mobilenet_v3_large": lambda: seg.lraspp_mobilenet_v3_large(weights="DEFAULT"),
    }
    for mname, ctor in models.items():
        model = ctor().eval().to(dev)
        for pname, prep in [("square520", prep_square), ("shortside+crop", prep_crop)]:
            ag = []
            with torch.no_grad():
                for n in names:
                    img = Image.open(os.path.join(IMG, n + ".jpg")).convert("RGB")
                    out = model(prep(img).unsqueeze(0).to(dev))["out"][0]
                    pred = out.argmax(0).cpu().numpy().astype(np.uint8)
                    ag.append(float((pred == refs[n]).mean()))
            print(f"  {mname:34s} {pname:16s} agreement={np.mean(ag):.3f} "
                  f"(min {min(ag):.2f}, max {max(ag):.2f})", flush=True)
        del model
        torch.cuda.empty_cache() if dev == "cuda" else None


if __name__ == "__main__":
    main()
