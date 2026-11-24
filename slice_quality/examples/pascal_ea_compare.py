"""Does a Pascal-EA material edit make the segmenter fail? Original vs edited, paired.

The images are VOC 2012 val images whose salient object was re-rendered with a
different material by a diffusion model (Pascal-EA, CVPR 2024; the copy in
~/datasets/private/pascal_ea_material). The edit keeps the object's mask, so the
VOC ground truth stays valid and every edited image has its own original as a
control: same scene, same object, same label, only the object's appearance changed.

The model under test is the thesis's: torchvision LR-ASPP MobileNetV3 (DEFAULT
weights), short side to 520 + centre crop, ImageNet normalisation -- the exact
preprocessing of scripts/_archive/gen_voc_predictions.py. The ground truth and the
subject mask are cut the same way (nearest neighbour).

Per image and version (orig, v3, v1):
  subject       the edited object's pixels: subject mask AND ground truth == its class
  rec           share of the subject predicted with its class
  wrong_top     what the subject's missed pixels become instead (label, share)
  iou_c         IoU of the subject's class over the whole image
  acc_out       pixel accuracy outside the subject (dilated by 15 px) -- the control:
                the edit should not change it
  miou          mean IoU over the classes in the image
  diff_in/out   mean absolute RGB change, edited vs original, inside the subject and
                outside it (how strong the edit is, and how much the background moved)
Predictions are saved as PNG (520x520, class index) for a later pipeline run.

Run (one GPU, or CPU):
  docker run ... -v $DATASETS:/ds segciani:dev \
     python slice_quality/examples/pascal_ea_compare.py
Env: EA (/ds/pascal_ea_material), VOC, OUT (outputs/pascal_ea), THREADS (8), LIMIT.
"""
import json
import os
import time
from collections import Counter

import numpy as np
import torch
import torchvision.models.segmentation as seg
from PIL import Image, ImageFilter
from torchvision import transforms as T

torch.backends.cudnn.enabled = False
torch.set_num_threads(int(os.environ.get("THREADS", "8")))
EA = os.environ.get("EA", "/ds/pascal_ea_material")
VOC = os.environ.get("VOC", "/data/voc2012/VOCdevkit/VOC2012")
OUT = os.environ.get("OUT", "outputs/pascal_ea")
LIMIT = int(os.environ.get("LIMIT", "0"))
S = 520
NORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
CLS = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
       "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
       "train", "tvmonitor"]


def geom(w, h):
    r = S / min(w, h)
    W, H = round(w * r), round(h * r)
    l, t = (W - S) // 2, (H - S) // 2
    return (W, H), (l, t, l + S, t + S)


def cut_rgb(img):
    size, box = geom(*img.size)
    return img.convert("RGB").resize(size).crop(box)       # same call as gen_voc_predictions


def cut_lab(img):
    size, box = geom(*img.size)
    return np.array(img.resize(size, Image.NEAREST).crop(box))


DEV = "cuda" if torch.cuda.is_available() else "cpu"


def predict(model, rgb):
    with torch.no_grad():
        x = NORM(T.ToTensor()(rgb)).unsqueeze(0).to(DEV)
        return model(x)["out"][0].argmax(0).cpu().numpy().astype(np.uint8)


def miou(pred, gt):
    v = gt != 255
    ious = []
    for c in set(np.unique(gt[v]).tolist()) | set(np.unique(pred[v]).tolist()):
        a, b = (pred == c) & v, (gt == c) & v
        u = (a | b).sum()
        if u:
            ious.append((a & b).sum() / u)
    return float(np.mean(ious)) if ious else float("nan")


def main():
    lab = json.load(open(f"{EA}/labels.json"))["images"]
    stems = sorted(lab)[: LIMIT or None]
    model = seg.lraspp_mobilenet_v3_large(weights="DEFAULT").eval().to(DEV)
    print(f"{len(stems)} images, device {DEV}", flush=True)
    for v in ("orig", "v3", "v1"):
        os.makedirs(f"{OUT}/pred_{v}", exist_ok=True)
    rows, t0 = [], time.time()
    agree = []
    for i, s in enumerate(stems, 1):
        r = lab[s]
        c = CLS.index(r["voc_class"]) if r.get("voc_class") in CLS else None
        gt = cut_lab(Image.open(f"{EA}/annotations_mmseg/{s}.png"))
        bm = cut_lab(Image.open(f"{EA}/binary_masks/{s}.png").convert("L")) > 127
        subj = bm & (gt == c) if c is not None else bm & False
        out_mask = np.array(Image.fromarray((bm * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(31))) == 0
        out_mask &= gt != 255
        orig = cut_rgb(Image.open(f"{VOC}/JPEGImages/{s}.jpg"))
        a0 = np.asarray(orig, dtype=np.float32)
        row = {"stem": s, "voc_class": r.get("voc_class"), "subject_px": int(subj.sum()),
               "material_v3": r.get("material_v3"), "material_v1": r.get("material_v1")}
        for v in ("orig", "v3", "v1"):
            if v == "orig":
                rgb = orig
            else:
                p = f"{EA}/edited_{v}/{s}.png"
                if not os.path.exists(p):
                    continue
                rgb = cut_rgb(Image.open(p))
            pp = f"{OUT}/pred_{v}/{s}.png"
            if os.path.exists(pp):
                pred = np.array(Image.open(pp))
            else:
                pred = predict(model, rgb)
                Image.fromarray(pred).save(pp)
            m = {"miou": miou(pred, gt), "acc_out": float((pred[out_mask] == gt[out_mask]).mean())
                 if out_mask.any() else None}
            if c is not None and subj.sum() >= 50:
                ps = pred[subj]
                m["rec"] = float((ps == c).mean())
                w = Counter(ps[ps != c].tolist())
                if w:
                    k, n = w.most_common(1)[0]
                    m["wrong_top"] = [CLS[k] if k < len(CLS) else str(k), round(n / len(ps), 3)]
                a, b = pred == c, (gt == c)
                u = ((a | b) & (gt != 255)).sum()
                m["iou_c"] = float((a & b).sum() / u) if u else None
            if v != "orig":
                a1 = np.asarray(rgb, dtype=np.float32)
                d = np.abs(a1 - a0).mean(-1)
                m["diff_in"] = float(d[subj].mean()) if subj.any() else None
                m["diff_out"] = float(d[out_mask].mean()) if out_mask.any() else None
            if v == "orig":
                ref = f"{VOC}/mobilenet_pred/mask_pr_{s}.png"
                if os.path.exists(ref):
                    agree.append(float((np.array(Image.open(ref))[..., 0] == pred).mean()))
            row[v] = m
        rows.append(row)
        if i % 50 == 0 or i == len(stems):
            print(f"  {i}/{len(stems)} ({time.time() - t0:.0f}s)", flush=True)
    json.dump({"rows": rows, "agreement_with_thesis_preds": agree}, open(f"{OUT}/per_image.json", "w"),
              indent=1)
    print(f"agreement of the original predictions with the thesis's mobilenet_pred: "
          f"mean {np.mean(agree):.4f} over {len(agree)} images", flush=True)
    print("PASCAL_EA_COMPARE_DONE", flush=True)


if __name__ == "__main__":
    main()
