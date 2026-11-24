"""Model-repair loop, part 1: fine-tune the reference model on the ONE failure
situation this thesis has the most and best evidence for, on BOTH models --
"an object cut off at the image border" (§6.6's boundary slice on model2, one of
model1's largest FN drivers, and the situation the eval pipeline itself induces
extra: every prediction is scored on a short-side-520 CENTER CROP of the image
[slicelens/adapters/voc_seg.py:_resize_center_crop], so an object fully visible
in the original photo can still be truncated by the CROP window at eval time.
The repair augmentation below targets exactly that crop-boundary geometry, not
just the original image's own border, to match the instrument the pipeline
actually measures with.

Two arms, same starting checkpoint, same VOC `train` split (1464 images --
completely untouched by any diagnosis so far; `val`, the 1449 images the whole
thesis's evidence is built on, stays held out and is never read here):
  POLICY=control  standard scale/crop/flip augmentation, no border bias --
                  isolates "does more training alone help" from the repair itself.
  POLICY=repair   same base recipe, but a fraction of crops (BORDER_FRAC) are
                  deliberately anchored so an object's bbox is cut by the crop
                  edge, using SegmentationObject instance masks to find one.

Run each arm to completion, then repair_model1_predict.py + the existing
pipeline re-diagnose `val` for each checkpoint and the comparison is: do the
ORIGINAL "cut off at the image border" slices/hypotheses shrink under repair
MORE than under the plain control, without new problems appearing elsewhere?

Run (GPU):
  PYTHONPATH=/workspace POLICY=control  python slice_quality/examples/repair_model1_train.py
  PYTHONPATH=/workspace POLICY=repair   python slice_quality/examples/repair_model1_train.py
Env: POLICY (control|repair, required), EPOCHS (default 18), BATCH (default 8),
     CROP (default 480), LR (default 1e-4), BORDER_FRAC (default 0.5, repair only),
     SEED (default 0), OUT_DIR (default outputs/slice_quality/repair).
"""
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from torchvision.models.segmentation import lraspp_mobilenet_v3_large

DATA = "/data/voc2012/VOCdevkit/VOC2012"
POLICY = os.environ.get("POLICY", "")
assert POLICY in ("control", "repair"), "set POLICY=control or POLICY=repair"
EPOCHS = int(os.environ.get("EPOCHS", "18"))
BATCH = int(os.environ.get("BATCH", "8"))
CROP = int(os.environ.get("CROP", "480"))
LR = float(os.environ.get("LR", "1e-4"))
BORDER_FRAC = float(os.environ.get("BORDER_FRAC", "0.5"))
SEED = int(os.environ.get("SEED", "0"))
OUT_DIR = os.environ.get("OUT_DIR", "outputs/slice_quality/repair")

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]
_NORM = T.Normalize(_MEAN, _STD)
_IGNORE = 255


def _instance_bboxes(name):
    """Object instance bboxes from SegmentationObject (index-per-object PNG),
    in the ORIGINAL image's own pixel coordinates."""
    p = f"{DATA}/SegmentationObject/{name}.png"
    if not os.path.exists(p):
        return []
    arr = np.array(Image.open(p))
    boxes = []
    for idx in np.unique(arr):
        if idx in (0, _IGNORE):
            continue
        ys, xs = np.where(arr == idx)
        if len(xs) < 50:            # skip slivers
            continue
        boxes.append((int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())))
    return boxes


def _random_resize(img, mask, rng):
    """Short-side scale by a random factor in [0.5, 2.0], same for image+mask."""
    scale = rng.uniform(0.5, 2.0)
    w, h = img.size
    nw, nh = max(CROP, round(w * scale)), max(CROP, round(h * scale))
    img = img.resize((nw, nh), Image.BILINEAR)
    mask = mask.resize((nw, nh), Image.NEAREST)
    return img, mask


def _pad_to_crop(img, mask):
    w, h = img.size
    if w >= CROP and h >= CROP:
        return img, mask
    pw, ph = max(0, CROP - w), max(0, CROP - h)
    fill_img = tuple(int(255 * m) for m in _MEAN)
    img = T.functional.pad(img, (0, 0, pw, ph), fill=fill_img)
    mask = T.functional.pad(mask, (0, 0, pw, ph), fill=_IGNORE)
    return img, mask


def _random_crop_box(w, h, rng):
    x0 = rng.randint(0, w - CROP)
    y0 = rng.randint(0, h - CROP)
    return x0, y0


def _border_crop_box(canvas_size, boxes, rng):
    """A crop window (within canvas_size) that cuts through a random object's
    bbox on at least one side, instead of framing it whole."""
    w, h = canvas_size
    candidates = [b for b in boxes if (b[2] - b[0]) < 3 * CROP and (b[3] - b[1]) < 3 * CROP]
    if not candidates:
        return None
    bx0, by0, bx1, by1 = candidates[rng.randrange(len(candidates))]
    side = rng.choice(["left", "right", "top", "bottom"])
    cx = rng.uniform(bx0, max(bx0 + 1, bx1))
    cy = rng.uniform(by0, max(by0 + 1, by1))
    x0 = cx if side == "left" else (cx - CROP if side == "right" else cx - CROP / 2)
    y0 = cy if side == "top" else (cy - CROP if side == "bottom" else cy - CROP / 2)
    x0 = int(max(0, min(x0, w - CROP)))
    y0 = int(max(0, min(y0, h - CROP)))
    return x0, y0


class VOCTrain(torch.utils.data.Dataset):
    def __init__(self, names, policy, seed):
        self.names = names
        self.policy = policy
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        rng = self.rng
        name = self.names[i]
        img = Image.open(f"{DATA}/JPEGImages/{name}.jpg").convert("RGB")
        mask = Image.open(f"{DATA}/SegmentationClass/{name}.png")
        orig_w, orig_h = img.size
        boxes = _instance_bboxes(name) if self.policy == "repair" else []

        img, mask = _random_resize(img, mask, rng)
        img, mask = _pad_to_crop(img, mask)
        w, h = img.size

        box = None
        if self.policy == "repair" and boxes and rng.random() < BORDER_FRAC:
            sx, sy = w / orig_w, h / orig_h
            scaled = [(bx0 * sx, by0 * sy, bx1 * sx, by1 * sy)
                      for bx0, by0, bx1, by1 in boxes]
            box = _border_crop_box((w, h), scaled, rng)
        if box is None:
            box = _random_crop_box(w, h, rng)
        x0, y0 = box
        img = img.crop((x0, y0, x0 + CROP, y0 + CROP))
        mask = mask.crop((x0, y0, x0 + CROP, y0 + CROP))

        if rng.random() < 0.5:
            img = T.functional.hflip(img)
            mask = T.functional.hflip(mask)

        x = _NORM(T.functional.to_tensor(img))
        y = torch.from_numpy(np.array(mask, dtype=np.int64))
        return x, y


def collate(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.stack(ys)


def main():
    torch.backends.cudnn.enabled = os.environ.get("USE_CUDNN") == "1"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUT_DIR, exist_ok=True)

    names = [l.strip() for l in
             open(f"{DATA}/ImageSets/Segmentation/train.txt", encoding="utf-8") if l.strip()]
    print(f"policy={POLICY} | {len(names)} train images | device={dev} | "
          f"epochs={EPOCHS} batch={BATCH} crop={CROP} lr={LR} "
          f"border_frac={BORDER_FRAC if POLICY == 'repair' else 'n/a'}", flush=True)

    ds = VOCTrain(names, POLICY, SEED)
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=True,
                                          num_workers=0, collate_fn=collate,
                                          drop_last=True)

    model = lraspp_mobilenet_v3_large(weights="DEFAULT").to(dev)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    steps_total = EPOCHS * max(1, len(loader))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps_total)

    t0 = time.time()
    for ep in range(EPOCHS):
        ep_loss, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            out = model(x)["out"]
            loss = F.cross_entropy(out, y, ignore_index=_IGNORE)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            ep_loss += loss.item()
            n += 1
        print(f"  epoch {ep + 1}/{EPOCHS}  loss={ep_loss / max(1, n):.4f}  "
              f"({time.time() - t0:.0f}s)", flush=True)
        if (ep + 1) % 6 == 0 or ep + 1 == EPOCHS:
            ckpt = os.path.join(OUT_DIR, f"model1_{POLICY}_ep{ep + 1}.pth")
            torch.save(model.state_dict(), ckpt)
            print(f"  saved {ckpt}", flush=True)

    final = os.path.join(OUT_DIR, f"model1_{POLICY}_final.pth")
    torch.save(model.state_dict(), final)
    print(f"REPAIR_TRAIN_DONE policy={POLICY} -> {final} ({time.time() - t0:.0f}s total)",
          flush=True)


if __name__ == "__main__":
    main()
