"""Model-repair loop, part 2: run a fine-tuned checkpoint (from
repair_model1_train.py) over VOC val and store its masks, in exactly the format
predict_second_model.py already established -- one mask_pr_<stem>.png per image,
grey value = class index, at ORIGINAL image resolution (the voc_seg adapter does
its own short-side-520 resize + center-crop, so writing full resolution keeps
this directly comparable to mobilenet_pred / deeplab_pred).

`val` (1449 images) is the ONLY split touched here -- it is the exact set every
number in this thesis is built on, so re-diagnosing the repaired checkpoint on
it is an apples-to-apples comparison with the original model1 result, not a new
evaluation protocol.

Run (GPU):
  PYTHONPATH=/workspace CKPT=outputs/slice_quality/repair/model1_control_final.pth \
    PRED_DIR=mobilenet_control_pred python slice_quality/examples/repair_model1_predict.py
  PYTHONPATH=/workspace CKPT=outputs/slice_quality/repair/model1_repair_final.pth \
    PRED_DIR=mobilenet_repair_pred python slice_quality/examples/repair_model1_predict.py
Env: CKPT (required, path to a .pth state_dict), PRED_DIR (required, under
     <voc-root>/), BATCH (default 8), LIMIT (0 = all).
"""
import os
import time

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.models.segmentation import lraspp_mobilenet_v3_large

DATA = "/data/voc2012/VOCdevkit/VOC2012"
CKPT = os.environ["CKPT"]
PRED_DIR = os.path.join(DATA, os.environ["PRED_DIR"])
BATCH = int(os.environ.get("BATCH", "8"))
LIMIT = int(os.environ.get("LIMIT", "0"))

_NORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def prep(img):
    return _NORM(T.functional.to_tensor(img))


def main():
    torch.backends.cudnn.enabled = os.environ.get("USE_CUDNN") == "1"
    names = [l.strip() for l in
             open(f"{DATA}/ImageSets/Segmentation/val.txt", encoding="utf-8") if l.strip()]
    if LIMIT:
        names = names[:LIMIT]
    os.makedirs(PRED_DIR, exist_ok=True)
    print(f"{len(names)} val images -> {PRED_DIR} | ckpt={CKPT}", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = lraspp_mobilenet_v3_large(weights=None, num_classes=21)
    state = torch.load(CKPT, map_location="cpu")
    model.load_state_dict(state)
    model = model.eval().to(dev)
    print("model on", dev, flush=True)

    t0 = time.time()
    done = 0
    with torch.no_grad():
        for i in range(0, len(names), BATCH):
            chunk = names[i:i + BATCH]
            imgs, sizes = [], []
            for stem in chunk:
                im = Image.open(f"{DATA}/JPEGImages/{stem}.jpg").convert("RGB")
                sizes.append(im.size)                  # (W, H)
                imgs.append(prep(im))
            h = max(t.shape[1] for t in imgs)
            w = max(t.shape[2] for t in imgs)
            batch = torch.zeros(len(imgs), 3, h, w)
            for j, t in enumerate(imgs):
                batch[j, :, :t.shape[1], :t.shape[2]] = t
            out = model(batch.to(dev))["out"].argmax(1).cpu().numpy().astype(np.uint8)
            for j, stem in enumerate(chunk):
                th, tw = imgs[j].shape[1], imgs[j].shape[2]
                pred = out[j, :th, :tw]
                pred_img = Image.fromarray(pred, mode="L").resize(sizes[j], Image.NEAREST)
                pred_img.save(os.path.join(PRED_DIR, f"mask_pr_{stem}.png"))
            done += len(chunk)
            if done % 200 < BATCH:
                el = time.time() - t0
                print(f"  {done}/{len(names)}  ({el:.0f}s, {el / max(1, done):.2f}s/img)",
                      flush=True)
    print(f"REPAIR_PREDICT_DONE {done} masks in {time.time() - t0:.0f}s -> {PRED_DIR}",
          flush=True)


if __name__ == "__main__":
    main()
