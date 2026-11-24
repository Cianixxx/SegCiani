"""Export original | ground truth | prediction strips for slides/report figures.

CPU only. Writes outputs/slice_quality/panels/<image_id>.png
Run: PYTHONPATH=/workspace python slice_quality/examples/export_example_panels.py 2007_002903 ...
"""
import os
import sys

from PIL import Image, ImageDraw

from slicelens.adapters.voc_seg import _resize_center_crop

DATA = "/data/voc2012/VOCdevkit/VOC2012"
OUT = "outputs/slice_quality/panels"
SIDE = 420
LABEL_H = 26
GAP = 10

IDS = sys.argv[1:] or ["2007_002903"]

val = [ln.strip() for ln in
       open(f"{DATA}/ImageSets/Segmentation/val.txt", encoding="utf-8") if ln.strip()]
PAL = Image.open(f"{DATA}/SegmentationClass/{val[0]}.png").getpalette()
os.makedirs(OUT, exist_ok=True)

for name in IDS:
    o = _resize_center_crop(Image.open(f"{DATA}/JPEGImages/{name}.jpg").convert("RGB"),
                            SIDE, Image.LANCZOS)
    g = _resize_center_crop(Image.open(f"{DATA}/SegmentationClass/{name}.png").convert("RGB"),
                            SIDE, Image.NEAREST)
    pr = Image.open(f"{DATA}/mobilenet_pred/mask_pr_{name}.png").split()[0]
    pr.putpalette(PAL)
    p = _resize_center_crop(pr.convert("RGB"), SIDE, Image.NEAREST)

    W = SIDE * 3 + GAP * 2
    canvas = Image.new("RGB", (W, SIDE + LABEL_H), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (lbl, im) in enumerate([("image", o), ("ground truth", g),
                                   ("model prediction", p)]):
        x = i * (SIDE + GAP)
        canvas.paste(im, (x, LABEL_H))
        draw.text((x + 4, 6), lbl, fill=(40, 40, 40))
    out = f"{OUT}/{name}.png"
    canvas.save(out)
    print("saved", out, canvas.size, flush=True)
print("PANELS_DONE", flush=True)
