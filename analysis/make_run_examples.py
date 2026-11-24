# -*- coding: utf-8 -*-
"""Image | ground truth | prediction strips for the 'three runs' section of both documents.

CPU, local: reads the 100 annotated images kept beside the thesis
(pascal_80/JPEGImages + SegmentationClass + mobilenet_pred) and writes
docs/panels/<image_id>.png, the same strip format export_example_panels.py
produces on the server (420 px, short side resized then centre-cropped, which is
the alignment the pipeline itself applies).

The ids are the examples §11 quotes; they were chosen because the three typings
say different things about them, not because they look good:

    2010_003383  containment: six "absorbed" cow parts -> one merge -> one cow
    2007_000129  occlusion: "six BICYCLE regions missed" -> "partly hidden behind"
    2008_006784  a global failure the reference called "missed entirely"
    2008_003703  the silent case: "No segmentation error was found"

Run:  python docs/make_run_examples.py [image_id ...]
"""
import os
import sys

from PIL import Image, ImageDraw, ImageFont

DOCS = os.path.dirname(os.path.abspath(__file__))
B = r"<datasets>\pascal_80"
OUT = os.path.join(DOCS, "panels")
SIDE, LABEL_H, GAP = 420, 26, 10

IDS = sys.argv[1:] or ["2010_003383", "2007_000129", "2008_006784", "2008_003703"]


def rcc(img, size, resample):
    w, h = img.size
    if w < h:
        nw, nh = size, int(round(h * size / w))
    else:
        nh, nw = size, int(round(w * size / h))
    img = img.resize((nw, nh), resample)
    l, t = (nw - size) // 2, (nh - size) // 2
    return img.crop((l, t, l + size, t + size))


try:
    FONT = ImageFont.truetype("arial.ttf", 15)
except OSError:                       # pragma: no cover - any box without arial
    FONT = ImageFont.load_default()

PAL = Image.open(os.path.join(B, "SegmentationClass", "2007_000033.png")).getpalette()
os.makedirs(OUT, exist_ok=True)

for name in IDS:
    o = rcc(Image.open(os.path.join(B, "JPEGImages", name + ".jpg")).convert("RGB"),
            SIDE, Image.LANCZOS)
    g = rcc(Image.open(os.path.join(B, "SegmentationClass", name + ".png")).convert("RGB"),
            SIDE, Image.NEAREST)
    pr = Image.open(os.path.join(B, "mobilenet_pred", "mask_pr_" + name + ".png")).split()[0]
    pr.putpalette(PAL)
    p = rcc(pr.convert("RGB"), SIDE, Image.NEAREST)

    canvas = Image.new("RGB", (SIDE * 3 + GAP * 2, SIDE + LABEL_H), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (lbl, im) in enumerate([("image", o), ("ground truth", g),
                                   ("model prediction", p)]):
        x = i * (SIDE + GAP)
        canvas.paste(im, (x, LABEL_H))
        draw.text((x + 4, 5), lbl, fill=(40, 40, 40), font=FONT)
    out = os.path.join(OUT, name + ".png")
    canvas.save(out)
    print("saved", out, canvas.size)
print("RUN_EXAMPLES_DONE")
