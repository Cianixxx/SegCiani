"""Contact sheet of the Pascal-EA cases where the edit makes the segmenter lose the object:
original | prediction on the original | edited (v3) | prediction on the edited image.
Rows: the largest recall drops in report.json, at most two per class. Run after
pascal_ea_compare.py and pascal_ea_report.py, in the same container."""
import json
import os
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont

EA = os.environ.get("EA", "/ds/pascal_ea_material")
VOC = os.environ.get("VOC", "/data/voc2012/VOCdevkit/VOC2012")
OUT = os.environ.get("OUT", "outputs/pascal_ea")
N = int(os.environ.get("N", "8"))
PER_CLASS = int(os.environ.get("PER_CLASS", "2"))
LANG = os.environ.get("LANG_FIG", "it")                  # en: labels in English, examples_worst_en.jpg
S, W = 520, 300
PAL = Image.open(f"{VOC}/SegmentationClass/2007_000032.png").getpalette()


def font(sz):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


F = font(14)


def cut(img):
    w, h = img.size
    r = S / min(w, h)
    img = img.convert("RGB").resize((round(w * r), round(h * r)))
    l, t = (img.width - S) // 2, (img.height - S) // 2
    return img.crop((l, t, l + S, t + S))


def colour(p):
    im = Image.fromarray(np.array(Image.open(p)).astype(np.uint8), "P")
    im.putpalette(PAL)
    return im.convert("RGB")


rep = json.load(open(f"{OUT}/report.json"))
picks, seen = [], {}
for r in rep["v3"]["worst"] + []:
    if seen.get(r["cls"], 0) < PER_CLASS:
        picks.append(r)
        seen[r["cls"]] = seen.get(r["cls"], 0) + 1
    if len(picks) == N:
        break
lab = 40
img = Image.new("RGB", (4 * (W + 8) + 12, len(picks) * (W + lab + 8) + 10), "white")
d = ImageDraw.Draw(img)
heads = (["original", "prediction on the original", "edited (v3)", "prediction on the edited image"]
         if LANG == "en" else
         ["originale", "predizione sull'originale", "modificata (v3)", "predizione sulla modificata"])
y = 8
for r in picks:
    s = r["stem"]
    if LANG == "en":
        becomes = (f", {100 * r['becomes'][1]:.0f}% of it becomes {r['becomes'][0]}" if r.get("becomes") else "")
        t = (f"{s} · {r['cls']} · {r['material']}: recall of the object {r['rec_orig']:.2f} -> {r['rec_edit']:.2f}"
             f"{becomes}")
    else:
        becomes = (f", diventa {r['becomes'][0]} per il {100 * r['becomes'][1]:.0f}%" if r.get("becomes") else "")
        t = (f"{s} · {r['cls']} · {r['material']}: recall dell'oggetto {r['rec_orig']:.2f} -> {r['rec_edit']:.2f}"
             f"{becomes}")
    d.text((8, y + 4), textwrap.shorten(t, 160), fill=(25, 25, 25), font=F)
    y += lab - 14
    panels = [cut(Image.open(f"{VOC}/JPEGImages/{s}.jpg")), colour(f"{OUT}/pred_orig/{s}.png"),
              cut(Image.open(f"{EA}/edited_v3/{s}.png")), colour(f"{OUT}/pred_v3/{s}.png")]
    for k, p in enumerate(panels):
        img.paste(p.resize((W, W)), (8 + k * (W + 8), y))
        d.text((12 + k * (W + 8), y + 4), heads[k], fill=(255, 255, 255), font=F, stroke_width=2,
               stroke_fill=(0, 0, 0))
    y += W + 22
name = "examples_worst_en.jpg" if LANG == "en" else "esempi_peggiori.jpg"
img.save(f"{OUT}/{name}", quality=88)
print(f"{OUT}/{name}", img.size, [p["stem"] for p in picks])
