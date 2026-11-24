# -*- coding: utf-8 -*-
"""Figure for the section on SliceLens's data-driven hypotheses (EN).

Draws the crops recorded in docs/dd_hyp/visual_check.json: the highest-confidence
error instances of each certified data-driven situation, re-grounded on all instances,
with the box the VLM was asked about and the verdict of the visual check.

Run: python docs/make_dd_hyp_figures.py   ->  docs/fig_dd_examples_en.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

DOCS = os.path.dirname(os.path.abspath(__file__))
VERDICT_COL = {"holds": "#16a34a", "mislabelled": "#d97706", "partial": "#d97706",
               "vague": "#d97706", "artefact": "#dc2626"}
VERDICT = {"holds": "holds", "mislabelled": "real, wrong label", "partial": "partial",
           "vague": "vague", "artefact": "artefact"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

vis = json.load(open(os.path.join(DOCS, "dd_hyp", "visual_check.json"), encoding="utf-8"))
by_slice = {s["slice"]: s for s in vis["slices"]}
ex = vis["examples"]
cols = 4
rows = (len(ex) + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(10.4, 2.95 * rows))
for ax in axes.flat:
    ax.axis("off")
for ax, e in zip(axes.flat, ex):
    s = by_slice[e["slice"]]
    col = VERDICT_COL[s["verdict"]]
    im = Image.open(os.path.join(DOCS, "dd_hyp", e["name"] + ".jpg")).convert("RGB")
    ImageDraw.Draw(im).rectangle(e["bbox"], outline=col, width=5)
    ax.imshow(im)
    ax.set_title(f"{s['class']} — {s['short']}\n{VERDICT[s['verdict']]}"
                 + (f" · {e['note']}" if e.get("note") else ""), fontsize=8, color=col, loc="left")
fig.tight_layout()
fig.savefig(os.path.join(DOCS, "fig_dd_examples_en.png"), dpi=170, bbox_inches="tight", facecolor="white")
print("   fig_dd_examples_en.png")
