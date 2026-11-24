"""Thesis figure (vector, English): one row per certified slice of the official VOC run, the IoU of the slice's class
on the slice's images (filled marker, value beside it) and on the class's other images (hollow marker), joined by a
line. Situations as circles, clusters as squares, so the figure reads in black and white too.

Data: slice_image_iou.json of the official run (slice_quality/examples/slice_image_iou.py), found by out_path().
Run:  python figs/fig_slice_iou.py  ->  Images/Chapter5/fig_slice_iou.pdf
"""
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "SlicelenceCode", "docs"))
from _revision_numbers import out_path  # noqa: E402

D = json.load(open(out_path("slice_quality_inst", "slice_image_iou.json"), encoding="utf-8"))
SIT, CLU, MUTED, DARK, LINE = "#B4470E", "#3B4A5E", "#5B6675", "#1E293B", "#C9D0DA"
SHORT = {"an object with holes or see-through parts": "holes or see-through parts",
         "an object cut off at the image border": "cut off at the border",
         "an object occluded by another object": "occluded by another object",
         "an object with thin or elongated parts": "thin or elongated parts",
         "several overlapping objects of the same type": "overlapping, same type",
         "an object that looks similar to a different object category": "looks like another category",
         "an object behind a fence, net, bars or branches": "behind a fence, net or branches",
         "an object that blends into its background": "blends into its background",
         "an object in an unusual setting for its kind": "unusual setting",
         "an object seen from an unusual viewpoint or in an unusual pose": "unusual viewpoint or pose",
         "a reflection of an object in a mirror, window or water": "reflection",
         "a very large object seen up close that fills most of the image": "very large, seen up close",
         "a picture, poster or screen showing an object": "picture, poster or screen",
         "a toy, statue or model of an object": "toy, statue or model",
         "a person partially hidden by clothing or a held object": "hidden by clothing or objects"}


def label(r):
    if r["branch"] == "cluster":
        return re.sub(r" \([^)]*\)", "", r["name"]).split(" #")[0].replace(" · ", ", ") + " [cluster]"
    cls, ph = r["name"].split(" — ")
    ph = ph.strip('"')
    return "%s: %s" % (cls, SHORT.get(ph, ph))


def short(t, n):
    return t if len(t) <= n else t[:n - 1].rstrip() + "…"


rows = sorted([r for r in D["rows"] if r["cls_iou_in"] is not None], key=lambda r: r["cls_iou_in"])
XLO = 20 if min(100 * r["cls_iou_in"] for r in rows) >= 24 else -12
MIOU = 100 * D["refs"]["all images"]["miou_pooled"]     # the dataset's mIoU (72.3), a dashed line on both columns
DASH = (0, (4, 3))
plt.rcParams.update({"font.family": "serif", "font.serif": ["Palatino Linotype", "Book Antiqua", "DejaVu Serif"],
                     "pdf.fonttype": 42})


def draw(ax, rr, fs=6.0, vs=5.2, dot=10, maxlab=34):
    for k, r in enumerate(rr):
        y = len(rr) - 1 - k
        col = SIT if r["branch"] == "language" else CLU
        xin = 100 * r["cls_iou_in"]
        xout = 100 * r["cls_iou_out"] if r["cls_iou_out"] is not None else None
        if xout is not None:
            ax.plot([xin, xout], [y, y], color=LINE, lw=1.2, zorder=1)
            ax.scatter([xout], [y], s=dot * 0.75, facecolors="white", edgecolors=col, linewidths=0.9, zorder=2,
                       marker="o" if r["branch"] == "language" else "s")
        ax.scatter([xin], [y], s=dot, color=col, zorder=3, marker="o" if r["branch"] == "language" else "s")
        left = xout is not None and xout < xin
        off = dot ** 0.5 / 2 + 1.5
        ax.annotate("%.0f" % xin, (xin, y), xytext=(off if left else -off, 0), textcoords="offset points",
                    ha="left" if left else "right", va="center", fontsize=vs, color=col,
                    zorder=1.5, bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none"))   # over the mIoU line
    ax.set_yticks(range(len(rr)))
    ax.set_yticklabels([short(label(r), maxlab) for r in reversed(rr)], fontsize=fs)
    ax.axvline(MIOU, color=DARK, lw=0.7, ls=DASH, zorder=0.8)
    ax.set_xlim(XLO, 100)
    ax.set_xticks(range(0 if XLO < 20 else 20, 101, 20))   # the mIoU's value is in the legend: 72 would touch 80
    ax.set_ylim(-0.8, len(rr) - 0.2)
    ax.set_xlabel("IoU of the slice's class (%)", color=MUTED, fontsize=fs)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.grid(axis="x", color="#EEF1F5", lw=0.6)
    ax.set_axisbelow(True)
    for side in ("right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["top"].set_color(LINE)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=MUTED, labelsize=fs - 0.5)


LEGEND = [Line2D([], [], marker="o", ls="", color=SIT, ms=4, label="situation, on its images"),
          Line2D([], [], marker="s", ls="", color=CLU, ms=4, label="cluster, on its images"),
          Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=MUTED, ms=4,
                 label="same class, on its other images"),
          Line2D([], [], color=DARK, lw=0.7, ls=DASH, label="mIoU of the dataset, %.1f" % MIOU)]
half = (len(rows) + 1) // 2
fig, axes = plt.subplots(1, 2, figsize=(5.2, 0.118 * half + 0.8))
for ax, rr in zip(axes, (rows[:half], rows[half:])):
    draw(ax, rr)
fig.legend(handles=LEGEND, loc="lower center", ncol=4, frameon=False, fontsize=6.0, columnspacing=1.2,
           handletextpad=0.4, handlelength=1.8)
fig.tight_layout(rect=(0, 0.025, 1, 1), w_pad=0.8)
out = os.path.join(HERE, "..", "Images", "Chapter5", "fig_slice_iou.pdf")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out)
print("saved", out, len(rows), "rows")
