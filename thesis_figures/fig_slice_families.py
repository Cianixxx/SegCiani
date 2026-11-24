"""Figure 2.1: the three ways of finding an error slice, in a schematic (10 November 2026).

The data are synthetic and only illustrate the idea: (a) slices defined by features, as on tabular data; (b) slices as
dense regions of errors in an embedding space; (c) slices as situations written in words and checked on every example.
Run: python figs/fig_slice_families.py  (writes Images/Chapter2/fig_slice_families.pdf)
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "Images", "Chapter2")
os.makedirs(OUT, exist_ok=True)
RED, GREY, BLUE, INK = "#c0392b", "#b8b8b8", "#2c6fbb", "#333333"


def panel_features(a):
    rows = [("all examples", 0.12), ("age < 30", 0.14), ("occupation = A", 0.11),
            ("age < 30  and  occupation = A", 0.41), ("age > 60", 0.09)]
    y = np.arange(len(rows))[::-1]
    for yi, (lab, e) in zip(y, rows):
        hot = e > 0.3
        a.barh(yi, e, color=RED if hot else GREY, height=0.6)
        a.text(-0.02, yi, lab, ha="right", va="center", fontsize=9, color=INK, weight="bold" if hot else None)
        a.text(e + 0.01, yi, f"{e:.0%}", va="center", fontsize=8, color=INK)
    a.set_xlim(0, 0.55)
    a.set_yticks([]); a.set_xticks([])
    for s in ("top", "right", "left"):
        a.spines[s].set_visible(False)
    a.set_xlabel("error rate", fontsize=9)
    a.set_title("(a) Slices from features\n(tabular data)", fontsize=10)


def panel_embedding(a):
    rng = np.random.default_rng(3)
    ok = rng.normal(0, 1, (260, 2)) * [1.6, 1.1]
    hot = rng.normal([1.6, 1.2], 0.28, (34, 2))
    err_bg = rng.normal(0, 1, (14, 2)) * [1.6, 1.1]
    ok_in = rng.normal([1.6, 1.2], 0.3, (6, 2))
    a.scatter(*ok.T, s=9, c=GREY, lw=0)
    a.scatter(*ok_in.T, s=9, c=GREY, lw=0)
    a.scatter(*err_bg.T, s=12, c=RED, lw=0)
    a.scatter(*hot.T, s=12, c=RED, lw=0)
    a.add_patch(Ellipse((1.6, 1.2), 2.0, 1.7, fill=False, ec=INK, lw=1.2, ls="--"))
    a.text(0.0, 2.75, "dense region of errors: water birds on land", ha="center", fontsize=8, color=INK,
           bbox=dict(fc="white", ec="none", pad=1))
    a.set_aspect("equal")
    a.set_xticks([]); a.set_yticks([])
    a.set_xlim(-4.2, 4.2); a.set_ylim(-3.2, 3.2)
    a.set_title("(b) Slices from an embedding\n(images)", fontsize=10)
    a.scatter([], [], s=12, c=RED, label="wrong"); a.scatter([], [], s=9, c=GREY, label="right")
    a.legend(loc="lower left", fontsize=8, frameon=False)


def panel_language(a):
    a.set_xlim(0, 1); a.set_ylim(0, 1); a.axis("off")
    a.add_patch(FancyBboxPatch((0.04, 0.78), 0.92, 0.14, boxstyle="round,pad=0.01", fc="#eef3fa", ec=BLUE))
    a.text(0.5, 0.85, "LLM: “a person partly\nhidden behind another”", ha="center", va="center",
           fontsize=9, color=INK)
    a.annotate("", xy=(0.5, 0.64), xytext=(0.5, 0.77), arrowprops=dict(arrowstyle="->", color=INK))
    a.text(0.55, 0.70, "VLM reads every object", fontsize=7.5, color=INK, va="center")
    ys = [0.55, 0.47, 0.39, 0.31, 0.23]
    conf = [0.94, 0.88, 0.71, 0.22, 0.08]
    wrong = [True, True, False, False, False]
    for y, c, w in zip(ys, conf, wrong):
        a.add_patch(plt.Rectangle((0.08, y - 0.03), 0.06, 0.06, color=RED if w else GREY))
        a.text(0.18, y, f"P(yes) = {c:.2f}", va="center", fontsize=8, color=INK)
        a.text(0.74, y, "wrong" if w else "right", va="center", fontsize=8, color=RED if w else "#666666")
    a.text(0.5, 0.06, "kept if the wrong objects\nget a higher P(yes)",
           ha="center", fontsize=8, color=INK)
    a.set_title("(c) Slices from words\n(language and VLMs)", fontsize=10)


def main():
    fig, ax = plt.subplots(1, 3, figsize=(8.4, 3.4), gridspec_kw=dict(width_ratios=[1.15, 1, 1], wspace=0.25))
    panel_features(ax[0]); panel_embedding(ax[1]); panel_language(ax[2])
    fig.subplots_adjust(left=0.17, right=0.99, top=0.84, bottom=0.08)
    p = os.path.join(OUT, "fig_slice_families.pdf")
    fig.savefig(p, bbox_inches="tight")
    fig.savefig(p.replace(".pdf", ".png"), dpi=80, bbox_inches="tight")
    print("written", p)


if __name__ == "__main__":
    main()
