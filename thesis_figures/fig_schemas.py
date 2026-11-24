# -*- coding: utf-8 -*-
"""Schematic figures of the thesis, drawn as vector PDFs (no data, only the structure of the method).
  python fig_schemas.py  ->  Images/Chapter2/fig_landscape.pdf, Images/Chapter3/fig_pipeline.pdf, fig_matching.pdf,
                             fig_bnd_ext.pdf, fig_embedding_schema.pdf, fig_certification.pdf
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "..", "Images")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42})

SLATE, AMBER, GREEN, DARK, MUTED = "#44546A", "#C2410C", "#1E7B4A", "#1E293B", "#5B6675"
SRC = {"Beyond IoU": "#7F8C9B", "SEG4SEG": "#7B4FA0", "MCSD": "#1F8A8A", "CB-SLICE": "#2F6DB5",
       "SliceLens": AMBER, "ours": GREEN}
TYPE_COL = {"missed": "#E07B1E", "hallucinated": "#C0392B", "merge": "#C2187A", "boundary": "#2F6DB5",
            "split": "#7B4FA0", "extent": "#1F8A8A", "wrong class": "#8A6D1F"}


def out(chapter, name):
    d = os.path.join(IMG, chapter)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


def box(ax, x, y, w, h, txt, ec=SLATE, fc="white", lw=1.2, size=8, bold_first=True, color=DARK, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", ec=ec, fc=fc, lw=lw,
                                ls=ls))
    lines = txt.split("\n")
    if bold_first and len(lines) > 1:
        ax.text(x + w / 2, y + h - 0.12, lines[0], ha="center", va="top", fontsize=size, fontweight="bold", color=color)
        ax.text(x + w / 2, y + (h - 0.3) / 2, "\n".join(lines[1:]), ha="center", va="center",
                fontsize=size - 0.6, color=color, linespacing=1.25)
    else:
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=size, color=color,
                fontweight="bold" if bold_first else "normal", linespacing=1.25)


def box_items(ax, x, y, w, h, title, items, ec=MUTED, lw=1.6, size=7, ls=(0, (4, 2))):
    """A block that mixes several works: dashed border, and each line in the colour of the work it comes from."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", ec=ec, fc="white", lw=lw,
                                ls=ls))
    ax.text(x + w / 2, y + h - 0.12, title, ha="center", va="top", fontsize=size, fontweight="bold", color=DARK)
    step = 0.29
    y0 = y + (h - 0.3) / 2 + step * (len(items) - 1) / 2
    for i, (s, c) in enumerate(items):
        ax.text(x + w / 2, y0 - i * step, s, ha="center", va="center", fontsize=size - 0.6, color=c,
                fontweight="normal")


def arrow(ax, p, q, color=SLATE, lw=1.1, style="-|>", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=9, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", ls=ls, shrinkA=1, shrinkB=1))


def canvas(w, h, fw, fh):
    fig = plt.figure(figsize=(fw, fh))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


# ===================================================================== the pipeline
def pipeline():
    fig, ax = canvas(17.4, 9.3, 7.2, 3.85)
    box(ax, 0.1, 3.7, 2.3, 1.8, "Input\nimage,\nground truth,\npredicted masks", ec=DARK, size=7)
    box(ax, 2.75, 3.3, 2.8, 2.6, "Block 1\nmissed\nhallucinated\nmerge\nsplit\n"
                                 "boundary\nextent\nwrong class", ec=SRC["Beyond IoU"], lw=1.6, size=7)
    arrow(ax, (2.4, 4.6), (2.75, 4.6))
    xs, BW = [6.05, 8.9, 11.75, 14.6], 2.65
    # cluster branch (top)
    y1 = 6.6
    ax.text(6.05, 9.2, "Cluster branch: how the model fails (no vocabulary)", fontsize=8.5, fontweight="bold",
            color=SLATE, va="top")
    cl = [("Features", "embedding:\nDINOv2 ViT-S/14\n+ error vector E\n+ type + γ(1−IoU)", SRC["SEG4SEG"]),
          ("Clustering", "one GMM per\nerror type,\nk chosen by BIC", SRC["SEG4SEG"]),
          ("Quality axes", [("ω², attribute purity", SRC["SEG4SEG"]), ("compactness", SRC["MCSD"]),
                            ("coherence", SRC["CB-SLICE"]), ("significance", GREEN)], None),
          ("Significance", [("mean z ≥ 0", GREEN), ("N ≥ 0.5%", SRC["SEG4SEG"]), ("attribute p ≤ 0.05", SRC["SEG4SEG"]),
                            ("BH q ≤ 0.05", DARK), ("coherence ≥ 0.6", SRC["CB-SLICE"])], None)]
    for (h, b, c), x in zip(cl, xs):
        if c is None:
            box_items(ax, x, y1, BW, 2.0, h, b)
        else:
            box(ax, x, y1, BW, 2.0, f"{h}\n{b}", ec=c, lw=1.6, size=7)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, (a + BW, y1 + 1.0), (b, y1 + 1.0))
    arrow(ax, (15.9, y1), (7.4, y1), rad=-0.22, ls="--", color=SRC["MCSD"])
    ax.text(10.6, 5.3, "peeling: remove the kept slices, cluster the rest again", fontsize=7, color=SRC["MCSD"],
            ha="center", style="italic")
    # language branch (bottom)
    y2 = 0.7
    ax.text(6.05, 3.05, "Language branch: when the model fails (situations in words)", fontsize=8.5,
            fontweight="bold", color=AMBER, va="top")
    lg = [("Hypotheses", "canonical\ngeneric\nclass-specific (Gemma)\ndata-driven", SRC["SliceLens"]),
          ("Grounding", "Qwen2.5-VL reads\neach object: c = P(yes),\nerrors and correct", SRC["SliceLens"]),
          ("Rank test", "AUC of c, errors vs\ncorrect objects;\nimage bootstrap;\nsample, then class", GREEN),
          ("Gates", "≥ 10 correct objects,\nBH q ≤ 0.05,\nmean c ≥ 0.6, size ≥ 3,\nmean z ≥ −0.25", GREEN)]
    for (h, b, c), x in zip(lg, xs):
        box(ax, x, y2, BW, 2.0, f"{h}\n{b}", ec=c, lw=1.6, size=7)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, (a + BW, y2 + 1.0), (b, y2 + 1.0))
    arrow(ax, (5.55, 5.3), (6.05, y1 + 1.0), rad=-0.2)
    arrow(ax, (5.55, 3.9), (6.05, y2 + 1.0), rad=0.2)
    # description
    box(ax, 14.25, 3.55, 3.1, 2.1, "Description\nof each image\'s errors:\nwhat went wrong (masks),\nhow (cluster),\nwhen (language)",
        ec=GREEN, lw=1.8, fc="#E4F1E9", size=7)
    arrow(ax, (15.9, y1), (15.9, 5.65))
    arrow(ax, (15.9, y2 + 2.0), (15.9, 3.55))
    arrow(ax, (5.55, 4.6), (14.25, 4.6), color=MUTED, ls="--")
    ax.text(9.9, 4.72, "typed errors", fontsize=7, color=MUTED, style="italic", ha="center")
    # legend of sources
    lx = 0.2
    ax.text(lx, 2.75, "Block from", fontsize=7.5, color=MUTED, fontweight="bold")
    for i, (k, c) in enumerate(SRC.items()):
        ax.add_patch(Rectangle((lx, 2.25 - 0.36 * i), 0.35, 0.22, fc="white", ec=c, lw=1.8))
        ax.text(lx + 0.5, 2.36 - 0.36 * i, "this thesis" if k == "ours" else k, fontsize=7, va="center", color=DARK)
    i = len(SRC)
    ax.add_patch(Rectangle((lx, 2.25 - 0.36 * i), 0.35, 0.22, fc="white", ec=MUTED, lw=1.6, ls=(0, (3, 1.5))))
    ax.text(lx + 0.5, 2.36 - 0.36 * i, "several: each line in its colour", fontsize=7, va="center", color=DARK)
    fig.savefig(out("Chapter3", "fig_pipeline.pdf"))
    plt.close(fig)


# ===================================================================== Block 1: matching
def _blob(cx, cy, rx, ry, n=60, wob=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = 1 + wob * np.sin(3 * t + rng.uniform(0, 6)) + wob * 0.5 * np.sin(5 * t)
    return np.c_[cx + rx * r * np.cos(t), cy + ry * r * np.sin(t)]


def matching():
    fig, axes = plt.subplots(1, 5, figsize=(7.2, 1.9))
    GT, PR = "#1E7B4A", "#E07B1E"
    titles = ["correct / boundary\n1 to 1, IoU ≥ 0.5", "split\nN to 1, each IoU ≥ 0.25",
              "merge\n1 to M (containment)", "missed\n0 to 1", "hallucinated\n1 to 0"]
    for ax, t in zip(axes, titles):
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 3.4)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(t, fontsize=7.5, color=DARK)

    def gt(ax, pts):
        ax.add_patch(Polygon(pts, closed=True, fc="none", ec=GT, lw=1.6))

    def pr(ax, pts):
        ax.add_patch(Polygon(pts, closed=True, fc=PR, alpha=0.45, ec=PR, lw=0.8))
    a = axes[0]
    gt(a, _blob(2, 1.7, 1.3, 1.1))
    pr(a, _blob(2.12, 1.62, 1.25, 1.12, wob=0.06))
    a = axes[1]
    gt(a, _blob(2, 1.7, 1.5, 1.1))
    pr(a, _blob(1.25, 1.7, 0.62, 0.95, wob=0.05))
    pr(a, _blob(2.8, 1.7, 0.62, 0.95, wob=0.05, seed=2))
    a.text(1.25, 0.3, "IoU 0.33", ha="center", fontsize=6.5, color=PR)
    a.text(2.8, 0.3, "IoU 0.33", ha="center", fontsize=6.5, color=PR)
    a = axes[2]
    gt(a, _blob(1.3, 1.7, 0.75, 1.0))
    gt(a, _blob(2.75, 1.7, 0.7, 0.95))
    pr(a, _blob(2.0, 1.7, 1.75, 1.15, wob=0.04))
    a = axes[3]
    gt(a, _blob(2, 1.7, 1.1, 1.0))
    a = axes[4]
    pr(a, _blob(2, 1.7, 1.1, 0.9, wob=0.08))
    fig.legend(handles=[Polygon([[0, 0]], fc="none", ec=GT, lw=1.6, label="ground-truth object"),
                        Polygon([[0, 0]], fc=PR, alpha=0.45, ec=PR, label="predicted mask")],
               loc="lower center", ncol=2, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.8, bottom=0.12, wspace=0.08)
    fig.savefig(out("Chapter3", "fig_matching.pdf"))
    plt.close(fig)


# ===================================================================== boundary vs extent
def bnd_ext():
    fig, ax = plt.subplots(figsize=(5.6, 2.7))
    ax.set_xlim(-1.2, 12.0)
    ax.set_ylim(-0.2, 5.9)
    ax.set_aspect("equal")
    ax.axis("off")
    g = _blob(4.2, 2.8, 2.6, 1.9)
    ax.add_patch(Polygon(_blob(4.2, 2.8, 2.95, 2.25), closed=True, fc="#DCE3EC", ec="none"))
    ax.add_patch(Polygon(_blob(4.2, 2.8, 2.25, 1.55), closed=True, fc="white", ec="none"))
    ax.add_patch(Polygon(_blob(4.35, 2.75, 2.55, 1.85, wob=0.05, seed=3), closed=True, fc="#E07B1E", alpha=0.35,
                         ec="#E07B1E", lw=0.8))
    ax.add_patch(Polygon(_blob(7.35, 3.4, 0.75, 0.55, seed=4), closed=True, fc="#C2187A", alpha=0.45,
                         ec="#C2187A", lw=0.8))
    ax.add_patch(Polygon(g, closed=True, fc="none", ec="#1E7B4A", lw=1.6))
    ax.annotate("pixels within 2d of the correctly\npredicted region P ∩ G\n(d = 1% of the image diagonal)",
                xy=(1.7, 4.3), xytext=(-1.1, 4.6),
                fontsize=7, color=SLATE, arrowprops=dict(arrowstyle="-", color=SLATE, lw=0.7))
    ax.annotate("wrong pixels inside the band:\nboundary error, E_bnd", xy=(6.75, 2.0), xytext=(7.5, 0.6),
                fontsize=7, color="#E07B1E", arrowprops=dict(arrowstyle="-", color="#E07B1E", lw=0.7))
    ax.annotate("wrong pixels outside the band:\nextent error, E_ext", xy=(7.9, 3.7), xytext=(8.3, 4.7), fontsize=7,
                color="#C2187A", arrowprops=dict(arrowstyle="-", color="#C2187A", lw=0.7))
    ax.text(4.2, 2.8, "ground-truth\nobject", ha="center", va="center", fontsize=7.5, color="#1E7B4A")
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(out("Chapter3", "fig_bnd_ext.pdf"))
    plt.close(fig)


# ===================================================================== embeddings, schematic
def embedding_schema():
    fig = plt.figure(figsize=(7.2, 3.5))
    ax = fig.add_axes([0.0, 0.0, 0.5, 1.0])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(0.1, 7.75, "(a) from one wrong object to one point", fontsize=8.5, fontweight="bold", color=DARK, va="top")
    ax.add_patch(Rectangle((0.2, 3.8), 2.4, 2.4, fc="#EEF2F6", ec=SLATE, lw=1))
    ax.add_patch(Polygon(_blob(1.4, 5.0, 0.8, 0.7, wob=0.1), closed=True, fc="#2F6DB5", alpha=0.55, ec="#2F6DB5"))
    ax.text(1.4, 3.45, "object region\n(its mask)", ha="center", va="top", fontsize=7, color=MUTED)
    for i in range(6):
        for j in range(6):
            inside = (i - 2.5) ** 2 / 4 + (j - 2.5) ** 2 / 3.2 < 1.5
            ax.add_patch(Rectangle((3.4 + 0.3 * i, 3.9 + 0.3 * j), 0.27, 0.27,
                                   fc="#2F6DB5" if inside else "#DCE3EC", alpha=0.75 if inside else 1, ec="none"))
    ax.text(4.25, 3.45, "DINOv2\npatch tokens", ha="center", va="top", fontsize=7, color=MUTED)
    arrow(ax, (2.65, 5.0), (3.35, 5.0))
    arrow(ax, (5.3, 5.0), (6.0, 5.0))
    for k in range(10):
        ax.add_patch(Rectangle((6.05, 3.6 + 0.28 * k), 0.38, 0.26, fc=plt.cm.Blues(0.25 + 0.07 * k), ec="white"))
    ax.text(6.24, 3.45, "mean over\nthe mask", ha="center", va="top", fontsize=7, color=MUTED)
    arrow(ax, (6.5, 5.0), (7.1, 5.0))
    y = 3.4
    for name, col, h in (("emb 384", "#2F6DB5", 1.6), ("E 3", "#E07B1E", 0.55), ("type 8", "#7B4FA0", 0.75),
                         ("γ(1−IoU) 1", "#C0392B", 0.5)):
        ax.add_patch(Rectangle((7.2, y), 0.5, h, fc=col, alpha=0.75, ec="white"))
        ax.text(7.85, y + h / 2, name, va="center", fontsize=6.5, color=col)
        y += h
    ax.text(7.45, 6.95, "x ∈ ℝ³⁹⁶", ha="center", va="bottom", fontsize=7.5, color=DARK, fontweight="bold")
    ax.text(0.2, 2.25, "x = [ emb ; αE ; onehot(type) ; γ(1 − IoU) ]\nblocks normalised to unit length, γ = 10 "
                       "(SEG4SEG);\nmissed objects are read on their ground-truth mask.",
            fontsize=7, color=DARK, va="top")
    ax.text(0.2, 0.75, "Objects that look alike and fail alike get close points;\nthe clustering works here, "
                       "one error type at a time.", fontsize=7, color=MUTED, va="top", style="italic")
    ax3 = fig.add_axes([0.5, 0.0, 0.5, 0.86], projection="3d")
    rng = np.random.default_rng(1)
    centers = {"missed": [(-2, 1.5, 1), (1.5, 2.5, 2.2)], "boundary": [(2.4, -1.8, -0.6), (0.2, -2.6, 1.8)],
               "merge": [(-2.2, -2.0, -1.5)], "hallucinated": [(2.6, 1.8, -1.9)]}
    for t, cs in centers.items():
        for c in cs:
            pts = rng.normal(c, (0.55, 0.45, 0.5), size=(45, 3))
            ax3.scatter(*pts.T, s=6, color=TYPE_COL[t], alpha=0.75, depthshade=False)
            u = np.linspace(0, 2 * np.pi, 40)
            ax3.plot(c[0] + 1.1 * np.cos(u), c[1] + 0.9 * np.sin(u), np.full_like(u, c[2]), color=TYPE_COL[t], lw=0.8)
        ax3.scatter([], [], color=TYPE_COL[t], s=12, label=t)
    ax3.set_xlabel("dim 1", fontsize=7, labelpad=-10)
    ax3.set_ylabel("dim 2", fontsize=7, labelpad=-10)
    ax3.set_zlabel("… dim 396", fontsize=7, labelpad=-10)
    ax3.set_xticklabels([])
    ax3.set_yticklabels([])
    ax3.set_zticklabels([])
    ax3.view_init(elev=18, azim=-58)
    ax3.legend(loc="upper right", fontsize=6.5, frameon=False, bbox_to_anchor=(1.0, 0.98))
    fig.text(0.52, 0.975, "(b) the feature space: each error type is clustered\napart; ellipses = Gaussian "
                          "components, chosen by BIC", fontsize=8, color=DARK, va="top", fontweight="bold")
    fig.savefig(out("Chapter3", "fig_embedding_schema.pdf"))
    plt.close(fig)


# ===================================================================== certification flow
def certification():
    fig, ax = canvas(16.4, 8.0, 7.2, 3.51)
    ax.text(0.2, 7.95, "Cluster branch: a candidate slice of errors", fontsize=8.5, fontweight="bold", color=SLATE,
            va="top")
    q = ["severe for its type?   mean z ≥ 0", "large enough?   |S| ≥ 0.5% of the objects",
         "shares an attribute beyond chance?\n≥ 5 informative tags, permutation p ≤ 0.05",
         "differs from the other errors?\nz-bootstrap or compactness permutation, BH q ≤ 0.05",
         "fails one way?   coherence ≥ 0.6 and type purity ≥ 0.6"]
    y = 6.75
    for t in q:
        box(ax, 0.2, y, 6.0, 0.9, t, ec=SLATE, bold_first=False, size=6.6)
        ax.text(6.3, y + 0.45, "no →", fontsize=6.8, color=AMBER, va="center")
        arrow(ax, (3.2, y), (3.2, y - 0.35))
        y -= 1.25
    box(ax, 0.2, y + 0.25, 6.0, 0.85, "KEEP: significant cluster\n(then peeling)", ec=GREEN, fc="#E4F1E9", size=7.5)
    box(ax, 7.0, 1.55, 2.4, 1.5, "significant, ω² ≥ 0.14\nbut incoherent:\nREFINE (split again)",
        ec=AMBER, fc="#FBECE2", bold_first=False, size=6.5)
    box(ax, 7.0, 4.6, 2.4, 1.0, "otherwise\nDISCARD", ec=AMBER, fc="#FBECE2", bold_first=False, size=7)
    ax.text(10.0, 7.95, "Language branch: one hypothesis", fontsize=8.5, fontweight="bold", color=AMBER,
            va="top")
    q2 = ["VLM reads it on a sample of 120 objects",
          "rank test on the sample\nAUC > 0.5, BH q ≤ 0.05",
          "VLM reads it on every object of the class",
          "class has ≥ 10 correct objects?\nrank test on the whole class, BH q ≤ 0.05",
          "members: c ≥ 0.5; mean c ≥ 0.6,\nsize ≥ 3, mean z ≥ −0.25"]
    y = 6.75
    for t in q2:
        box(ax, 10.1, y, 6.1, 0.9, t, ec=AMBER, bold_first=False, size=6.6)
        arrow(ax, (13.15, y), (13.15, y - 0.35))
        y -= 1.25
    box(ax, 10.1, y + 0.25, 6.1, 0.85, "significant situation\nenters the descriptions", ec=GREEN, fc="#E4F1E9",
        size=7.5)
    fig.savefig(out("Chapter3", "fig_certification.pdf"))
    plt.close(fig)


# ===================================================================== the landscape of related work
def landscape():
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    tasks = ["image\nclassification", "detection,\npose", "instance\nsegmentation", "medical\nsegmentation",
             "semantic\nsegmentation"]
    levels = ["error types\nonly", "slices of\nerrors", "slices\n+ names", "names verified\non each object",
              "+ test against\nchance"]
    works = [("GEORGE, MCSD", 0, 1), ("Domino, Jain et al.,\nCB-SLICE, LADDER", 0, 2),
             ("HiBug2 (names from a\nvocabulary of attributes)", 1, 2), ("SliceLens / GH-ESD", 2.0, 3),
             ("SEG4SEG (slices\nof images)", 3, 1), ("Beyond IoU", 4, 0),
             ("Singh et al. (one caption\nper patch, no slices)", 4, 0.5), ("This thesis", 4, 4)]
    for name, x, yv in works:
        col = GREEN if name == "This thesis" else (AMBER if name.startswith("SliceLens") else SLATE)
        ax.scatter([x], [yv], s=70 if name == "This thesis" else 40, color=col, zorder=3)
        right = x < 3.5 or name == "Beyond IoU"
        level = name.startswith("Singh") or name == "Beyond IoU"
        ax.text(x + (0.1 if right else -0.1), yv + (0 if level else 0.08), name, fontsize=7.2, color=col,
                va="center" if level else "bottom", ha="left" if right else "right",
                fontweight="bold" if col != SLATE else "normal")
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(tasks, fontsize=7.5)
    ax.set_yticks(range(len(levels)))
    ax.set_yticklabels(levels, fontsize=7.5)
    ax.set_xlim(-0.4, 4.9)
    ax.set_ylim(-0.5, 4.6)
    ax.grid(color="#E5E9EF", lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.add_patch(Rectangle((3.6, 2.5), 1.3, 1.25, fc="#FBECE2", ec="none", zorder=0))
    ax.text(4.25, 2.95, "empty before\nthis work", fontsize=7, color=AMBER, ha="center", style="italic")
    ax.set_xlabel("task the method is tested on", fontsize=8, color=MUTED)
    ax.set_ylabel("what the method returns", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(out("Chapter2", "fig_landscape.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    pipeline()
    matching()
    bnd_ext()
    embedding_schema()
    certification()
    landscape()
    print("ok")
