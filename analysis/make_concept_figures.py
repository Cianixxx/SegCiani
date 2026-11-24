# -*- coding: utf-8 -*-
"""Concept figures for the implementation report.

Each figure exists to carry an argument the text makes, not to decorate it:

  fig_taxonomy.png    what the 7 typed-error cases actually look like, and why a
                      SECOND matching threshold is needed for split/merge at all
  fig_audit_trail.png how the certified-slice count moved, and what changed each time
  fig_purity_null.png the permutation null of the purity gate against the observed
                      value for a real slice (needs outputs/slice_quality/purity_null.json,
                      produced by slice_quality/examples/export_purity_null.py)

Run: python docs/make_concept_figures.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

DOCS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(DOCS)
sys.path.insert(0, os.path.join(ROOT, "docs"))
from _revision_numbers import out_path as _out_path  # noqa: E402  (RUN_UFFICIALI, RUN_VECCHI or outputs/)
NULL_JSON = _out_path("slice_quality", "purity_null.json")
MODELS_JSON = _out_path("slice_quality", "model_comparison.json")

INK, MUTED = "#1f2933", "#7b8794"
GT_EDGE = "#2f6fb2"          # ground truth: outline only
PRED = "#e8a33d"             # prediction: filled
PRED2 = "#c05b4d"            # prediction of a different class


def _blob(cx, cy, r, wobble=0.0, n=14, seed=0):
    """A closed roundish polygon; wobble>0 makes the edge ragged."""
    import math
    import random
    rng = random.Random(seed)
    pts = []
    for i in range(n):
        a = 2 * math.pi * i / n
        rr = r * (1 + wobble * (rng.random() - 0.5) * 2)
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a) * 0.85))
    return pts


def taxonomy():
    cases = [
        ("correct", "IoU >= 0.85", "kept as reference"),
        ("boundary", "matched, ragged edge", "same object, imprecise outline"),
        ("extent", "matched, part missing", "a chunk of the object is lost"),
        ("confusion", "matched, wrong class", "right pixels, wrong label"),
        ("FN (missed)", "no prediction claims it", "the object is simply not there"),
        ("FP (spurious)", "claims nothing real", "an object that does not exist"),
        ("split", "2 fragments, each < 0.5", "one object broken in two"),
        ("merge", "1 prediction over 2 objects", "two objects fused into one"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(12.6, 6.0), dpi=200)
    for ax, (name, rule, gloss) in zip(axes.ravel(), cases):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect("equal")
        ax.axis("off")
        gt = dict(closed=True, fill=False, ec=GT_EDGE, lw=1.8, ls=(0, (4, 2)))
        pr = dict(closed=True, fc=PRED, ec="none", alpha=.75)

        if name == "correct":
            ax.add_patch(Polygon(_blob(5, 5.4, 2.7, 0.03, seed=1), **pr))
            ax.add_patch(Polygon(_blob(5, 5.4, 2.8, 0.03, seed=1), **gt))
        elif name == "boundary":
            ax.add_patch(Polygon(_blob(5, 5.4, 2.7, 0.30, seed=3), **pr))
            ax.add_patch(Polygon(_blob(5, 5.4, 2.8, 0.03, seed=1), **gt))
        elif name == "extent":
            pts = [p for p in _blob(5, 5.4, 2.7, 0.03, seed=1) if p[1] < 6.2]
            ax.add_patch(Polygon(pts, **pr))
            ax.add_patch(Polygon(_blob(5, 5.4, 2.8, 0.03, seed=1), **gt))
        elif name == "confusion":
            pr2 = dict(pr, fc=PRED2)
            ax.add_patch(Polygon(_blob(5, 5.4, 2.7, 0.05, seed=1), **pr2))
            ax.add_patch(Polygon(_blob(5, 5.4, 2.8, 0.03, seed=1), **gt))
        elif name.startswith("FN"):
            ax.add_patch(Polygon(_blob(5, 5.4, 2.8, 0.03, seed=1), **gt))
        elif name.startswith("FP"):
            ax.add_patch(Polygon(_blob(5, 5.4, 2.5, 0.10, seed=7), **pr))
        elif name == "split":
            ax.add_patch(Polygon(_blob(3.4, 5.4, 1.35, 0.10, seed=4), **pr))
            ax.add_patch(Polygon(_blob(6.7, 5.4, 1.35, 0.10, seed=5), **pr))
            ax.add_patch(Polygon(_blob(5, 5.4, 3.0, 0.03, seed=1), **gt))
        elif name == "merge":
            ax.add_patch(Polygon(_blob(5, 5.4, 3.2, 0.08, seed=6), **pr))
            ax.add_patch(Polygon(_blob(3.3, 5.4, 1.5, 0.03, seed=1), **gt))
            ax.add_patch(Polygon(_blob(6.8, 5.4, 1.5, 0.03, seed=2), **gt))

        ax.text(5, 1.9, name, ha="center", fontsize=12, color=INK, fontweight="bold")
        ax.text(5, 1.05, rule, ha="center", fontsize=9.5, color="#3f6fa8")
        ax.text(5, 0.25, gloss, ha="center", fontsize=9, color=MUTED, style="italic")

    fig.suptitle("The seven typed-error cases (plus the correct reference)",
                 fontsize=14, color=INK, fontweight="bold", y=0.985)
    fig.text(0.5, 0.925,
             "dashed outline = ground-truth object   ·   filled = model prediction"
             "   ·   red fill = predicted as a different class",
             ha="center", fontsize=10, color=MUTED)
    fig.text(0.5, 0.02,
             "SPLIT and MERGE are why a single threshold is not enough: two fragments of one "
             "object can never BOTH reach IoU 0.5, so with one threshold those cases are "
             "invisible by construction. A second, lower threshold (0.25) lets a fragment "
             "'touch' an object without claiming it.",
             ha="center", fontsize=9.5, color=INK, wrap=True)
    fig.tight_layout(rect=[0, 0.06, 1, 0.90])
    out = os.path.join(DOCS, "fig_taxonomy.png")
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    print("saved", out)


def audit_trail():
    steps = [
        (22, "purity gate silently\nfell back to a measure\nthat is 1.0 by construction",
         "under-tested", "#c9d4de"),
        (6, "fixed the fallback, kept\nthe absolute 0.80 threshold\non a noisy VLM channel",
         "wrong test", "#c9d4de"),
        (9, "threshold replaced by a\npermutation test against\nchance (self-calibrating)",
         "right test", "#8fb4d9"),
        (19, "attribute tags re-elicited\nby verification — but only\nhalf the pool (mixed)",
         "unfair null", "#e8a33d"),
        (14, "re-elicitation completed:\nslice and null measured\nby the SAME instrument",
         "final", "#2f6fb2"),
    ]
    fig, ax = plt.subplots(figsize=(11.5, 4.4), dpi=200)
    xs = range(len(steps))
    ax.bar(list(xs), [s[0] for s in steps], width=.55,
           color=[s[3] for s in steps], zorder=3)
    for i, (n, why, tag, _c) in enumerate(steps):
        ax.text(i, n + .6, str(n), ha="center", fontsize=15, fontweight="bold", color=INK)
        ax.text(i, -1.4, why, ha="center", va="top", fontsize=8.6, color=MUTED)
        ax.text(i, n + 2.4, tag, ha="center", fontsize=9, color="#3f6fa8",
                fontweight="bold")
    ax.set_xticks([])
    ax.set_ylim(0, 26)
    ax.set_ylabel("significant cluster slices", fontsize=10, color=MUTED)
    ax.yaxis.grid(True, color="#eef2f6", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.tick_params(colors=MUTED, length=0)
    ax.set_title("Every change to the count was a change to the MEASUREMENT, "
                 "not to the discovery code",
                 fontsize=13, color=INK, loc="left", pad=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.16, 1, 1])
    out = os.path.join(DOCS, "fig_audit_trail.png")
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    print("saved", out)


def purity_null():
    if not os.path.exists(NULL_JSON):
        print("skipped fig_purity_null.png (no", NULL_JSON + ")")
        return
    d = json.load(open(NULL_JSON, encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(9.4, 4.0), dpi=200)
    ax.hist(d["null"], bins=28, color="#c9d4de", edgecolor="white", zorder=3,
            label="random groups of same-subtype errors (%d draws)" % len(d["null"]))
    ax.axvline(d["observed"], color="#2f6fb2", lw=2.6, zorder=4,
               label="this slice (%s, %d instances)" % (d["slice_id"], d["size"]))
    ax.axvline(0.80, color="#c05b4d", lw=1.6, ls="--", zorder=4,
               label="the absolute threshold we abandoned (0.80)")
    ax.set_xlabel("attribute agreement among informative tags", fontsize=10, color=MUTED)
    ax.set_ylabel("draws", fontsize=10, color=MUTED)
    ax.set_xlim(0, 1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    # inline annotations below replace the legend: a legend box here would sit on
    # top of them and repeat what they already say
    ax.legend().set_visible(False)
    ax.set_title("Why purity is tested against chance, not against a fixed number",
                 fontsize=13, color=INK, loc="left", pad=12, fontweight="bold")
    mean_null = sum(d["null"]) / len(d["null"])
    top = ax.get_ylim()[1]
    ax.annotate("chance level\n%.2f" % mean_null, xy=(mean_null, top * 0.30),
                xytext=(mean_null - 0.17, top * 0.62), fontsize=9.5,
                color=MUTED, ha="center",
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
    ax.annotate("this slice: %.2f  (p = %.3f)\n%s, %d instances, %d informative tags"
                % (d["observed"], d["p"], d["slice_id"], d["size"], d["n_informative"]),
                xy=(d["observed"] + 0.02, top * 0.93), fontsize=10,
                color="#2f6fb2", fontweight="bold", va="top")
    ax.annotate("what the abandoned\nabsolute threshold demanded", xy=(0.60, top * 0.55),
                fontsize=9.5, color="#c05b4d", va="center")
    ax.annotate("%d random groups of\nsame-subtype errors" % len(d["null"]),
                xy=(0.045, top * 0.30), fontsize=9.5, color=MUTED, va="center")
    fig.tight_layout()
    out = os.path.join(DOCS, "fig_purity_null.png")
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    print("saved", out)


def complementarity():
    """How much of the covered error space each branch reaches ALONE.

    This is the empirical answer to the obvious objection: if the two branches
    largely agreed, one of them would be redundant and the design would be
    over-engineered. They do not agree, and the figure says by how much.
    """
    path = os.path.join(DOCS, "complementarity.json")
    if not os.path.exists(path):
        print("skipped fig_complementarity.png (no", path + ")")
        return
    d = json.load(open(path, encoding="utf-8"))
    rows = [("error instances", d["inst"]), ("images", d["imgs"])]
    C_ONLY, BOTH, L_ONLY = "#2f6fb2", "#7b8794", "#e8a33d"

    fig, ax = plt.subplots(figsize=(10.6, 3.5), dpi=200)
    for i, (label, (c, b, l)) in enumerate(rows):
        y = len(rows) - 1 - i
        tot = c + b + l
        ax.barh(y, c, color=C_ONLY, height=.5, zorder=3)
        ax.barh(y, b, left=c, color=BOTH, height=.5, zorder=3)
        ax.barh(y, l, left=c + b, color=L_ONLY, height=.5, zorder=3)
        for x, n, col in ((c / 2, c, "white"), (c + b / 2, b, "white"),
                          (c + b + l / 2, l, "white")):
            ax.text(x, y, str(n), ha="center", va="center", color=col,
                    fontsize=11, fontweight="bold")
        ax.text(tot * 1.02, y, "  %s (%d covered)" % (label, tot), va="center",
                fontsize=10.5, color=INK)

    ax.set_xlim(0, max(sum(r[1]) for r in rows) * 1.30)
    ax.set_ylim(-0.6, len(rows) - 0.3)
    ax.axis("off")
    ax.set_title("The two branches cover different errors", fontsize=13.5,
                 color=INK, loc="left", pad=26, fontweight="bold")
    # colour-coded key, so the reader does not have to infer it from bar order
    xmax = ax.get_xlim()[1]
    for frac, txt, col in ((0.00, "found ONLY by the cluster branch", C_ONLY),
                           (0.30, "found by BOTH", BOTH),
                           (0.44, "found ONLY by the language branch", L_ONLY)):
        ax.text(xmax * frac, len(rows) - 0.35, txt, fontsize=9.5, color=col,
                fontweight="bold")
    fig.text(0.012, -0.06,
             "Only %d of the %d covered instances are reached by both branches (%.0f%%). "
             "Dropping either one would cost roughly %d and %d instances respectively — "
             "the redundancy objection does not survive the measurement. Cross-branch "
             "corroboration is correspondingly rare: %d of the %d certified cluster "
             "slices carry a verified SliceLens hypothesis."
             % (d["inst"][1], sum(d["inst"]), 100 * d["inst"][1] / sum(d["inst"]),
                d["inst"][0], d["inst"][2], d["corroborated"], d["n_cluster_slices"]),
             fontsize=9.5, color=INK, wrap=True)
    fig.tight_layout()
    out = os.path.join(DOCS, "fig_complementarity.png")
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    print("saved", out)


def model_comparison():
    """Same pipeline, two segmenters: does the KIND of failure change?

    A single IoU number would report 0.48 -> 0.82 and stop there. The typed
    taxonomy shows the composition moving, which is the actual finding.
    """
    if not os.path.exists(MODELS_JSON):
        print("skipped fig_model_comparison.png (no", MODELS_JSON + ")")
        return
    d = json.load(open(MODELS_JSON, encoding="utf-8"))
    a, b = d["mobilenet_pred"], d["deeplab_pred"]
    order = ["fn", "fp", "boundary", "extent", "merge", "confusion", "split"]
    label = {"fn": "missed (FN)", "fp": "invented (FP)", "boundary": "imprecise edge",
             "extent": "part lost", "merge": "objects fused", "confusion": "wrong class",
             "split": "object broken"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.3), dpi=200,
                                   gridspec_kw={"width_ratios": [1.25, 1]})

    xs = range(len(order))
    wa = [a["taxonomy"].get(k, 0) for k in order]
    wb = [b["taxonomy"].get(k, 0) for k in order]
    ax1.bar([x - .19 for x in xs], wa, width=.38, color="#c9d4de",
            label="LR-ASPP MobileNetV3  (mIoU 0.48)", zorder=3)
    ax1.bar([x + .19 for x in xs], wb, width=.38, color="#2f6fb2",
            label="DeepLabV3-ResNet101  (mIoU 0.82)", zorder=3)
    for x, (va, vb) in enumerate(zip(wa, wb)):
        if va:
            ax1.text(x + .19, vb + 60, "%+.0f%%" % (100 * (vb - va) / va), ha="center",
                     fontsize=8.6, color=("#2f8f5b" if vb < va else "#c05b4d"),
                     fontweight="bold")
    ax1.set_xticks(list(xs))
    ax1.set_xticklabels([label[k] for k in order], fontsize=9, rotation=18, ha="right")
    ax1.set_ylabel("typed errors", fontsize=10, color=MUTED)
    ax1.legend(frameon=False, fontsize=9.5)
    ax1.yaxis.grid(True, color="#eef2f6", zorder=0)
    ax1.set_axisbelow(True)
    for sp in ("top", "right"):
        ax1.spines[sp].set_visible(False)
    ax1.tick_params(colors=MUTED)
    ax1.set_title("Counts", fontsize=11.5, color=INK, loc="left", fontweight="bold")

    for i, (name, m) in enumerate((("MobileNet", a), ("DeepLab", b))):
        left = 0
        for k, col in zip(order, ["#2f6fb2", "#e8a33d", "#7b8794", "#a8c4de",
                                  "#c05b4d", "#8fb4d9", "#d9d9d9"]):
            sh = m["taxonomy_share"].get(k, 0)
            ax2.barh(1 - i, sh, left=left, color=col, height=.45, zorder=3)
            if sh > .05:
                ax2.text(left + sh / 2, 1 - i, "%.0f%%" % (100 * sh), ha="center",
                         va="center", color="white", fontsize=9.5, fontweight="bold")
            left += sh
        ax2.text(-0.02, 1 - i, name, ha="right", va="center", fontsize=10, color=INK)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-0.55, 1.5)
    ax2.axis("off")
    ax2.set_title("Composition", fontsize=11.5, color=INK, loc="left", fontweight="bold")
    ax2.text(0, 1.42, "  ".join("%s" % label[k] for k in order[:3]) + "  ...",
             fontsize=8.5, color=MUTED)

    fig.suptitle("A stronger segmenter does not just make fewer errors — it makes "
                 "different ones", fontsize=13.5, color=INK, fontweight="bold",
                 x=0.012, ha="left", y=1.03)
    fig.text(0.012, -0.10,
             "Errors fall 25% (6406 → 4818) and correct references rise 69% "
             "(522 → 883). But invented objects nearly halve while imprecise edges "
             "grow in absolute number and rise from 12% to 17% of all errors, and fused "
             "objects grow 27%: the failure mode shifts from perception (is there an "
             "object?) to delineation (where does it end, and is it one or two?).",
             fontsize=9.4, color=INK, wrap=True)
    fig.tight_layout()
    out = os.path.join(DOCS, "fig_model_comparison.png")
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    taxonomy()
    audit_trail()
    purity_null()
    complementarity()
    model_comparison()
