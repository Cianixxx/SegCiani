"""Figure 3.1: which work has which property, as a matrix (10 November 2026; replaces the x-y landscape).

A filled dot: the work has the property; a half dot: in part; empty: not. The judgements are those of Table 3.1 and
Chapter 2.  Run: python figs/fig_gaps_matrix.py  (writes Images/Chapter2/fig_gaps_matrix.pdf)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "Images", "Chapter2")
PROPS = ["semantic\nsegmentation", "object\nas unit", "returns\nslices", "names\nthe slices",
         "verifies\nmembers", "test against\nchance", "types\nthe error"]
# 1 yes, 0.5 in part, 0 no
WORKS = [
    ("GEORGE",            [0, 0, 1, 0, 0, 0, 0]),
    ("Domino",            [0, 0, 1, 1, 0, 0, 0]),
    ("Jain et al.",       [0, 0, 1, 1, 0, 0, 0]),
    ("MCSD",              [0, 0, 1, 0, 0, 0, 0]),
    ("CB-SLICE",          [0, 0, 1, 1, 0, 0, 0]),
    ("LADDER",            [0, 0, 1, 1, 0.5, 0, 0]),
    ("HiBug2",            [0, 0.5, 1, 1, 0.5, 0, 0]),
    ("SliceLens / GH-ESD", [0, 1, 1, 1, 1, 0.5, 0]),
    ("SEG4SEG",           [1, 0, 1, 0, 0, 0, 0]),
    ("Singh et al.",      [1, 0.5, 0, 0.5, 0, 0, 0]),
    ("Beyond IoU",        [1, 0, 0, 0, 0, 0, 1]),
    ("This thesis",       [1, 1, 1, 1, 1, 1, 1]),
]
INK, ACC = "#333333", "#c0392b"


def main():
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    n = len(WORKS)
    for i, (name, row) in enumerate(WORKS):
        y = n - 1 - i
        mine = name == "This thesis"
        col = ACC if mine else INK
        for j, v in enumerate(row):
            ax.scatter(j, y, s=120, facecolors="white", edgecolors=col, linewidths=1.2, zorder=2)
            if v == 1:
                ax.scatter(j, y, s=120, c=col, zorder=3)
            elif v == 0.5:
                ax.scatter(j, y, s=120, c=col, marker=matplotlib.markers.MarkerStyle("o", fillstyle="left"), zorder=3)
        ax.text(-0.6, y, name, ha="right", va="center", fontsize=10.5, color=col, weight="bold" if mine else None)
    ax.axhline(0.5, color="#bbbbbb", lw=0.8)
    ax.set_xticks(range(len(PROPS)))
    ax.set_xticklabels(PROPS, fontsize=9.5)
    ax.xaxis.tick_top()
    ax.set_yticks([])
    ax.set_xlim(-0.5, len(PROPS) - 0.5)
    ax.set_ylim(-0.6, n - 0.4)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    fig.subplots_adjust(left=0.25, right=0.99, top=0.86, bottom=0.02)
    p = os.path.join(OUT, "fig_gaps_matrix.pdf")
    fig.savefig(p)
    fig.savefig(p.replace(".pdf", ".png"), dpi=80)
    print("written", p)


if __name__ == "__main__":
    main()
