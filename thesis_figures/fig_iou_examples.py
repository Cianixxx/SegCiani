"""Figure 1.2: four objects of VOC val with almost the same IoU and very different errors (10 November 2026).

The object is a connected component of its class in the ground truth; the prediction is every predicted component of
the same class that touches it; void pixels (255) are left out, as in the VOC evaluation. Rows: the image, the ground truth and the prediction, in the VOC colours.
Run: python figs/fig_iou_examples.py  (writes Images/Chapter1/fig_iou_examples.pdf and prints the numbers)
"""
import os
import sys

import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig_images as F

VOC = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
       'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
# (image, class, component, title); the types are those Block 1 gives to the same objects in the official run
CASES = [("2011_002124", "person", 1, "(a) person: legs missed\nin the dark"),
         ("2007_000129", "bicycle", 8, "(b) bicycle: thin parts\nlost"),
         ("2009_002487", "diningtable", 1, "(c) table: base missed,\npart called chair"),
         ("2009_002727", "boat", 1, "(d) boat: merged with\nthe boat next to it")]
GREEN, RED, BLUE, YELLOW = (0.20, 0.75, 0.30), (0.90, 0.20, 0.20), (0.20, 0.45, 0.95), (0.95, 0.80, 0.10)


def obj(name, cls, k):
    img, _, _, gt, pr = F.p80(name)
    c = VOC.index(cls)
    lab, _ = ndimage.label(gt == c)
    G = lab == k
    plab, _ = ndimage.label(pr == c)
    touch = set(np.unique(plab[G])) - {0}
    P = np.isin(plab, list(touch)) & (gt != 255)
    other = G & ~P & (pr != 0) & (pr != 255) & (pr != c)
    return img, G, P, other


def overlay(img, G, P, other):
    base = img.astype(float) / 255
    col = np.zeros_like(base)
    col[G & P] = GREEN
    col[G & ~P] = RED
    col[P & ~G] = BLUE
    col[other] = YELLOW
    m = (G | P)[..., None]
    return np.where(m, 0.3 * base + 0.7 * col, 0.6 * base)


def main():
    fig, ax = plt.subplots(3, 4, figsize=(7.2, 6.4), gridspec_kw=dict(hspace=0.04, wspace=0.05))
    for j, (n, c, k, title) in enumerate(CASES):
        img, G, P, other = obj(n, c, k)
        _, gt_rgb, pr_rgb, _, _ = F.p80(n)
        iou = (G & P).sum() / (G | P).sum()
        print(n, c, k, "IoU %.3f" % iou, "gt", G.sum(), "missed", (G & ~P).sum(), "outside", (P & ~G).sum(),
              "other class", other.sum())
        ys, xs = np.where(G | P)
        pad = 20
        y0, y1 = max(0, ys.min() - pad), min(520, ys.max() + pad)
        x0, x1 = max(0, xs.min() - pad), min(520, xs.max() + pad)
        # square-ish crops so the four columns line up
        h, w = y1 - y0, x1 - x0
        if w < h * 0.75:
            cx = (x0 + x1) // 2; half = int(h * 0.375); x0, x1 = max(0, cx - half), min(520, cx + half)
        for r, im in enumerate((img, gt_rgb, pr_rgb)):
            ax[r, j].imshow(im[y0:y1, x0:x1], interpolation="nearest")
        ax[0, j].set_title(title, fontsize=9)
        ax[2, j].set_xlabel("IoU = %.2f" % iou, fontsize=10)
        for a in ax[:, j]:
            a.set_xticks([]); a.set_yticks([])
            for s in a.spines.values():
                s.set_visible(False)
    for r, lab in enumerate(("image", "ground truth\n(correct)", "prediction\n(wrong)")):
        ax[r, 0].set_ylabel(lab, fontsize=9)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.92, bottom=0.05)
    out = F.out("Chapter1", "fig_iou_examples.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=80, bbox_inches="tight")
    print("written", out)


if __name__ == "__main__":
    main()
