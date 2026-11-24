# -*- coding: utf-8 -*-
"""Figures of the thesis made of real images (CPU only). Sources: the 100 annotated images with their masks
(UNI/TESI/DATASETS/pascal_80), the HTML pages of the runs (HTML/...), which embed every picture they show, and
result_tagged.pkl of the official run (boxes, embeddings).
  python fig_images.py [name ...]  ->  Images/Chapter*/fig_*.pdf|jpg, generated/human_examples.tex
"""
import base64
import html
import io
import json
import os
import pickle
import re
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "SlicelenceCode"))
sys.path.insert(0, HERE)
import gallery_extract as G  # noqa: E402

IMG = os.path.join(HERE, "..", "Images")
GEN = os.path.join(HERE, "..", "generated")
P80 = os.path.join(os.path.dirname(ROOT), "UNI", "TESI", "DATASETS", "pascal_80")
HT = os.path.join(ROOT, "HTML")
INST = os.path.join(ROOT, "RUN_UFFICIALI", "slice_quality_inst")
SCR = os.environ.get("FIG_SCRATCH", os.path.join(HERE, "cache"))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42})
DARK, MUTED, GREEN, AMBER, SLATE = "#1E293B", "#5B6675", "#1E7B4A", "#C2410C", "#44546A"
VOC = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
       "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


def out(chapter, name):
    d = os.path.join(IMG, chapter)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


def palette():
    p = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        c, r, g, b = i, 0, 0, 0
        for j in range(8):
            r |= ((c >> 0) & 1) << (7 - j)
            g |= ((c >> 1) & 1) << (7 - j)
            b |= ((c >> 2) & 1) << (7 - j)
            c >>= 3
        p[i] = (r, g, b)
    p[255] = (224, 224, 192)
    return p


PAL = palette()


def crop520(im, nearest=False):
    """The evaluation transform of the thesis: short side to 520 pixels, centre crop 520 x 520."""
    w, h = im.size
    s = 520 / min(w, h)
    im = im.resize((round(w * s), round(h * s)), Image.NEAREST if nearest else Image.BILINEAR)
    w, h = im.size
    l, t = (w - 520) // 2, (h - 520) // 2
    return im.crop((l, t, l + 520, t + 520))


def p80(name):
    """(image, ground truth, prediction) of an annotated image, as RGB arrays in the 520 crop."""
    img = crop520(Image.open(os.path.join(P80, "JPEGImages", name + ".jpg")).convert("RGB"))
    gt = np.array(crop520(Image.open(os.path.join(P80, "SegmentationClass", name + ".png")), nearest=True))
    pr = np.array(Image.open(os.path.join(P80, "mobilenet_pred", f"mask_pr_{name}.png")))[..., 0]
    return np.array(img), PAL[gt], PAL[pr], gt, pr


def b2a(b):
    return np.array(Image.open(io.BytesIO(b)).convert("RGB"))


def triptych_axes(fig, rows, cols=3, top=0.93, bottom=0.02, left=0.11, right=0.995, hspace=0.04, wspace=0.02):
    gs = fig.add_gridspec(rows, cols, top=top, bottom=bottom, left=left, right=right, hspace=hspace, wspace=wspace)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(cols)] for r in range(rows)]
    for row in axes:
        for a in row:
            a.set_xticks([])
            a.set_yticks([])
            for s in a.spines.values():
                s.set_visible(False)
    return axes


# ===================================================================== the comparison page (human notes)
def compare_cards():
    h = open(os.path.join(HT, "VOC_ufficiale", "confronto_umano_pipeline_inst.html"), encoding="utf-8").read()
    out_ = {}
    for c in re.findall(r'<article class="card".*?</article>', h, re.S):
        nm = re.search(r'<span class="nm">([^<]+)</span>', c).group(1)
        note = re.search(r"<h4>Annotazione umana</h4><p>(.*?)</p>", c, re.S)
        gen = c.split('class="gen"')[1] if 'class="gen"' in c else ""
        narr = re.findall(r"<p>(.*?)</p>", gen, re.S)
        facts = []
        for li in re.findall(r'<li class="fv-[a-z]+">.*?</li>', c, re.S):
            v = re.search(r'class="fv-([a-z]+)"', li).group(1)
            gem = re.search(r"\(Gemma: ([a-z ]+)\)", li)
            t = re.sub(r"<[^>]+>", "", re.sub(r'<span class="fs".*?</span>', "", re.sub(r"<i.*?</i>", "", li)))
            facts.append((v, gem.group(1) if gem else v, html.unescape(t).strip()))
        patt = re.search(r"<pre>(.*?)</pre>", c, re.S)
        plaus = re.findall(r'<div class="cs[^"]*"><span class="sc">([\d.]+)', c)
        out_[nm] = {"idx": int(re.search(r'<span class="idx">(\d+)</span>', c).group(1)),
                    "split": re.search(r'data-split="([^"]*)"', c).group(1),
                    "note": html.unescape(note.group(1)).strip() if note else "",
                    "narr": html.unescape(re.sub(r"<[^>]+>", "", narr[-1])).strip() if narr else "",
                    "facts": facts, "patterns": html.unescape(patt.group(1)) if patt else "",
                    "plaus": [float(x) for x in plaus[:2]]}
    return out_


def tex_escape(s):
    return (s.replace("\\", "/").replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("#", "\\#")
            .replace("$", "\\$").replace("→", "$\\rightarrow$").replace("—", "---").replace("“", "``").replace("”", "''")
            .replace("�", "---").replace("·", "$\\cdot$"))


def fig_horse():
    img, gt, pr, _, _ = p80("2007_000332")
    fig = plt.figure(figsize=(6.3, 2.25))
    axes = triptych_axes(fig, 1, left=0.005, top=0.9)
    for a, im, t in zip(axes[0], (img, gt, pr), ("image 2007_000332", "ground truth", "prediction (LR-ASPP MobileNetV3)")):
        a.imshow(im)
        a.set_title(t, fontsize=8, color=DARK)
    fig.savefig(out("Chapter5", "fig_horse18.pdf"), dpi=250)
    plt.close(fig)


def fig_human():
    cards = compare_cards()
    pick = ["2007_000033", "2007_000332", "2011_002223"]
    fig = plt.figure(figsize=(6.3, 6.4))
    axes = triptych_axes(fig, 3, top=0.965, left=0.075)
    for r, nm in enumerate(pick):
        img, gt, pr, _, _ = p80(nm)
        for a, im in zip(axes[r], (img, gt, pr)):
            a.imshow(im)
        axes[r][0].set_ylabel(f"({'abc'[r]}) {nm}", fontsize=8, color=DARK)
    for a, t in zip(axes[0], ("image", "ground truth", "prediction")):
        a.set_title(t, fontsize=8.5, color=DARK)
    fig.savefig(out("Chapter5", "fig_human_examples.pdf"), dpi=220)
    plt.close(fig)
    # the texts, written from the page so that nothing is retyped
    mark = {"stated": "stated", "partial": "partly stated", "absent": "absent", "contradicted": "contradicted",
            "riportato": "stated", "in parte": "partly stated", "assente": "absent", "contraddetto": "contradicted"}
    lines = ["% written by figs/fig_images.py from HTML/VOC_ufficiale/confronto_umano_pipeline_inst.html -- do not edit"]
    for r, nm in enumerate(pick):
        c = cards[nm]
        lines.append("\\item[(%s) %s.] \\emph{Note:} ``%s''\\\\ \\emph{Description:} ``%s''\\\\ \\emph{Facts of the note:} %s."
                     % ("abc"[r], tex_escape(nm), tex_escape(c["note"]), tex_escape(c["narr"]),
                        "; ".join("``%s'' (%s%s)" % (tex_escape(t.rstrip(".")), mark.get(v, v),
                                                     "" if mark.get(g, g) == mark.get(v, v) else ", automatic judge: " + mark.get(g, g))
                                  for v, g, t in c["facts"])))
    os.makedirs(GEN, exist_ok=True)
    open(os.path.join(GEN, "human_examples.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    # the horse card, for the text
    json.dump(cards["2007_000332"], open(os.path.join(HERE, "cache", "card18.json"), "w", encoding="utf-8"), indent=1)


# ===================================================================== best and worst images per dataset
def _ea_cards():
    f = os.path.join(SCR, "ea_images.pkl")
    if os.path.exists(f):
        return pickle.load(open(f, "rb"))
    im = G.images(os.path.join(HT, "Pascal-EA", "gallery_val_ea_full.html"))
    pickle.dump(im, open(f, "wb"))
    return im


def fig_best_worst_voc_ea():
    rows = json.load(open(os.path.join(ROOT, "RUN_UFFICIALI", "pascal_ea", "per_image.json"), encoding="utf-8"))["rows"]
    rows = [r for r in rows if r.get("v3")]
    cards = _ea_cards()
    have = [r for r in rows if r["stem"] in cards and len(cards[r["stem"]]["panels"]) >= 5]
    # VOC val: the original images, by the mIoU of the image
    by = sorted(have, key=lambda r: r["orig"]["miou"])
    picks = [("worst", by[0]), ("worst", by[1]), ("best", by[-1]), ("best", by[-2])]
    fig = plt.figure(figsize=(6.3, 8.0))
    axes = triptych_axes(fig, 4, top=0.97, left=0.08)
    for r, (tag, row) in enumerate(picks):
        p = cards[row["stem"]]["panels"]
        for a, k in zip(axes[r], (0, 3, 1)):            # original, ground truth, prediction on the original
            a.imshow(b2a(p[k][1]))
        axes[r][0].set_ylabel(f"{tag}: {row['stem']}\nimage mIoU {row['orig']['miou']:.2f}", fontsize=7.5, color=DARK)
    for a, t in zip(axes[0], ("image", "ground truth", "prediction")):
        a.set_title(t, fontsize=8.5, color=DARK)
    fig.savefig(out("Chapter4", "fig_bw_voc.pdf"), dpi=220)
    plt.close(fig)
    # val-EA: the edits that cost most (victims) and edits that change nothing
    drop = lambda r: r["orig"]["rec"] - r["v3"]["rec"]
    vic = sorted([r for r in have if drop(r) >= 0.3], key=lambda r: -drop(r))
    calm = sorted([r for r in have if abs(drop(r)) < 0.02 and r["v3"]["rec"] > 0.9], key=lambda r: -r["v3"]["miou"])
    picks = [("victim", vic[0]), ("victim", vic[3]), ("unaffected", calm[0]), ("unaffected", calm[1])]
    fig = plt.figure(figsize=(6.3, 6.5))
    axes = triptych_axes(fig, 4, cols=4, top=0.965, left=0.1)
    for r, (tag, row) in enumerate(picks):
        p = cards[row["stem"]]["panels"]
        for a, k in zip(axes[r], (0, 1, 2, 4)):         # original, its prediction, edited, prediction on the edit
            a.imshow(b2a(p[k][1]))
        axes[r][0].set_ylabel(f"{tag}\n{row['stem']}, {row['voc_class']}\nin {row['material_v3']}\nrecall "
                              f"{row['orig']['rec']:.2f}→{row['v3']['rec']:.2f}", fontsize=6.6, color=DARK)
    for a, t in zip(axes[0], ("original", "prediction", "edited (val-EA)", "prediction")):
        a.set_title(t, fontsize=8.5, color=DARK)
    fig.savefig(out("Chapter4", "fig_bw_ea.pdf"), dpi=220)
    plt.close(fig)


def _sbd_cards():
    f = os.path.join(SCR, "sbd_cards.pkl")
    if os.path.exists(f):
        return pickle.load(open(f, "rb"))
    path = os.path.join(SCR, "sbdgal", "gallery_sbd_share.html")
    h = open(path, encoding="utf-8").read()
    i = h.find("const IMGS=")
    j = h.find("];", i)
    imgs = json.loads(h[i + len("const IMGS="):j + 1])
    cards = {}
    for m in re.finditer(r'<div class="icard"><h3>([0-9_]+)', h):
        end = h.find('<div class="icard">', m.end())
        body = h[m.end():end if end > 0 else len(h)]
        narr = re.search(r'<div class="inarr"><span class="lbl">[^<]*</span>(.*?)</div>', body, re.S)
        pan = re.findall(r'<figure><img src="" data-img="(\d+)"><figcaption>(.*?)</figcaption>', body)
        cards[m.group(1)] = {"narr": html.unescape(re.sub(r"<[^>]+>", "", narr.group(1))) if narr else "",
                             "panels": [(cap, base64.b64decode(imgs[int(k)].split(",", 1)[1])) for k, cap in pan]}
    pickle.dump(cards, open(f, "wb"))
    return cards


def fig_best_worst_sbd():
    cards = _sbd_cards()

    def score(n):
        ious = [float(x) for x in re.findall(r"\(IoU (\d\.\d+)", n)]
        ious += [0.0] * len(re.findall(r"missed entirely", n))
        return (np.mean(ious) if ious else None), len(ious)
    sc = [(k, *score(v["narr"])) for k, v in cards.items() if len(v["panels"]) >= 3]
    sc = [s for s in sc if s[1] is not None and s[2] >= 1]
    sc.sort(key=lambda s: s[1])
    best = [s for s in sc if s[2] >= 2][-2:][::-1]
    picks = [("worst", sc[0]), ("worst", sc[1]), ("best", best[0]), ("best", best[1])]
    fig = plt.figure(figsize=(6.3, 8.0))
    axes = triptych_axes(fig, 4, top=0.97, left=0.08)
    for r, (tag, (k, s, n)) in enumerate(picks):
        p = cards[k]["panels"]
        for a, (cap, b) in zip(axes[r], p[:3]):
            a.imshow(b2a(b))
        axes[r][0].set_ylabel(f"{tag}: {k}\nmean IoU of its\n{n} wrong object{'' if n == 1 else 's'} {s:.2f}", fontsize=7.2, color=DARK)
    for a, t in zip(axes[0], ("image", "ground truth", "prediction")):
        a.set_title(t, fontsize=8.5, color=DARK)
    fig.savefig(out("Chapter4", "fig_bw_sbd.pdf"), dpi=220)
    plt.close(fig)
    VALS["sbd_bw"] = [(t, k, round(s, 2), n) for t, (k, s, n) in picks]


def fig_gesd_examples():
    ex = [("carrot", "Sliced carrot under-segmented\ndue to shape change"),
          ("bananas", "Clustered bananas merged,\nboundaries unclear"), ("paper", "Paper mis-segmented as book"),
          ("bedpet", "Bed over-segmented with pet"), ("watch", "Watch missed when\nsegmenting clock"),
          ("umbrella", "Closed umbrella missed\n(non-typical pose)"),
          ("keyboard", "Laptop keyboard merged\nwith laptop body"), ("logo", "Apple logo mis-segmented\nas apple"),
          ("na", "a correct object (no slice)")]
    d = os.path.join(ROOT, "SlicelenceCode", "docs", "gesd", "deck_examples")
    fig, axes = plt.subplots(3, 3, figsize=(6.3, 5.6))
    for a, (k, lab) in zip(axes.ravel(), ex):
        a.imshow(Image.open(os.path.join(d, f"ex_{k}.png")).convert("RGB"))
        a.set_title(lab, fontsize=7, color=DARK)
        a.axis("off")
    fig.tight_layout(pad=0.4)
    fig.savefig(out("Chapter4", "fig_gesd_examples.pdf"), dpi=220)
    plt.close(fig)


# ===================================================================== certified groups: clusters and situations
def fig_clusters():
    P = os.path.join(ROOT, "presentazione", "ott2026")
    meta = {m["k"]: m for m in json.load(open(os.path.join(P, "fig_cluster_examples.json"), encoding="utf-8"))}
    comp = json.load(open(os.path.join(P, "fig_cluster_composition.json"), encoding="utf-8"))
    TN = {"fn": "missed", "boundary": "boundary", "merge": "merge", "fp": "hallucinated"}
    fig, axes = plt.subplots(4, 1, figsize=(6.3, 6.3))
    for a, k in zip(axes, sorted(meta)):
        m = meta[k]
        a.imshow(Image.open(os.path.join(P, f"fig_cluster_{k}.png")).convert("RGB"))
        c = comp.get(m["slice_id"], [])
        cls = ", ".join(f"{n} {name}" for name, n in c[:3]) + (f", {sum(n for _, n in c[3:])} others" if c[3:] else "")
        a.set_title(f"{m['name'].replace(' · ', ', ')}: {m['size']} errors ({TN.get(m['type'], m['type'])}); {cls}",
                    fontsize=7.2, color=DARK, loc="left")
        a.axis("off")
    fig.tight_layout(pad=0.3, h_pad=0.6)
    fig.savefig(out("Chapter5", "fig_clusters.pdf"), dpi=220)
    plt.close(fig)


def fig_situations():
    sys.modules.setdefault("slice_quality", __import__("slice_quality"))
    res = pickle.load(open(os.path.join(INST, "result_tagged.pkl"), "rb"))
    idx = {e["slice_id"]: e for e in json.load(open(os.path.join(INST, "slices_index.json"), encoding="utf-8"))["slices"]}
    cert = {s.slice_id: s for s, q in res["result"].scored if idx.get(s.slice_id, {}).get("in_descriptions")
            and s.source == "language"}
    cards = G.images(os.path.join(HT, "VOC_ufficiale", "gallery_full_tagged.html"))
    want = ["person_0", "car_0", "bird_extra_0", "horse_0", "sheep_extra_2", "bottle_pend_9", "chair_0", "dog_pend_0"]
    rows, used, used_h = [], set(), set()
    for k in want + sorted(cert, key=lambda k: -idx[k]["size"]):
        if k not in cert or k in [r[0] for r in rows] or k.split("_")[0] in used or idx[k]["hypothesis"] in used_h:
            continue
        ms = [m for m in cert[k].members if m.image_id in cards and m.bbox]
        seen, pick = set(), []
        for m in sorted(ms, key=lambda m: -(m.bbox[2] - m.bbox[0]) * (m.bbox[3] - m.bbox[1])):
            if m.image_id not in seen:
                seen.add(m.image_id)
                pick.append(m)
        if len(pick) >= 3:
            rows.append((k, pick[:4]))
            used.add(k.split("_")[0])
            used_h.add(idx[k]["hypothesis"])
        if len(rows) == 6:
            break
    # the gallery shows in full only the annotated images: a row with fewer than four of them takes the next members by
    # box area, drawn on the image of VOC val copied into figs/cache/voc_jpg
    JPG = os.path.join(HERE, "cache", "voc_jpg")
    for i, (k, pick) in enumerate(rows):
        shown = {m.image_id for m in pick}
        rest = sorted((m for m in cert[k].members if m.bbox and m.image_id not in shown),
                      key=lambda m: -(m.bbox[2] - m.bbox[0]) * (m.bbox[3] - m.bbox[1]))
        for m in rest:
            if len(pick) == 4:
                break
            if m.image_id not in {x.image_id for x in pick}:
                assert os.path.exists(os.path.join(JPG, m.image_id + ".jpg")), m.image_id
                pick.append(m)
        rows[i] = (k, pick)
    COL = {"fn": (224, 123, 30), "fp": (192, 57, 43), "merge": (194, 24, 122), "boundary": (47, 109, 181),
           "split": (123, 79, 160), "extent": (31, 138, 138), "confusion": (138, 109, 31)}
    n = len(rows)
    fig = plt.figure(figsize=(6.3, 1.78 * n))
    gs = fig.add_gridspec(n, 4, left=0.005, right=0.995, top=1 - 0.2 / (1.78 * n), bottom=0.005, hspace=0.16,
                          wspace=0.02)
    for r, (k, ms) in enumerate(rows):
        for c, m in enumerate(ms):
            if m.image_id in cards:
                im = Image.open(io.BytesIO(cards[m.image_id]["panels"][0][1])).convert("RGB")
            else:
                im = crop520(Image.open(os.path.join(JPG, m.image_id + ".jpg")).convert("RGB"))
            VALS.setdefault("situations_types", []).append(m.seg_subtype.value)
            s = im.size[0] / 520.0
            d = ImageDraw.Draw(im)
            x0, y0, x1, y1 = [v * s for v in m.bbox]
            d.rectangle([x0, y0, x1, y1], outline=COL.get(m.seg_subtype.value, (255, 140, 30)), width=5)
            a = fig.add_subplot(gs[r, c])
            a.imshow(im)
            a.axis("off")
            if c == 0:
                h = idx[k]["hypothesis"] or k
                typ = (idx[k].get("dominant_error_type") or "").replace("fn", "missed").replace("fp", "hallucinated")
                a.set_title(f"{k.split('_')[0]}: “{h}”  ({idx[k]['size']} errors, mostly {typ})", fontsize=7.2,
                            color=DARK, loc="left")
    fig.savefig(out("Chapter5", "fig_situations.pdf"), dpi=220)
    plt.close(fig)
    VALS["situations_shown"] = [k for k, _ in rows]


def fig_situation_thumbs():
    """Eight certified situations of eight classes, each with eight members as the gallery shows them."""
    f = os.path.join(SCR, "voc_slices.pkl")
    sl = pickle.load(open(f, "rb")) if os.path.exists(f) else G.slices(os.path.join(HT, "VOC_ufficiale",
                                                                                    "gallery_full_tagged.html"))
    idx = {e["slice_id"]: e for e in json.load(open(os.path.join(INST, "slices_index.json"), encoding="utf-8"))["slices"]}
    cert = sorted([k for k, e in idx.items() if e["in_descriptions"] and not k.startswith("cluster") and k in sl],
                  key=lambda k: -idx[k]["size"])
    pref = ["horse_0", "sheep_extra_2", "bird_extra_0", "cow_0", "dog_pend_0", "motorbike_extra_0", "boat_0", "cat_0"]
    rows, used = [], set()
    for k in [p for p in pref if p in cert] + cert:
        c = k.split("_")[0]
        hc = sum(1 for x in rows if idx[x]["hypothesis"] == idx[k]["hypothesis"])
        if c in used or len(sl[k]["thumbs"]) < 8 or hc >= 2:
            continue
        rows.append(k)
        used.add(c)
        if len(rows) == 8:
            break
    fig = plt.figure(figsize=(6.3, 0.98 * len(rows)))
    gs = fig.add_gridspec(len(rows), 8, left=0.005, right=0.995, top=1 - 0.18 / (0.98 * len(rows)), bottom=0.005,
                          hspace=0.42, wspace=0.05)
    for r, k in enumerate(rows):
        for c in range(8):
            a = fig.add_subplot(gs[r, c])
            a.imshow(letterbox(Image.open(io.BytesIO(sl[k]["thumbs"][c][1])).convert("RGB")))
            a.axis("off")
            if c == 0:
                typ = (idx[k].get("dominant_error_type") or "").replace("fn", "missed").replace("fp", "hallucinated")
                a.set_title(f"{k.split('_')[0]}: “{idx[k]['hypothesis']}” ({idx[k]['size']} errors, mostly {typ})",
                            fontsize=7, color=DARK, loc="left")
    fig.savefig(out("Chapter5", "fig_situation_thumbs.pdf"), dpi=220)
    plt.close(fig)
    VALS["situation_thumbs"] = rows


def letterbox(im, size=128):
    """The picture on a white square, with its proportions kept, so that a thin crop stays thin."""
    w, h = im.size
    s = size / max(w, h)
    w, h = max(1, round(w * s)), max(1, round(h * s))
    box = Image.new("RGB", (size, size), "white")
    box.paste(im.resize((w, h)), ((size - w) // 2, (size - h) // 2))
    return box


def fig_compact_nn():
    """For one member of two significant clusters: the object and its ten nearest neighbours in the DINOv2 space.
    The members are fixed: every neighbour has a thumbnail in the gallery, and the share of neighbours in the cluster
    (5 and 2 of 10) is close to the compactness of the cluster (0.62 and 0.19), so neither row is a flattering outlier."""
    from sklearn.neighbors import NearestNeighbors
    res = pickle.load(open(os.path.join(INST, "result_tagged.pkl"), "rb"))
    errs, ref = res["errors"], res["reference"]
    allin = [i for i in list(errs) + list(ref) if i.embedding is not None]
    E = np.stack([i.embedding for i in allin])
    nb = NearestNeighbors(n_neighbors=11, metric="cosine").fit(E).kneighbors(E)[1]
    pos = {i.instance_id: k for k, i in enumerate(allin)}
    thumbs = pickle.load(open(os.path.join(SCR, "voc_thumbs.pkl"), "rb"))
    sl = {s.slice_id: s for s, q in res["result"].scored}
    names = json.load(open(os.path.join(INST, "cluster_names.json"), encoding="utf-8"))
    rows = []
    for k, q in (("cluster_merge_12_r0", "2007_005813#g2"), ("cluster_fn_4_r0", "2007_000663#g1")):
        nn = [allin[j].instance_id for j in nb[pos[q]][1:]]
        rows.append((k, q, nn, {m.instance_id for m in sl[k].members}))
    n = len(rows)
    fig = plt.figure(figsize=(6.3, 0.95 * n))
    gs = fig.add_gridspec(n, 11, left=0.005, right=0.995, top=1 - 0.2 / (0.95 * n), bottom=0.01, hspace=0.45,
                          wspace=0.08)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(11)] for r in range(n)]
    for r, (k, q, nn, mem) in enumerate(rows):
        for c, iid in enumerate([q] + nn):
            a = axes[r][c]
            a.set_xticks([])
            a.set_yticks([])
            a.imshow(letterbox(Image.open(io.BytesIO(thumbs[iid])).convert("RGB")))
            col = DARK if c == 0 else (GREEN if iid in mem else "#9AA8BA")
            for s in a.spines.values():
                s.set_color(col)
                s.set_linewidth(2.2 if c == 0 else 1.6)
        n_in = sum(1 for x in nn if x in mem)
        axes[r][0].set_title(f"{names.get(k, k).replace(' · ', ', ')}: a member (left) and its 10 nearest "
                             f"neighbours, {n_in} in the cluster", fontsize=7, color=DARK, loc="left")
    fig.savefig(out("Chapter3", "fig_compact_nn.pdf"), dpi=220)
    plt.close(fig)


def fig_branches():
    src = os.path.join(ROOT, "presentazione", "ott2026", "fig_rami_2007_000663.jpg")
    Image.open(src).convert("RGB").resize((700, 700)).save(out("Chapter3", "fig_branches_663.jpg"), quality=90)


TYPING_EX = [("2010_003383", "A cow in pieces: what the containment claims buy"),
             ("2007_000129", "Bicycles behind their riders: what the ground-truth objects buy"),
             ("2008_006784", "A sofa read as background: the failure said first"),
             ("2008_003703", "A well-segmented aeroplane: the silence that cost points")]


def fig_typings():
    """Appendix A: four annotated images, and the description each version of Block 1 and of the text wrote for them,
    with the rubric score of the judge. The texts are read from the narrative files of the runs."""
    sys.path.insert(0, os.path.join(ROOT, "SlicelenceCode", "docs"))
    import _revision_numbers as R
    fig = plt.figure(figsize=(6.3, 8.2))
    axes = triptych_axes(fig, 4, top=0.97, left=0.075, hspace=0.05)
    for r, (nm, _) in enumerate(TYPING_EX):
        img, gt, pr, _, _ = p80(nm)
        for a, im in zip(axes[r], (img, gt, pr)):
            a.imshow(im)
        axes[r][0].set_ylabel(f"({'abcd'[r]}) {nm}", fontsize=8, color=DARK)
    for a, t in zip(axes[0], ("image", "ground truth", "prediction")):
        a.set_title(t, fontsize=8.5, color=DARK)
    d = os.path.join(IMG, "AppendixA")
    os.makedirs(d, exist_ok=True)
    fig.savefig(os.path.join(d, "fig_typing_examples.pdf"), dpi=220)
    plt.close(fig)
    ex = R.load_examples([nm for nm, _ in TYPING_EX])
    cols = (("ref", "Components, IoU claims"), ("cov05b", "+ containment"), ("inst", "+ ground-truth objects"),
            ("inst_v3", "+ revised description"))
    lines = ["% written by figs/fig_images.py from the narrative and judge files of the three runs -- do not edit"]
    for r, (nm, title) in enumerate(TYPING_EX):
        e = ex[nm]
        lines.append("\\paragraph{(%s) %s, image %s.}" % ("abcd"[r], title, tex_escape(nm)))
        lines.append("\\emph{Note written by a person:} ``%s''" % tex_escape(e["human"]))
        lines.append("\\begin{description}")
        for k, lab in cols:
            x = e["runs"][k]
            lines.append("\\item[%s (%s/10):] ``%s''" % (lab, "{:g}".format(x["score"]), tex_escape(x["text"])))
        lines.append("\\end{description}")
    open(os.path.join(GEN, "typing_examples.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")


# ===================================================================== the pipeline, drawn on one image
def fig_pipeline_visual():
    """The pipeline followed by the car of image 2010_004472: its real masks, embedding, VLM readings, slices and description."""
    import math
    import random
    import textwrap
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
    sys.modules.setdefault("slice_quality", __import__("slice_quality"))
    res = pickle.load(open(os.path.join(INST, "result_tagged.pkl"), "rb"))
    errs = res["errors"]
    obj = {e.instance_id: e for e in errs}
    sl = {s.slice_id: s for s, q in res["result"].scored}
    idx = {e["slice_id"]: e for e in json.load(open(os.path.join(INST, "slices_index.json"), encoding="utf-8"))["slices"]}
    f = os.path.join(SCR, "voc_slices.pkl")
    gal = pickle.load(open(f, "rb")) if os.path.exists(f) else G.slices(os.path.join(HT, "VOC_ufficiale",
                                                                                    "gallery_full_tagged.html"))
    Y = np.load(os.path.join(HERE, "cache", "tsne_errors.npy"))
    assert len(Y) == len(errs)
    pos = {e.instance_id: k for k, e in enumerate(errs)}

    SRC = {"Beyond IoU": "#7F8C9B", "SEG4SEG": "#7B4FA0", "MCSD": "#1F8A8A", "CB-SLICE": "#2F6DB5", "SliceLens": AMBER,
           "this thesis": GREEN}
    COL = {"fn": (224, 123, 30), "fp": (192, 57, 43), "merge": (194, 24, 122), "boundary": (47, 109, 181),
           "split": (123, 79, 160), "extent": (31, 138, 138), "confusion": (138, 109, 31)}
    hexc = lambda c: "#%02X%02X%02X" % c
    ID = {c: i for i, c in enumerate(VOC)}
    W, H = 72.0, 36.0
    fig = plt.figure(figsize=(7.2, H / 10))
    ax = fig.add_axes([0, 0, 1, 1])

    def panel(x, y, w, h, title, ec, fc="white", lw=1.3, size=7.2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.45", ec=ec, fc=fc, lw=lw, zorder=1))
        if title:
            ax.text(x + w / 2, y + h - 0.45, title, ha="center", va="top", fontsize=size, fontweight="bold", color=DARK,
                    zorder=5)

    def pic(im, x, y, w, ec=None, lw=0.9):
        if not isinstance(im, Image.Image):
            im = Image.fromarray(im.astype(np.uint8))
        im = im.convert("RGB").resize((int(w * 32), int(w * 32)), Image.LANCZOS)
        ax.imshow(np.array(im), extent=(x, x + w, y, y + w), zorder=3, aspect="auto", interpolation="none")
        if ec:
            ax.add_patch(Rectangle((x, y), w, w, fc="none", ec=ec, lw=lw, zorder=4))

    def txt(x, y, s, size=6.2, ha="center", va="top", color=DARK, **k):
        ax.text(x, y, s, fontsize=size, ha=ha, va=va, color=color, linespacing=1.15, zorder=5, **k)

    def arrow(p, q, color=SLATE, lw=1.1, rad=0.0, ls="-"):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, color=color, lw=lw, ls=ls, zorder=2,
                                     connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))

    def square(arr, box, margin=1.15):
        x0, y0, x1, y1 = [int(v) for v in box]
        s = min(arr.shape[0], int(max(x1 - x0, y1 - y0) * margin))
        l = max(0, min(arr.shape[1] - s, (x0 + x1 - s) // 2))
        t = max(0, min(arr.shape[0] - s, (y0 + y1 - s) // 2))
        return arr[t:t + s, l:l + s]

    def paint(img, mask, c, a=0.62):
        o = img.astype(float).copy()
        o[mask] = (1 - a) * o[mask] + a * np.array(c)
        return o.astype(np.uint8)

    def thumb(k, i):
        return Image.open(io.BytesIO(gal[k]["thumbs"][i][1]))

    # ---- the image and what Block 1 finds on it
    NM = "2010_004472"
    cache = os.path.join(HERE, "cache")
    img = np.array(crop520(Image.open(os.path.join(cache, NM + ".jpg")).convert("RGB")))
    gt = np.array(crop520(Image.open(os.path.join(cache, NM + "_gt.png")), nearest=True))
    pr = np.array(Image.open(os.path.join(cache, NM + "_pred.png")).convert("RGB"))[..., 0]
    assert gt.shape == pr.shape
    gtc, prc = PAL[gt], PAL[pr]
    car = obj[NM + "#g0"]
    cid = ID["car"]
    wrong = np.zeros(gt.shape, dtype=bool)
    for c in (cid, ID["person"]):
        wrong |= (gt == c) != (pr == c)
    typed = paint(img, wrong & (gt != 255), (225, 29, 72), 0.8)
    faded = img.astype(float).copy()
    faded[gt != cid] = 0.25 * faded[gt != cid] + 0.75 * 255
    patt = {p["slice_id"]: p["description"] for p in
            json.load(open(os.path.join(INST, "image_descriptions.json"), encoding="utf-8"))[NM]["patterns"]}
    narr = json.load(open(os.path.join(INST, "error_narratives.json"), encoding="utf-8"))[NM]["text"]
    what = re.search(r"The ground truth CAR[^.]*\.", narr).group(0) + " ..."
    when_cluster = patt["cluster_fn_4_r0"]
    when_lang = "“%s”" % sl["car_4"].hypothesis

    # ---- input
    txt(3.6, 35.6, "Input", 7.2, fontweight="bold")
    for y, im, t in ((26.6, img, "image"), (15.2, gtc, "ground truth"), (3.8, prc, "prediction")):
        pic(im, 0.2, y, 6.8, ec="#9AA8BA")
        txt(3.6, y - 0.2, t, 6.4)
    arrow((7.2, 18.6), (8.7, 18.6))

    # ---- Block 1: the cases of fig_taxonomy, drawn pixel by pixel (dashed: ground truth; filled: predicted pixels)
    def blob(cx, cy, r, wobble=0.0, n=14, seed=0):
        rng = random.Random(seed)
        pts = []
        for i in range(n):
            a = 2 * math.pi * i / n
            rr = r * (1 + wobble * (rng.random() - 0.5) * 2)
            pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a) * 0.85))
        return pts

    GT_EDGE, PRED, PRED2 = "#2F6FB2", (232, 163, 61), (192, 91, 77)
    whole, big = blob(5, 5, 2.8, 0.03, seed=1), blob(5, 5, 3.0, 0.03, seed=1)
    cases = [("missed", [], [whole]),
             ("hallucinated", [blob(5, 5, 2.5, 0.10, seed=7)], []),
             ("merge", [blob(5, 5, 3.2, 0.08, seed=6)], [blob(3.3, 5, 1.5, 0.03, seed=1), blob(6.8, 5, 1.5, 0.03, seed=2)]),
             ("split", [blob(3.0, 5, 1.15, 0.08, seed=4), blob(7.0, 5, 1.15, 0.08, seed=5)], [big]),
             ("boundary", [blob(5, 5, 2.7, 0.30, seed=3)], [whole]),
             ("extent", [[p for p in blob(5, 5, 2.7, 0.03, seed=1) if p[1] < 5.8]], [whole]),
             ("wrong class", [blob(5, 5, 2.7, 0.05, seed=1)], [whole]),
             ("correct", [blob(5, 5, 2.7, 0.03, seed=1)], [whole])]
    N, CELL, TS = 16, 10, 6.0
    panel(8.7, 1.0, 14.3, 33.2, "Block 1: typed errors", SRC["Beyond IoU"], lw=1.5)
    for i, (name, preds, gts) in enumerate(cases):
        cells = Image.new("L", (N, N), 0)
        for poly in preds:
            ImageDraw.Draw(cells).polygon([(px * N / 10, N - py * N / 10) for px, py in poly], fill=255)
        tile = np.full((N * CELL, N * CELL, 3), 240, dtype=np.uint8)
        tile[np.kron(np.array(cells) > 0, np.ones((CELL, CELL), dtype=bool))] = PRED2 if name == "wrong class" else PRED
        tile[::CELL, :] = 255
        tile[:, ::CELL] = 255
        x, y = 9.5 + 6.7 * (i % 2), 25.7 - 7.6 * (i // 2)
        pic(tile, x, y, TS, ec="#CBD3DC", lw=0.6)
        for poly in gts:
            ax.add_patch(Polygon([(x + px * TS / 10, y + py * TS / 10) for px, py in poly], closed=True, fill=False,
                                 ec=GT_EDGE, lw=0.9, ls=(0, (3, 1.5)), zorder=4))
        txt(x + TS / 2, y - 0.1, name, 6.2)

    # ---- cluster branch
    y1, h1 = 20.4, 13.8
    txt(24.4, 35.6, "Cluster branch: how the model fails", 7.6, ha="left", color=SLATE, fontweight="bold")
    panel(24.4, y1, 8.6, h1, "Features", SRC["SEG4SEG"])
    pic(square(faded, car.bbox, 1.0), 24.8, 22.6, 7.8, ec="#9AA8BA")
    panel(34.2, y1, 11.6, h1, "Embedding space", SRC["SEG4SEG"])
    a2 = fig.add_axes([34.7 / W, 20.9 / H, 10.6 / W, 11.2 / H])
    a2.scatter(Y[:, 0], Y[:, 1], s=0.7, color="#CBD3DC", lw=0, rasterized=True)
    kept = [k for k, e in idx.items() if e["in_descriptions"] and k.startswith("cluster")]
    for k in kept:
        p = [pos[m.instance_id] for m in sl[k].members if m.instance_id in pos]
        a2.scatter(Y[p, 0], Y[p, 1], s=1.5, color=GREEN, lw=0, rasterized=True)
    hc = "cluster_fn_4_r0"
    p = [pos[m.instance_id] for m in sl[hc].members if m.instance_id in pos]
    a2.scatter(Y[p, 0], Y[p, 1], s=7, facecolor="none", edgecolor=GREEN, lw=0.5, rasterized=True)
    hx, hy = Y[pos[car.instance_id]]
    a2.scatter([hx], [hy], s=46, marker="*", color=DARK, edgecolor="white", lw=0.4, zorder=6)
    a2.set_xticks([])
    a2.set_yticks([])
    for s in a2.spines.values():
        s.set_color("#CBD3DC")
    panel(47.0, y1, 10.0, h1, "Cluster situation", SRC["this thesis"], size=6.8)
    txt(52.0, 32.55, "manifold compactness", 5.6, color=SLATE)
    txt(52.0, 31.65, "coherence", 5.6, color=SLATE)
    rows = [("✓", GREEN, GREEN, (square(img, car.bbox, 1.0), thumb(hc, 13))),
            ("✗", "#8A94A3", "#8A94A3", (thumb("cluster_extent_0_r0", 3), thumb("cluster_extent_0_r0", 13)))]
    for r, (mark, cm, cf, tiles) in enumerate(rows):
        y = 26.7 - 4.1 * r
        txt(48.1, y + 1.95, mark, 9, color=cm, va="center", fontweight="bold")
        for j, t in enumerate(tiles):
            pic(t, 48.95 + 4.0 * j, y, 3.9, ec=cf, lw=1.1)

    # ---- language branch
    y2, h2 = 1.0, 14.4
    txt(24.4, 17.2, "Language branch: when the model fails", 7.6, ha="left", color=AMBER, fontweight="bold")
    panel(24.4, y2, 8.6, h2, "Same image", SRC["SliceLens"])
    boxed = Image.fromarray(img.copy())
    ImageDraw.Draw(boxed).rectangle([int(v) for v in car.bbox], outline=(255, 140, 30), width=8)
    pic(boxed, 24.8, 3.5, 7.8, ec="#9AA8BA")
    panel(34.2, y2, 12.0, h2, "VLM + hypotheses", SRC["SliceLens"])
    hyps = ["car_4", "car_0", "car_extra_2"]
    band_y = []
    y = 13.45
    for k in hyps:
        c = sl[k].membership[car.instance_id]
        s = textwrap.fill(sl[k].hypothesis, 21)
        nl = s.count("\n") + 1
        txt(34.65, y, "“%s”" % s, 6.4, ha="left")
        yb = y - 1.02 * nl - 0.55
        ax.add_patch(Rectangle((34.65, yb), 8.2, 0.5, fc="#E5E9EF", ec="none", zorder=3))
        ax.add_patch(Rectangle((34.65, yb), 8.2 * c, 0.5, fc=AMBER, ec="none", zorder=4))
        txt(43.1, yb + 0.25, f"{c:.2f}", 6.2, ha="left", va="center")
        band_y.append(yb + 0.25)
        y = yb - 0.95
    panel(46.5, y2, 2.9, h2, None, SRC["this thesis"])
    txt(47.95, 15.0, "rank\ntest", 6.0, fontweight="bold")
    for k, y in zip(hyps, band_y):
        ok = idx[k]["in_descriptions"]
        txt(47.95, y, "✓" if ok else "✗", 9, va="center", color=GREEN if ok else "#8A94A3", fontweight="bold")
    panel(49.9, y2, 7.1, h2, "Language\nsituation", SRC["this thesis"], size=6.8)
    for i, t in enumerate((11, 0, 12, 18, 19, 20)):
        pic(thumb("car_4", t), 50.2 + 3.3 * (i % 2), 8.5 - 3.3 * (i // 2), 3.2, ec=GREEN)
    arrow((49.4, band_y[0]), (49.9, band_y[0]), color=GREEN)

    # ---- description
    panel(58.2, y2, 13.6, 33.2, "Output", SRC["this thesis"], fc="#E4F1E9", lw=1.7)
    pic(typed, 58.7, 19.6, 12.6, ec="#9AA8BA")
    txt(58.7, 18.9, "Description", 7.0, ha="left", fontweight="bold")
    w1 = textwrap.fill(what, 24)
    txt(58.7, 17.6, w1, 6.8, ha="left")
    yw = 17.6 - 1.09 * (w1.count("\n") + 1) - 1.0
    txt(58.7, yw, "Cluster description", 7.0, ha="left", color=SLATE, fontweight="bold")
    w2 = textwrap.fill(when_cluster, 24)
    txt(58.7, yw - 1.3, w2, 6.8, ha="left", color=SLATE)
    yl = yw - 1.3 - 1.09 * (w2.count("\n") + 1) - 1.0
    txt(58.7, yl, "Language description", 7.0, ha="left", color=AMBER, fontweight="bold")
    txt(58.7, yl - 1.3, textwrap.fill(when_lang, 24), 6.8, ha="left", color=AMBER)

    # ---- the flow
    arrow((23.0, 27.3), (24.4, 27.3))
    arrow((45.8, 27.3), (47.0, 27.3))
    arrow((57.0, 27.3), (58.2, 27.3))
    arrow((23.0, 10.0), (24.4, 10.0))
    arrow((33.0, 10.0), (34.2, 10.0))
    arrow((57.0, 5.4), (58.2, 5.4))
    arrow((52.0, y1), (40.0, y1), color=MUTED, rad=-0.09, ls="--")
    txt(46.0, 19.5, "peeling", 6.0, color=MUTED, style="italic")
    arrow((23.0, 18.0), (58.2, 18.0), color=MUTED, ls="--")
    txt(28.7, 19.35, "typed errors", 6.0, color=MUTED, style="italic")
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    # the object projected on its point of the embedding space
    star = fig.transFigure.inverted().transform(a2.transData.transform((hx, hy)))
    fig.add_artist(FancyArrowPatch((32.6 / W, 27.3 / H), tuple(star), arrowstyle="-|>", mutation_scale=7,
                                   color=DARK, lw=0.9, shrinkA=0, shrinkB=4, transform=fig.transFigure, zorder=8))
    fig.savefig(out("Chapter3", "fig_pipeline_visual.pdf"), dpi=300)
    plt.close(fig)




# ===================================================================== looser cluster gates (Section 6.9)
RELAX_ROWS = (("A", "cluster_merge_14_r3", "Occluded sheep, merged", ("occluded", "sheep")),
              ("A", "cluster_merge_0_r2", "Occluded cows, merged", ("occluded", "cow")),
              ("A", "cluster_merge_17_r1", "Occluded chairs, merged", ("occluded", "chair")),
              ("B", "cluster_fn_3_r0", "Small bottles, missed", ("small", "bottle")))


def _relax_run(v):
    d = os.path.join(ROOT, "RUN_UFFICIALI", "slice_quality_inst_relax" + v)
    res = pickle.load(open(os.path.join(d, "result_tagged.pkl"), "rb"))
    return res, {s.slice_id: s for s, q in res["result"].scored}


def fig_relaxed_slices():
    """Members of four clusters that only the looser gates keep, drawn at random among the members whose crop has a
    gallery thumbnail and is at least 40 pixels on its short side; titles from the report of the variants."""
    rep = json.load(open(os.path.join(ROOT, "RUN_UFFICIALI", "relaxed_cluster_report.json"), encoding="utf-8"))
    thumbs = pickle.load(open(os.path.join(SCR, "voc_thumbs.pkl"), "rb"))
    runs = {v: _relax_run(v) for v in ("A", "B")}
    rng = np.random.default_rng(0)
    n, k = len(RELAX_ROWS), 7
    fig = plt.figure(figsize=(6.3, 1.0 * n))
    gs = fig.add_gridspec(n, k, left=0.005, right=0.995, top=1 - 0.2 / n, bottom=0.01, hspace=0.5, wspace=0.08)
    for r, (v, sid, title, words) in enumerate(RELAX_ROWS):
        row = next(x for x in rep[v]["rows"] if x["slice_id"] == sid)
        assert all(w in row["description"] for w in words), (sid, row["description"])
        mem = [m for m in runs[v][1][sid].members if m.instance_id in thumbs
               and min(m.bbox[2] - m.bbox[0], m.bbox[3] - m.bbox[1]) >= 40]
        pick = [mem[i] for i in sorted(rng.choice(len(mem), size=min(k, len(mem)), replace=False))]
        for c, m in enumerate(pick):
            a = fig.add_subplot(gs[r, c])
            a.imshow(letterbox(Image.open(io.BytesIO(thumbs[m.instance_id])).convert("RGB")))
            a.set_xticks([])
            a.set_yticks([])
            for s_ in a.spines.values():
                s_.set_color(SLATE)
                s_.set_linewidth(1.2)
            if c == 0:
                a.set_title(f"{title} (variant {v}): {row['size']} errors on {row['n_images']} images; class IoU "
                            f"{100 * row['cls_iou_in']:.0f} on its images, {100 * row['cls_iou_out']:.0f} on the others",
                            fontsize=7, color=DARK, loc="left")
    fig.savefig(out("Chapter5", "fig_relaxed_slices.pdf"), dpi=220)
    plt.close(fig)


def fig_relaxed_example():
    """Image 2009_003059: two calves merged into one mask; the slice of variant A and the significant situation that
    hold the same error. Image and masks copied from the server into figs/cache."""
    NM = "2009_003059"
    cache = os.path.join(HERE, "cache")
    img = np.array(crop520(Image.open(os.path.join(cache, NM + ".jpg")).convert("RGB")))
    gt = PAL[np.array(crop520(Image.open(os.path.join(cache, NM + "_gt.png")), nearest=True))]
    pr = PAL[np.array(Image.open(os.path.join(cache, NM + "_pred.png")).convert("RGB"))[..., 0]]
    res, sc = _relax_run("A")
    assert {m.instance_id for m in sc["cluster_merge_0_r2"].members if m.image_id == NM} == \
        {NM + "#p0", NM + "#g0", NM + "#g1"}
    assert NM + "#p0" in {m.instance_id for m in sc["cow_2"].members}
    obj = {i.instance_id: i for i in res["errors"] if i.image_id == NM}
    fig = plt.figure(figsize=(6.3, 2.75))
    gs = fig.add_gridspec(1, 3, left=0.005, right=0.995, top=0.9, bottom=0.2, wspace=0.03)
    for c, (a_img, t) in enumerate(((img, "image"), (gt, "ground truth: two cows"), (pr, "prediction: one mask"))):
        a = fig.add_subplot(gs[0, c])
        a.imshow(a_img)
        a.set_xticks([])
        a.set_yticks([])
        a.set_title(t, fontsize=8, color=DARK)
        if c < 2:
            for g in ("#g0", "#g1"):
                x0, y0, x1, y1 = obj[NM + g].bbox
                a.add_patch(plt.Rectangle((x0 + 3, y0 + 3), x1 - x0 - 6, y1 - y0 - 6, fill=False, ec=SLATE, lw=1.6))
    fig.text(0.005, 0.11, "cluster branch, variant A: occluded cows, merged with a neighbour  \u2192  how the model fails",
             fontsize=8, color=SLATE, weight="bold")
    fig.text(0.005, 0.03, "language branch: \u201can object occluded by another object\u201d, significant for the cows  "
             "\u2192  when it fails", fontsize=8, color=AMBER, weight="bold")
    fig.savefig(out("Chapter5", "fig_relaxed_example.pdf"), dpi=220)
    plt.close(fig)


VALS = {}
ALL = {"horse": fig_horse, "human": fig_human, "bw_voc_ea": fig_best_worst_voc_ea, "bw_sbd": fig_best_worst_sbd,
       "gesd": fig_gesd_examples, "clusters": fig_clusters, "situations": fig_situations,
       "compact_nn": fig_compact_nn, "branches": fig_branches, "sit_thumbs": fig_situation_thumbs,
       "typings": fig_typings, "pipeline_visual": fig_pipeline_visual,
       "relaxed_slices": fig_relaxed_slices, "relaxed_example": fig_relaxed_example}

if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "cache"), exist_ok=True)
    for n in sys.argv[1:] or list(ALL):
        ALL[n]()
        print("done", n)
    vf = os.path.join(HERE, "cache", "fig_images_values.json")
    old = json.load(open(vf, encoding="utf-8")) if os.path.exists(vf) else {}
    old.update(VALS)
    json.dump(old, open(vf, "w", encoding="utf-8"), indent=1)
