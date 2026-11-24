# -*- coding: utf-8 -*-
"""Figures for the 'widening the hypothesis vocabulary' sections (EN + IT).

Every number is read through _revision_numbers.load_extra_hyp(); the example
panel draws the crops recorded in docs/extra_hyp/visual_check.json. The
full-population bars appear only once outputs/slice_quality_inst_extra_full
exists, so the script can run before and after the re-grounding.

Run: python docs/make_extra_hyp_figures.py   ->  docs/fig_extra_*_{en,it}.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from _revision_numbers import load_extra_hyp

DOCS = os.path.dirname(os.path.abspath(__file__))
INK, MUTED = "#1f2937", "#6b7280"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#d1d5db",
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False})
VERDICT_COL = {"holds": "#16a34a", "mislabelled": "#d97706", "partial": "#d97706",
               "vague": "#d97706", "artefact": "#dc2626"}
SHORT = {
    "a picture, poster or screen showing an object": "picture, poster, screen",
    "a reflection of an object in a mirror, window or water": "reflection",
    "a toy, statue or model of an object": "toy, statue, model",
    "an object with holes or see-through parts": "holes / see-through parts",
    "an object behind a fence, net, bars or branches": "behind a fence, net, bars",
    "an object seen from an unusual viewpoint or in an unusual pose": "unusual viewpoint or pose",
    "a very large object seen up close that fills most of the image": "large close-up",
    "an object in an unusual setting for its kind": "unusual setting",
}
T = {
    "en": {"canon": "7 canonical\n(generic)", "spec": "class-specific\n(Gemma, ~15/class)",
           "samp": "8 extra generic\n(120-instance sample)", "full": "8 extra generic\n(after full re-grounding)",
           "rate": "certified / evaluated (class × hypothesis)",
           "yield_t": "How often a hypothesis becomes a certified situation",
           "sys": "slope-systematic on the sample", "cert": "certified on the sample",
           "held": "still systematic on ALL instances", "ncls": "classes (of 20)",
           "per_t": "The 8 extra hypotheses, class by class",
           "types_t": "Error type that dominates each certified situation",
           "before": "before (inst run)", "after_s": "with the extra (sample)",
           "after_f": "with the extra (after re-grounding)", "nsit": "certified situations",
           "types": {"fn": "missed", "extent": "extent", "confusion": "wrong\nclass", "boundary": "boundary",
                     "fp": "hallucinated", "merge": "merge", "split": "split"},
           "verdict": {"holds": "holds", "mislabelled": "real, wrong label", "partial": "partial",
                       "vague": "vague", "artefact": "artefact"}},
    "it": {"canon": "7 canoniche\n(generiche)", "spec": "specifiche per classe\n(Gemma, ~15/classe)",
           "samp": "8 generiche nuove\n(campione di 120)", "full": "8 generiche nuove\n(dopo la riverifica)",
           "rate": "certificate / valutate (classe × ipotesi)",
           "yield_t": "Quanto spesso un'ipotesi diventa una situazione certificata",
           "sys": "sistematica sul campione", "cert": "certificata sul campione",
           "held": "ancora sistematica su TUTTE le istanze", "ncls": "classi (su 20)",
           "per_t": "Le 8 ipotesi nuove, classe per classe",
           "types_t": "Tipo d'errore che domina ogni situazione certificata",
           "before": "prima (run inst)", "after_s": "con le nuove (campione)",
           "after_f": "con le nuove (dopo la riverifica)", "nsit": "situazioni certificate",
           "types": {"fn": "mancato", "extent": "estensione", "confusion": "classe\nsbagliata",
                     "boundary": "bordo", "fp": "allucinato", "merge": "merge", "split": "split"},
           "verdict": {"holds": "regge", "mislabelled": "vera, frase sbagliata", "partial": "parziale",
                       "vague": "vaga", "artefact": "artefatto"}},
}


def save(fig, name):
    fig.savefig(os.path.join(DOCS, name), dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  ", name)


def fig_yield(X, L, lang):
    bars = [(L["canon"], X["canon"]["certified"], X["canon"]["evaluated"], "#2563eb"),
            (L["spec"], X["specific"]["certified"], X["specific"]["evaluated"], "#9ca3af"),
            (L["samp"], X["sample_certified"], X["pairs"], "#f59e0b")]
    if "full" in X:
        bars.append((L["full"], X["full_certified"], X["pairs"], "#b45309"))
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    for i, (lab, c, n, col) in enumerate(bars):
        ax.bar(i, 100 * c / n, color=col, width=0.6)
        ax.text(i, 100 * c / n + 0.8, f"{100 * c / n:.1f}%\n({c}/{n})", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(range(len(bars)), [b[0] for b in bars], fontsize=8)
    ax.set_ylabel(L["rate"], fontsize=8)
    ax.set_ylim(0, max(100 * c / n for _l, c, n, _c in bars) * 1.3)
    ax.set_title(L["yield_t"], loc="left", fontsize=10, color=INK)
    save(fig, f"fig_extra_yield_{lang}.png")


def fig_per_hyp(X, L, lang):
    hyps = X["extra"]
    per = X["sample"]["per_hypothesis"]
    sys_ = [len(per[h]["systematic_classes"]) for h in hyps]
    cert = [len(per[h]["certified"]) for h in hyps]
    held = [len(X["full_held"].get(h, [])) for h in hyps] if "full" in X else None
    order = sorted(range(len(hyps)), key=lambda i: (cert[i], sys_[i]))
    y = range(len(hyps))
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    k = 3 if held is not None else 2
    hgt = 0.8 / k
    ax.barh([j + hgt for j in y], [sys_[i] for i in order], height=hgt, color="#fcd34d", label=L["sys"])
    ax.barh(list(y), [cert[i] for i in order], height=hgt, color="#f59e0b", label=L["cert"])
    if held is not None:
        ax.barh([j - hgt for j in y], [held[i] for i in order], height=hgt, color="#b45309", label=L["held"])
    for j, i in enumerate(order):
        cl = ", ".join(c["class"] for c in per[hyps[i]]["certified"])
        if cl:
            ax.text(max(sys_[i], cert[i]) + 0.15, j, cl, va="center", fontsize=7.5, color=MUTED)
    ax.set_yticks(list(y), [SHORT[hyps[i]] for i in order], fontsize=8)
    ax.set_xlabel(L["ncls"])
    ax.set_xlim(0, max(sys_) + 3.5)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    ax.set_title(L["per_t"], loc="left", fontsize=10, color=INK)
    save(fig, f"fig_extra_per_hyp_{lang}.png")


def fig_types(X, L, lang):
    keys = ["fn", "extent", "confusion", "boundary", "fp", "merge", "split"]
    series = [(L["before"], X["inst_types"], "#9ca3af"),
              (L["after_s"], X["sample"]["totals"]["types_after"], "#f59e0b")]
    if "full" in X:
        series.append((L["after_f"], X["full"]["totals"]["types_after"], "#b45309"))
    w = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(7.4, 3.1))
    for k, (lab, d, col) in enumerate(series):
        xs = [i + (k - (len(series) - 1) / 2) * w for i in range(len(keys))]
        v = [d.get(t, 0) for t in keys]
        ax.bar(xs, v, width=w, color=col, label=lab)
        for x, val in zip(xs, v):
            ax.text(x, val + 0.3, str(val), ha="center", fontsize=7, color=INK)
    ax.set_xticks(range(len(keys)), [L["types"][t] for t in keys])
    ax.set_ylabel(L["nsit"])
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.set_title(L["types_t"], loc="left", fontsize=10, color=INK)
    save(fig, f"fig_extra_types_{lang}.png")


def fig_examples(X, L, lang):
    vis = X.get("visual")
    if not vis:
        return
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
        im = Image.open(os.path.join(DOCS, "extra_hyp", e["name"] + ".jpg")).convert("RGB")
        ImageDraw.Draw(im).rectangle(e["bbox"], outline=col, width=5)
        ax.imshow(im)
        ax.set_title(f"{s['class']} — {SHORT[s['hypothesis']]}\n{L['verdict'][s['verdict']]}",
                     fontsize=8, color=col, loc="left")
    fig.tight_layout()
    save(fig, f"fig_extra_examples_{lang}.png")


if __name__ == "__main__":
    X = load_extra_hyp()
    for lang, L in T.items():
        fig_yield(X, L, lang)
        fig_per_hyp(X, L, lang)
        fig_types(X, L, lang)
        fig_examples(X, L, lang)
