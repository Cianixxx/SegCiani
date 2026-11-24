"""Scientific-paper-style pipeline figure (PNG), EN + IT.

Improvements over v1:
* the arrows leaving Block 1 (Beyond IoU) are labelled "typed error instances" to
  make explicit that BOTH branches share the same typed-instance substrate;
* each branch carries a label of WHAT it describes (cluster = the ERROR,
  language = the SITUATION), mirroring the thesis idea;
* box text stacks title-then-body downward, so no title/body overlap;
* bilingual: build("en") and build("it").
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

C = {"input": "#e8eaed", "beyond": "#f6c65b", "mcsd": "#5b9bd5",
     "seg4seg": "#7fb069", "cbslice": "#b085d1", "slicelens": "#68b3a3",
     "new": "#ed9b40", "out": "#e07a5f"}
EDGE = "#2b2f36"

LAB = {
    "en": {
        "title": "Error-slice discovery & per-image failure-description pipeline",
        "input": ("Input", ["seg. model + VOC val", "(img, GT, pred)"]),
        "block1": ("1. Typed error", ["6-case matching:", "boundary / extent / seg",
                                       "FP · FN · split · merge"]),
        "emb": ("2. Embeddings", ["DINOv2", "mask-pooled"]),
        "clu": ("3–4. Cluster", ["feat = emb ⊕ E", "γ-weighted GMM"]),
        "sl1": ("SliceLens LLM", ["situational", "hypotheses / class"]),
        "sl2": ("Qwen grounding", ["P(yes | q, r)", "slice confidence"]),
        "sqv": ("5–6. Slice-Quality Vector", ["ω² · manifold ·", "mispred-coherence ·",
                                              "err/slice purity · signif."]),
        "refine": ("7. Refine + peeling", []),
        "expl": ("8. Explanation", ["Qwen attribute tags", "→ per-image text:",
                                    "“situation” → error"]),
        "out": ("Output", ["per-image failure", "descriptions +", "HTML gallery + JSON"]),
        "top_branch": "Cluster branch — describes the ERROR (how it fails)",
        "bot_branch": "Language branch (SliceLens) — describes the SITUATION (when it fails)",
        "edge_lbl": "typed error instances",
        "valid": "Def.1 (cluster: type-aware severity + attr. purity + signif.)   •   language: slope-trend + severity + confidence",
        "legend_new": "new (this work)",
    },
    "it": {
        "title": "Pipeline: scoperta delle error-slice e descrizione dei fallimenti per immagine",
        "input": ("Input", ["modello seg. + VOC val", "(img, GT, pred)"]),
        "block1": ("1. Errore tipato", ["matching a 6 casi:", "boundary / extent / seg",
                                        "FP · FN · split · merge"]),
        "emb": ("2. Embedding", ["DINOv2", "mask-pooled"]),
        "clu": ("3–4. Cluster", ["feat = emb ⊕ E", "GMM pesata (γ)"]),
        "sl1": ("SliceLens LLM", ["ipotesi", "situazionali / classe"]),
        "sl2": ("Qwen grounding", ["P(sì | q, r)", "slice confidence"]),
        "sqv": ("5–6. Slice-Quality Vector", ["ω² · manifold ·", "coerenza-errore ·",
                                              "purezza · signif."]),
        "refine": ("7. Refine + peeling", []),
        "expl": ("8. Spiegazione", ["tag attributi (Qwen)", "→ testo per immagine:",
                                    "“situazione” → errore"]),
        "out": ("Output", ["descrizioni di", "fallimento per immagine", "gallery HTML + JSON"]),
        "top_branch": "Ramo cluster — descrive l’ERRORE (come fallisce)",
        "bot_branch": "Ramo linguistico (SliceLens) — descrive la SITUAZIONE (quando fallisce)",
        "edge_lbl": "istanze d’errore tipate",
        "valid": "Def.1 (cluster: severità per-tipo + purezza attributi + signif.)   •   language: slope-trend + severità + confidence",
        "legend_new": "nuovo (questo lavoro)",
    },
}


def build(lang):
    L = LAB[lang]
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    def box(x, y, w, h, spec, color, fs=10.5, bfs=8.4, tag="", tcolor="#111"):
        title, body = spec
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.02,rounding_size=0.10",
                     linewidth=1.4, edgecolor=EDGE, facecolor=color, zorder=2))
        ty = y + h - 0.30
        ax.text(x + w / 2, ty, title, ha="center", va="center", fontsize=fs,
                fontweight="bold", color=tcolor, zorder=3)
        for i, line in enumerate(body):
            ax.text(x + w / 2, ty - 0.40 - i * 0.28, line, ha="center", va="center",
                    fontsize=bfs, color="#22252b", zorder=3)
        if tag:
            ax.text(x + w - 0.07, y + h - 0.03, tag, ha="right", va="top",
                    fontsize=7.0, style="italic", color="#3a3f47", zorder=3)
        return (x, y, w, h)

    def arrow(a, b, rad=0.0, color=EDGE):
        ax.add_patch(FancyArrowPatch((a[0] + a[2], a[1] + a[3] / 2),
                     (b[0], b[1] + b[3] / 2), connectionstyle=f"arc3,rad={rad}",
                     arrowstyle="-|>", mutation_scale=14, linewidth=1.5, color=color, zorder=1))

    def apts(p0, p1, rad=0.0, color=EDGE, lw=1.5):
        ax.add_patch(FancyArrowPatch(p0, p1, connectionstyle=f"arc3,rad={rad}",
                     arrowstyle="-|>", mutation_scale=14, linewidth=lw, color=color, zorder=1))

    ax.text(8, 8.68, L["title"], ha="center", fontsize=15, fontweight="bold")

    # ---- input + block 1 (shared typed-instance substrate) ----
    b_in = box(0.25, 3.90, 2.05, 1.45, L["input"], C["input"], fs=11)
    b1 = box(2.75, 3.70, 2.35, 1.85, L["block1"], C["beyond"], tag="Beyond IoU")
    arrow(b_in, b1)

    # ---- cluster branch (top) ----
    ax.text(8.35, 8.05, L["top_branch"], ha="center", fontsize=9.6,
            fontweight="bold", color="#4a6a2f")
    b2 = box(5.70, 6.15, 2.05, 1.45, L["emb"], C["mcsd"], tag="MCSD")
    b34 = box(8.10, 6.15, 2.35, 1.45, L["clu"], C["seg4seg"], tag="SEG4SEG")
    arrow(b2, b34)

    # ---- language branch (bottom) ----
    ax.text(8.45, 0.98, L["bot_branch"], ha="center", fontsize=9.6,
            fontweight="bold", color="#2f6a5f")
    bL1 = box(5.70, 1.45, 2.05, 1.45, L["sl1"], C["slicelens"], tag="SliceLens")
    bL2 = box(8.10, 1.45, 2.35, 1.45, L["sl2"], C["slicelens"], tag="SliceLens")
    arrow(bL1, bL2)

    # ---- BoU feeds BOTH branches with the SAME typed instances ----
    apts((5.15, 4.95), (5.70, 6.55), rad=-0.22)
    apts((5.15, 4.30), (5.70, 2.35), rad=0.22)
    ax.text(5.42, 4.62, L["edge_lbl"], ha="center", va="center", fontsize=8.0,
            style="italic", color="#333", rotation=90,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))

    # ---- SQV (merge) ----
    b56 = box(10.95, 3.35, 2.55, 2.20, L["sqv"], C["cbslice"], fs=9.8, bfs=8.2,
              tag="MCSD/CB-SLICE/SEG4SEG")
    apts((10.45, 6.55), (10.95, 5.15), rad=0.20)     # cluster -> SQV
    apts((10.45, 2.35), (10.95, 3.75), rad=-0.20)    # language -> SQV
    ax.text(12.22, 3.20, L["valid"], ha="center", va="top", fontsize=7.5,
            style="italic", color="#333")

    # ---- refine + peeling loop ----
    box(10.95, 6.05, 2.55, 0.78, L["refine"], C["new"], fs=9.5, tag="new/MCSD")
    apts((12.22, 5.55), (12.22, 6.05))
    # feedback: peeling re-clusters the residual -> lands ON the 3-4 box edge
    apts((10.95, 6.55), (10.45, 6.85), rad=0.35, color="#8a8a8a", lw=1.2)

    # ---- explanation + output ----
    b8 = box(13.75, 3.35, 2.10, 2.20, L["expl"], C["new"], fs=9.8, bfs=8.2,
             tag="CB-SLICE / new")
    arrow(b56, b8)
    box(13.75, 1.15, 2.10, 1.55, L["out"], C["out"], fs=10.5, bfs=8.4)
    apts((14.80, 3.35), (14.80, 2.70))

    # ---- legend ----
    leg = [("Beyond IoU", C["beyond"]), ("MCSD", C["mcsd"]), ("SEG4SEG", C["seg4seg"]),
           ("CB-SLICE", C["cbslice"]), ("SliceLens", C["slicelens"]),
           (L["legend_new"], C["new"])]
    lx = 0.30
    for name, col in leg:
        ax.add_patch(FancyBboxPatch((lx, 0.20), 0.28, 0.28, boxstyle="round,pad=0.01",
                     facecolor=col, edgecolor=EDGE, linewidth=1))
        ax.text(lx + 0.36, 0.34, name, va="center", fontsize=8.5)
        lx += 1.9 + 0.052 * len(name)

    here = os.path.dirname(os.path.abspath(__file__))
    fname = "pipeline_figure.png" if lang == "en" else f"pipeline_figure_{lang}.png"
    out = os.path.join(here, fname)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    build("en")
    build("it")
