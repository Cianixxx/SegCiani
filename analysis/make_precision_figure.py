"""Bar figure for the validation slide: description precision vs matched chance.

Reads outputs/slice_quality/eval_precision.json so the picture can never drift
from the measurement. Writes docs/precision_figure[_it].png.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DOCS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(DOCS)
sys.path.insert(0, os.path.join(ROOT, "docs"))
from _revision_numbers import out_path as _out_path  # noqa: E402  (RUN_UFFICIALI, RUN_VECCHI or outputs/)
SRC = _out_path("slice_quality", "eval_precision.json")

INK, MUTED = "#1f2933", "#9aa5b1"
POS, NULL = "#2f6fb2", "#c9d4de"

LABELS = {
    "en": {"rows": [("language", "SliceLens situations\n“an object cut off at the border”"),
                    ("cluster_situational", "cluster + attribute\n“a small bicycle”"),
                    ("cluster_class_only", "cluster, class only\n“a bottle”")],
           "pos": "verified on its own instances",
           "null": "same class, same error type, outside the slice (chance)",
           "ttl": "Is the description true of the object it is attached to?",
           "sub": "{n} VLM verifications · {p} instances · {g} phrases"},
    "it": {"rows": [("language", "situazioni SliceLens\n«an object cut off at the border»"),
                    ("cluster_situational", "cluster + attributo\n«a small bicycle»"),
                    ("cluster_class_only", "cluster, solo classe\n«a bottle»")],
           "pos": "verificata sulle sue istanze",
           "null": "stessa classe, stesso tipo d'errore, fuori dalla slice (caso)",
           "ttl": "La descrizione è vera per l'oggetto a cui è attaccata?",
           "sub": "{n} verifiche VLM · {p} istanze · {g} frasi"},
}


def build(lang):
    data = json.load(open(SRC, encoding="utf-8"))["summary"]
    L = LABELS[lang]
    keys = [k for k, _ in L["rows"]]
    names = [n for _, n in L["rows"]]
    prec = [data[k]["precision"] for k in keys]
    ctrl = [data[k]["control_rate_same_class"] for k in keys]

    fig, ax = plt.subplots(figsize=(9.2, 3.9), dpi=200)
    y = range(len(keys))
    h = 0.34
    ax.barh([v + h / 2 for v in y], prec, height=h, color=POS, label=L["pos"], zorder=3)
    ax.barh([v - h / 2 for v in y], ctrl, height=h, color=NULL, label=L["null"], zorder=3)
    for i, (p, c) in enumerate(zip(prec, ctrl)):
        ax.text(p + .012, i + h / 2, f"{p:.2f}", va="center", fontsize=10,
                color=POS, fontweight="bold")
        ax.text(c + .012, i - h / 2, f"{c:.2f}", va="center", fontsize=10, color="#66727f")
        ax.annotate(f"{p - c:+.2f}", xy=(max(p, c) + .07, i), va="center",
                    fontsize=11, color=INK, fontweight="bold")

    ax.set_yticks(list(y))
    ax.set_yticklabels(names, fontsize=10.5)
    ax.invert_yaxis()                       # strongest family on top
    ax.set_xlim(0, 1.12)
    ax.set_xticks([0, .2, .4, .6, .8])
    ax.xaxis.grid(True, color="#e7ecf1", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors="#66727f", length=0)
    ax.set_title(L["ttl"], fontsize=13, color=INK, loc="left", pad=26,
                 fontweight="bold")
    sub = L["sub"].format(n=data["n_evaluations"], p=data["overall"]["n_pos"],
                          g=data["overall"]["groups"])
    ax.text(0, 1.045, sub, transform=ax.transAxes, fontsize=9.5, color="#66727f")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2,
              frameon=False, fontsize=9.5)
    fig.tight_layout()
    out = os.path.join(DOCS, f"precision_figure{'_it' if lang == 'it' else ''}.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print("saved", out)


if __name__ == "__main__":
    build("en")
    build("it")
