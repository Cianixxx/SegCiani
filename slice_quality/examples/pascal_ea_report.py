"""Report of pascal_ea_compare.py: does the material edit make the segmenter fail?

Reads <OUT>/per_image.json and writes <OUT>/report.{txt,json} and two figures:
the paired change of the subject's recall (edited minus original) and its relation
with how strongly the edit changed the object's pixels.
Run locally: python slice_quality/examples/pascal_ea_report.py
"""
import json
import os
from collections import Counter, defaultdict

import numpy as np

OUT = os.environ.get("OUT", "outputs/pascal_ea")
DROP = 0.30          # a "failure caused by the edit": the subject's recall falls by 0.30 or more
d = json.load(open(f"{OUT}/per_image.json"))
rows = d["rows"]
lines = []
say = lambda s="": (lines.append(s), print(s))
rep = {"agreement_with_thesis_preds": float(np.mean(d["agreement_with_thesis_preds"]))}


def paired(v):
    return [r for r in rows if r.get(v) and "rec" in r["orig"] and "rec" in r[v]]


def stats(rs, v):
    o = np.array([r["orig"]["rec"] for r in rs])
    e = np.array([r[v]["rec"] for r in rs])
    dl = e - o
    out = dict(n=len(rs), rec_orig=float(o.mean()), rec_edit=float(e.mean()), delta=float(dl.mean()),
               median_delta=float(np.median(dl)), drop_010=float((dl <= -0.10).mean()),
               drop_030=float((dl <= -DROP).mean()), gain_010=float((dl >= 0.10).mean()),
               n_drop_030=int((dl <= -DROP).sum()))
    try:
        from scipy import stats as sst
        out["wilcoxon_p"] = float(sst.wilcoxon(e, o).pvalue)
        di = [r[v].get("diff_in") for r in rs]
        ok = [(a, b) for a, b in zip(di, dl) if a is not None]
        if len(ok) > 5:
            rho, p = sst.spearmanr([a for a, _ in ok], [b for _, b in ok])
            out["spearman_diffin_delta"] = [float(rho), float(p)]
    except ImportError:
        pass
    for k in ("iou_c", "miou", "acc_out"):
        pr = [(r["orig"].get(k), r[v].get(k)) for r in rs]
        pr = [(a, b) for a, b in pr if a is not None and b is not None]
        if pr:
            out[f"{k}_orig"] = float(np.mean([a for a, _ in pr]))
            out[f"{k}_edit"] = float(np.mean([b for _, b in pr]))
    for k in ("diff_in", "diff_out"):
        x = [r[v].get(k) for r in rs if r[v].get(k) is not None]
        out[k] = float(np.mean(x)) if x else None
    return out


say("=" * 78)
say("PASCAL-EA (materiale) -- LR-ASPP MobileNetV3, originale vs modificata, stessa immagine")
say("=" * 78)
say(f"predizioni sugli originali identiche a quelle della tesi: {100 * rep['agreement_with_thesis_preds']:.2f}% "
    f"dei pixel ({len(d['agreement_with_thesis_preds'])} immagini)")
for v in ("v3", "v1"):
    rs = paired(v)
    st = stats(rs, v)
    rep[v] = st
    say()
    say(f"-- {v}: {st['n']} immagini con originale e modificata " + "-" * 40)
    say(f"   oggetto modificato riconosciuto (recall della sua classe): originale {st['rec_orig']:.3f} -> "
        f"modificata {st['rec_edit']:.3f}  (media {st['delta']:+.3f}, mediana {st['median_delta']:+.3f}"
        + (f", Wilcoxon p={st['wilcoxon_p']:.1e})" if "wilcoxon_p" in st else ")"))
    say(f"   immagini in cui l'oggetto perde >=0.10: {100 * st['drop_010']:.0f}%  | >=0.30: "
        f"{100 * st['drop_030']:.0f}% ({st['n_drop_030']})  | guadagna >=0.10: {100 * st['gain_010']:.0f}%")
    say(f"   IoU della classe dell'oggetto: {st.get('iou_c_orig', float('nan')):.3f} -> {st.get('iou_c_edit', float('nan')):.3f}"
        f"   | mIoU dell'immagine: {st.get('miou_orig', float('nan')):.3f} -> {st.get('miou_edit', float('nan')):.3f}")
    say(f"   CONTROLLO fuori dall'oggetto: accuratezza {st.get('acc_out_orig', float('nan')):.3f} -> "
        f"{st.get('acc_out_edit', float('nan')):.3f}; variazione dei pixel dentro {st['diff_in']:.1f} vs fuori "
        f"{st['diff_out']:.1f} (su 255)")
    if "spearman_diffin_delta" in st:
        say(f"   piu' la modifica cambia l'oggetto, piu' il recall scende? Spearman "
            f"{st['spearman_diffin_delta'][0]:+.2f} (p={st['spearman_diffin_delta'][1]:.1e})")
    # what the subject becomes when it is lost
    lost = [r for r in rs if r[v]["rec"] - r["orig"]["rec"] <= -DROP]
    w = Counter(r[v]["wrong_top"][0] for r in lost if r[v].get("wrong_top"))
    rep[v]["lost_becomes"] = w.most_common()
    say(f"   quando l'oggetto si perde (calo >= {DROP}), diventa soprattutto: "
        + ", ".join(f"{k} {n}" for k, n in w.most_common(6)))
    # by material and by class
    for key, lab in ((f"material_{v}", "materiale"), ("voc_class", "classe")):
        g = defaultdict(list)
        for r in rs:
            g[r.get(key)].append(r)
        tab = []
        for k, rr in g.items():
            o = np.mean([r["orig"]["rec"] for r in rr])
            e = np.mean([r[v]["rec"] for r in rr])
            dr = np.mean([(r[v]["rec"] - r["orig"]["rec"]) <= -DROP for r in rr])
            tab.append((k, len(rr), o, e, e - o, dr))
        tab.sort(key=lambda t: t[4])
        rep[v][f"by_{'material' if lab == 'materiale' else 'class'}"] = [
            dict(key=k, n=n, rec_orig=o, rec_edit=e, delta=dd, drop_030=dr) for k, n, o, e, dd, dr in tab]
        say(f"   per {lab}:")
        for k, n, o, e, dd, dr in tab:
            if n >= 5:
                say(f"      {str(k):12s} n={n:3d}  {o:.3f} -> {e:.3f}  ({dd:+.3f})  calo>=0.30 in {100 * dr:3.0f}%")
    rep[v]["worst"] = [dict(stem=r["stem"], cls=r["voc_class"], material=r.get(f"material_{v}"),
                            rec_orig=r["orig"]["rec"], rec_edit=r[v]["rec"], becomes=r[v].get("wrong_top"))
                       for r in sorted(rs, key=lambda r: r[v]["rec"] - r["orig"]["rec"])[:24]]

json.dump(rep, open(f"{OUT}/report.json", "w"), indent=1)
open(f"{OUT}/report.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for v, col in (("v3", "#c0392b"), ("v1", "#2c7fb8")):
        rs = paired(v)
        dl = [r[v]["rec"] - r["orig"]["rec"] for r in rs]
        ax[0].hist(dl, bins=np.linspace(-1, 0.6, 33), alpha=0.55, color=col, label=f"{v} (n={len(rs)})")
        x = [r[v].get("diff_in") for r in rs]
        ax[1].scatter([a for a in x if a is not None], [b for a, b in zip(x, dl) if a is not None], s=8,
                      alpha=0.45, color=col, label=v)
    ax[0].axvline(0, color="k", lw=0.8)
    ax[0].axvline(-DROP, color="k", lw=0.8, ls="--")
    ax[0].set_xlabel("recall dell'oggetto: modificata - originale")
    ax[0].set_ylabel("immagini")
    ax[0].legend()
    ax[0].set_title("Quanto la modifica fa perdere l'oggetto")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xlabel("variazione media dei pixel dell'oggetto (0-255)")
    ax[1].set_ylabel("variazione del recall")
    ax[1].set_title("Modifica piu' forte, errore piu' grande?")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_delta.png", dpi=130)
    print(f"figura: {OUT}/fig_delta.png")
except ImportError:
    pass
