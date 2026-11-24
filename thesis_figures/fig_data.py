# -*- coding: utf-8 -*-
"""Figures of the thesis drawn from the stored runs (CPU only). Every number comes from the run files, read with the
loaders of the technical report (SlicelenceCode/docs/_revision_numbers.py) or from result_tagged.pkl of the official run.
  python fig_data.py [name ...]   ->  Images/Chapter*/fig_*.pdf, and figs/cache/fig_data_values.json with the values
                                      the text quotes (compactness, BIC choices, slope of the car example)
"""
import json
import os
import pickle
import sys
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
CODE = os.path.join(ROOT, "SlicelenceCode")
sys.path.insert(0, CODE)
sys.path.insert(0, os.path.join(CODE, "docs"))
import _revision_numbers as R  # noqa: E402

IMG = os.path.join(HERE, "..", "Images")
CACHE = os.path.join(HERE, "cache")
os.makedirs(CACHE, exist_ok=True)
INST = os.path.join(ROOT, "RUN_UFFICIALI", "slice_quality_inst")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "pdf.fonttype": 42, "axes.titlesize": 9,
                     "axes.labelsize": 8.5})
SLATE, AMBER, GREEN, DARK, MUTED, GRID = "#44546A", "#C2410C", "#1E7B4A", "#1E293B", "#5B6675", "#E5E9EF"
TYPE_COL = {"fn": "#E07B1E", "fp": "#C0392B", "merge": "#C2187A", "boundary": "#2F6DB5", "split": "#7B4FA0",
            "extent": "#1F8A8A", "confusion": "#8A6D1F"}
TYPE_NAME = {"fn": "missed", "fp": "hallucinated", "merge": "merge", "boundary": "boundary", "split": "split",
             "extent": "extent", "confusion": "wrong class"}
VALUES = {}


def out(chapter, name):
    d = os.path.join(IMG, chapter)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


def clean(ax, grid_y=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#C9D0DA")
    ax.tick_params(colors=MUTED)
    if grid_y:
        ax.yaxis.grid(color=GRID, lw=0.7)
        ax.set_axisbelow(True)


_RES = None


def result():
    global _RES
    if _RES is None:
        _RES = pickle.load(open(os.path.join(INST, "result_tagged.pkl"), "rb"))
    return _RES


def certified():
    idx = {e["slice_id"]: e for e in json.load(open(os.path.join(INST, "slices_index.json"), encoding="utf-8"))["slices"]}
    return idx, {s.slice_id: s for s, q in result()["result"].scored if idx.get(s.slice_id, {}).get("in_descriptions")}


def tsne():
    f = os.path.join(CACHE, "tsne_errors.npy")
    errs = result()["errors"]
    if os.path.exists(f):
        return np.load(f)
    from sklearn.manifold import TSNE
    X = np.stack([e.embedding for e in errs])
    Y = TSNE(n_components=2, metric="cosine", init="pca", perplexity=40, random_state=0).fit_transform(X)
    np.save(f, Y)
    return Y


# ===================================================================== embeddings in two dimensions
def fig_embedding():
    errs = result()["errors"]
    Y = tsne()
    idx, cert = certified()
    names = json.load(open(os.path.join(INST, "cluster_names.json"), encoding="utf-8"))
    pos = {e.instance_id: k for k, e in enumerate(errs)}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.1))
    ax = axes[0]
    for t in ("fp", "merge", "boundary", "split", "fn", "extent", "confusion"):
        m = np.array([e.seg_subtype.value == t for e in errs])
        ax.scatter(Y[m, 0], Y[m, 1], s=2.2, color=TYPE_COL[t], alpha=0.7, lw=0, label=f"{TYPE_NAME[t]} ({m.sum()})")
    ax.set_title("(a) the 4,967 wrong objects of VOC val,\nby error type", loc="left", fontsize=8.5, color=DARK)
    ax.legend(fontsize=6.5, frameon=False, markerscale=3, loc="upper left", ncol=2, handletextpad=0.2,
              columnspacing=0.6, bbox_to_anchor=(0.0, 0.0))
    ax = axes[1]
    ax.scatter(Y[:, 0], Y[:, 1], s=1.5, color="#D5DBE3", lw=0)
    cols = ["#E07B1E", "#7B4FA0", "#2F6DB5", "#C2187A"]
    clus = sorted([k for k in cert if k.startswith("cluster")], key=lambda k: -len(cert[k].members))
    for c, k in zip(cols, clus):
        p = [pos[m.instance_id] for m in cert[k].members if m.instance_id in pos]
        ax.scatter(Y[p, 0], Y[p, 1], s=5, color=c, lw=0, label=f"{names.get(k, k)} ({len(p)})")
    lang = cert.get("person_0")
    if lang is not None:
        p = [pos[m.instance_id] for m in lang.members if m.instance_id in pos]
        ax.scatter(Y[p, 0], Y[p, 1], s=7, facecolor="none", edgecolor=DARK, lw=0.4,
                   label=f"situation: person, “cut off at the image border” ({len(p)})")
    ax.set_title("(b) the significant clusters (filled)\nand one significant situation (rings)", loc="left", fontsize=8.5,
                 color=DARK)
    ax.legend(fontsize=6.3, frameon=False, markerscale=2, loc="upper left", bbox_to_anchor=(0.0, 0.0))
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#C9D0DA")
    fig.tight_layout()
    fig.savefig(out("Chapter3", "fig_embedding_tsne.pdf"), dpi=300)
    plt.close(fig)


# ===================================================================== BIC per error type
def fig_bic():
    from sklearn.mixture import GaussianMixture
    from slice_quality.config import ClusterConfig, FeatureConfig
    from slice_quality.features import build_features
    errs = result()["errors"]
    cfg = ClusterConfig()
    X, kept = build_features(errs, FeatureConfig())
    X = np.concatenate([X, np.array([[cfg.gamma * (1 - e.iou)] for e in kept])], axis=1)
    strata = defaultdict(list)
    for k, e in enumerate(kept):
        strata[e.seg_subtype.value].append(k)
    fig, ax = plt.subplots(figsize=(5.2, 2.9))
    chosen = {}
    for t in ("fp", "merge", "boundary", "split", "fn", "extent", "confusion"):
        ix = strata.get(t, [])
        if len(ix) < cfg.min_stratum_fit:
            chosen[t] = {"n": len(ix), "k": 1, "whole": True}
            continue
        kmax = max(1, len(ix) // cfg.min_points_per_component)
        ks = [k for k in cfg.k_grid if k <= kmax] or [1]
        bics = [GaussianMixture(n_components=k, covariance_type="diag", random_state=0).fit(X[ix]).bic(X[ix])
                for k in ks]
        b = np.array(bics)
        kbest = ks[int(np.argmin(b))]
        chosen[t] = {"n": len(ix), "k": kbest, "ks": ks, "bic": bics}
        ax.plot(ks, (b - b.min()) / 1000, "-o", ms=3, lw=1.4, color=TYPE_COL[t],
                label=f"{TYPE_NAME[t]} (n = {len(ix)}, k = {kbest})")
        ax.scatter([kbest], [0], s=40, facecolor="none", edgecolor=TYPE_COL[t], lw=1.4, zorder=4)
    ax.set_xlabel("number of Gaussian components k")
    ax.set_ylabel("BIC − min BIC  (thousands)")
    ax.set_xticks(cfg.k_grid)
    clean(ax)
    ax.legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    fig.savefig(out("Chapter3", "fig_bic.pdf"))
    plt.close(fig)
    VALUES["bic"] = {t: {k: v for k, v in d.items() if k in ("n", "k", "whole")} for t, d in chosen.items()}


# ===================================================================== manifold compactness
def fig_compactness():
    from sklearn.neighbors import NearestNeighbors
    res = result()
    errs, ref = res["errors"], res["reference"]
    allin = [i for i in list(errs) + list(ref) if i.embedding is not None]
    E = np.stack([i.embedding for i in allin])
    nn = NearestNeighbors(n_neighbors=11, metric="cosine").fit(E)
    _, nb = nn.kneighbors(E)
    knn = [set(r[1:]) for r in nb]
    pos = {i.instance_id: k for k, i in enumerate(allin)}

    def mc(ix):
        s = set(ix)
        return float(np.mean([len(knn[i] & s) / len(knn[i]) for i in ix]))
    idx, cert = certified()
    names = json.load(open(os.path.join(INST, "cluster_names.json"), encoding="utf-8"))
    rng = np.random.default_rng(0)
    pools = defaultdict(list)
    cls_pool = defaultdict(list)
    for e in errs:
        pools[e.seg_subtype.value].append(pos[e.instance_id])
        cls_pool[e.gt_class or e.pred_class].append(pos[e.instance_id])
    vals = {}
    for k, s in cert.items():
        ix = [pos[m.instance_id] for m in s.members if m.instance_id in pos]
        if k.startswith("cluster"):
            dom = Counter(m.seg_subtype.value for m in s.members).most_common(1)[0][0]
            pool = pools[dom]
        else:
            c = Counter((m.gt_class or m.pred_class) for m in s.members).most_common(1)[0][0]
            pool = cls_pool[c]
        if len(pool) < len(ix) + 5:
            continue
        null = [mc(list(rng.choice(pool, size=len(ix), replace=False))) for _ in range(200 if not
                                                                                      k.startswith("cluster") else 1000)]
        obs = mc(ix)
        vals[k] = {"n": len(ix), "obs": obs, "null_mean": float(np.mean(null)),
                   "p": float((1 + sum(v >= obs for v in null)) / (len(null) + 1)), "null": null}
    VALUES["compactness"] = {k: {kk: vv for kk, vv in v.items() if kk != "null"} for k, v in vals.items()}
    lang = [k for k in vals if not k.startswith("cluster")]
    VALUES["compactness_lang_summary"] = {
        "n": len(lang), "above_chance": sum(1 for k in lang if vals[k]["p"] <= 0.05),
        "median_ratio": float(np.median([vals[k]["obs"] / max(vals[k]["null_mean"], 1e-9) for k in lang]))}
    # figure: (a) the kNN neighbourhood in 2-D, (b) the null of a cluster, (c) the null of a situation
    Y = tsne()
    epos = {e.instance_id: k for k, e in enumerate(errs)}
    kc = "cluster_fn_4_r0" if "cluster_fn_4_r0" in vals else sorted(k for k in vals if k.startswith("cluster"))[0]
    ks = "person_0" if "person_0" in vals else lang[0]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), gridspec_kw={"width_ratios": [1.2, 1, 1]})
    ax = axes[0]
    fn = [epos[e.instance_id] for e in errs if e.seg_subtype.value == "fn"]
    ax.scatter(Y[fn, 0], Y[fn, 1], s=2, color="#D5DBE3", lw=0)
    mem = [m for m in cert[kc].members if m.instance_id in epos]
    mset = {pos[m.instance_id] for m in mem}
    ax.scatter([Y[epos[m.instance_id], 0] for m in mem], [Y[epos[m.instance_id], 1] for m in mem], s=4,
               color=TYPE_COL["fn"], lw=0)
    inv = {pos[e.instance_id]: epos[e.instance_id] for e in errs}
    shown = [mem[i] for i in rng.choice(len(mem), size=6, replace=False)]
    for m in shown:
        a = pos[m.instance_id]
        for b in knn[a]:
            if b in inv:
                pa, pb = Y[epos[m.instance_id]], Y[inv[b]]
                ax.plot([pa[0], pb[0]], [pa[1], pb[1]], lw=0.9, color=GREEN if b in mset else "#7F8C9B", alpha=0.9)
    pts = np.array([Y[epos[m.instance_id]] for m in mem])
    lo, hi = pts.min(0), pts.max(0)
    pad = 0.15 * (hi - lo)
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#C9D0DA")
    ax.set_title("(a) 10 nearest neighbours of six\nmembers: inside (green), outside", loc="left", fontsize=7.2,
                 color=DARK)
    for ax, k, lab in ((axes[1], kc, "(b) cluster: small objects,\nmissed"),
                       (axes[2], ks, "(c) situation: person “cut off\nat the image border”")):
        v = vals[k]
        ax.hist(v["null"], bins=25, color="#C9D0DA")
        ax.axvline(v["obs"], color=GREEN if k.startswith("cluster") else AMBER, lw=1.8)
        ax.set_title(lab, loc="left", fontsize=7.2, color=DARK)
        ax.set_xlabel("manifold compactness")
        ax.text(0.03, 0.95, f"observed {v['obs']:.2f}\nrandom {v['null_mean']:.2f}\np = {v['p']:.3f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=6.8, color=DARK)
        clean(ax)
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out("Chapter3", "fig_compactness.pdf"), dpi=300)
    plt.close(fig)


# ===================================================================== the car example: AUC, ROC, slope rule
def _slope_rule(conf, err, num_bins=10, min_bin=10, num_thr=10, q_max=0.8, min_win=20, min_bins=3, min_spread=0.05):
    pts = sorted(zip(conf, err))
    cmin, cmax = min(conf), max(conf)
    width = (cmax - cmin) / num_bins
    sums, cnt, er = [0.0] * num_bins, [0] * num_bins, [0.0] * num_bins
    for c, e in pts:
        i = min(int((c - cmin) / width), num_bins - 1)
        sums[i] += c
        cnt[i] += 1
        er[i] += e
    bins = [(sums[i] / cnt[i], er[i] / cnt[i], cnt[i]) for i in range(num_bins) if cnt[i] >= min_bin]
    cs = sorted(conf)

    def q(p):
        x = p * (len(cs) - 1)
        lo = int(x)
        hi = min(lo + 1, len(cs) - 1)
        return cs[lo] * (1 - (x - lo)) + cs[hi] * (x - lo)
    wins = []
    for i in range(num_thr):
        t = q(q_max * i / (num_thr - 1))
        sel = [b for b in bins if b[0] >= t]
        if sum(b[2] for b in sel) < min_win or len(sel) < min_bins:
            continue
        xs, ys = np.array([b[0] for b in sel]), np.array([b[1] for b in sel])
        if xs.max() - xs.min() < min_spread:
            continue
        slope = float(np.polyfit(xs, ys, 1)[0])
        wins.append((t, slope, xs, ys))
    return bins, wins


def fig_rank_car():
    d = json.load(open(os.path.join(ROOT, "presentazione", "ott2026", "fig_auc_car.json"), encoding="utf-8"))
    err, cor = np.array(d["conf_error"]), np.array(d["conf_correct"])
    auc = (err[:, None] > cor[None, :]).mean() + 0.5 * (err[:, None] == cor[None, :]).mean()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45), gridspec_kw={"width_ratios": [1.25, 0.9, 1.1]})
    ax = axes[0]
    rng = np.random.default_rng(0)
    for y, v, col, lab in ((1.0, err, AMBER, f"wrong cars ({len(err)})"), (0.0, cor, SLATE, f"correct cars ({len(cor)})")):
        ax.scatter(v, y + rng.uniform(-0.28, 0.28, len(v)), s=5, color=col, alpha=0.55, lw=0)
        ax.plot([np.median(v)] * 2, [y - 0.36, y + 0.36], color=col, lw=2)
        ax.text(0.5, y + 0.42, lab, ha="center", color=col, fontsize=7.2, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 1.6)
    ax.set_yticks([])
    ax.set_xlabel("VLM confidence c = P(yes)")
    ax.set_title("(a) “cut off at the image border”", loc="left", color=DARK)
    clean(ax, grid_y=False)
    ax = axes[1]
    th = np.r_[1.01, np.sort(np.unique(np.r_[err, cor]))[::-1], -0.01]
    tpr = [(err >= t).mean() for t in th]
    fpr = [(cor >= t).mean() for t in th]
    ax.fill_between(fpr, tpr, step=None, color=AMBER, alpha=0.15)
    ax.plot(fpr, tpr, color=AMBER, lw=1.6)
    ax.plot([0, 1], [0, 1], ls="--", color=MUTED, lw=0.8)
    ax.text(0.95, 0.06, f"AUC = {auc:.2f}", ha="right", fontsize=8, color=AMBER, fontweight="bold")
    ax.set_xlabel("share of correct cars above c")
    ax.set_ylabel("share of wrong cars above c")
    ax.set_title("(b) the rank test: ROC", loc="left", color=DARK)
    ax.set_aspect("equal")
    clean(ax, grid_y=False)
    ax = axes[2]
    conf = np.r_[err, cor]
    lab = np.r_[np.ones(len(err)), np.zeros(len(cor))]
    bins, wins = _slope_rule(list(conf), list(lab))
    ax.plot([b[0] for b in bins], [b[1] for b in bins], "o", color=SLATE, ms=4)
    best = max(wins, key=lambda w: w[1]) if wins else None
    for t, slope, xs, ys in wins:
        b0 = np.polyfit(xs, ys, 1)
        ax.plot(xs, np.polyval(b0, xs), color="#C9D0DA" if (best and slope != best[1]) else AMBER, lw=1)
    if best:
        b0 = np.polyfit(best[2], best[3], 1)
        ax.plot(best[2], np.polyval(b0, best[2]), color=AMBER, lw=2)
        ax.text(0.97, 0.06, f"max slope {best[1]:.2f}\n(rule: ≥ 0.20)", transform=ax.transAxes, fontsize=7,
                color=AMBER, ha="right")
    ax.set_xlabel("mean c of the bin")
    ax.set_ylabel("error rate in the bin")
    ax.set_title("(c) the slope rule of SliceLens", loc="left", color=DARK)
    ax.set_ylim(0.5, 1.05)
    clean(ax)
    fig.tight_layout()
    fig.savefig(out("Chapter3", "fig_rank_car.pdf"))
    plt.close(fig)
    VALUES["car"] = {"auc": float(auc), "n_err": len(err), "n_cor": len(cor), "median_err": float(np.median(err)),
                     "median_cor": float(np.median(cor)), "q": d["q"], "slope_stored": d.get("slope"),
                     "slope_recomputed": best[1] if best else None, "windows": len(wins)}


# ===================================================================== the simulation
def fig_sim():
    SIM = R.out_path("slope_vs_auc_sim.json")
    RULES = R.out_path("slope_rules_sim.json")
    T = json.load(open(SIM, encoding="utf-8"))["table"]
    rc = {(r["b"], r["n"]): r for r in json.load(open(RULES, encoding="utf-8"))["table"]}
    T = [dict(r, slope=rc[(r["b"], r["n"])]["v1"], auc=rc[(r["b"], r["n"])]["AUC"]) if (r["b"], r["n"]) in rc else r
         for r in T]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    for ax, (b, title) in zip(axes, ((0.0, "(a) no association: false alarms"), (0.25, "(b) a weak, real association"))):
        rows = sorted((r for r in T if r["b"] == b), key=lambda r: r["n"])
        xs = [r["n"] for r in rows]
        for key, col, lab in (("slope", AMBER, "slope rule (SliceLens)"), ("auc", GREEN, "rank test")):
            ys = [100 * r[key] for r in rows]
            other = "auc" if key == "slope" else "slope"
            ax.plot(xs, ys, "-o", color=col, lw=2, ms=5, label=lab)
            for x, y, r in zip(xs, ys, rows):
                below = abs(y - 100 * r[other]) < 12 and (y < 100 * r[other] or (y == 100 * r[other] and key == "auc"))
                ax.annotate(f"{y:.0f}%", (x, y), textcoords="offset points", xytext=(0, -11 if below else 6), ha="center",
                            color=col, fontsize=7)
        if b == 0.0:
            ax.axhline(5, color=MUTED, lw=0.8, ls="--")
            ax.text(xs[-1], 9, "5% expected", ha="right", color=MUTED, fontsize=7)
        else:
            title += f" (AUC {np.mean([r['auc_mean'] for r in rows]):.2f})"
        ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{x:,}" for x in xs])
        ax.minorticks_off()
        ax.set_ylim(-5, 110)
        ax.set_title(title, loc="left", color=DARK)
        ax.set_xlabel("objects in the class")
        clean(ax)
    axes[0].set_ylabel("verdict “systematic” (%)")
    axes[1].legend(frameon=False, fontsize=7.5, loc="center right")
    fig.tight_layout()
    fig.savefig(out("Chapter5", "fig_sim.pdf"))
    plt.close(fig)


# ===================================================================== facts score, plausibility
def fig_judges():
    j = json.load(open(os.path.join(INST, "judge_claims.json"), encoding="utf-8"))["judgements"]
    g = [x["score"] for x in j if x.get("kind") == "narrative" and x["pairing"] == "genuine" and x["rubric"] == "facts"]
    c = [x["score"] for x in j if x.get("kind") == "narrative" and x["pairing"] == "control" and x["rubric"] == "facts"]
    cz = R.load_cause(val_only=True)["all"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axes[0]
    bins = np.linspace(0, 10, 11)
    ax.hist([c, g], bins=bins, color=["#C9D0DA", GREEN], label=[f"note vs description of another image (mean {np.mean(c):.2f})",
                                                               f"note vs its own description (mean {np.mean(g):.2f})"],
            rwidth=0.85)
    ax.set_xlabel("facts score (0–10), one value per annotated image")
    ax.set_ylabel("images")
    ax.legend(frameon=False, fontsize=6.6, loc="upper center")
    ax.set_title(f"(a) facts score, {len(g)} images", loc="left", color=DARK)
    clean(ax)
    ax = axes[1]
    bb = cz["by_branch"]["pipeline"]
    labs = ["language\nbranch", "cluster\nbranch", "both", "control"]
    vals = [bb["language"][0], bb["cluster"][0], bb["both"][0], cz["pipeline"]["control"]]
    ns = [bb["language"][1], bb["cluster"][1], bb["both"][1], cz["pipeline"]["n_control"]]
    ax.bar(labs, vals, color=[AMBER, SLATE, GREEN, "#C9D0DA"], width=0.6)
    for i, (v, n) in enumerate(zip(vals, ns)):
        ax.text(i, v + 0.15, f"{v:.1f}", ha="center", fontsize=7.5, color=DARK)
        ax.text(i, 0.25, f"n={n}", ha="center", fontsize=6, color="white" if i < 3 else DARK)
    ax.set_ylim(0, 10)
    ax.set_ylabel("plausibility (0–10)")
    ax.set_title("(b) plausibility of the blamed situation", loc="left", color=DARK)
    clean(ax)
    fig.tight_layout()
    fig.savefig(out("Chapter5", "fig_judges.pdf"))
    plt.close(fig)
    from scipy.stats import wilcoxon
    gi = {x["image"]: x["score"] for x in j if x.get("kind") == "narrative" and x["pairing"] == "genuine"
          and x["rubric"] == "facts"}
    ci_ = {x["image"]: x["score"] for x in j if x.get("kind") == "narrative" and x["pairing"] == "control"
           and x["rubric"] == "facts"}
    both = sorted(set(gi) & set(ci_))
    w = wilcoxon([gi[k] for k in both], [ci_[k] for k in both], alternative="greater")
    VALUES["facts"] = {"n": len(g), "genuine_ge8": sum(1 for x in g if x >= 8), "control_le1": sum(1 for x in c if x <= 1),
                       "genuine_zero": sum(1 for x in g if x == 0), "paired_n": len(both), "p_wilcoxon": float(w.pvalue),
                       "wins": sum(1 for k in both if gi[k] > ci_[k]), "ties": sum(1 for k in both if gi[k] == ci_[k])}


# ===================================================================== SliceLens against the pipeline
def fig_cmp():
    c = R.load_slicelens_cmp()
    cov, ea = c["stats"]["coverage"], c["stats"]["ea"]
    K = ("SliceLens", "+ rank test", "pipeline")
    COL = ("#C9D0DA", AMBER, GREEN)
    panels = [("errors in a significant slice,\nVOC val (%)", [100 * cov["VOC val"]["rate"][k] for k in K], 50),
              ("errors in a significant slice,\nSBD (%)", [100 * cov["SBD"]["rate"][k] for k in K], 50),
              ("Pascal-EA victims found (of 144)", [ea["found"][k] for k in K], 144),
              ("facts score vs human notes", [c["facts"]["sl"], (c.get("facts_auc") or {}).get("sl", np.nan),
                                              c["facts"]["ours"]], 10)]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.2))
    for ax, (t, v, top) in zip(axes, panels):
        ax.bar(range(3), v, color=COL, width=0.65)
        for i, x in enumerate(v):
            if not np.isnan(x):
                ax.text(i, x + top * 0.02, f"{x:.1f}" if top != 144 else f"{x:.0f}", ha="center", fontsize=7)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["SliceLens", "+ rank\ntest", "SegCiani"], fontsize=6.8)
        ax.set_ylim(0, top * 1.12)
        ax.set_title(t, fontsize=7.3, color=DARK, loc="left")
        clean(ax)
    fig.tight_layout()
    fig.savefig(out("Chapter5", "fig_cmp.pdf"))
    plt.close(fig)


def fig_gesd():
    g = json.load(open(os.path.join(CODE, "docs", "gesd", "gesd_cmp.json"), encoding="utf-8"))
    GS = R.load_gesd()
    cols, ca, why = g["cols"], g["cols_all"], g["why"]
    rows = [("SliceLens (slope rule)", cols["SliceLens, verdict only"]["p10"], ca["SliceLens, verdict only"]["p10"], AMBER),
            (f"SliceLens cut at random to {why['slicelens_cut_to_rank_count']['n']} slices",
             why["slicelens_cut_to_rank_count"]["p10_mean"], ca["slicelens_cut_to_rank_count"]["p10_mean"], AMBER),
            ("SliceLens, members shuffled", why["SliceLens, verdict only, members shuffled"]["p10_mean"], None, AMBER),
            ("SliceLens + rank test", cols["+ rank test, verdict only"]["p10"], ca["+ rank test, verdict only"]["p10"],
             SLATE),
            ("SegCiani", cols["pipeline"]["p10"], ca["pipeline"]["p10"], GREEN),
            ("class × error type, no phrase, no VLM", GS["baselines"]["baselines"]["C"]["p_at_k"], None, MUTED),
            ("GH-ESD, published", GS["ghesd_p10"], None, MUTED)]
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    y = np.arange(len(rows))[::-1]
    for yy, (lab, v, vk, col) in zip(y, rows):
        ax.barh(yy + 0.18, v, height=0.34, color=col)
        ax.text(v + 0.01, yy + 0.18, f"{v:.3f}", va="center", fontsize=6.8)
        if vk is not None:
            ax.barh(yy - 0.18, vk, height=0.34, color=col, alpha=0.4)
            ax.text(vk + 0.01, yy - 0.18, f"{vk:.3f}", va="center", fontsize=6.5, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Precision@10 on the 21 expert slices")
    clean(ax, grid_y=False)
    ax.xaxis.grid(color=GRID, lw=0.7)
    fig.tight_layout()
    fig.savefig(out("Chapter5", "fig_gesd_p10.pdf"))
    plt.close(fig)


def fig_gesd_gap():
    """Precision@10 of SliceLens and of the pipeline on each expert slice: on top the slices where both have one,
    below those where the pipeline has none, with the reason (gesd_cmp.json "missing")."""
    g = json.load(open(os.path.join(CODE, "docs", "gesd", "gesd_cmp.json"), encoding="utf-8"))
    SL, PL = g["cols"]["SliceLens, verdict only"]["per_slice"], g["cols"]["pipeline"]["per_slice"]
    why = {}
    for v in g["missing"]["classes"].values():
        for s in v["expert_slices"]:
            if v["p10_pipeline"].get(s, 1) == 0:
                why[s] = "rank test, sample" if v["reason"] == "sample" else "severity gate"
    both = sorted((s for s in SL if PL[s] > 0), key=lambda s: (-SL[s], -PL[s]))
    none = sorted((s for s in SL if PL[s] == 0), key=lambda s: (why.get(s, "z"), -SL[s]))
    for s in none:
        why.setdefault(s, "no phrase")
    mean = lambda d, ss: sum(d[s] for s in ss) / len(ss)
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    y, rows, heads = 0, [], []
    for title, ss in ((f"{len(both)} expert slices where both have a slice: SliceLens {mean(SL, both):.2f}, "
                       f"SegCiani {mean(PL, both):.2f}", both),
                      (f"{len(none)} expert slices where SegCiani has none: SliceLens {mean(SL, none):.2f}, SegCiani 0",
                       none)):
        heads.append((y, title))
        y += 1
        top = y
        for s in ss:
            rows.append((y, s))
            y += 1
        if ss is none:
            ax.axhspan(top - 0.5, y - 0.5, color="#FBECE2", zorder=0, lw=0)
        y += 0.4
    for yy, s in rows:
        ax.plot([PL[s], SL[s]], [yy, yy], color="#C9D0DA", lw=1.4, zorder=1)
        ax.scatter([SL[s]], [yy - 0.13], s=22, color=AMBER, zorder=3, edgecolors="white", linewidths=0.6, clip_on=False)
        ax.scatter([PL[s]], [yy + 0.13], s=20, color=SLATE, marker="s", zorder=3, edgecolors="white", linewidths=0.6,
                   clip_on=False)
        ax.text(-0.03, yy, s, ha="right", va="center", fontsize=6.6, color=DARK)
        if s in none:
            ax.text(1.04, yy, why[s], ha="left", va="center", fontsize=6.4, color=MUTED)
    for yy, t in heads:
        ax.text(-1.2, yy, t, ha="left", va="center", fontsize=7, color=DARK, fontweight="bold")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(y - 0.6, -0.6)
    ax.set_yticks([])
    ax.xaxis.tick_top()
    ax.set_xlabel("Precision@10", color=MUTED)
    ax.xaxis.set_label_position("top")
    ax.xaxis.grid(color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for side in ("right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["top"].set_color("#C9D0DA")
    ax.tick_params(axis="x", colors=MUTED, length=0)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=AMBER, ms=5, label="SliceLens (slope rule)"),
                        Line2D([], [], marker="s", ls="", color=SLATE, ms=5, label="SegCiani")],
               loc="lower center", ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.62, 0))
    fig.subplots_adjust(left=0.45, right=0.84, top=0.9, bottom=0.07)
    fig.savefig(out("Chapter5", "fig_gesd_gap.pdf"))
    plt.close(fig)


def fig_overlap():
    s = R.load_inst_summary()
    sb = R.load_sbd()["run"]
    fig, ax = plt.subplots(figsize=(6.4, 1.5))
    for yy, d, lab in ((1, s, "VOC val"), (0, sb, "SBD")):
        n = d["n_err"]
        parts = [(d["cov_language"] - d["overlap"], AMBER, "language branch only"),
                 (d["overlap"], GREEN, "both branches"), (d["cov_cluster"] - d["overlap"], SLATE, "cluster branch only"),
                 (n - d["cov_union"], "#E5E9EF", "in no significant slice")]
        x = 0
        for v, col, name in parts:
            ax.barh(yy, 100 * v / n, left=x, color=col, height=0.6, label=name if yy == 1 else None)
            if 100 * v / n > 4:
                ax.text(x + 50 * v / n, yy, f"{100 * v / n:.1f}%", ha="center", va="center", fontsize=6.8,
                        color="white" if col != "#E5E9EF" else DARK)
            x += 100 * v / n
        ax.text(101, yy, f"{n:,} errors", va="center", fontsize=6.8, color=MUTED)
    ax.set_yticks([1, 0])
    ax.set_yticklabels(["VOC val", "SBD"])
    ax.set_xlim(0, 115)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("share of the wrong objects (%)")
    ax.legend(ncol=4, fontsize=6.5, frameon=False, loc="lower center", bbox_to_anchor=(0.45, 1.0))
    clean(ax, grid_y=False)
    fig.tight_layout()
    fig.savefig(out("Chapter5", "fig_overlap.pdf"))
    plt.close(fig)


def fig_types():
    s = R.load_inst_summary()["tax"]
    sb = R.load_sbd()["run"]["tax"]
    order = ["fp", "merge", "boundary", "split", "fn", "extent", "confusion"]
    fig, ax = plt.subplots(figsize=(6.0, 2.1))
    x = np.arange(len(order))
    tv, ts = sum(s.values()), sum(sb.values())
    ax.bar(x - 0.2, [100 * s.get(t, 0) / tv for t in order], width=0.4, color=SLATE, label=f"VOC val ({tv:,} errors)")
    ax.bar(x + 0.2, [100 * sb.get(t, 0) / ts for t in order], width=0.4, color="#9AA8BA", label=f"SBD ({ts:,} errors)")
    ax.set_xticks(x)
    ax.set_xticklabels([TYPE_NAME[t] for t in order], fontsize=7.5)
    ax.set_ylabel("share of the errors (%)")
    ax.legend(frameon=False, fontsize=7)
    clean(ax)
    fig.tight_layout()
    fig.savefig(out("Chapter4", "fig_types.pdf"))
    plt.close(fig)


def fig_ea_repair():
    ER = R.load_ea_repair()
    M, S = ER["means"], ER["start"]
    labs = ["start", "control", "diagnosed\nimages only", "full repair", "original\nimages (ref.)"]
    v = [S["recall.B_victims_edited"], M["control"]["recall.B_victims_edited"], M["diagnosed"]["recall.B_victims_edited"],
         M["repair"]["recall.B_victims_edited"], S["recall.B_victims_original"]]
    fig, ax = plt.subplots(figsize=(5.0, 2.3))
    ax.bar(labs, v, color=[MUTED, MUTED, GREEN, GREEN, SLATE], width=0.6)
    for i, x in enumerate(v):
        ax.text(i, x + 0.02, f"{x:.2f}", ha="center", fontsize=7.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("recall of the held-out victims")
    ax.tick_params(axis="x", labelsize=7.2)
    clean(ax)
    fig.tight_layout()
    fig.savefig(out("Chapter6", "fig_ea_repair.pdf"))
    plt.close(fig)


ALL = {"embedding": fig_embedding, "bic": fig_bic, "compactness": fig_compactness, "rank_car": fig_rank_car,
       "sim": fig_sim, "judges": fig_judges, "cmp": fig_cmp, "gesd": fig_gesd, "gesd_gap": fig_gesd_gap, "overlap": fig_overlap,
       "types": fig_types, "ea_repair": fig_ea_repair}

if __name__ == "__main__":
    names = sys.argv[1:] or list(ALL)
    vf = os.path.join(CACHE, "fig_data_values.json")
    if os.path.exists(vf):
        VALUES.update(json.load(open(vf, encoding="utf-8")))
    for n in names:
        ALL[n]()
        print("done", n)
    json.dump(VALUES, open(vf, "w", encoding="utf-8"), indent=1, default=float)
