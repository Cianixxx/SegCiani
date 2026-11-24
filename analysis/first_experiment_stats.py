"""The first experiment of the thesis (May 2026), recomputed from its own files: can a VLM describe a segmentation
error in free text? 80 VOC images, each shown with its ground-truth mask and a prediction; four VLMs (a fifth,
gemini-3.5-flash, added on 26 May) write a paragraph on how the prediction differs; a judge LLM scores each paragraph
0-10 against a human description of the same image (answer_gts.jsonl, the annotations used again in §6.4).

The cells of the design, as stored in UNI/TESI/MYCODES/PASCAL_BestVLMModel:
  JOIN-D1  one composite image (RGB | ground truth | prediction), prompt wording 1: descriptions
           1ex_pascalVLM/1<model>_descriptions.txt, judged three times (1-1, 1-2, 1-3: the same paragraphs re-scored)
  JOIN-D2  the same with prompt wording 2: 2ex_pascalVLM, judged three times (2-1, 2-2, 2-3)
  SEP      the three images as separate attachments, wording 1: 1sepPascal, judged twice
Primary judge: MiniMax-M2.7 (config.py JUDGE_MODEL); second judge: Gemini (the *_gemini columns, added 29 May).
A missing description or a failed call is N/A (also stored as -1.0).

Out: docs/first_experiment/first_experiment.{json,txt}. CPU, a few seconds.
"""
import csv
import json
import os
import re
from itertools import combinations

import numpy as np
from scipy import stats

BASE = os.environ.get("FIRST_EXP", r"<home>\Desktop\UNI\TESI\MYCODES\PASCAL_BestVLMModel")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "first_experiment")
M4 = ["gemma-3n-E4B-it", "gemma-4", "Kimi-K2.5", "Qwen3.5-9B"]
M5 = M4 + ["gemini-3.5-flash"]
FILES = {"JOIN-D1": ["1ex_pascalVLM/1-1scores_table.csv", "1ex_pascalVLM/1-2scores_table.csv",
                     "1ex_pascalVLM/1-3scores_table.csv"],
         "JOIN-D2": ["2ex_pascalVLM/2-1scores_table.csv", "2ex_pascalVLM/2-2scores_table.csv",
                     "2ex_pascalVLM/2-3scores_table.csv"],
         "SEP": ["1sepPascal/scores_table_sep.csv", "1sepPascal/2scores_table_sep.csv"]}
DESC = {"JOIN-D1": "1ex_pascalVLM/1{m}_descriptions.txt", "JOIN-D2": "2ex_pascalVLM/2{m}_descriptions.txt",
        "SEP": "1sepPascal/{m}_sep_descriptions.txt"}
DESC_RE = re.compile(r"\[idx=(\d+)\]\s*\[[^\]]+\]\s*\n(.*?)(?=\n-{20,}|\Z)", re.DOTALL)
lines = []
say = lambda s="": (lines.append(s), print(s))


def val(x):
    try:
        v = float(x)
    except ValueError:
        return None
    return None if v < 0 else v


def read(path):
    """{model: {idx: score}} for the primary judge, and {model: {idx: score}} for the Gemini judge (the second
    *_gemini column: the one left N/A where the description itself is missing)"""
    rows = [r for r in csv.reader(open(os.path.join(BASE, path), encoding="utf-8")) if r and not r[0].startswith("#")]
    head, body = rows[0], [r for r in rows[1:] if r[0].isdigit()]          # the summary rows (MEAN_PER_MODEL...) out
    prim, gem = {}, {}
    for m in M5:
        j = head.index(m)
        g = [k for k, h in enumerate(head) if h == m + "_gemini"]
        prim[m] = {int(r[0]): val(r[j]) for r in body}
        gem[m] = {int(r[0]): val(r[g[-1]]) for r in body} if g else {}
    return prim, gem


T = {cell: [read(f) for f in fs] for cell, fs in FILES.items()}
IDX = sorted(T["JOIN-D1"][0][0][M4[0]])
cell_score = {}          # (cell, model) -> {idx: mean over the judge runs, None if every run is N/A}
for cell, runs in T.items():
    for m in M5:
        cell_score[(cell, m)] = {}
        for i in IDX:
            v = [r[0][m].get(i) for r in runs if r[0][m].get(i) is not None]
            cell_score[(cell, m)][i] = float(np.mean(v)) if v else None

say("FIRST EXPERIMENT (May 2026): FREE-TEXT DESCRIPTIONS OF A SEGMENTATION ERROR, JUDGED AGAINST A HUMAN ONE")
say(f"{len(IDX)} images; judge MiniMax-M2.7, scores 0-10")
out = {"n_images": len(IDX), "models": {}, "cells": list(FILES)}

# ---- per model and cell: mean on the answers given, how many missing, mean with a missing answer as 0
say(f"\n{'model':18s} " + " ".join(f"{c:>22s}" for c in FILES))
for m in M5:
    row = {}
    for c in FILES:
        v = [x for x in cell_score[(c, m)].values() if x is not None]
        na = len(IDX) - len(v)
        row[c] = {"mean": float(np.mean(v)) if v else None, "na": na,
                  "penalised": float(np.sum(v) / len(IDX))}
    out["models"][m] = row
    say(f"{m:18s} " + " ".join(f"{row[c]['mean'] if row[c]['mean'] is not None else float('nan'):6.2f} "
                               f"(N/A {row[c]['na']:2d}, pen {row[c]['penalised']:4.2f})" for c in FILES))


def per_image(cell, models):
    """mean over the models that answered, per image"""
    r = {}
    for i in IDX:
        v = [cell_score[(cell, m)][i] for m in models if cell_score[(cell, m)][i] is not None]
        r[i] = float(np.mean(v)) if v else None
    return r


def paired(a, b):
    k = [i for i in IDX if a[i] is not None and b[i] is not None]
    x, y = np.array([a[i] for i in k]), np.array([b[i] for i in k])
    return {"n": len(k), "mean_a": float(x.mean()), "mean_b": float(y.mean()), "mean_abs_diff": float(np.abs(x - y).mean()),
            "pearson": float(stats.pearsonr(x, y)[0]), "p_wilcoxon": float(stats.wilcoxon(x, y).pvalue)}


# ---- the sources of variation: input format, prompt wording, judge re-scoring
J1, J2, S = per_image("JOIN-D1", M4), per_image("JOIN-D2", M4), per_image("SEP", M4)
out["format"] = paired(J1, S)
out["wording"] = paired(J1, J2)
noise = []
for cell in ("JOIN-D1", "JOIN-D2"):
    for m in M4:
        for i in IDX:
            v = [r[0][m].get(i) for r in T[cell] if r[0][m].get(i) is not None]
            if len(v) == 3:
                noise.append(float(np.std(v, ddof=1)))
rr = []                                           # two judge runs of the same paragraphs, per image
for cell in ("JOIN-D1", "JOIN-D2"):
    for a, b in combinations(range(3), 2):
        ra = {i: np.mean([T[cell][a][0][m][i] for m in M4 if T[cell][a][0][m].get(i) is not None] or [np.nan]) for i in IDX}
        rb = {i: np.mean([T[cell][b][0][m][i] for m in M4 if T[cell][b][0][m].get(i) is not None] or [np.nan]) for i in IDX}
        k = [i for i in IDX if not (np.isnan(ra[i]) or np.isnan(rb[i]))]
        rr.append(stats.pearsonr([ra[i] for i in k], [rb[i] for i in k])[0])
out["judge_rerun"] = {"sd_mean": float(np.mean(noise)), "pearson_mean": float(np.mean(rr))}
f, w = out["format"], out["wording"]
say(f"\ninput format, JOIN-D1 vs SEP (same prompt): mean {f['mean_a']:.2f} -> {f['mean_b']:.2f} on {f['n']} images, "
    f"Wilcoxon p {f['p_wilcoxon']:.1e}; per-image Pearson {f['pearson']:.2f}")
say(f"prompt wording, JOIN-D1 vs JOIN-D2 (new paragraphs, same format): mean {w['mean_a']:.2f} vs {w['mean_b']:.2f}, "
    f"Wilcoxon p {w['p_wilcoxon']:.2f}; per-image |difference| {w['mean_abs_diff']:.2f}; Pearson {w['pearson']:.2f}")
say(f"judge re-scoring of the same paragraphs: SD per image and model {out['judge_rerun']['sd_mean']:.2f}; "
    f"per-image Pearson between two re-scorings {out['judge_rerun']['pearson_mean']:.2f}")

# ---- the second judge
pa, pb = [], []
for cell, runs in T.items():
    for m in M5:
        for i in IDX:
            a, g = cell_score[(cell, m)][i], runs[0][1][m].get(i)
            if a is not None and g is not None:
                pa.append(a)
                pb.append(g)
out["judges"] = {"n": len(pa), "pearson": float(stats.pearsonr(pa, pb)[0]), "spearman": float(stats.spearmanr(pa, pb)[0]),
                 "mean_minimax": float(np.mean(pa)), "mean_gemini": float(np.mean(pb))}
j = out["judges"]
say(f"\ntwo judges on the same paragraphs ({j['n']} pairs): MiniMax {j['mean_minimax']:.2f}, Gemini {j['mean_gemini']:.2f}; "
    f"Pearson {j['pearson']:.2f}, Spearman {j['spearman']:.2f}")

# ---- length: do longer paragraphs score higher?
nw, sc, sg = [], [], []
for cell, pat in DESC.items():
    for m in M5:
        fp = os.path.join(BASE, pat.format(m=m))
        if not os.path.exists(fp):
            continue
        for mm in DESC_RE.finditer(open(fp, encoding="utf-8", errors="replace").read()):
            i, t = int(mm.group(1)), mm.group(2).strip()
            if not t or i not in cell_score[(cell, m)] or cell_score[(cell, m)][i] is None:
                continue
            nw.append(len(t.split()))
            sc.append(cell_score[(cell, m)][i])
            g = T[cell][0][1][m].get(i)
            sg.append(g if g is not None else np.nan)
nw, sc, sg = np.array(nw), np.array(sc), np.array(sg)
ok = ~np.isnan(sg)
out["length"] = {"n": int(len(nw)), "median_words": float(np.median(nw)),
                 "spearman_minimax": float(stats.spearmanr(nw, sc)[0]),
                 "spearman_gemini": float(stats.spearmanr(nw[ok], sg[ok])[0])}
L = out["length"]
say(f"\nlength: {L['n']} judged paragraphs, median {L['median_words']:.0f} words; Spearman words-score "
    f"MiniMax {L['spearman_minimax']:.2f}, Gemini {L['spearman_gemini']:.2f}")
os.makedirs(OUT, exist_ok=True)
json.dump(out, open(os.path.join(OUT, "first_experiment.json"), "w"), indent=1)
open(os.path.join(OUT, "first_experiment.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("FIRST_EXP_DONE")
