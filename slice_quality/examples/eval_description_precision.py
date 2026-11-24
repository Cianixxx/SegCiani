"""Precision of the produced descriptions, verified INDEPENDENTLY by the VLM.

Every other check in the pipeline asks "is this group statistically unusual?".
This one asks the missing question: *when we tell the user "a cow cut off at the
image border", is that actually true of the object we said it about?*

Protocol (chance-calibrated, like the purity gate):
  positives  = the instances the description was actually attached to;
  controls   = random error instances of the SAME seg_subtype NOT in that slice
               (so a query that Qwen says "yes" to for everything scores no lift);
  metric     = P(yes | query, region) >= 0.5, per group, plus the positive-vs-
               control gap and a one-sided Fisher exact test (BH-FDR over groups).

Queries are the very phrases shipped to the user: the SliceLens situation for
language patterns, "<attribute> <class>" for cluster patterns (class-only when
the slice carries no attribute -- those are reported apart, being near-tautological).

Run (GPU):
  PYTHONPATH=/workspace python slice_quality/examples/eval_description_precision.py
Env knobs: PREC_K (positives per query, default 10), PREC_SEED, PREC_MAX_EVALS.
"""
import json
import os
import pickle
import random
from collections import defaultdict

from slicelens.data_types import Region
from slice_quality.describe import (_attr_word, _class_phrase, _art,
                                    build_image_descriptions)
from slice_quality.metrics.significance import benjamini_hochberg

SQ_OUT = os.environ.get("SQ_OUT", "outputs/slice_quality")     # the run whose shipped phrases are verified
PKL = f"{SQ_OUT}/result_tagged.pkl"
OUT_JSON = f"{SQ_OUT}/eval_precision.json"
OUT_TXT = f"{SQ_OUT}/eval_precision.txt"
K = int(os.environ.get("PREC_K", "10"))
SEED = int(os.environ.get("PREC_SEED", "0"))
MAX_EVALS = int(os.environ.get("PREC_MAX_EVALS", "2000"))
YES = 0.5

rng = random.Random(SEED)


def _cluster_query(members):
    """Same claim the user reads, as a noun phrase; None if it claims nothing."""
    attr = _attr_word(members)
    clsp = _class_phrase(members)
    cls = None
    if clsp and clsp.startswith("of class "):        # unanimous class only
        cls = clsp.split("'")[1]
    if attr and cls:
        return f"{_art(attr)} {attr} {cls}", "attribute+class"
    if attr:
        return f"{_art(attr)} {attr} object", "attribute"
    # No attribute: the description no longer claims one (describe.py says
    # "grouped by appearance only"), so there is nothing verifiable to score.
    # Measuring the bare class was what showed it to be empty (lift −0.021).
    return None, None


def _lang_query(description):
    """'"a cow cut off ..." (SliceLens) -> ...'  ->  'a cow cut off ...'."""
    if '"' not in description:
        return None
    return description.split('"')[1]


def main():
    with open(PKL, "rb") as f:
        data = pickle.load(f)
    res, errors, image_paths = data["result"], data["errors"], data["image_paths"]

    # the descriptions actually shipped (same gate, same wording)
    descs = build_image_descriptions(res)

    # slice_id -> image_id -> members   (mirrors build_image_descriptions' dedup)
    members_of = defaultdict(lambda: defaultdict(list))
    hypothesis_of = {}
    slice_ids = set()
    seen = set()
    for s, q in res.scored:
        if q.slice_id in seen:
            continue
        seen.add(q.slice_id)
        slice_ids.add(q.slice_id)
        if s.source == "language" and s.hypothesis:
            hypothesis_of[q.slice_id] = s.hypothesis
        for m in s.members:
            members_of[q.slice_id][m.image_id].append(m)

    # group evaluation units by the exact phrase the user is shown
    groups = defaultdict(lambda: {"branch": "", "claim": "", "pos": [], "sids": set(),
                                  "generic": None})
    for image_id, d in descs.items():
        for p in d["patterns"]:
            members = members_of[p["slice_id"]][image_id]
            if not members:
                continue
            if p["branch"] == "language":
                query, claim = _lang_query(p["description"]), "situation"
            else:
                query, claim = _cluster_query(members)
            if not query:
                continue
            g = groups[query]
            g["branch"], g["claim"] = p["branch"], claim
            g["sids"].add(p["slice_id"])
            g["pos"].extend(members)
            if p["slice_id"] in hypothesis_of:      # pre-rewrite wording
                g["generic"] = hypothesis_of[p["slice_id"]]

    in_slice = {sid: {m.instance_id for ms in members_of[sid].values() for m in ms}
                for sid in slice_ids}
    by_subtype = defaultdict(list)
    for e in errors:
        if e.bbox is not None and e.image_id in image_paths:
            by_subtype[e.seg_subtype.value].append(e)

    def _cls(inst):
        return inst.gt_class or inst.pred_class

    # Positives + TWO nulls. Most phrases name a class ("a cow cut off at the
    # image border"), so a control drawn from any class would let the class term
    # alone produce the gap -- the lift would measure class recognition, not the
    # situation. ctrl_cls (same subtype AND same class, outside the slice) is the
    # strict null: it isolates the SITUATIONAL claim. ctrl_any (same subtype only)
    # is kept as the loose null, i.e. "does the phrase say anything at all".
    plan = []
    for query, g in sorted(groups.items()):
        pos = [m for m in g["pos"] if m.bbox is not None and m.image_id in image_paths]
        if not pos:
            continue
        rng.shuffle(pos)
        pos = pos[:K]
        excluded = set().union(*(in_slice[s] for s in g["sids"]))
        ctrl_any, ctrl_cls = [], []
        for m in pos:
            pool = [e for e in by_subtype[m.seg_subtype.value]
                    if e.instance_id not in excluded]
            if pool:
                ctrl_any.append(rng.choice(pool))
            same = [e for e in pool if _cls(e) == _cls(m)]
            if same:
                ctrl_cls.append(rng.choice(same))
        plan.append({"query": query, "branch": g["branch"], "claim": g["claim"],
                     "slices": sorted(g["sids"]), "pos": pos,
                     "ctrl": ctrl_any, "ctrl_cls": ctrl_cls,
                     "generic": g["generic"]})

    # PAIRED ablation: same instances, same seed, but scored with the hypothesis
    # BEFORE the class rewrite ("an object cut off..." vs "a dog cut off...").
    # Isolates what the class term costs or buys in verifiability.
    if os.environ.get("PREC_GENERIC"):
        plan = [dict(p, query=p["generic"]) for p in plan
                if p["generic"] and p["generic"] != p["query"]]
        OUT_JSON_G = OUT_JSON.replace(".json", "_generic.json")
        OUT_TXT_G = OUT_TXT.replace(".txt", "_generic.txt")
        globals()["OUT_JSON"], globals()["OUT_TXT"] = OUT_JSON_G, OUT_TXT_G

    n_evals = sum(len(p["pos"]) + len(p["ctrl"]) + len(p["ctrl_cls"]) for p in plan)
    print(f"groups={len(plan)}  evaluations={n_evals} (cap {MAX_EVALS})", flush=True)
    if n_evals > MAX_EVALS:
        raise SystemExit(f"budget exceeded ({n_evals} > {MAX_EVALS}); lower PREC_K")
    if os.environ.get("PREC_DRYRUN"):     # plan check on CPU, no model loaded
        for p in plan:
            print(f"  [{p['branch']}/{p['claim']}] {p['query']}  "
                  f"pos={len(p['pos'])} ctrl_any={len(p['ctrl'])} "
                  f"ctrl_cls={len(p['ctrl_cls'])}", flush=True)
        return

    from slicelens.backends import QwenVLM
    vlm = QwenVLM(model=os.environ.get("VLM_MODEL", "/models/qwen"), grounding="bbox")

    def score(inst, query):
        return float(vlm.slice_confidence(image_paths[inst.image_id],
                                          Region(bbox=tuple(inst.bbox)), query))

    rows, done = [], 0
    for gi, p in enumerate(plan, 1):
        sp = [score(m, p["query"]) for m in p["pos"]]
        sc = [score(m, p["query"]) for m in p["ctrl"]]
        sk = [score(m, p["query"]) for m in p["ctrl_cls"]]
        done += len(sp) + len(sc) + len(sk)
        hp = sum(v >= YES for v in sp)
        hc = sum(v >= YES for v in sc)
        hk = sum(v >= YES for v in sk)
        rows.append({
            "query": p["query"], "branch": p["branch"], "claim": p["claim"],
            "slices": p["slices"],
            "n_pos": len(sp), "hits_pos": hp,
            "precision": round(hp / len(sp), 3) if sp else None,
            "mean_conf_pos": round(sum(sp) / len(sp), 3) if sp else None,
            "n_ctrl": len(sc), "hits_ctrl": hc,
            "control_rate": round(hc / len(sc), 3) if sc else None,
            "mean_conf_ctrl": round(sum(sc) / len(sc), 3) if sc else None,
            "n_ctrl_cls": len(sk), "hits_ctrl_cls": hk,
            "control_rate_same_class": round(hk / len(sk), 3) if sk else None,
        })
        print(f"[{gi}/{len(plan)}] {p['query'][:48]:<48} "
              f"prec={rows[-1]['precision']} ctrl={rows[-1]['control_rate']} "
              f"ctrlC={rows[-1]['control_rate_same_class']} "
              f"({done}/{n_evals})", flush=True)

    # one-sided Fisher exact (positives more often "yes" than controls) + BH-FDR.
    # The reported test uses the STRICT null (same class) when it exists.
    try:
        from scipy.stats import fisher_exact

        def _fisher(hp, np_, hc, nc):
            return round(float(fisher_exact([[hp, np_ - hp], [hc, nc - hc]],
                                            alternative="greater")[1]), 5)
        for r in rows:
            if r["n_ctrl"]:
                r["p_value_any_class"] = _fisher(r["hits_pos"], r["n_pos"],
                                                 r["hits_ctrl"], r["n_ctrl"])
            if r["n_ctrl_cls"]:
                r["p_value"] = _fisher(r["hits_pos"], r["n_pos"],
                                       r["hits_ctrl_cls"], r["n_ctrl_cls"])
            else:
                r["p_value"] = r.get("p_value_any_class", 1.0)
        pv = [r.get("p_value", 1.0) for r in rows]
        for r, ok in zip(rows, benjamini_hochberg(pv, alpha=0.05)):
            r["significant_vs_chance"] = bool(ok)
    except ImportError:
        print("scipy missing: no Fisher test", flush=True)

    def agg(sel):
        sel = [r for r in rows if sel(r)]
        if not sel:
            return None
        hp, np_ = sum(r["hits_pos"] for r in sel), sum(r["n_pos"] for r in sel)
        hc, nc = sum(r["hits_ctrl"] for r in sel), sum(r["n_ctrl"] for r in sel)
        hk, nk = (sum(r["hits_ctrl_cls"] for r in sel),
                  sum(r["n_ctrl_cls"] for r in sel))
        return {"groups": len(sel), "n_pos": np_, "precision": round(hp / np_, 3),
                "control_rate_any_class": round(hc / nc, 3) if nc else None,
                "control_rate_same_class": round(hk / nk, 3) if nk else None,
                "lift_vs_same_class": round(hp / np_ - hk / nk, 3) if nk else None,
                "groups_above_chance": sum(1 for r in sel
                                           if r.get("significant_vs_chance"))}

    summary = {
        "K": K, "seed": SEED, "yes_threshold": YES, "n_evaluations": done,
        "overall": agg(lambda r: True),
        "language": agg(lambda r: r["branch"] == "language"),
        "cluster_situational": agg(lambda r: r["branch"] == "cluster"
                                   and r["claim"] != "class-only"),
        "cluster_class_only": agg(lambda r: r["claim"] == "class-only"),
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "groups": rows}, f, indent=2)
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("DESCRIPTION PRECISION (VLM re-verification, chance-calibrated)\n\n")
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")
        f.write("\nper description\n")
        for r in sorted(rows, key=lambda x: -(x["precision"] or 0)):
            f.write(f"  {r['precision']:.2f} (ctrl same-class "
                    f"{r['control_rate_same_class']}, any {r['control_rate']}) "
                    f"[{r['branch']}/{r['claim']}] {r['query']}\n")
    print("SUMMARY", json.dumps(summary, indent=2), flush=True)
    print("PRECISION_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
