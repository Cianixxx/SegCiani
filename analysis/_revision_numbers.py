# -*- coding: utf-8 -*-
"""Numbers for the 'Block 1 revision' sections of both pipeline documents.

Read from the run files, never typed in: the reference run (outputs/slice_quality,
IoU claim rule) and the revised run (outputs/slice_quality_cov05b, containment
claims + swallowed objects kept as instances + merges measured against their
union). Every number the two documents quote about the revision comes from here,
so the text cannot drift from the files the way the legend of the comparison
page once did.

The one exception is the taxonomy of the reference run on val, which is not in
any local file: it is the 'cached taxonomy' line printed by retype_pickle.py
(outputs/chain_cov05b.log) and is copied here with that provenance.
"""
import json
import os
import pickle
import statistics as st
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
# Where the runs live on this computer since the reorganisation of 23 October 2026: the ones the thesis reports,
# the earlier ones it compares with, and the old place (the server keeps ~/storage/SLICELENS/outputs/<run>).
SLICELENS = os.path.dirname(ROOT)
RUN_BASES = [os.path.join(SLICELENS, "RUN_UFFICIALI"), os.path.join(SLICELENS, "RUN_VECCHI"),
             os.path.join(ROOT, "outputs")]


def out_path(*parts):
    """A run directory, or a file in it, named as under outputs/ on the server: the first of RUN_BASES that
    has it (RUN_UFFICIALI when none does, so that a new file goes with the official runs)."""
    for b in RUN_BASES:
        p = os.path.join(b, *parts)
        if os.path.exists(p):
            return p
    return os.path.join(RUN_BASES[0], *parts)

REF = out_path("slice_quality")
NEW = out_path("slice_quality_cov05b")

# retype_pickle.py, 'cached taxonomy' (outputs/chain_cov05b.log, 2026-09-15)
TAX_BEFORE = {"fn": 2651, "fp": 2628, "boundary": 799, "extent": 188,
              "merge": 98, "confusion": 38, "split": 4}
REF_BEFORE = 522


def _judge(path):
    return {(r["image"], r["kind"], r["pairing"]): r
            for r in json.load(open(path, encoding="utf-8"))["judgements"]}


def _paired(A, B, kind, sel=lambda i: True):
    imgs = sorted({k[0] for k in B if k[1] == kind and k[2] == "genuine"})
    pa = [(A[(i, kind, "genuine")]["score"], B[(i, kind, "genuine")]["score"])
          for i in imgs if sel(i) and (i, kind, "genuine") in A]
    d = [b - a for a, b in pa]
    return dict(n=len(pa), a=st.mean(x for x, _ in pa), b=st.mean(y for _, y in pa),
                up=sum(1 for x in d if x > 0), dn=sum(1 for x in d if x < 0),
                eq=sum(1 for x in d if x == 0))


def _cls(slice_id):
    """The class of a language slice id: '<cls>_<k>', '<cls>_extra_<k>', '<cls>_dd_<k>', '<cls>_pend_<k>'."""
    import re
    return re.sub(r"_(extra|dd|pend)$", "", slice_id.rsplit("_", 1)[0])


def _slices(path):
    idx = json.load(open(path, encoding="utf-8"))["slices"]
    used = [s for s in idx if s.get("in_descriptions")]
    return dict(cluster=sum(1 for s in used if s.get("source") == "cluster"),
                language=sum(1 for s in used if s.get("source") != "cluster"),
                cluster_types=dict(Counter(s.get("dominant_error_type") for s in used
                                           if s.get("source") == "cluster")))


def _coverage(path):
    d = json.load(open(path, encoding="utf-8"))
    cov = [v["n_systematic"] / v["n_errors"] for v in d.values() if v.get("n_errors")]
    return dict(images=len(d), with_pattern=sum(1 for v in d.values() if v.get("n_systematic")),
                share=sum(cov) / len(cov))


def _synth(path):
    rows = json.load(open(path, encoding="utf-8"))["judgements"]
    out = {}
    for kind in ("narrative", "pattern"):
        g = [r for r in rows if r.get("kind", "narrative") == kind and r.get("pairing") == "genuine"
             and r.get("score") is not None]
        good = [r["score"] for r in g if r.get("declared_good")]
        bad = [r["score"] for r in g if not r.get("declared_good")]
        out[kind] = dict(all=st.mean(r["score"] for r in g), good=st.mean(good), bad=st.mean(bad))
    return out


def load():
    n = {"tax_before": TAX_BEFORE, "ref_before": REF_BEFORE}
    pk = pickle.load(open(os.path.join(NEW, "result_tagged.pkl"), "rb"))
    err, ref = pk["errors"], pk["reference"]
    n["tax_after"] = dict(Counter(e.seg_subtype.value for e in err))
    n["merge_gt_side"] = sum(1 for e in err if e.seg_subtype.value == "merge"
                             and e.metadata.get("from_gt_mask"))
    n["merge_blobs"] = n["tax_after"]["merge"] - n["merge_gt_side"]
    n["merge_mixed"] = sum(1 for e in err if e.seg_subtype.value == "merge"
                           and not e.metadata.get("from_gt_mask")
                           and len(set(e.metadata.get("gt_classes", []))) > 1)
    n["merge_no_primary"] = sum(1 for e in err if e.seg_subtype.value == "merge"
                                and not e.metadata.get("from_gt_mask")
                                and e.metadata.get("primary_iou") is None)
    n["pairs"] = sum(1 for e in err + ref if e.metadata.get("paired_by") == "containment")
    n["pairs_by_type"] = dict(Counter(e.seg_subtype.value for e in err + ref
                                      if e.metadata.get("paired_by") == "containment"))
    n["ref_after"] = len(ref)
    n["ref_moved"] = sum(1 for e in err if e.seg_subtype.value == "merge"
                         and (e.metadata.get("primary_iou") or 0.0) >= 0.85)
    n["errors_before"] = sum(TAX_BEFORE.values())
    n["errors_after"] = len(err)

    A = _judge(os.path.join(REF, "judge_descriptions_scored.json"))
    B = _judge(os.path.join(NEW, "judge_descriptions_scored.json"))
    split = {k[0]: r.get("split") for k, r in B.items()}
    n["judge"] = {}
    for kind in ("narrative", "pattern"):
        n["judge"][kind] = dict(
            all=_paired(A, B, kind),
            val=_paired(A, B, kind, lambda i: split.get(i) == "val"),
            train=_paired(A, B, kind, lambda i: split.get(i) == "train"),
            control_a=st.mean(r["score"] for k, r in A.items() if k[1] == kind and k[2] == "control"),
            control_b=st.mean(r["score"] for k, r in B.items() if k[1] == kind and k[2] == "control"))
    n["criteria"] = {}
    for c in ("completeness", "correctness", "specificity"):
        n["criteria"][c] = (st.mean(r[c] for k, r in A.items() if k[1] == "narrative" and k[2] == "genuine"),
                            st.mean(r[c] for k, r in B.items() if k[1] == "narrative" and k[2] == "genuine"))

    n["slices_before"] = _slices(os.path.join(REF, "slices_index.json"))
    n["slices_after"] = _slices(os.path.join(NEW, "slices_index.json"))
    n["cov_before"] = _coverage(os.path.join(REF, "image_descriptions.json"))
    n["cov_after"] = _coverage(os.path.join(NEW, "image_descriptions.json"))
    n["synth_before"] = _synth(os.path.join(REF, "synth", "judge_synth_v2.json"))
    n["synth_after"] = _synth(os.path.join(NEW, "synth", "judge_synth.json"))
    return n


def load_narrative_stage():
    """Numbers for the 'narratives and judge' section: the reference run only."""
    A = _judge(os.path.join(REF, "judge_descriptions_scored.json"))
    n = {}
    for kind in ("narrative", "pattern"):
        g = [r for k, r in A.items() if k[1] == kind and k[2] == "genuine"]
        c = [r for k, r in A.items() if k[1] == kind and k[2] == "control"]
        n[kind] = dict(n=len(g), genuine=st.mean(r["score"] for r in g),
                       control=st.mean(r["score"] for r in c),
                       crit_g={x: st.mean(r[x] for r in g) for x in ("completeness", "correctness", "specificity")},
                       crit_c={x: st.mean(r[x] for r in c) for x in ("completeness", "correctness", "specificity")})
    gaps = [r["gap_sum_holistic"] for r in A.values() if r.get("gap_sum_holistic") is not None]
    n["holistic"] = dict(n=len(gaps), mean_gap=st.mean(gaps),
                         below=sum(1 for g in gaps if g < 0), above=sum(1 for g in gaps if g > 0),
                         equal=sum(1 for g in gaps if g == 0))
    by_v = {}
    for k, r in A.items():
        if k[1] == "narrative" and k[2] == "genuine" and r.get("verdict"):
            by_v.setdefault(r["verdict"], []).append(r["score"])
    n["by_verdict"] = {v: (st.mean(x), len(x)) for v, x in by_v.items()}
    n["split"] = {sp: st.mean(r["score"] for k, r in A.items() if k[1] == "narrative"
                              and k[2] == "genuine" and r.get("split") == sp) for sp in ("val", "train")}
    # contour notes on CORRECT regions: images whose narrative carries one, how many
    # of them are val (the file holds val + train36), and how many have no typed
    # error at all (those produced no text before the notes)
    nar = json.load(open(os.path.join(REF, "error_narratives.json"), encoding="utf-8"))
    typed = json.load(open(os.path.join(REF, "typed_per_image.json"), encoding="utf-8"))
    tr36 = set(json.load(open(os.path.join(REF, "train36", "descriptions_train36_full.json"),
                              encoding="utf-8")))
    con = [k for k, v in nar.items() if "correctly located" in v["text"] or "outline is" in v["text"]]
    n["contour"] = dict(all=len(con), val=sum(1 for k in con if k not in tr36),
                        noerr=sum(1 for k in con if not sum(typed.get(k, {}).values())))
    n["synth"] = _synth(os.path.join(REF, "synth", "judge_synth_v2.json"))
    rows = json.load(open(os.path.join(REF, "synth", "judge_synth_v2.json"), encoding="utf-8"))["judgements"]
    good = {r["image"] for r in rows if r.get("declared_good")}
    cn = [r for r in rows if r.get("kind", "narrative") == "narrative" and r.get("pairing") == "control"]
    cg = [r for r in cn if r["image"] in good]
    same = [r for r in cg if r.get("control_from") in good]
    n["synth_control"] = dict(same=len(same), good_n=len(cg),
                              floor_with=st.mean(r["score"] for r in cg),
                              floor_without=st.mean(r["score"] for r in cg if r not in same),
                              floor_bad=st.mean(r["score"] for r in cn if r["image"] not in good))
    return n


INST = out_path("slice_quality_inst")
# the inst run as it was certified by SliceLens's slope rule (kept by chain_auc_official.sh on 21 October): the
# comparisons between typings hold the certification fixed, and every other run in them used the slope rule
INST_SLOPE = out_path("slice_quality_inst_slope")
_INST_TYPING = INST_SLOPE if os.path.exists(os.path.join(INST_SLOPE, "slices_index.json")) else INST


def load_inst():
    """Numbers for the 'GT instances = objects' subsection (run inst vs the two others)."""
    n = {}
    amb = json.load(open(os.path.join(NEW, "instance_ambiguity.json"), encoding="utf-8"))
    t = amb["totals"]
    per = amb["per_image"]
    n["amb"] = dict(components=t["components"], objects=t["objects"], cut=t["cut_objects"],
                    parts=t["parts_of_cut"], touching=t["touching_components"],
                    touching_objects=t["objects_in_touching"], images=len(per),
                    img_cut=sum(1 for r in per.values() if r.get("cut_objects")),
                    img_touch=sum(1 for r in per.values() if r.get("touching_components")),
                    false_merge=amb["false_merge"], real_merge=amb["real_merge"],
                    unknown_merge=amb["unknown_merge"], partial_miss=amb["partial_miss"],
                    whole_miss=amb["whole_miss"])
    pk = pickle.load(open(os.path.join(INST, "result_tagged.pkl"), "rb"))
    err, ref = pk["errors"], pk["reference"]
    n["tax"] = dict(Counter(e.seg_subtype.value for e in err))
    n["ref"] = len(ref)
    n["gt_side"] = sum(1 for e in err if e.metadata.get("from_gt_mask"))
    cut = [e for e in err + ref if (e.metadata.get("gt_parts") or 1) > 1]
    n["cut_inst"] = len(cut)
    occ = Counter(e.metadata.get("occluded_by") for e in cut if e.metadata.get("occluded_by"))
    n["occluded"] = sum(occ.values())
    n["occluders"] = occ.most_common(4)
    A = _judge(os.path.join(REF, "judge_descriptions_scored.json"))
    B = _judge(os.path.join(NEW, "judge_descriptions_scored.json"))
    # frozen right after the object run, BEFORE the narrative revision (load_narr)
    C = _judge(os.path.join(INST, "judge_descriptions_scored_objects.json"))
    split = {k[0]: r.get("split") for k, r in C.items()}
    n["judge"] = {}
    for kind in ("narrative", "pattern"):
        n["judge"][kind] = dict(
            vs_ref=_paired(A, C, kind), vs_cov=_paired(B, C, kind),
            val=_paired(A, C, kind, lambda i: split.get(i) == "val"),
            train=_paired(A, C, kind, lambda i: split.get(i) == "train"),
            control=st.mean(r["score"] for k, r in C.items() if k[1] == kind and k[2] == "control"))
    n["criteria"] = {c: st.mean(r[c] for k, r in C.items() if k[1] == "narrative" and k[2] == "genuine")
                     for c in ("completeness", "correctness", "specificity")}
    n["slices"] = _slices(os.path.join(_INST_TYPING, "slices_index.json"))
    n["cov"] = _coverage(os.path.join(_INST_TYPING, "image_descriptions.json"))
    n["synth"] = _synth(os.path.join(INST, "synth", "judge_synth_objects.json"))
    nar = json.load(open(os.path.join(INST, "error_narratives_objects.json"), encoding="utf-8"))
    n["hidden"] = sum(1 for v in nar.values() if "hidden behind" in v["text"])
    n["hidden79"] = sum(1 for i in {k[0] for k in C} if "hidden behind" in nar.get(i, {}).get("text", ""))
    return n


def stale_cards(run_dir=None, judge="judge_descriptions_scored.json",
                narr="error_narratives.json"):
    """Judged cards whose narrative has been REGENERATED since the judgement.

    The score on a comparison card belongs to the text that was sent to the
    judge; when the text is rebuilt afterwards (the sub-threshold sentences were
    removed on 2026-09-26) the two stop being a matched pair, and both the page
    and the documents have to say so rather than look consistent.
    """
    d = run_dir or INST
    A = _judge(os.path.join(d, judge))
    nar = json.load(open(os.path.join(d, narr), encoding="utf-8"))
    return sum(1 for k, r in A.items()
               if k[1] == "narrative" and k[2] == "genuine" and k[0] in nar
               and (r.get("cand") or "").strip() != nar[k[0]]["text"].strip())


def load_narr():
    """Numbers for the narrative revision (direction, sides, correct regions,
    spills, sub-threshold patches, verdict, ordering) on the object run."""
    A = _judge(os.path.join(INST, "judge_descriptions_scored_objects.json"))
    C = _judge(os.path.join(INST, "judge_descriptions_scored.json"))
    R = _judge(os.path.join(REF, "judge_descriptions_scored.json"))
    split = {k[0]: r.get("split") for k, r in C.items()}
    n = {"judge": {}}
    for kind in ("narrative", "pattern"):
        n["judge"][kind] = dict(
            vs_obj=_paired(A, C, kind), vs_ref=_paired(R, C, kind),
            val=_paired(A, C, kind, lambda i: split.get(i) == "val"),
            train=_paired(A, C, kind, lambda i: split.get(i) == "train"),
            control_a=st.mean(r["score"] for k, r in A.items() if k[1] == kind and k[2] == "control"),
            control=st.mean(r["score"] for k, r in C.items() if k[1] == kind and k[2] == "control"))
    n["criteria"] = {c: (st.mean(r[c] for k, r in A.items() if k[1] == "narrative" and k[2] == "genuine"),
                         st.mean(r[c] for k, r in C.items() if k[1] == "narrative" and k[2] == "genuine"))
                     for c in ("completeness", "correctness", "specificity")}
    imgs = sorted({k[0] for k in C if k[1] == "narrative" and k[2] == "genuine"})
    n["dist_before"] = dict(Counter(int(A[(i, "narrative", "genuine")]["score"]) for i in imgs))
    n["dist_after"] = dict(Counter(int(C[(i, "narrative", "genuine")]["score"]) for i in imgs))
    n["low_before"] = sum(1 for i in imgs if A[(i, "narrative", "genuine")]["score"] <= 6)
    n["low_after"] = sum(1 for i in imgs if C[(i, "narrative", "genuine")]["score"] <= 6)
    n["ten_before"] = sum(1 for i in imgs if A[(i, "narrative", "genuine")]["score"] >= 10)
    n["ten_after"] = sum(1 for i in imgs if C[(i, "narrative", "genuine")]["score"] >= 10)
    nar = json.load(open(os.path.join(INST, "error_narratives.json"), encoding="utf-8"))
    txt = [v["text"] for v in nar.values()]
    n["counts"] = dict(images=len(txt),
                       spill=sum(1 for t in txt if "spilling over" in t),
                       missing=sum(1 for t in txt if "parts missing" in t or "of its area missing" in t),
                       correct=sum(1 for t in txt if "segmented closely" in t or "segmented well" in t),
                       spill_patch=sum(1 for t in txt if "also spills beyond" in t),
                       verdict=sum(1 for t in txt if t.startswith("The prediction is largely wrong") or t.startswith("The prediction misses most")),
                       half=sum(1 for t in txt if "half of the scene" in t),
                       muto=sum(1 for t in txt if t.startswith("No segmentation error")))
    n["synth_before"] = _synth(os.path.join(INST, "synth", "judge_synth_objects.json"))
    n["synth_after"] = _synth(os.path.join(INST, "synth", "judge_synth.json"))
    return n


def _tax(path):
    """Typed-error population of a run, val only (its own pickle)."""
    pk = pickle.load(open(path, "rb"))
    err, ref = pk["errors"], pk["reference"]
    return dict(Counter(e.seg_subtype.value for e in err)), len(err), len(ref)


def load_runs():
    """Everything the 'three runs side by side' section quotes, per run.

    Four columns, because the narrative revision is a change of TEXT applied to
    the object typing only: reference (IoU claims) / cov05b (containment claims)
    / inst (containment + GT instances = objects) / inst + narrative v3. The
    typing-side numbers (taxonomy, slices, coverage) belong to the first three;
    the fourth repeats the third's and differs only in what the judge reads.
    """
    runs = {}
    runs["ref"] = dict(dir=REF, judge="judge_descriptions_scored.json",
                       synth="judge_synth_v2.json", typing=REF,
                       narr="error_narratives.json",
                       tax=dict(TAX_BEFORE), n_err=sum(TAX_BEFORE.values()), n_ref=REF_BEFORE)
    for key, d in (("cov05b", NEW), ("inst", INST)):
        tax, n_err, n_ref = _tax(os.path.join(d, "result_tagged.pkl"))
        runs[key] = dict(dir=d, judge="judge_descriptions_scored.json",
                         synth="judge_synth.json", typing=d,
                         narr="error_narratives.json",
                         tax=tax, n_err=n_err, n_ref=n_ref)
    runs["inst"]["typing"] = _INST_TYPING                                # slope rule, as the other columns
    runs["inst"]["judge"] = "judge_descriptions_scored_objects.json"   # before the narrative revision
    runs["inst"]["synth"] = "judge_synth_objects.json"
    runs["inst"]["narr"] = "error_narratives_objects.json"
    runs["inst_v3"] = dict(runs["inst"], judge="judge_descriptions_scored.json",
                           synth="judge_synth.json", narr="error_narratives.json")

    for key, r in runs.items():
        A = _judge(os.path.join(r["dir"], r["judge"]))
        for kind in ("narrative", "pattern"):
            g = [x for k, x in A.items() if k[1] == kind and k[2] == "genuine"]
            c = [x for k, x in A.items() if k[1] == kind and k[2] == "control"]
            r[kind] = dict(n=len(g), score=st.mean(x["score"] for x in g),
                           control=st.mean(x["score"] for x in c))
        g = [x for k, x in A.items() if k[1] == "narrative" and k[2] == "genuine"]
        r["criteria"] = {c: st.mean(x[c] for x in g)
                         for c in ("completeness", "correctness", "specificity")}
        r["split"] = {sp: st.mean(x["score"] for x in g if x.get("split") == sp)
                      for sp in ("val", "train")}
        r["ten"] = sum(1 for x in g if x["score"] >= 10)
        r["low"] = sum(1 for x in g if x["score"] <= 6)
        r["slices"] = _slices(os.path.join(r["typing"], "slices_index.json"))
        r["cov"] = _coverage(os.path.join(r["typing"], "image_descriptions.json"))
        r["synth_scores"] = _synth(os.path.join(r["dir"], "synth", r["synth"]))
        nar = json.load(open(os.path.join(r["dir"], r["narr"]), encoding="utf-8"))
        txt = [v["text"] for v in nar.values()]
        r["text_counts"] = dict(
            images=len(txt),
            missed=sum(1 for t in txt if "missed entirely" in t),
            absorbed=sum(1 for t in txt if "absorbed into" in t),
            merges=sum(1 for t in txt if "merges" in t),
            union=sum(1 for t in txt if "taken together" in t),
            hidden=sum(1 for t in txt if "hidden behind" in t),
            spill=sum(1 for t in txt if "spilling over" in t),
            verdict=sum(1 for t in txt if t.startswith("The prediction is largely wrong")
                        or t.startswith("The prediction misses most")),
            silent=sum(1 for t in txt if t.startswith("No segmentation error")))
    return runs


def load_examples(ids):
    """For each image id: the hand-written annotation, the narrative of every run
    and the judge's score on it. Nothing typed in -- the quotes in the documents
    are read from the same files the comparison pages render."""
    mp = {int(k): v.replace(".jpg", "") for k, v in json.load(open(
        os.path.join(ROOT, "_evaldata", "dataset", "voc_patch_matches.json"),
        encoding="utf-8")).items()}
    ann = {}
    for line in open(os.path.join(ROOT, "_evaldata", "dataset", "answer_gts.jsonl"),
                     encoding="utf-8"):
        if line.strip():
            o = json.loads(line)
            ann[mp[o["img_idx"]]] = o["content"].strip()
    runs = load_runs()
    out = {}
    for i in ids:
        rec = {"human": ann[i], "runs": {}}
        for key, r in runs.items():
            nar = json.load(open(os.path.join(r["dir"], r["narr"]), encoding="utf-8"))
            A = _judge(os.path.join(r["dir"], r["judge"]))
            typed = json.load(open(os.path.join(r["typing"], "typed_per_image.json"),
                                   encoding="utf-8"))
            rec["runs"][key] = dict(text=nar[i]["text"],
                                    score=A[(i, "narrative", "genuine")]["score"],
                                    typed=typed.get(i, {}))
        out[i] = rec
    return out


def _branch(txt):
    """Which branch proposed this cause: a SliceLens hypothesis, a cluster group, or both."""
    lang, clus = "SliceLens situation" in txt, "visual group" in txt
    return "both" if lang and clus else ("language" if lang else "cluster")


def _branch_means(recs):
    acc = {}
    for r in recs.values():
        acc.setdefault(_branch(r["cause"]), []).append(r["score"])
    return {b: (st.mean(v), len(v)) for b, v in acc.items()}


def _etype_means(recs):
    """The error type the cause claims -- the wording comes from describe.py."""
    keys = ("missed", "spurious", "merged", "split", "imprecise boundaries",
            "wrong extent", "wrong class")
    acc = {}
    for r in recs.values():
        t = r["cause"].lower()
        acc.setdefault(next((k for k in keys if k in t), "other"), []).append(r["score"])
    return {k: (st.mean(v), len(v)) for k, v in acc.items()}


def load_cause(run_dir=None, val_only=False):
    """Numbers for the 'plausibility of the cause' section (judge_cause*.json).

    The same certified situation judged twice, against the human annotation and
    against the pipeline narrative of the same errors, plus the deranged control.
    The VOC images and the synthetic ones are kept apart: the synthetic
    predictions are other images' ground truths, so a situation certified on val
    cannot explain them, and mixing the two would hide that.
    """
    d = run_dir or INST
    out = {}
    for tag, path in (("voc", os.path.join(d, "judge_cause.json")),
                      ("synth", os.path.join(d, "synth", "judge_cause_synth.json")),
                      ("all", os.path.join(d, "judge_cause_val.json"))):
        if not os.path.exists(path):
            continue
        rows = [r for r in json.load(open(path, encoding="utf-8"))["judgements"]
                if "score" in r and not (val_only and tag == "all" and r.get("split") != "val")]
        n = {}
        kinds = sorted({r["kind"] for r in rows})
        for kind in kinds:
            g = [r for r in rows if r["kind"] == kind and r["pairing"] == "genuine"]
            c = [r for r in rows if r["kind"] == kind and r["pairing"] == "control"]
            n[kind] = dict(n=len(g), n_control=len(c), score=st.mean(r["score"] for r in g),
                           control=st.mean(r["score"] for r in c),
                           cons=st.mean(r["consistency"] for r in g),
                           expl=st.mean(r["explanatory_power"] for r in g),
                           words=st.mean(r["err_words"] for r in g))
        by = {k: {r["image"]: r for r in rows
                  if r["kind"] == k and r["pairing"] == "genuine"}
              for k in kinds}
        if "human" not in by:                    # full coverage: the narrative alone
            n["by_branch"] = {"pipeline": _branch_means(by["pipeline"])}
            n["n_val"] = sum(1 for r in by["pipeline"].values() if r.get("split") == "val")
            n["n_train"] = sum(1 for r in by["pipeline"].values() if r.get("split") == "train")
            n["split"] = {sp: st.mean(r["score"] for r in by["pipeline"].values()
                                      if r.get("split") == sp)
                          for sp in ("val", "train")
                          if any(r.get("split") == sp for r in by["pipeline"].values())}
            v = [r["score"] for r in by["pipeline"].values()]
            n["bands"] = dict(high=sum(1 for x in v if x >= 7),
                              mid=sum(1 for x in v if 5 <= x < 7),
                              low=sum(1 for x in v if x < 5))
            n["by_etype"] = _etype_means(by["pipeline"])
            # how many certified patterns the image carries: a cause that covers more
            # of the image explains more of what the narrative reports
            _recs = json.load(open(os.path.join(d, "image_descriptions.json"), encoding="utf-8"))
            _recs.update(json.load(open(os.path.join(d, "train36",
                                                     "descriptions_train36_full.json"),
                                        encoding="utf-8")))
            acc = {}
            for i, r in by["pipeline"].items():
                k = min(len(_recs.get(i, {}).get("patterns", [])), 4)
                acc.setdefault(k, []).append(r["score"])
            n["by_npatterns"] = {k: (st.mean(v), len(v)) for k, v in sorted(acc.items())}
            out[tag] = n
            continue
        imgs = sorted(set(by["human"]) & set(by["pipeline"]))
        pa = [(by["human"][i]["score"], by["pipeline"][i]["score"]) for i in imgs]
        dif = [b - a for a, b in pa]
        ma, mb = st.mean(a for a, _ in pa), st.mean(b for _, b in pa)
        sa = (sum((a - ma) ** 2 for a, _ in pa) / len(pa)) ** 0.5
        sb = (sum((b - mb) ** 2 for _, b in pa) / len(pa)) ** 0.5
        cov = sum((a - ma) * (b - mb) for a, b in pa) / len(pa)
        n["agree"] = dict(n=len(pa), diff=st.mean(dif),
                          absdiff=st.mean(abs(x) for x in dif),
                          within1=sum(1 for x in dif if abs(x) <= 1),
                          within2=sum(1 for x in dif if abs(x) <= 2),
                          r=(cov / (sa * sb)) if sa and sb else float("nan"),
                          both_high=sum(1 for a, b in pa if a >= 7 and b >= 7),
                          both_low=sum(1 for a, b in pa if a < 7 and b < 7))
        n["worst"] = sorted(((abs(by["pipeline"][i]["score"] - by["human"][i]["score"]),
                              i, by["human"][i]["score"], by["pipeline"][i]["score"])
                             for i in imgs), reverse=True)[:3]
        n["split"] = {sp: st.mean(r["score"] for r in rows
                                  if r["kind"] == "pipeline" and r["pairing"] == "genuine"
                                  and r.get("split") == sp)
                      for sp in ("val", "train")
                      if any(r.get("split") == sp for r in rows)}
        n["by_branch"] = {k: _branch_means({i: by[k][i] for i in imgs}) for k in kinds}
        n["n_val"] = sum(1 for i in imgs if by["human"][i].get("split") == "val")
        n["n_train"] = sum(1 for i in imgs if by["human"][i].get("split") == "train")
        if tag == "synth":
            n["declared_good"] = sum(1 for i in imgs if by["human"][i].get("declared_good"))
        out[tag] = n

    # how the 80 annotated images divide: a cause to judge, errors but no certified
    # slice (nothing is blamed, nothing to judge), or no typed error at all
    mp = {int(k): v.replace(".jpg", "") for k, v in json.load(open(
        os.path.join(ROOT, "_evaldata", "dataset", "voc_patch_matches.json"),
        encoding="utf-8")).items()}
    ann = set()
    for line in open(os.path.join(ROOT, "_evaldata", "dataset", "answer_gts.jsonl"),
                     encoding="utf-8"):
        if line.strip():
            ann.add(mp[json.loads(line)["img_idx"]])
    recs = json.load(open(os.path.join(d, "image_descriptions.json"), encoding="utf-8"))
    recs.update(json.load(open(os.path.join(d, "train36", "descriptions_train36_full.json"),
                               encoding="utf-8")))
    typed = json.load(open(os.path.join(d, "typed_per_image.json"), encoding="utf-8"))
    known = [i for i in ann if i in recs]
    out["coverage"] = dict(
        annotated=len(ann), known=len(known),
        with_cause=sum(1 for i in known if recs[i].get("patterns")),
        no_pattern=sum(1 for i in known
                       if not recs[i].get("patterns") and sum(typed.get(i, {}).values())),
        # the rest: no typed error at all, plus the one image with no segmentation GT
        no_error=len(ann) - sum(1 for i in known if recs[i].get("patterns"))
        - sum(1 for i in known
              if not recs[i].get("patterns") and sum(typed.get(i, {}).values())))
    return out


def load_gallery(run_dir=None):
    """What the HTML gallery of a run carries, counted from the same files it reads.

    Not parsed out of the 39 MB page: the gallery is a rendering of these files,
    so the document quotes the sources and stays true if the page is rebuilt.
    """
    d = run_dir or INST
    n = {}
    nar = json.load(open(os.path.join(d, "error_narratives.json"), encoding="utf-8"))
    n["narratives"] = sum(1 for v in nar.values() if (v.get("text") or "").strip())
    scores = {}
    for f in ("judge_cause_val.json", "judge_cause.json"):
        path = os.path.join(d, f)
        if not os.path.exists(path):
            continue
        for r in json.load(open(path, encoding="utf-8"))["judgements"]:
            if r.get("kind") == "pipeline" and r.get("pairing") == "genuine" and "score" in r:
                scores.setdefault(r["image"], r["score"])
    n["scored_images"] = len(scores)
    pk = pickle.load(open(os.path.join(d, "result_tagged.pkl"), "rb"))
    res = pk["result"]
    pairs = list(getattr(res, "scored", None) or
                 [(ks.slice_, ks.quality) for ks in res.kept])
    seen, cards, with_mean = set(), 0, 0
    for sl, q in pairs:
        if q.slice_id in seen:
            continue
        seen.add(q.slice_id)
        cards += 1
        if any(m.image_id in scores for m in sl.members):
            with_mean += 1
    n["slice_cards"] = cards
    n["slice_cards_scored"] = with_mean
    n["image_cards"] = 80          # regen_gallery.MAX_IMAGES: the cards carry pictures
    return n


EXTRA = out_path("slice_quality_inst_extra")
EXTRA_FULL = out_path("slice_quality_inst_extra_full")
VISUAL = os.path.join(ROOT, "docs", "extra_hyp", "visual_check.json")


def load_extra_hyp(run_dir=None):
    """Numbers for the 'widening the hypothesis vocabulary' sections.

    Sources: the inst slice index (yield of the 7 canonical vs the class-specific
    hypotheses), the two reports of compare_extra_hypotheses.py (on the July
    120-instance sample, and after reground_validated.py on all instances), the
    slice index of the full-population rescore, and the visual check recorded in
    docs/extra_hyp/visual_check.json.
    """
    from slice_quality.describe import _CANONICAL, _GENERIC_EXTRA
    out = {"canonical": list(_CANONICAL), "extra": list(_GENERIC_EXTRA)}
    d = run_dir or INST
    sl = json.load(open(os.path.join(d, "slices_index.json"), encoding="utf-8"))["slices"]
    # the July vocabulary only: once the extra hypotheses are merged into the inst
    # run (6 Oct 2026) their slices sit in the same index and must not be counted
    # as 'class-specific' nor as the 'before' state
    lang = [s for s in sl if s.get("hypothesis") and s["hypothesis"] not in _GENERIC_EXTRA]
    for key, grp in (("canon", [s for s in lang if s["hypothesis"] in _CANONICAL]),
                     ("specific", [s for s in lang if s["hypothesis"] not in _CANONICAL])):
        out[key] = {"evaluated": len(grp), "certified": sum(1 for s in grp if s.get("in_descriptions"))}
    out["specific_examples"] = [s["hypothesis"] for s in lang
                                if s["hypothesis"] not in _CANONICAL][:40]
    out["specific_certified"] = sorted({s["hypothesis"] for s in lang
                                        if s["hypothesis"] not in _CANONICAL and s.get("in_descriptions")})
    out["inst_types"] = dict(Counter(s["dominant_error_type"] for s in lang if s.get("in_descriptions")))
    # the extra study's own slices ('<cls>_extra_<k>'); the same phrases certified in other classes once the rank
    # test also chose what to ground (23 October, '<cls>_pend_<k>') are counted apart
    merged = [s for s in sl if s.get("hypothesis") in _GENERIC_EXTRA and s.get("in_descriptions")
              and "_pend_" not in s["slice_id"]]
    out["merged_pend"] = sorted((_cls(s["slice_id"]), s["hypothesis"]) for s in sl
                                if s.get("hypothesis") in _GENERIC_EXTRA and s.get("in_descriptions")
                                and "_pend_" in s["slice_id"])
    out["merged"] = [{"slice_id": s["slice_id"], "hypothesis": s["hypothesis"], "size": s["size"],
                      "n_images": s["n_images"], "type": s["dominant_error_type"]} for s in merged]
    # what the cause-plausibility judge says about the merged situations (pipeline
    # narrative side, genuine pairs, images whose cause names the hypothesis)
    jv = os.path.join(d, "judge_cause_val.json")
    if merged and os.path.exists(jv):
        rows = [r for r in json.load(open(jv, encoding="utf-8"))["judgements"]
                if r.get("kind") == "pipeline" and r.get("pairing") == "genuine" and "score" in r]
        allm = st.mean(r["score"] for r in rows) if rows else None
        out["merged_plaus"] = {"all": (allm, len(rows))}
        for s in merged:          # the images whose cause names this slice: its class and its phrase
            tag = '%s — "%s"' % (_cls(s["slice_id"]), s["hypothesis"])
            v = [r["score"] for r in rows if tag in (r.get("cause") or "")]
            out["merged_plaus"][s["slice_id"]] = out["merged_plaus"][s["hypothesis"]] = (st.mean(v) if v else None, len(v))

    rep = json.load(open(os.path.join(EXTRA, "extra_hypotheses_report.json"), encoding="utf-8"))
    out["sample"] = rep
    out["n_classes"] = rep["n_classes_grounded"]
    out["pairs"] = rep["n_classes_grounded"] * len(_GENERIC_EXTRA)
    out["sample_systematic"] = sum(len(v["systematic_classes"]) for v in rep["per_hypothesis"].values())
    out["sample_certified"] = rep["totals"]["new_extra_certified"]

    full_rep = os.path.join(EXTRA_FULL, "extra_hypotheses_report.json")
    if os.path.exists(full_rep):
        fr = json.load(open(full_rep, encoding="utf-8"))
        out["full"] = fr
        out["full_held"] = {h: sorted(c for c, ok in v.get("full_reground", {}).items() if ok)
                            for h, v in fr["per_hypothesis"].items()}
        out["full_dropped"] = {h: sorted(c for c, ok in v.get("full_reground", {}).items() if not ok)
                               for h, v in fr["per_hypothesis"].items()}
        out["full_certified"] = fr["totals"]["new_extra_certified"]
    if os.path.exists(VISUAL):
        out["visual"] = json.load(open(VISUAL, encoding="utf-8"))
    return out


DD = out_path("slice_quality_inst_dd")
DD_PILOT = os.path.join(ROOT, "docs", "dd_hyp", "pilot_june.json")
DD_VISUAL = os.path.join(ROOT, "docs", "dd_hyp", "visual_check.json")


def load_dd_hyp():
    """Numbers for the section on SliceLens's data-driven hypotheses (October 2026).

    Sources: outputs/slice_quality_inst_dd/dd_summary.json (written on the server by
    slice_quality/examples/dd_summary.py from the generation records, the groundings and
    the two rescores), the June pilot's hypotheses (docs/dd_hyp/pilot_june.json) looked
    up in the inst slice index, and the visual check in docs/dd_hyp/visual_check.json.
    """
    out = json.load(open(os.path.join(DD, "dd_summary.json"), encoding="utf-8"))
    hy = [h for c in out["classes"] for h in out["per_class"][c]["hypotheses"]]
    out["n_hyp"] = len(hy)
    out["n_sample_sys"] = sum(h["sample_systematic"] for h in hy)
    out["n_sample_cert"] = sum(h["sample_certified"] for h in hy)
    out["n_reground"] = sum(h["full_slope"] is not None for h in hy)
    out["n_cert"] = sum(bool(h["certified"]) for h in hy)
    out["seeds"] = Counter(out["per_class"][c]["seeds_from"] for c in out["classes"])
    out["n_values"] = sum(a["n_values"] for c in out["classes"] for a in out["per_class"][c]["attributes"])
    out["n_captions"] = sum(out["per_class"][c]["n_sampled"] * len(out["per_class"][c]["attributes"])
                            for c in out["classes"])
    # why the slope stays flat: a phrase the VLM confirms on most of the sample describes the
    # usual look of the class, one it confirms on almost none describes a single caption
    flat = [h for h in hy if h["sample_slope"] is not None and h["sample_slope"] < 0.005]
    out["n_flat"] = len(flat)
    out["flat_common"] = sorted((h for h in flat if h["sample_match"] >= h["sample_scored"] / 2),
                                key=lambda h: -h["sample_match"])
    out["flat_rare"] = sorted((h for h in flat if h["sample_match"] <= 5), key=lambda h: h["sample_match"])
    out["light"] = sum(1 for h in hy if any(w in h["query"].lower() for w in
                                            ("light", "shadow", "contrast", "glare", "dim ", "dark")))
    out["collapsed"] = [(c, a["attribute"], max(a["clusters"].values()), a["n_values"])
                        for c in out["classes"] for a in out["per_class"][c]["attributes"]
                        if len(a["clusters"]) == 1 and max(a["clusters"].values()) >= 10]
    pil = json.load(open(DD_PILOT, encoding="utf-8"))["hypotheses"]
    sl = json.load(open(os.path.join(INST, "slices_index.json"), encoding="utf-8"))["slices"]
    lang = {(_cls(s["slice_id"]), s.get("hypothesis")): s for s in sl if s.get("source") == "language"}
    # the generator's hypotheses certified in the official run, where the rank test also chose what to ground
    out["n_cert_official"] = sum(bool(lang.get((c, h["query"]), {}).get("in_descriptions"))
                                 for c in out["classes"] for h in out["per_class"][c]["hypotheses"])
    out["pilot"] = {"classes": sorted(pil), "n": sum(len(v) for v in pil.values()),
                    "evaluated": sum((c, h) in lang for c, v in pil.items() for h in v),
                    "certified": sum(bool(lang.get((c, h), {}).get("in_descriptions")) for c, v in pil.items() for h in v)}
    if os.path.exists(DD_VISUAL):
        out["visual"] = json.load(open(DD_VISUAL, encoding="utf-8"))
    return out


SBD_RUN = out_path("slice_quality_sbd")
EA_RUN = out_path("slice_quality_ea")
EA_SEG = out_path("pascal_ea")


def _cert_situations(run_dir):
    """the certified slices of a run: language ones by (class, hypothesis), cluster ones with
    the phrase most often shown for them in the image descriptions."""
    sl = json.load(open(os.path.join(run_dir, "slices_index.json"), encoding="utf-8"))["slices"]
    cert = [s for s in sl if s.get("in_descriptions")]
    phr = {}
    descs = json.load(open(os.path.join(run_dir, "image_descriptions.json"), encoding="utf-8"))
    for v in descs.values():
        for p in v.get("patterns", []):
            if p.get("branch") == "cluster":
                phr.setdefault(p["slice_id"], Counter())[p["description"]] += 1
    lang = {}
    for s in cert:
        if s.get("source") == "language":
            c = s["slice_id"].split("_")[0]
            lang[(c, s["hypothesis"])] = {"size": s["size"], "n_images": s["n_images"],
                                          "type": s.get("dominant_error_type")}
    clus = [{"slice_id": s["slice_id"], "size": s["size"], "n_images": s["n_images"],
             "type": s.get("dominant_error_type"),
             "phrase": phr[s["slice_id"]].most_common(1)[0][0] if s["slice_id"] in phr else ""}
            for s in cert if s.get("source") == "cluster"]
    return lang, sorted(clus, key=lambda x: -x["size"])


def load_sbd():
    """Numbers for the chapter on SBD (October 2026).

    Sources: outputs/slice_quality_sbd/sbd_summary.json (written on the server by
    slice_quality/examples/doc_numbers_sbd_ea.py: mIoU, the run's headline numbers through
    load_inst_summary, the language-branch funnel), slope_robustness.json
    (slice_quality/examples/slope_robustness.py), and the slice indexes and image
    descriptions of the SBD and inst runs.
    """
    out = json.load(open(os.path.join(SBD_RUN, "sbd_summary.json"), encoding="utf-8"))
    out["robust"] = json.load(open(os.path.join(SBD_RUN, "slope_robustness.json"), encoding="utf-8"))["results"]
    out["lang_sbd"], out["clus_sbd"] = _cert_situations(SBD_RUN)
    out["lang_voc"], out["clus_voc"] = _cert_situations(INST)
    return out


def load_ea():
    """Numbers for the chapter on the planted cause (Pascal-EA material edits of VOC val).

    Sources: outputs/pascal_ea/report.json (pascal_ea_compare.py + pascal_ea_report.py: the
    segmenter on each original/edited pair), outputs/slice_quality_ea/planted_eval.json
    (ea_planted_eval.py: victims and whether a certified slice holds them),
    judge_ea_patterns.json (Gemma: do the patterns capture the edit?), ea_summary.json
    (doc_numbers_sbd_ea.py: mIoU of val and val-EA) and gallery_notes.json.
    """
    out = {"seg": json.load(open(os.path.join(EA_SEG, "report.json"), encoding="utf-8")),
           "planted": json.load(open(os.path.join(EA_RUN, "planted_eval.json"), encoding="utf-8")),
           "miou": json.load(open(os.path.join(EA_RUN, "ea_summary.json"), encoding="utf-8"))}
    jf = os.path.join(EA_RUN, "judge_ea_patterns.json")
    if os.path.exists(jf):
        out["judge"] = json.load(open(jf, encoding="utf-8"))
    out["lang_ea"], out["clus_ea"] = _cert_situations(EA_RUN)
    # victims and their coverage by class and by material (per_image.json of the segmenter run)
    rows = json.load(open(os.path.join(EA_SEG, "per_image.json"), encoding="utf-8"))["rows"]
    meta = {r["stem"]: (r["voc_class"], r["material_v3"]) for r in rows}
    pl = out["planted"]
    cov = set(pl["covered"])
    for key, i in (("by_class", 0), ("by_material", 1)):
        agg = {}
        for v in pl["victims"]:
            a = agg.setdefault(meta.get(v.split("#")[0], ("?", "?"))[i], [0, 0])
            a[0] += v in cov
            a[1] += 1
        out[key] = agg
    # the lines of ea_planted_eval.py's report that carry dictionaries or ratios not stored
    # as fields (its own fixed format)
    import ast
    import re
    for line in (x.strip() for x in pl["report"]):
        if line.startswith("victims' error type in val-EA:"):
            out["victim_types_ea"] = ast.literal_eval(line.split(":", 1)[1].strip())
        elif line.startswith("the same objects on the ORIGINAL image"):
            out["victim_types_orig"] = ast.literal_eval(line.split(":", 1)[1].strip())
        elif line.startswith("CONTROL: survivors"):
            m = re.search(r"(\d+)/(\d+) \(they are errors in val-EA: (\d+)\)", line)
            out["survivors_cov"] = tuple(map(int, m.groups()))
        elif line.startswith("edited images"):
            m = re.search(r"edited images (\d+); subject instance found in (\d+)", line)
            out["n_edited"], out["n_found"] = map(int, m.groups())
        elif line.startswith("-- certified situations: inst"):
            m = re.search(r"inst (\d+) language \+ (\d+) cluster; val-EA (\d+) \+ (\d+)", line)
            out["cert_inst"], out["cert_inst_cl"], out["cert_ea"], out["cert_ea_cl"] = map(int, m.groups())
    if "judge" in out:                  # rubric components, a judgement without patterns counts 0
        js = out["judge"]["judgements"]
        out["judge_means"] = {}
        for pairing in ("genuine", "control", "original"):
            g = [r for r in js if r.get("pairing") == pairing]
            wp = [r for r in g if not r.get("auto")]
            out["judge_means"][pairing] = {
                "n": len(g), "n_with_patterns": len(wp),
                "coverage": st.mean((r.get("coverage") or 0) for r in g),
                "cause": st.mean((r.get("cause") or 0) for r in g),
                "coverage_wp": st.mean(r.get("coverage") or 0 for r in wp) if wp else None,
                "cause_wp": st.mean(r.get("cause") or 0 for r in wp) if wp else None}
    return out


SLOPE_DIR = out_path("slope_permutation")


def load_slope_verdicts():
    """Numbers for the size-aware certification section (October 2026).

    Sources: outputs/slope_permutation/results.json (slice_quality/examples/slope_permutation.py
    PHASE=test + compare: SliceLens's statistic with a permutation null) and results_rank.json
    (PHASE=rank + compare: the AUC verdict with an image-block bootstrap), both computed on the
    saved groundings, the rescores on copies of the inst and SBD runs.
    """
    out = {}
    for tag, f in (("perm", "results.json"), ("rank", "results_rank.json")):
        r = json.load(open(os.path.join(SLOPE_DIR, f), encoding="utf-8"))
        st_ = {}
        for run, rows in r["results"].items():
            for stage in ("sample", "full"):
                g = [x for x in rows if x["stage"] == stage]
                n = len(g)
                st_[(run, stage)] = {
                    "n": n, "old": sum(x["old_sys"] for x in g), "new": sum(x["new_sys"] for x in g),
                    "null_new": sum(x["null_p"] < 0.05 for x in g) / n if n else 0.0,
                    "null_old": (sum(x["null_slope"] >= 0.2 for x in g) / n) if n and "null_slope" in g[0] else None}
        out[tag] = {"stages": st_, "size_check": r.get("size_check"), "compare": r.get("compare", {}),
                    "draws": r.get("B") or r.get("BB")}
    return out


def load_ea_naming():
    """Numbers for the per-object naming section (Pascal-EA, October 2026).

    Sources: outputs/slice_quality_ea/per_instance_naming_tau0.5.json (the 120-instance samples,
    three description rules) and per_instance_naming_all.json (every victim and every edited
    subject the segmenter keeps, rule 'auc', groundings of examples/ground_ea_naming.py).
    """
    d = out_path("slice_quality_ea")
    s = json.load(open(os.path.join(d, "per_instance_naming_tau0.5.json"), encoding="utf-8"))
    a = json.load(open(os.path.join(d, "per_instance_naming_all.json"), encoding="utf-8"))
    rk = json.load(open(os.path.join(SLOPE_DIR, "results_rank.json"), encoding="utf-8"))["results"]["VOC val"]
    toy = "a toy, statue or model of an object"
    admitted = {r["cls"] for r in rk if r["stage"] == "sample" and r["query"] == toy and r["new_sys"]}
    tried = {r["cls"] for r in rk if r["stage"] == "sample" and r["query"] == toy}
    return {"sample": s["results"], "all": a["results"]["auc_all"], "tau": s["tau"],
            "cause_words": s["cause_words"], "examples": a.get("examples", []),
            "toy_not_admitted": sorted(tried - admitted)}


def load_official_validations(run_dir=None):
    """The validations re-run on the official inst run after the AUC verdict became official
    (outputs/chain_night_inst.sh, 2026-10-21). Sources in outputs/slice_quality_inst:
    split_half.json (split_half.py), eval_precision.json (eval_description_precision.py),
    corruption_results_p25/p50.json and night/corruption_p*.log (corruption_experiment.py with
    TAG_MODE=verify; rule B is only in the log), model2/slices_index.json and model2/dd_auc_full.json
    (the second segmenter typed like inst: run_second_model_cluster.py, reground_validated.py on its
    objects, dd_auc.py with RANK_COPY, run_second_model_language.py) and night/model2_cluster.log.
    """
    import re
    d = run_dir or INST
    out = {"split": json.load(open(os.path.join(d, "split_half.json"), encoding="utf-8")),
           "prec": json.load(open(os.path.join(d, "eval_precision.json"), encoding="utf-8"))}
    out["corr"] = {}
    for dose in ("p25", "p50"):
        r = json.load(open(os.path.join(d, f"corruption_results_{dose}.json"), encoding="utf-8"))
        log = open(os.path.join(d, "night", f"corruption_{dose}.log"), encoding="utf-8").read()
        m = re.search(r"=== RULE B.*?\ninjected: (\d+) \| found in archived grounding: (\d+) \| "
                      r"pass confidence gate \(>= ([\d.]+)\): (\d+)", log, re.S)
        r["rule_b"] = dict(zip(("injected", "found", "gate", "passed"),
                               (int(m.group(1)), int(m.group(2)), float(m.group(3)), int(m.group(4))))) if m else None
        vb = re.findall(r"verdict breakdown of ALL host slices \(incl\. REFINE/DISCARD\): (\{.*?\})", log)
        r["host_verdicts"] = __import__("ast").literal_eval(vb[-1]) if vb else {}
        out["corr"][dose] = r
    key = lambda x: (_cls(x["slice_id"]), x.get("hypothesis"))
    off = [x for x in json.load(open(os.path.join(d, "slices_index.json"), encoding="utf-8"))["slices"]
           if x.get("in_descriptions")]
    m2 = [x for x in json.load(open(os.path.join(d, "model2", "slices_index.json"), encoding="utf-8"))["slices"]
          if x.get("in_descriptions")]
    ol = {key(x) for x in off if x["source"] == "language"}
    ml = {key(x) for x in m2 if x["source"] == "language"}
    auc2 = json.load(open(os.path.join(d, "model2", "dd_auc_full.json"), encoding="utf-8"))
    log = open(os.path.join(d, "night", "model2_cluster.log"), encoding="utf-8").read()
    m = re.search(r"(\d+) errors, (\d+) reference", log)
    out["model2"] = {"official_language": len(ol), "recertified": len(ol & ml), "not_recertified": sorted(ol - ml),
                     "cluster_certified": sum(1 for x in m2 if x["source"] == "cluster"),
                     "auc_sys": sum(bool(r["auc_sys"]) for r in auc2), "n_regrounded": len(auc2),
                     "n_err": int(m.group(1)) if m else None, "n_ref": int(m.group(2)) if m else None}
    # the same validations as measured before, on the reference run (outputs/slice_quality, §6)
    ref = out_path("slice_quality")
    ro = [x for x in json.load(open(os.path.join(ref, "slices_index.json"), encoding="utf-8"))["slices"]
          if x.get("in_descriptions") and x["source"] == "language"]
    rm = [x for x in json.load(open(os.path.join(ref, "model2", "slices_index.json"), encoding="utf-8"))["slices"]
          if x.get("in_descriptions") and x["source"] == "language"]
    out["before"] = {"split": json.load(open(os.path.join(ref, "split_half.json"), encoding="utf-8")),
                     "prec": json.load(open(os.path.join(ref, "eval_precision.json"), encoding="utf-8")),
                     "corr": json.load(open(os.path.join(ref, "corruption_results.json"), encoding="utf-8")),
                     "model2": {"official_language": len({key(x) for x in ro}),
                                "recertified": len({key(x) for x in ro} & {key(x) for x in rm})}}
    # hypotheses significant for the rank test on the 120-instance sample but never re-grounded on the
    # whole class (the re-grounding followed the slope rule): the rank verdict cannot reach them
    rr = json.load(open(os.path.join(SLOPE_DIR, "results_rank.json"), encoding="utf-8"))["results"]
    out["pending"] = {}
    for run, rows in rr.items():
        full = {(x["cls"], x["query"]) for x in rows if x["stage"] == "full"}
        out["pending"][run] = sum(1 for x in rows if x["stage"] == "sample" and x["new_sys"]
                                  and (x["cls"], x["query"]) not in full)
    return out


SBD_DD = out_path("slice_quality_sbd_dd")


def load_sbd_dd():
    """Numbers for SliceLens's data-driven hypotheses on SBD (outputs/chain_dd_sbd.sh, chain_dd_sbd_full.sh).

    Sources, in outputs/slice_quality_sbd_dd: dd_auc_cc.json and dd_auc_full.json (dd_auc.py: slope rule and
    rank test on the 120-instance sample and on the whole classes) and dd_hypotheses_report.json
    (compare_dd_hypotheses.py after the whole-class re-grounding: the copy of the SBD run with and without them).
    """
    cc = json.load(open(os.path.join(SBD_DD, "dd_auc_cc.json"), encoding="utf-8"))
    full = json.load(open(os.path.join(SBD_DD, "dd_auc_full.json"), encoding="utf-8"))
    cmp_ = json.load(open(os.path.join(SBD_DD, "dd_hypotheses_report.json"), encoding="utf-8"))
    return {"n_hyp": len(cc), "n_classes": len({r["cls"] for r in cc}),
            "sample_slope": sum(bool(r["slope_sys"]) for r in cc), "sample_auc": sum(bool(r["auc_sys"]) for r in cc),
            "n_full": len(full), "full_slope": sum(bool(r["slope_sys"]) for r in full),
            "full_auc": sum(bool(r["auc_sys"]) for r in full),
            "full_median_auc": st.median(r["auc"] for r in full),
            "full_slope_queries": [(r["cls"], r["query"]) for r in full if r["slope_sys"]],
            "cmp": cmp_}


EA_DD = out_path("slice_quality_ea_dd")
EA_DD_BASE = out_path("slice_quality_ea_dd_base")


def load_ea_dd():
    """Numbers for SliceLens's data-driven hypotheses on val-EA (outputs/chain_dd_ea.sh, October 2026).

    Sources, all in outputs/slice_quality_ea_dd: dd_hypotheses_report.json (compare_dd_hypotheses.py:
    the two rescores of copies of the val-EA pickle, without and with the data-driven groundings),
    dd_auc_cc.json (dd_auc.py: slope rule and rank test on the 120-instance sample),
    dd_ea_summary.json (dd_ea_summary.py: material words in the phrases and in the captions), and
    planted_eval.json of both copies (ea_planted_eval.py; the copy without the data-driven
    hypotheses is in outputs/slice_quality_ea_dd_base).
    """
    out = {"cmp": json.load(open(os.path.join(EA_DD, "dd_hypotheses_report.json"), encoding="utf-8")),
           "auc": json.load(open(os.path.join(EA_DD, "dd_auc_cc.json"), encoding="utf-8")),
           "sum": json.load(open(os.path.join(EA_DD, "dd_ea_summary.json"), encoding="utf-8"))}
    pl = {k: json.load(open(os.path.join(d, "planted_eval.json"), encoding="utf-8"))
          for k, d in (("base", EA_DD_BASE), ("dd", EA_DD))}
    out["victims"] = len(pl["dd"]["victims"])
    out["covered_base"], out["covered_dd"] = len(pl["base"]["covered"]), len(pl["dd"]["covered"])
    pc = out["sum"]["per_class"]
    out["n_classes"] = len(pc)
    out["n_hyp"] = sum(len(v["hypotheses"]) for v in pc.values())
    out["seeds_fallback"] = sum(v["seeds_from"] == "fallback" for v in pc.values())
    out["attributes"] = next(iter(pc.values()))["attributes"]
    out["material_phrases"] = [(c, h) for c in sorted(pc) for h in pc[c]["material_phrases"]]
    out["captions"] = {k: [sum(v["captions"][k][i] for v in pc.values()) for i in (0, 1)]
                       for k in ("edited", "untouched")}
    out["n_slope_sys"] = sum(bool(r["slope_sys"]) for r in out["auc"])
    out["n_auc_sys"] = sum(bool(r["auc_sys"]) for r in out["auc"])
    out["median_auc"] = st.median(r["auc"] for r in out["auc"])
    out["auc_by_query"] = {(r["cls"], r["query"]): r for r in out["auc"]}
    return out


def load_ea_repair(seg_dir=None):
    """Numbers for the Pascal-EA repair experiment (SEGIMPROVER, October 2026).

    Sources: SEGIMPROVER/runs/ea/report.json (ea_repair_report.py: paired differences by seed,
    criteria fixed in advance), runs/ea/model_<arc>_s<seed>_eval.json and start_eval.json
    (evaluate_ea.py), groups/ea_split.json (ea_split.py).
    """
    import glob
    d = seg_dir or SEG
    rep = json.load(open(os.path.join(d, "runs", "ea", "report.json"), encoding="utf-8"))
    ev = {}
    for f in glob.glob(os.path.join(d, "runs", "ea", "model_*_s*_eval.json")):
        pol, seed = os.path.basename(f)[len("model_"):-len("_eval.json")].rsplit("_s", 1)
        ev.setdefault(pol, []).append(json.load(open(f, encoding="utf-8")))
    mean = lambda pol, grp, key: st.mean(e[grp][key] for e in ev[pol])
    keys = [("recall", "B_victims_edited"), ("miou", "unedited"), ("recall", "B_victims_original"),
            ("recall", "B_others_edited"), ("miou", "B_edited")]
    start = json.load(open(os.path.join(d, "runs", "ea", "start_eval.json"), encoding="utf-8"))
    sp = json.load(open(os.path.join(d, "groups", "ea_split.json"), encoding="utf-8"))
    v = set(sp["victims"])
    return {"paired": rep["paired"], "seeds": {p: len(e) for p, e in ev.items()},
            "means": {p: {f"{g}.{k}": mean(p, g, k) for g, k in keys} for p in ev},
            "start": {f"{g}.{k}": start[g][k] for g, k in keys},
            "n_A": len(sp["A"]), "n_B": len(sp["B"]), "victims_A": sum(s in v for s in sp["A"]),
            "victims_B": start.get("n_victims_B")}


GESD_DOCS = os.path.join(ROOT, "docs", "gesd")
# the published GH-ESD figure for COCO instance segmentation (Precision@10, arXiv 2512.24592 v4,
# supplementary table of best predicted slice per expert slice): a literature value, not a run file
GHESD_SEG_P10 = 0.805
# point A's expected type per word of the expert slice name (gesd_block1.py EXPECTED, fixed in advance)
GESD_WORDS = [("merged", "merged", {"merge"}), ("missed", "missed", {"fn"}),
              ("mis-segmented as", "mis-segmented as (hallucination)", {"fp", "confusion"}),
              ("over-segmented", "over-segmented / expanded", {"extent", "merge"}),
              ("expanded", "over-segmented / expanded", {"extent", "merge"}),
              ("boundary unclear", "boundary unclear", {"boundary", "extent"}),
              ("under-segmented", "under-segmented", {"extent", "boundary", "split"})]


def load_gesd():
    """Numbers for the GESD chapter (the benchmark published with SliceLens / GH-ESD, Oct 2026).

    Sources (local copies in docs/gesd/): baselines.json (outputs/gesd_baselines.py: Precision@10
    of groupings without hypotheses or VLM); block1_vs_experts.json (point A,
    slice_quality/examples/gesd_block1.py); pointB_keywords.json (the naming rules, fixed before the
    run); hypotheses/ (the 36 hypotheses per class and the four JSON repairs); pointB_cc.json and
    pointB_full.json (slice_quality/examples/gesd_pointB.py on the sample and on whole classes);
    slices_index_gesd_full.json (the final rescore).
    """
    D = GESD_DOCS
    J = lambda f: json.load(open(os.path.join(D, f), encoding="utf-8"))
    out = {"ghesd_p10": GHESD_SEG_P10, "baselines": J("baselines.json")}
    # ---- point A: Block 1 against the expert word
    a = J("block1_vs_experts.json")
    rows = [r for r in a["rows"] if r["slice"] != "NA"]
    na = [r for r in a["rows"] if r["slice"] == "NA"]
    def word(s):
        for w, g, t in GESD_WORDS:
            if w in s.lower():
                return g, t
        return None, set()
    groups = {}
    for r in rows:
        g, t = word(r["slice"])
        e = groups.setdefault(g, {"n": 0, "block1": 0, "naive": 0, "slices": set(), "expected": sorted(t)})
        e["n"] += 1
        e["block1"] += r["type"] in t
        nat = {"boundary/extent": {"boundary", "extent"}, "fn": {"fn"}, "fp": {"fp"}, "none": set()}[r["naive"]]
        e["naive"] += bool(nat & t)
        e["slices"].add(r["slice"])
    per_slice = {s: dict(v, rate=v["agree"] / v["n"]) for s, v in a["per_slice"].items()}
    for g in groups.values():
        g["slices"] = len(g["slices"])
    rng = {}
    for s, v in per_slice.items():
        g = word(s)[0]
        rng.setdefault(g, []).append(v["rate"])
    for g, v in rng.items():
        groups[g]["range"] = (min(v), max(v))
    missed_pred = [r for r in rows if "missed" in r["slice"].lower() and r.get("iou") is not None]
    paper = per_slice.get("Paper mis-segmented as book", {})
    laptop = [r for r in rows if r["slice"].startswith("Laptop keyboard") and r["type"] != "merge"
              and r.get("max_cov_other") is not None]
    post = [("merged", {"merge", "extent"}), ("missed", {"fn"}), ("mis-segmented as", {"fp", "confusion"}),
            ("over-segmented", {"extent", "merge"}), ("expanded", {"extent", "merge", "boundary"}),
            ("boundary unclear", {"boundary", "extent"}), ("under-segmented", {"extent", "boundary", "split", "fn"})]
    post_ok = sum(r["type"] in next(t for w, t in post if w in r["slice"].lower()) for r in rows)
    under = [r for r in rows if "under-segmented" in r["slice"].lower()]
    out["pointA"] = {
        "n": a["overall"]["n"], "block1": a["overall"]["block1"], "naive": a["overall"]["naive"],
        "groups": groups, "per_slice": per_slice,
        "na_n": len(na), "na_correct": sum(r["type"] == "correct" for r in na),
        "na_merge": sum(r["type"] == "merge" for r in na),
        "n_typed": len(a["rows"]),
        "images": len({r["image"] for r in a["rows"]}),
        "missed_with_pred": len(missed_pred), "missed_with_pred_iou50": sum(r["iou"] >= 0.5 for r in missed_pred),
        "paper_books": paper.get("types", {}).get("correct", 0), "paper_n": paper.get("n", 0),
        "under_n": len(under), "under_no_pred": sum(r["naive"] == "fn" for r in under),
        "laptop_partial": (sum(0.1 <= r["max_cov_other"] < 0.5 for r in laptop), len(laptop),
                           st.median(r["max_cov_other"] for r in laptop) if laptop else None),
        "posthoc": post_ok}
    # ---- point B: vocabulary (level 1), from the hypothesis files and the fixed rules
    kw = J("pointB_keywords.json")
    out["rules_fixed"] = kw["fixed"]
    try:
        sys.path.insert(0, ROOT)
        from slice_quality.describe import _GENERIC_EXTRA as extra
    except Exception:
        extra = []
    canon = ["a small object", "an object occluded by another object", "an object that blends into its background",
             "an object cut off at the image border", "an object with thin or elongated parts",
             "several overlapping objects of the same type", "an object that looks similar to a different object category"]
    H = os.path.join(D, "hypotheses")
    vocab, n_hyp = {}, Counter()
    for s, spec in kw["slices"].items():
        c = spec["class"]
        dd = [l.strip() for l in open(os.path.join(H, f"dd_{c}.txt"), encoding="utf-8") if l.strip() and not l.startswith("#")]
        comb = [l.strip() for l in open(os.path.join(H, f"combined_{c}.txt"), encoding="utf-8") if l.strip() and not l.startswith("#")]
        hits = {}
        for q in comb + [q for q in extra if q not in comb]:
            src = "canonical" if q in canon else "extra" if q in extra else "data-driven" if q in dd else "gemma"
            if any(k in q.lower() for k in spec["keywords"]):
                hits.setdefault(src, []).append(q)
        vocab[s] = {"class": c, "hits": hits}
    for c in {v["class"] for v in vocab.values()}:
        comb = [l.strip() for l in open(os.path.join(H, f"combined_{c}.txt"), encoding="utf-8") if l.strip() and not l.startswith("#")]
        dd = {l.strip() for l in open(os.path.join(H, f"dd_{c}.txt"), encoding="utf-8") if l.strip() and not l.startswith("#")}
        n_hyp["canonical"] += sum(q in canon for q in comb)
        n_hyp["gemma"] += sum(q not in canon and q not in dd for q in comb)
        n_hyp["data-driven"] += sum(q in dd for q in comb)
        n_hyp["extra"] += len([q for q in extra if q not in comb])
    out["vocab"] = vocab
    out["n_hyp"] = dict(n_hyp)
    out["n_classes"] = len({v["class"] for v in vocab.values()})
    out["repairs"] = {f[len("repair_"):-5]: J(os.path.join("hypotheses", f)) for f in os.listdir(H) if f.startswith("repair_")}
    # ---- point B: the sample and the whole classes
    for stage in ("cc", "full"):
        r = J(f"pointB_{stage}.json")
        v = r["verdicts"]
        out[stage] = {"n_hyp": len(v), "slope": sum(x["slope_sys"] for x in v), "auc": sum(x["auc_sys"] for x in v),
                      "both": sum(x["slope_sys"] and x["auc_sys"] for x in v),
                      "by_src": {s: (sum(x["src"] == s for x in v), sum(x["slope_sys"] for x in v if x["src"] == s),
                                     sum(x["auc_sys"] for x in v if x["src"] == s))
                                 for s in ("canonical", "gemma", "data-driven", "extra")},
                      "level2": r["level2"], "level2b": r["level2b"], "level3": r["level3"],
                      "ours": Counter(o[0] for o in r["ours"])}
    full = J("slices_index_gesd_full.json")["slices"]
    out["certified_full"] = Counter(s["source"] for s in full if s.get("in_descriptions"))
    # SliceLens against the pipeline on GESD with the benchmark's measure (gesd_cmp.py, 23 October)
    if os.path.exists(os.path.join(D, "gesd_cmp.json")):
        out["cmp"] = J("gesd_cmp.json")
    return out


def load_inst_summary(run_dir=None):
    """The headline numbers of the reference run of the thesis (inst), for the
    summary and the results chapter: population, certified slices per branch,
    images described, and TRUE coverage -- the union of the instances of every
    slice that ships in the descriptions, both branches (the definition fixed in
    the second-model study; res.kept alone is cluster-only)."""
    d = run_dir or INST
    pk = pickle.load(open(os.path.join(d, "result_tagged.pkl"), "rb"))
    res, err, ref = pk["result"], pk["errors"], pk["reference"]
    descs = json.load(open(os.path.join(d, "image_descriptions.json"), encoding="utf-8"))
    used = {p["slice_id"] for v in descs.values() for p in v.get("patterns", [])}
    seen, cl, la = set(), set(), set()
    for s, q in res.scored:
        if q.slice_id in seen or q.slice_id not in used:
            continue
        seen.add(q.slice_id)
        (cl if s.source == "cluster" else la).update(m.instance_id for m in s.members)
    n = dict(n_err=len(err), n_ref=len(ref), tax=dict(Counter(e.seg_subtype.value for e in err)),
             images=len(descs), with_pattern=sum(1 for v in descs.values() if v.get("patterns")),
             cov_union=len(cl | la), cov_cluster=len(cl), cov_language=len(la), overlap=len(cl & la))
    n["slices"] = _slices(os.path.join(d, "slices_index.json"))
    n["situational_images"] = sum(1 for v in descs.values()
                                  if any("SliceLens" in p["description"] for p in v.get("patterns", [])))
    return n


def load_claims(run_dir=None):
    """The length-neutral facts score (judge_claims.py) and its length checks."""
    p = os.path.join(run_dir or INST, "judge_claims.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p, encoding="utf-8"))
    nf = [len(v["facts"]) for v in d["facts"].values()]
    rows = d["judgements"]
    verd = {}
    for s in ("voc", "synth"):
        for kind in ("narrative", "pattern"):
            c = Counter(v for r in rows if r.get("rubric") == "facts" and r.get("pairing") == "genuine"
                        and r.get("set") == s and r.get("kind") == kind for v in r.get("verdicts", []))
            tot = sum(c.values())
            if tot:
                verd[f"{s}_{kind}"] = {k: c[k] / tot for k in ("stated", "partial", "contradicted", "absent")}
    return dict(summary=d["summary"], duplicate=d["duplicate"], correlation=d["correlation"],
                summary_all=d.get("summary_all", {}), v1=d.get("v1", {}),
                manual=(json.load(open(os.path.join(run_dir or INST, "judge_claims_manual.json"),
                                       encoding="utf-8"))
                        if os.path.exists(os.path.join(run_dir or INST, "judge_claims_manual.json"))
                        else None),
                fact_types=d.get("fact_types", {}),
                split=d["split"], n_annotations=len(nf), facts_mean=st.mean(nf), facts_min=min(nf),
                facts_max=max(nf), n_judgements=len(rows),
                n_valid=sum(1 for r in rows if "score" in r), verdicts=verd)


SEG = os.environ.get("SEGIMPROVER", r"<home>\Desktop\SEGIMPROVER")
T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23,
       11: 2.20, 12: 2.18, 13: 2.16, 14: 2.14, 15: 2.13, 16: 2.12, 17: 2.11, 18: 2.10, 19: 2.09,
       20: 2.09, 21: 2.08, 22: 2.07, 23: 2.07, 24: 2.06, 25: 2.06, 26: 2.06, 27: 2.05, 28: 2.05,
       29: 2.05, 30: 2.04}


def load_segimprover(seg_dir=None):
    """Numbers for the 'using the diagnosis to improve the segmenter' chapter.

    Read from the SEGIMPROVER folder (runs/*_eval.json, diag/*/index.json,
    groups/, the arm-A config, rediag_compare.json, report_assets/picks.json):
    the same files report_segimprover.docx is built from.
    """
    import glob
    import math
    import re as _re
    d = seg_dir or SEG
    J = lambda *p: json.load(open(os.path.join(d, *p), encoding="utf-8"))

    def arc(patterns):
        out = {}
        for pat in patterns:
            for f in glob.glob(os.path.join(d, pat)):
                m = _re.search(r"_s(\d+)[/\\]", f)
                out[int(m.group(1)) if m else 0] = json.load(open(f, encoding="utf-8"))
        return dict(sorted(out.items()))

    def paired(a, b, key="all"):
        seeds = sorted(set(a) & set(b))
        v = [b[s][key]["miou"] - a[s][key]["miou"] for s in seeds]
        m = st.mean(v) if v else float("nan")
        sd = st.stdev(v) if len(v) > 1 else 0.0
        t = m / (sd / math.sqrt(len(v))) if len(v) > 1 and sd > 0 else float("nan")
        crit = T95.get(len(v) - 1, float("nan"))
        return dict(seeds=seeds, d=v, mean=m, sd=sd, t=t, crit=crit, signif=abs(t) > crit,
                    same_sign=bool(v) and (all(x > 0 for x in v) or all(x < 0 for x in v)))

    def needed(P):
        if not P["sd"] or not P["mean"]:
            return None
        k = abs(P["mean"]) / P["sd"]
        return next((n for n in range(2, 31) if k * math.sqrt(n) > T95[n - 1]), None)

    mi = lambda ev, key="all": [e[key]["miou"] for e in ev.values()]
    ms = lambda v: (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0)
    out = {}
    diag = J("diag", "train", "index.json")
    per = diag["per_image"]
    out["diag"] = dict(n_images=diag["n_images"], by_type=diag["errors_by_subtype"],
                       px=diag["pixels_by_code"], n_inst=sum(diag["errors_by_subtype"].values()),
                       no_err=sum(1 for r in per.values() if r["n_errors"] == 0),
                       median=st.median(r["n_errors"] for r in per.values()),
                       img_with={t: sum(1 for r in per.values() if r["by_subtype"].get(t, 0) > 0)
                                 for t in diag["errors_by_subtype"]})
    out["groups"] = {}
    for s in ("border", "occlusion", "same_class_overlap", "thin"):
        g = J("groups", f"groups_{s}.json")["splits"]
        out["groups"][s] = dict(train=len(g["train"]["in_group"]), n_train=g["train"]["n_images"],
                                val=set(g["val"]["in_group"]), n_val=g["val"]["n_images"])
    A1C = arc(["runs/same_class_overlap/model_control_final_eval.json"])
    A1R = arc(["runs/same_class_overlap/model_repair_final_eval.json"])
    AAC = arc(["runs/errorweighted/model_control_final_eval.json", "runs/ew_s*/model_control_final_eval.json"])
    AAR = arc(["runs/errorweighted/model_repair_final_eval.json", "runs/ew_s*/model_repair_final_eval.json"])
    A2C = arc(["runs/border_s*/model_control_final_eval.json"])
    A2O = arc(["runs/border_s*/model_objcentre_final_eval.json"])
    A2R = arc(["runs/border_s*/model_repair_final_eval.json"])
    out["arm1"] = {k: {"control": A1C[0][k]["miou"], "repair": A1R[0][k]["miou"]}
                   for k in ("all", "group", "rest")} if A1C and A1R else None
    out["armA"] = dict(per_seed=[(s, AAC[s]["all"]["miou"], AAR[s]["all"]["miou"]) for s in sorted(set(AAC) & set(AAR))],
                       control=ms(mi(AAC)), repair=ms(mi(AAR)), diff=paired(AAC, AAR),
                       diff_group=paired(AAC, AAR, "group"))
    cfg = J("runs", "errorweighted", "config_repair.json")
    out["armA"]["weights"] = [(n, share, target, alpha) for n, _px, share, target, alpha in cfg["report"]]
    seeds = sorted(set(A2C) & set(A2O) & set(A2R))
    out["arm2"] = dict(per_seed=[(s, A2C[s]["all"]["miou"], A2O[s]["all"]["miou"], A2R[s]["all"]["miou"]) for s in seeds],
                       control=ms(mi(A2C)), objcentre=ms(mi(A2O)), repair=ms(mi(A2R)),
                       frame=paired(A2C, A2O), cut=paired(A2O, A2R), total=paired(A2C, A2R),
                       total_group=paired(A2C, A2R, "group"), total_rest=paired(A2C, A2R, "rest"),
                       cut_group=paired(A2O, A2R, "group"), cut_rest=paired(A2O, A2R, "rest"),
                       n_group=A2C[seeds[0]]["n_group"], n_val=A2C[seeds[0]]["n_val"])
    first3 = seeds[:3]
    sub = lambda ev: {s: ev[s] for s in first3}
    out["arm2"]["first3"] = dict(seeds=first3, total=paired(sub(A2C), sub(A2R)), cut=paired(sub(A2O), sub(A2R)),
                                 frame=paired(sub(A2C), sub(A2O)))
    out["arm2"]["later"] = seeds[3:]
    out["arm2"]["needed"] = needed(out["arm2"]["total"])
    sds = [ms(mi(X))[1] for X in (AAC, AAR, A2C, A2O, A2R) if len(X) > 1]
    out["noise"] = (min(sds), max(sds))
    out["ctrl_gap"] = abs(ms(mi(AAC))[0] - ms(mi(A2C))[0])
    for key, (a, b) in (("rediag_arm1", ("val_mobilenet_arm1_control_pred", "val_mobilenet_arm1_repair_pred")),
                        ("rediag_armA", ("val_mobilenet_ew_control_pred", "val_mobilenet_ew_repair_pred"))):
        pa, pb = os.path.join(d, "diag", a, "index.json"), os.path.join(d, "diag", b, "index.json")
        if os.path.exists(pa) and os.path.exists(pb):
            ca, cb = J("diag", a, "index.json")["errors_by_subtype"], J("diag", b, "index.json")["errors_by_subtype"]
            out[key] = dict(total=(sum(ca.values()), sum(cb.values())), fp=(ca["fp"], cb["fp"]))
    rp = os.path.join(d, "rediag_compare.json")
    if os.path.exists(rp):
        s = json.load(open(rp, encoding="utf-8"))["summary"]
        out["rediag"] = dict(summary=s, edge_px=json.load(open(rp, encoding="utf-8"))["edge_px"])
        if "objcentre" in s and "repair" in s:
            de = [(y - x) / x for x, y in zip(s["objcentre"]["edge"], s["repair"]["edge"])]
            di = [(y - x) / x for x, y in zip(s["objcentre"]["inner"], s["repair"]["inner"])]
            out["rediag"].update(edge=st.mean(de), inner=st.mean(di), met=st.mean(de) < st.mean(di),
                                 per_seed_met=sum(1 for x, y in zip(de, di) if x < y), n=len(de),
                                 noise=max(ms(s[a]["edge"])[1] / ms(s[a]["edge"])[0] for a in s))
    out["picks"] = J("report_assets", "picks.json")
    return out


def load_slicelens_cmp():
    """SliceLens alone against the pipeline on the same data, and the slope rule against the AUC test
    (outputs/chain_slicelens_cmp.sh, 21 October). Pieces not yet computed come back as None.

    sim    planted effects (slope_vs_auc_sim.py): table rows {b, n, slope, auc, auc_mean}, and the
           whole-class phrases the slope rule certifies although AUC < 0.5 ("real")
    ea     Pascal-EA victims by branch (ea_branch_coverage.py)
    facts  facts judge on the images with both texts: ours (narrative) vs SliceLens-style (pattern)
    rubric the rubric judge on the same pairs (judge_descriptions.py), when done
    disagree  where the two tests disagree on VOC (disagreement_report.py), when done
    gesd   the expert slices whose naming phrase each test flags (docs/gesd/pointB_full.json)
    """
    O, C = None, "slice_quality_slcmp"          # O: the outputs folder, wherever out_path() finds a name
    J = lambda *p: (lambda f: json.load(open(f, encoding="utf-8")) if os.path.exists(f) else None)(
        out_path(*[x for x in p if x is not None]))
    out = {"sim": J(O, "slope_vs_auc_sim.json"), "ea": J(O, "slice_quality_ea", "branch_coverage.json"),
           "disagree": J(O, "slope_vs_auc_disagreement.json"),
           # night of 21-22 October (chain_night_pending.sh): Pascal-EA rescored with the rank test, and every
           # SliceLens rule (first and final version of the paper) in a larger simulation (slope_rules_sim.py)
           "ea_auc": J(O, "slice_quality_ea_auc", "branch_coverage.json"), "rules": J(O, "slope_rules_sim.json"),
           # 23 October: every difference of the comparison tested as paired (slicelens_cmp_stats.py)
           "stats": J(O, "slicelens_cmp_stats.json")}
    if out["sim"]:
        cell = {(r["b"], r["n"]): r for r in out["sim"]["table"]}
        if out["rules"]:              # the larger simulation replaces the first one's rates (same design)
            rc = {(r["b"], r["n"]): r for r in out["rules"]["table"]}
            cell = {k: dict(v, slope=rc[k]["v1"], auc=rc[k]["AUC"], draws=rc[k]["draws"]) if k in rc else v
                    for k, v in cell.items()}
            out["rules"]["cell"] = rc
        out["sim"]["cell"] = cell
        for ds in ("VOC", "SBD"):
            rr = [r for r in out["sim"]["real"] if r["ds"] == ds and r["auc"] is not None]
            ss = [r for r in rr if r["slope_sys"]]
            out["sim"][ds] = dict(n=len(rr), slope_sys=len(ss), inverted=[r for r in ss if r["auc"] < 0.5])
    rows = []
    for k in (0, 1):
        sh = J(C, f"judge_claims_shard{k}.json")
        if sh:
            rows += sh["judgements"]
    if rows:
        rows = [r for r in rows if r.get("rubric") == "facts" and "score" in r and r["set"] == "voc"]
        g = lambda kind, pairing: {r["image"]: r for r in rows if r["kind"] == kind and r["pairing"] == pairing}
        NG, PG, NC, PC = g("narrative", "genuine"), g("pattern", "genuine"), g("narrative", "control"), g("pattern", "control")
        common = sorted(set(NG) & set(PG))

        def verd(G):
            c = Counter(v for i in common for v in G[i]["verdicts"])
            t = sum(c.values())
            return {k: c[k] / t for k in ("stated", "partial", "contradicted", "absent")}
        out["facts"] = dict(n=len(common), ours=st.mean(NG[i]["score"] for i in common),
                            ours_ctrl=st.mean(NC[i]["score"] for i in common if i in NC),
                            sl=st.mean(PG[i]["score"] for i in common),
                            sl_ctrl=st.mean(PC[i]["score"] for i in common if i in PC),
                            ours_v=verd(NG), sl_v=verd(PG),
                            ours_wins=sum(1 for i in common if NG[i]["score"] > PG[i]["score"]),
                            sl_wins=sum(1 for i in common if NG[i]["score"] < PG[i]["score"]))
    # the same SliceLens-style text built from the official run (rank test): judged alone, same facts, same
    # control derangement (judge_claims.py KINDS=pattern; judge_descriptions.py with an empty narrative)
    ca = J(O, "slice_quality_slcmp_auc", "judge_claims_shard0.json")
    if ca and out.get("facts"):
        pr = [r for r in ca["judgements"] if r.get("rubric") == "facts" and "score" in r and r["set"] == "voc"
              and r["kind"] == "pattern"]
        PG = {r["image"]: r["score"] for r in pr if r["pairing"] == "genuine"}
        PC = {r["image"]: r["score"] for r in pr if r["pairing"] == "control"}
        out["facts_auc"] = dict(n=len(PG), sl=st.mean(PG.values()), sl_ctrl=st.mean(PC.values()) if PC else None)
    ra = J(O, "slice_quality_slcmp_auc", "judge_descriptions.json")
    if ra and ra.get("judgements"):
        ok = [r for r in ra["judgements"] if "score" in r and r["kind"] == "pattern"]
        G = [r["score"] for r in ok if r["pairing"] == "genuine"]
        Cc = [r["score"] for r in ok if r["pairing"] == "control"]
        out["rubric_auc"] = dict(n=len(G), sl=st.mean(G), sl_ctrl=st.mean(Cc) if Cc else None)
    rb = J(C, "judge_descriptions.json")
    if rb and rb.get("judgements"):
        ok = [r for r in rb["judgements"] if "score" in r]
        gi = lambda kind, pairing: {r["image"]: r["score"] for r in ok if r["kind"] == kind and r["pairing"] == pairing}
        NG, PG, NC, PC = gi("narrative", "genuine"), gi("pattern", "genuine"), gi("narrative", "control"), gi("pattern", "control")
        common = sorted(set(NG) & set(PG))
        if common and len(common) == len(PG):
            out["rubric"] = dict(n=len(common), ours=st.mean(NG[i] for i in common),
                                 ours_ctrl=st.mean(NC[i] for i in common if i in NC),
                                 sl=st.mean(PG[i] for i in common), sl_ctrl=st.mean(PC[i] for i in common if i in PC))
    kw = json.load(open(os.path.join(ROOT, "docs", "gesd", "pointB_keywords.json"), encoding="utf-8"))["slices"]
    V = json.load(open(os.path.join(ROOT, "docs", "gesd", "pointB_full.json"), encoding="utf-8"))["verdicts"]
    named = {s: [v for v in V if v["cls"] == spec["class"] and v["src"] in ("gemma", "data-driven")
                 and any(k in v["q"].lower() for k in spec["keywords"])] for s, spec in kw.items()}
    named = {s: hs for s, hs in named.items() if hs}
    out["gesd"] = dict(n=len(named), slope=sum(any(h["slope_sys"] for h in hs) for hs in named.values()),
                       auc=sum(any(h["auc_sys"] for h in hs) for hs in named.values()),
                       slope_inverted=sum(any(h["slope_sys"] for h in hs) and not any(h["slope_sys"] and h["auc"] >= 0.5
                                                                                      for h in hs)
                                          for hs in named.values()))
    return out


def load_slice_image_iou(run_dir=None):
    """How badly the images of every certified slice are segmented (slice_quality/examples/slice_image_iou.py):
    the reference rows, and per branch the class IoU on the slice's images against the class's other images
    (how many slices are worse, the mean and median drop, a one-sided Wilcoxon test), the mean drop per phrase
    of the language branch and the slices whose class is not worse. None if the analysis has not been run."""
    f = os.path.join(run_dir or INST, "slice_image_iou.json")
    if not os.path.exists(f):
        return None
    from scipy.stats import spearmanr, wilcoxon
    d = json.load(open(f, encoding="utf-8"))
    out = {"refs": d["refs"], "rows": d["rows"]}
    for br in ("language", "cluster"):
        rr = [r for r in d["rows"] if r["branch"] == br]
        pr = [r for r in rr if r["cls_iou_in"] is not None and r["cls_iou_out"] is not None]
        drop = [r["cls_iou_out"] - r["cls_iou_in"] for r in pr]
        b = dict(n=len(rr), worse=sum(1 for x in drop if x > 0), n_pairs=len(pr),
                 cls_in=st.mean(r["cls_iou_in"] for r in pr), cls_out=st.mean(r["cls_iou_out"] for r in pr),
                 drop_mean=st.mean(drop), drop_median=st.median(drop),
                 miou_img=st.mean(r["miou_img"] for r in rr),
                 not_worse=[r for r, x in zip(pr, drop) if x <= 0])
        if len(pr) > 5:
            b["wilcoxon_p"] = float(wilcoxon([r["cls_iou_in"] for r in pr], [r["cls_iou_out"] for r in pr],
                                             alternative="less").pvalue)
            b["size_rho"], b["size_p"] = (float(v) for v in spearmanr([r["n_images"] for r in pr], drop))
        out[br] = b
    ph = defaultdict(list)
    for r in d["rows"]:
        if r["branch"] == "language" and r["cls_iou_in"] is not None and r["cls_iou_out"] is not None:
            ph[r["name"].split(" — ")[1].strip('"')].append(r["cls_iou_out"] - r["cls_iou_in"])
    out["by_phrase"] = sorted(((k, st.mean(v), len(v)) for k, v in ph.items()), key=lambda t: -t[1])
    return out


def load_rule_min(q=0.05, min_ok=10):
    """The rank test with a minimum of correct objects, and the data-driven hypotheses in the official run
    (24-25 October; outputs/chain_dd_official_v3.sh). Sources, in outputs/slice_quality_inst/rule_min_correct
    (copied from the server's outputs/ and outputs/slice_quality_inst_withdd):
    rank_few_correct.json  every whole-class rank verdict of the official families, with an exact one-sided
                           Mann-Whitney p (rank_few_correct.py) and whether the run of 23 October shipped it
    rank_min_{main,pending}.txt  the families re-verdicted with the minimum (rank_min_correct.py)
    dd_auc_{cc,full}.json  the data-driven hypotheses on the sample and on whole classes (dd_auc.py)
    new_situations.txt     the situations the rule and the data-driven hypotheses add (+) and withdraw (-)
    and, in the run itself, validation_report.json (the 6 new situations against the others).
    Pascal-EA: slice_quality_ea_auc/rank_min_correct.txt and branch_coverage.json, before the rule in
    slice_quality_ea_auc_nomin. None when the folder is missing."""
    import re
    d = out_path("slice_quality_inst", "rule_min_correct")
    if not os.path.isdir(d):
        return None
    J = lambda f: json.load(open(os.path.join(d, f), encoding="utf-8"))
    rows = J("rank_few_correct.json")
    conf = lambda r: r["certified"] and r["q_exact"] is not None and r["q_exact"] <= q
    now = Counter(_cls(x["slice_id"]) for x in json.load(open(os.path.join(INST, "slices_index.json"),
                                                                 encoding="utf-8"))["slices"]
                  if x.get("in_descriptions") and x["source"] == "language")
    classes = []
    for c in sorted({r["cls"] for r in rows}):
        rc = [r for r in rows if r["cls"] == c]
        classes.append(dict(cls=c, n_ok=rc[0]["n_ok"], n_err=rc[0]["n_err"],
                            cert=sum(r["certified"] for r in rc), confirmed=sum(conf(r) for r in rc),
                            shipped=sum(r["shipped"] for r in rc), shipped_conf=sum(r["shipped"] and conf(r) for r in rc),
                            now=now.get(c, 0)))
    classes.sort(key=lambda r: r["n_ok"])
    fam = {}
    for k in ("main", "pending"):
        t = open(os.path.join(d, f"rank_min_{k}.txt"), encoding="utf-8").read()
        m = re.search(r"(\d+) hypotheses tested.*?certified before (\d+), now (\d+) \((\d+) hypotheses", t, re.S)
        fam[k] = dict(n=int(m.group(1)), before=int(m.group(2)), now=int(m.group(3)), untested=int(m.group(4)),
                      lost=Counter(x for x in re.findall(r"lost (\w+)", t)))
    cc, full = J("dd_auc_cc.json"), J("dd_auc_full.json")
    sig = [r for r in cc if r["auc_sys"]]
    ch = [l.strip() for l in open(os.path.join(d, "new_situations.txt"), encoding="utf-8") if l.strip()]
    split = lambda s: tuple(x.strip() for x in s[1:].split("|", 1))
    hi = [c for c in classes if c["n_ok"] >= min_ok]
    out = dict(classes=classes, below=[c for c in classes if c["n_ok"] < min_ok], fam=fam,
               ge=dict(cert=sum(c["cert"] for c in hi), confirmed=sum(c["confirmed"] for c in hi),
                       not_conf=[r for r in rows if r["n_ok"] >= min_ok and r["certified"] and not conf(r)]),
               shipped=sum(c["shipped"] for c in classes), shipped_conf=sum(c["shipped_conf"] for c in classes),
               dd=dict(n=len(cc), n_classes=len({r["cls"] for r in cc}), sample_auc=len(sig),
                       sample_slope=sum(bool(r["slope_sys"]) for r in cc),
                       excluded=Counter(r["cls"] for r in sig if r["cls"] not in {x["cls"] for x in full}),
                       n_full=len(full), full_auc=sum(bool(r["auc_sys"]) for r in full),
                       full_slope=sum(bool(r["slope_sys"]) for r in full),
                       full_median_auc=st.median(r["auc"] for r in full), full=full,
                       min_ok=min(r["n_ok"] for r in full)),
               gained=[split(l) for l in ch if l.startswith("+")], lost=[split(l) for l in ch if l.startswith("-")],
               val=json.load(open(os.path.join(INST, "validation_report.json"), encoding="utf-8")))
    ea = out_path("slice_quality_ea_auc", "rank_min_correct.txt")
    if os.path.exists(ea):
        t = open(ea, encoding="utf-8").read()
        m = re.search(r"(\d+) hypotheses tested.*?certified before (\d+), now (\d+) \((\d+) hypotheses", t, re.S)
        bc = lambda run: json.load(open(out_path(run, "branch_coverage.json"), encoding="utf-8"))
        out["ea"] = dict(n=int(m.group(1)), before=int(m.group(2)), now=int(m.group(3)), untested=int(m.group(4)),
                         lost=Counter(re.findall(r"lost (\w+)", t)),
                         n_ok=dict((c, int(k)) for c, k in re.findall(r"lost (\w+)\s+correct\s+(\d+)", t)),
                         cov=bc("slice_quality_ea_auc"), cov_before=bc("slice_quality_ea_auc_nomin"))
    return out


if __name__ == "__main__":
    import pprint
    pprint.pprint(load_narrative_stage(), width=110)
    pprint.pprint(load(), width=110)
