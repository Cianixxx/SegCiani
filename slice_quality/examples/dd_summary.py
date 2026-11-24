"""Numbers for the thesis section on the data-driven hypotheses. CPU only, reads only.

Collects, for the 16 VOC classes that got data-driven hypotheses in October 2026
(examples/gen_datadriven.py, outputs/chain_dd_gen.sh + chain_dd_voc.sh):
  per class   the attributes captioned, their clusters, the 6 condensed hypotheses, and for each
              hypothesis its slope on the 120-instance sample, on all instances (if re-grounded)
              and whether it ends up certified
  yield       slope-systematic / evaluated on the SAME 120-instance sample, by source:
              canonical, class-specific (Gemma), the 8 extra generic, data-driven
  totals      certified language slices and images with a situational description, base vs dd
  examples    the highest-confidence error instances of every certified data-driven slice
              (for the visual check)
Run: python slice_quality/examples/dd_summary.py  ->  outputs/slice_quality_inst_dd/dd_summary.json
Env: GEN (/data/outputs_dd), GROUND (/data/outputs_dd_ground), KNOW (/data/outputs),
     EXTRA (/data/outputs_extra), BASE, DD (the two rescores)
"""
import json
import os

from slice_quality.describe import _CANONICAL, _GENERIC_EXTRA

GEN = os.environ.get("GEN", "/data/outputs_dd")
GROUND = os.environ.get("GROUND", "/data/outputs_dd_ground")
KNOW = os.environ.get("KNOW", "/data/outputs")
EXTRA = os.environ.get("EXTRA", "/data/outputs_extra")
BASE = os.environ.get("BASE", "outputs/slice_quality_inst_dd_base")
DD = os.environ.get("DD", "outputs/slice_quality_inst_dd")
CLASSES = sorted(f[3:-4] for f in os.listdir(GEN) if f.startswith("dd_") and f.endswith(".txt"))


def run(d, c, suf):
    f = f"{d}/voc_paper_{c}{suf}.json"
    return json.load(open(f, encoding="utf-8")) if os.path.exists(f) else None


def cert(d):
    sl = json.load(open(f"{d}/slices_index.json", encoding="utf-8"))["slices"]
    return {s["slice_id"]: s for s in sl if s.get("in_descriptions") and s["source"] == "language"}


cb, cd = cert(BASE), cert(DD)
dd_cert = {(s["slice_id"].split("_dd_")[0], s["hypothesis"]): s for k, s in cd.items() if "_dd_" in k}
_vf = f"{DD}/dd_validated_index.json"                     # what the sample stage certified
validated = json.load(open(_vf, encoding="utf-8"))["slices"] if os.path.exists(_vf) else []
val_key = {(s["slice_id"].split("_dd_")[0], s["hypothesis"]): s for s in validated}

out = {"classes": CLASSES, "per_class": {}, "examples": {}}
yl = {k: [0, 0] for k in ("canonical", "class-specific", "extra", "data-driven")}
for c in CLASSES:
    g = json.load(open(f"{GEN}/dd_{c}.json", encoding="utf-8"))
    s, f = run(GROUND, c, "_cc"), run(GROUND, c, "_full")
    sc = {x["query"]: x for x in (s or {}).get("candidates", [])}
    fc = {x["query"]: x for x in (f or {}).get("candidates", [])
          if f and len(x["scored"]) == f["num_instances"]}           # re-grounded, not carried over
    hyps = []
    for h in (x["hypothesis"] for x in g["condensed"]):
        a, b = sc.get(h), fc.get(h)
        cs = dd_cert.get((c, h))
        hyps.append({"query": h,
                     "sample_slope": a and a["max_positive_slope"],
                     "sample_systematic": bool(a and a["is_systematic"]),
                     "sample_match": a and sum(x["conf"] > 0.5 for x in a["scored"]),   # VLM says yes
                     "sample_scored": a and len(a["scored"]),
                     "sample_certified": (c, h) in val_key,
                     "full_slope": b and b["max_positive_slope"],
                     "full_systematic": bool(b and b["is_systematic"]) if b else None,
                     "certified": cs and {"slice_id": cs["slice_id"], "size": cs["size"],
                                          "n_images": cs["n_images"], "type": cs.get("dominant_error_type")}})
        if cs and b:
            out["examples"][cs["slice_id"]] = {
                "class": c, "hypothesis": h,
                "top": [{"name": t["name"], "bbox": t["bbox"], "conf": round(t["slice_confidence"], 3),
                         "is_error": bool(t["is_error"])} for t in b["top_instances"]]}
    out["per_class"][c] = {
        "n_instances": g["n_instances"], "n_sampled": len(g["sample"]), "seeds_from": g["seeds_from"],
        "attributes": [{"attribute": a["attribute"],
                        "n_values": sum(1 for v in a["values"] if v and v != "none"),
                        "clusters": {k: len(v) for k, v in a["clusters"].items()}} for a in g["attributes"]],
        "hypotheses": hyps,
        "sample_n": s and s["num_instances"], "sample_errors": s and s["num_errors"],
        "full_n": f and f["num_instances"]}
    for src, r in (("know", run(KNOW, c, "_cc")), ("extra", run(EXTRA, c, "_cc")), ("dd", s)):
        for x in (r or {}).get("candidates", []):
            q = x["query"]
            k = ("data-driven" if src == "dd" else "extra" if q in _GENERIC_EXTRA
                 else "canonical" if q in _CANONICAL else "class-specific")
            yl[k][0] += int(bool(x["is_systematic"]))
            yl[k][1] += 1
out["yield_sample"] = yl
if out["examples"]:                     # the crop each bbox lives in, for the visual check
    import pickle
    paths = pickle.load(open(os.environ.get("SUBSTRATE", "/data/voc2012/substrate_cc.pkl"), "rb"))["image_paths"]
    for e in out["examples"].values():
        for t in e["top"]:
            t["path"] = str(paths.get(t["name"]))
img = lambda cs: {im for s in cs.values() for im in s["images"]}
out["totals"] = {"certified_base": len(cb), "certified_dd": len(cd),
                 "certified_sample_stage": len(validated),
                 "lost": sorted({(s["slice_id"].rsplit("_", 1)[0], s.get("hypothesis")) for s in cb.values()}
                                - {(s["slice_id"].rsplit("_", 1)[0], s.get("hypothesis")) for s in cd.values()}),
                 "images_base": len(img(cb)), "images_dd": len(img(cd)),
                 "types_base": {}, "types_dd": {}}
for k, cs in (("types_base", cb), ("types_dd", cd)):
    for s in cs.values():
        t = s.get("dominant_error_type")
        out["totals"][k][t] = out["totals"][k].get(t, 0) + 1
json.dump(out, open(f"{DD}/dd_summary.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(json.dumps(out["yield_sample"]), json.dumps({k: v for k, v in out["totals"].items() if "types" not in k}))
print("DD_SUMMARY_DONE")
