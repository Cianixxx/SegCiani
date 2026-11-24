"""What the 8 extra generic hypotheses add to the inst run. CPU only.

Compares two rescores of the SAME inst pickle that differ only in the language
runs they read:
  BASE  = outputs/slice_quality_inst_extra_base  (LANG_DIRS=/data/outputs)
  EXTRA = outputs/slice_quality_inst_extra       (LANG_DIRS=/data/outputs:/data/outputs_extra)
so code drift since the inst run cannot pass for an effect of the hypotheses.
The official outputs/slice_quality_inst is only read, as a sanity check that
BASE reproduces it.

Reports, per extra hypothesis: in how many classes SliceLens's slope-trend test
calls it systematic on the 120-instance sample (the raw runs in RUNS_DIR), in how
many it is CERTIFIED (ships in the descriptions), with which dominant error type,
how many images it reaches; then the totals before/after, the images that gain a
situational description, the old certified slices that lose it (more slices
also means a stricter multiple-testing threshold for everyone), and the extra
slices that dedup merged into an old one.

Run: python slice_quality/examples/compare_extra_hypotheses.py
Env: BASE, EXTRA, OFFICIAL, RUNS_DIR, OUT (default <EXTRA>/extra_hypotheses_report)
"""
import json
import os
from collections import Counter, defaultdict

from slice_quality.describe import _CANONICAL, _GENERIC_EXTRA

BASE = os.environ.get("BASE", "outputs/slice_quality_inst_extra_base")
EXTRA = os.environ.get("EXTRA", "outputs/slice_quality_inst_extra")
OFFICIAL = os.environ.get("OFFICIAL", "outputs/slice_quality_inst")
RUNS_DIR = os.environ.get("RUNS_DIR", "/data/outputs_extra")
JULY_DIR = os.environ.get("JULY_DIR", "/data/outputs")
OUT = os.environ.get("OUT", f"{EXTRA}/extra_hypotheses_report")


def index(d):
    return json.load(open(f"{d}/slices_index.json", encoding="utf-8"))["slices"]


def lang_cert(sl):
    return {s["slice_id"]: s for s in sl if s.get("hypothesis") and s.get("in_descriptions")}


def images(cert):
    out = set()
    for s in cert.values():
        out |= set(s.get("images") or [])
    return out


def cls_of(slice_id):
    return slice_id.split("_extra")[0] if "_extra" in slice_id else slice_id.rsplit("_", 1)[0]


def main():
    base, extra = index(BASE), index(EXTRA)
    cb, ce = lang_cert(base), lang_cert(extra)
    lines = []
    say = lambda s="": (lines.append(s), print(s))

    # sanity: BASE must reproduce the official inst run
    if os.path.exists(f"{OFFICIAL}/slices_index.json"):
        co = lang_cert(index(OFFICIAL))
        same = set(co) == set(cb)
        say(f"controllo: BASE riproduce la run inst ufficiale? {'si' if same else 'NO'} "
            f"({len(cb)} vs {len(co)} slice linguistiche certificate)")
        if not same:
            say(f"  solo in BASE: {sorted(set(cb) - set(co))}   solo in inst: {sorted(set(co) - set(cb))}")

    # raw SliceLens verdicts on the sample
    raw = defaultdict(dict)      # hyp -> cls -> (is_systematic, slope, top10 error rate)
    for f in sorted(os.listdir(RUNS_DIR)):
        if f.startswith("voc_paper_") and f.endswith("_cc.json"):
            d = json.load(open(os.path.join(RUNS_DIR, f), encoding="utf-8"))
            for c in d["candidates"]:
                raw[c["query"]][d["label"]] = (c["is_systematic"], c["max_positive_slope"],
                                               c["top_k_error_rate"])
    n_cls = len({c for h in raw.values() for c in h})
    # verdicts of the full-population re-grounding (reground_validated.py), if run
    full = defaultdict(dict)     # hyp -> cls -> is_systematic on ALL instances of the class
    for f in sorted(os.listdir(RUNS_DIR)):
        if f.startswith("voc_paper_") and f.endswith("_full.json"):
            d = json.load(open(os.path.join(RUNS_DIR, f), encoding="utf-8"))
            for c in d["candidates"]:
                if "scored" in c and len(c["scored"]) == d["num_instances"]:   # re-grounded, not carried
                    full[c["query"]][d["label"]] = c["is_systematic"]

    # same sample as the July discovery runs? (Jaccard of the scored instance ids)
    same = {}
    for f in sorted(os.listdir(RUNS_DIR)):
        old = os.path.join(JULY_DIR, f)
        if f.startswith("voc_paper_") and f.endswith("_cc.json") and os.path.exists(old):
            ids = lambda p: {it["instance_id"] for c in json.load(open(p, encoding="utf-8"))["candidates"][:1]
                             for it in c.get("scored", [])}
            a, b = ids(old), ids(os.path.join(RUNS_DIR, f))
            same[f[10:-8]] = round(len(a & b) / max(1, len(a | b)), 3)
    if same:
        diff = {c: j for c, j in same.items() if j < 1}
        say(f"campione identico a quello di luglio in {len(same) - len(diff)}/{len(same)} classi"
            + (f"; diverso in: {diff}" if diff else ""))

    say(f"\n{'ipotesi':64s} {'sistem.':>8s} {'certif.':>8s} {'imm.':>5s}  tipo dominante / classi")
    per_hyp = {}
    for h in _GENERIC_EXTRA:
        sys_cls = sorted(c for c, v in raw.get(h, {}).items() if v[0])
        cert = {k: s for k, s in ce.items() if s["hypothesis"] == h}
        imgs = images(cert)
        types = Counter(s["dominant_error_type"] for s in cert.values())
        per_hyp[h] = {"systematic_classes": sys_cls, "full_reground": dict(full.get(h, {})),
                      "certified": [{"slice_id": k, "class": cls_of(k), "size": s["size"],
                                     "type": s["dominant_error_type"], "n_images": s["n_images"]}
                                    for k, s in sorted(cert.items())],
                      "n_images": len(imgs), "types": dict(types)}
        detail = ", ".join(f"{cls_of(k)}:{s['dominant_error_type']}({s['size']})"
                           for k, s in sorted(cert.items()))
        say(f"{h[:64]:64s} {len(sys_cls):>3d}/{n_cls:<4d} {len(cert):>8d} {len(imgs):>5d}  {detail}")
        if full.get(h):
            held = sorted(c for c, v in full[h].items() if v)
            gone = sorted(c for c, v in full[h].items() if not v)
            say(f"   riverifica sull'intera popolazione: regge in {held or '-'}, cade in {gone or '-'}")

    # totals
    ib, ie = images(cb), images(ce)
    new_extra = {k: s for k, s in ce.items() if s["hypothesis"] in _GENERIC_EXTRA}
    lost = sorted(set(cb) - set(ce))
    say(f"\nslice linguistiche certificate: {len(cb)} -> {len(ce)} "
        f"(nuove ipotesi {len(new_extra)}, vecchie perse {len(lost)})")
    say(f"immagini con una descrizione situazionale: {len(ib)} -> {len(ie)} "
        f"(+{len(ie - ib)} nuove, -{len(ib - ie)} perse)")
    tb = Counter(s["dominant_error_type"] for s in cb.values())
    te = Counter(s["dominant_error_type"] for s in ce.values())
    say("tipo dominante delle certificate: " + ", ".join(
        f"{t} {tb.get(t, 0)}->{te.get(t, 0)}" for t in sorted(set(tb) | set(te))))
    if lost:
        say("\nvecchie slice che perdono la certificazione:")
        for k in lost:
            s = cb[k]
            alias = [x for x in ce.values() if s["hypothesis"] in (x.get("aliases") or [])]
            why = (f"assorbita da {alias[0]['slice_id']} ({alias[0]['hypothesis']})" if alias
                   else "non piu' certificata")
            say(f"  {k}: {s['hypothesis']} -> {why}")

    # extra slices absorbed by dedup into an old representative
    absorbed = [(k, s["hypothesis"], a) for k, s in ce.items() for a in (s.get("aliases") or [])
                if a in _GENERIC_EXTRA and s["hypothesis"] not in _GENERIC_EXTRA]
    if absorbed:
        say("\nipotesi nuove assorbite dalla deduplica in una slice esistente:")
        for k, h, a in absorbed:
            say(f"  {a!r} dentro {k} ({h})")

    json.dump({"base": BASE, "extra": EXTRA, "runs_dir": RUNS_DIR, "n_classes_grounded": n_cls,
               "per_hypothesis": per_hyp,
               "totals": {"certified_before": len(cb), "certified_after": len(ce),
                          "new_extra_certified": len(new_extra), "old_lost": lost,
                          "images_before": len(ib), "images_after": len(ie),
                          "images_gained": sorted(ie - ib), "images_lost": sorted(ib - ie),
                          "types_before": dict(tb), "types_after": dict(te)},
               "absorbed": absorbed,
               "canonical": list(_CANONICAL), "extra": list(_GENERIC_EXTRA)},
              open(f"{OUT}.json", "w", encoding="utf-8"), indent=1)
    open(f"{OUT}.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"\nscritto {OUT}.json/.txt")
    print("COMPARE_EXTRA_DONE")


if __name__ == "__main__":
    main()
