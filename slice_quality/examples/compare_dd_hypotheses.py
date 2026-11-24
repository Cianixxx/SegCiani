"""What the data-driven hypotheses add to the inst run. CPU only.

Two rescores of the SAME inst pickle that differ only in the language runs they read:
  BASE = LANG_DIRS /data/outputs:/data/outputs_extra                 (= the inst run)
  DD   = LANG_DIRS /data/outputs:/data/outputs_extra:/data/outputs_dd_ground, LANG_TAGS :extra:dd
so code drift cannot pass for an effect of the hypotheses. Reports, per class, the
data-driven hypotheses grounded, those SliceLens's slope-trend test calls systematic, those
certified (slice ids "<cls>_dd_<n>"), the images that gain a situational description and the
old certified slices that lose it.
Run: python slice_quality/examples/compare_dd_hypotheses.py
Env: BASE, DD, RUNS_DIR (/data/outputs_dd_ground), OUT (<DD>/dd_hypotheses_report)
"""
import json
import os
from collections import defaultdict

BASE = os.environ.get("BASE", "outputs/slice_quality_inst_dd_base")
DD = os.environ.get("DD", "outputs/slice_quality_inst_dd")
RUNS_DIR = os.environ.get("RUNS_DIR", "/data/outputs_dd_ground")
OUT = os.environ.get("OUT", f"{DD}/dd_hypotheses_report")


def cert(run):
    d = json.load(open(f"{run}/slices_index.json", encoding="utf-8"))["slices"]
    return {s["slice_id"]: s for s in d if s.get("in_descriptions") and s["source"] == "language"}


def images(cs):
    return {im for s in cs.values() for im in s["images"]}


cb, cd = cert(BASE), cert(DD)
new = {k: s for k, s in cd.items() if "_dd_" in k}
old_key = lambda s: (s["slice_id"].rsplit("_", 1)[0], s.get("hypothesis"))
lost = sorted({old_key(s) for s in cb.values()} - {old_key(s) for s in cd.values() if "_dd_" not in s["slice_id"]})
lines = []
say = lambda s="": (lines.append(s), print(s, flush=True))
say("=" * 78)
say("DATA-DRIVEN HYPOTHESES (SliceLens Sec. 3.1 part 2) ON %s: what they add to the run" % os.environ.get("DATASET", "VOC val"))
say("=" * 78)
per = defaultdict(lambda: [0, 0])
for f in sorted(os.listdir(RUNS_DIR)) if os.path.isdir(RUNS_DIR) else []:
    if f.startswith("voc_paper_") and (f.endswith("_cc.json") or f.endswith("_full.json")):
        r = json.load(open(os.path.join(RUNS_DIR, f), encoding="utf-8"))
        cls = r["label"]
        tag = "full" if f.endswith("_full.json") else "sample"
        per[(cls, tag)] = [len(r["candidates"]), len(r["systematic_error_slices"])]
for (cls, tag), (n, k) in sorted(per.items()):
    say(f"   {cls:12s} {tag:6s} hypotheses {n} | systematic {k}")
say()
say(f"certified language slices: inst-equivalent base {len(cb)} -> with data-driven {len(cd)} "
    f"(new data-driven {len(new)}, old lost {len(lost)})")
for k, s in sorted(new.items()):
    say(f"   + {k:22s} {s.get('hypothesis')} | {s['size']} errors, {s['n_images']} images, {s.get('dominant_error_type')}")
for c, h in lost:
    say(f"   - {c}: {h}")
gain = images(cd) - images(cb)
say(f"images with a situational description: {len(images(cb))} -> {len(images(cd))} (+{len(gain)})")
json.dump({"per_class": {f"{c}|{t}": v for (c, t), v in per.items()}, "base": len(cb), "dd": len(cd),
           "new": {k: {"hypothesis": s.get("hypothesis"), "size": s["size"], "n_images": s["n_images"],
                       "type": s.get("dominant_error_type")} for k, s in new.items()},
           "lost": lost, "images_gained": len(gain), "report": lines},
          open(OUT + ".json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
open(OUT + ".txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("DD_COMPARE_DONE", flush=True)
