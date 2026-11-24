"""Planted-cause test: does the pipeline, run blind on val-EA, find the failures the
Pascal-EA material edit caused, and what does it call them?

Ground truth (never seen by the pipeline):
  victim     the edited subject's GT instance in an image where the edit made the
             segmenter lose it: the subject's recall falls by >= DROP from the
             original to the edited image (pascal_ea_compare.py), and the instance
             is an error in val-EA
  survivor   an edited subject whose recall moved by less than 0.05 -- the edit did
             not hurt (the control: it should not be called a failure)
The subject's GT instance is the val-EA object instance (inst configuration) that
overlaps the subject mask most; its id "<image>#g<J>" is the same in the inst run,
whose fate on the ORIGINAL image is the baseline.

Measured, from the two slice indexes (certified = in_descriptions):
  recall      share of victims inside at least one certified slice of val-EA (and,
              for comparison, share inside a certified slice of the inst run --
              mostly 0, since on the original they were mostly correct)
  per slice   victims, edited-image members and size of every certified slice
              that holds a victim, with its description (what the pipeline says)
  new         certified (class, hypothesis) situations of val-EA absent from inst
Run (CPU, after chain_ea.sh):  python slice_quality/examples/ea_planted_eval.py
"""
import json
import os
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

from slice_quality.adapters.voc_seg_samples import build_voc_samples

EA_RUN = os.environ.get("EA_RUN", "outputs/slice_quality_ea")
INST_RUN = os.environ.get("INST_RUN", "outputs/slice_quality_inst")
ROOT = "/data/voc2012_ea/VOCdevkit/VOC2012"
EAD = os.environ.get("EA", "/ds/pascal_ea_material")
CMP = os.environ.get("CMP", "outputs/pascal_ea/per_image.json")
DROP = float(os.environ.get("DROP", "0.30"))
S = 520

edited = [l.strip() for l in open("/data/voc2012_ea/edited_stems.txt") if l.strip()]
rows = {r["stem"]: r for r in json.load(open(CMP))["rows"]}
lab = json.load(open(f"{EAD}/labels.json"))["images"]


def cut(img):
    w, h = img.size
    r = S / min(w, h)
    img = img.resize((round(w * r), round(h * r)), Image.NEAREST)
    l, t = (img.width - S) // 2, (img.height - S) // 2
    return np.array(img.crop((l, t, l + S, t + S)))


# the subject's GT instance id in the object configuration
os.environ["SQ_GT_INSTANCES"] = "object"
samples, _ = build_voc_samples(ROOT, "/workspace/voc_crops_full_ea", edited, gt_instances="object")
subject_id = {}
for s in samples:
    bm = cut(Image.open(f"{EAD}/binary_masks/{s['image_id']}.png").convert("L")) > 127
    best, bi = 0.0, None
    for j, g in enumerate(s["gts"]):
        m = np.asarray(g["mask"], bool)
        u = (m | bm).sum()
        iou = (m & bm).sum() / u if u else 0
        if iou > best:
            best, bi = iou, j
    if bi is not None and best >= 0.3:
        subject_id[s["image_id"]] = f"{s['image_id']}#g{bi}"


def index(run):
    d = json.load(open(f"{run}/slices_index.json", encoding="utf-8"))
    cert = [s for s in d["slices"] if s.get("in_descriptions")]
    member = defaultdict(list)
    for s in cert:
        for img, ids in s["images"].items():
            for i in ids:
                member[i].append(s["slice_id"])
    return cert, member


def describe(run):
    """The most frequent per-image phrase of each slice (a cluster slice is phrased image by
    image, with that image's class and attribute)."""
    ph = defaultdict(Counter)
    for img, d in json.load(open(f"{run}/image_descriptions.json", encoding="utf-8")).items():
        for p in d.get("patterns", []):
            ph[p["slice_id"]][p["description"]] += 1
    return {k: v.most_common(1)[0][0] for k, v in ph.items()}


def linked(errors):
    """GT instance id -> the error instances that concern it: the GT-side instance itself (a
    miss) and the prediction-side ones paired with it (boundary, extent, merge ...)."""
    out = defaultdict(list)
    for t in errors.values():
        keys = {t.instance_id, t.metadata.get("gt_id"), t.metadata.get("primary_gt")}
        keys |= set(t.metadata.get("gt_ids") or [])
        for k in keys:
            if k:
                out[k].append(t.instance_id)
    return out


ea_cert, ea_mem = index(EA_RUN)
in_cert, in_mem = index(INST_RUN)
ea_desc = describe(EA_RUN)
import pickle
ea_err = {t.instance_id: t for t in pickle.load(open(f"{EA_RUN}/result_tagged.pkl", "rb"))["errors"]}
in_err = {t.instance_id: t for t in pickle.load(open(f"{INST_RUN}/result_tagged.pkl", "rb"))["errors"]}
ea_link, in_link = linked(ea_err), linked(in_err)
in_slice = lambda gid, link, mem: any(mem.get(i) for i in link.get(gid, []))

victims, survivors = [], []
for stem, iid in subject_id.items():
    r = rows.get(stem, {})
    if "rec" not in r.get("orig", {}) or "rec" not in r.get("v3", {}):
        continue
    dl = r["v3"]["rec"] - r["orig"]["rec"]
    if dl <= -DROP and ea_link.get(iid):
        victims.append(iid)
    elif abs(dl) < 0.05:
        survivors.append(iid)

lines = []
say = lambda s="": (lines.append(s), print(s))
say("=" * 78)
say(f"PLANTED-CAUSE TEST (val-EA): victims = edited subjects lost by the segmenter (recall drop >= {DROP})")
say("=" * 78)
say(f"edited images {len(edited)}; subject instance found in {len(subject_id)}; victims {len(victims)}; "
    f"survivors (|change| < 0.05) {len(survivors)}")
vt = Counter(ea_err[i].seg_subtype.value for v in victims for i in ea_link[v])
say(f"victims' error type in val-EA: {dict(vt.most_common())}")
base = Counter(("+".join(sorted({in_err[i].seg_subtype.value for i in in_link[v]})) if in_link.get(v)
                else "correct") for v in victims)
say(f"the same objects on the ORIGINAL image (inst run): {dict(base.most_common())}")
cov_ea = [v for v in victims if in_slice(v, ea_link, ea_mem)]
cov_in = [v for v in victims if in_slice(v, in_link, in_mem)]
say(f"\nRECALL: victims inside a certified slice of val-EA: {len(cov_ea)}/{len(victims)} "
    f"({100 * len(cov_ea) / max(len(victims), 1):.0f}%)  | inside a certified slice of inst: "
    f"{len(cov_in)}/{len(victims)}")
by_branch = Counter()
for v in cov_ea:
    br = {("language" if not sid.startswith("cluster_") else "cluster")
          for i in ea_link[v] for sid in ea_mem.get(i, [])}
    for b in br:
        by_branch[b] += 1
say(f"   by branch: {dict(by_branch)}")
vcls = Counter(lab[v.split('#')[0]].get("voc_class") for v in victims)
ccls = Counter(lab[v.split('#')[0]].get("voc_class") for v in cov_ea)
say("   by class (covered/victims): " + ", ".join(f"{c} {ccls[c]}/{n}" for c, n in vcls.most_common()))
vmat = Counter(lab[v.split('#')[0]].get("material_v3") for v in victims)
cmat = Counter(lab[v.split('#')[0]].get("material_v3") for v in cov_ea)
say("   by material (covered/victims): " + ", ".join(f"{c} {cmat[c]}/{n}" for c, n in vmat.most_common()))
sv = [v for v in survivors if in_slice(v, ea_link, ea_mem)]
say(f"\nCONTROL: survivors inside a certified slice of val-EA: {len(sv)}/{len(survivors)} "
    f"(they are errors in val-EA: {sum(1 for v in survivors if ea_link.get(v))})")

say("\n-- certified slices of val-EA that hold victims " + "-" * 29)
vs = {i for v in victims for i in ea_link[v]}
es = set(edited)
stats = []
for s in ea_cert:
    ids = [i for ids in s["images"].values() for i in ids]
    nv = sum(1 for i in ids if i in vs)
    if not nv:
        continue
    ne = sum(1 for i in ids if i.split("#")[0] in es)
    cls = Counter(ea_err[i].gt_class or ea_err[i].pred_class for i in ids if i in ea_err).most_common(4)
    stats.append((nv, s, len(ids), ne, cls))
for nv, s, n, ne, cls in sorted(stats, key=lambda x: -x[0]):
    say(f"   {s['slice_id']:24s} victims {nv:3d} | members {n:4d} (from edited images {ne}) | "
        f"{s.get('dominant_error_type')} | classes {', '.join(f'{c} {k}' for c, k in cls)}")
    say("   " + " " * 24 + " phrase: " + ea_desc.get(s["slice_id"], s.get("hypothesis") or "")[:110])

key = lambda s: (s["slice_id"].rsplit("_", 1)[0], s.get("hypothesis"))
in_lang = {key(s) for s in in_cert if s.get("source") == "language"}
new_lang = [s for s in ea_cert if s.get("source") == "language" and key(s) not in in_lang]
gone = [s for s in in_cert if s.get("source") == "language"
        and key(s) not in {key(x) for x in ea_cert if x.get("source") == "language"}]
say(f"\n-- certified situations: inst {sum(1 for s in in_cert if s.get('source') == 'language')} language + "
    f"{sum(1 for s in in_cert if s.get('source') == 'cluster')} cluster; val-EA "
    f"{sum(1 for s in ea_cert if s.get('source') == 'language')} + {sum(1 for s in ea_cert if s.get('source') == 'cluster')}")
say("   new in val-EA (language): " + ("; ".join(f"{k[0]}: {k[1]}" for k in map(key, new_lang)) or "none"))
say("   lost in val-EA (language): " + ("; ".join(f"{k[0]}: {k[1]}" for k in map(key, gone)) or "none"))

json.dump({"victims": victims, "survivors": survivors, "covered": cov_ea, "covered_inst": cov_in,
           "slices_with_victims": [dict(slice_id=s["slice_id"], victims=nv, members=n, from_edited=ne,
                                        description=ea_desc.get(s["slice_id"]), hypothesis=s.get("hypothesis"))
                                   for nv, s, n, ne, _c in stats],
           "new_language": [list(key(s)) for s in new_lang], "lost_language": [list(key(s)) for s in gone],
           "report": lines}, open(f"{EA_RUN}/planted_eval.json", "w"), indent=1)
open(f"{EA_RUN}/planted_eval.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print(f"\nwritten {EA_RUN}/planted_eval.json / .txt")
