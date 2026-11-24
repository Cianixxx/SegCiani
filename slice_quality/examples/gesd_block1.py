"""Block 1 against the experts of GESD (the SliceLens / GH-ESD benchmark), instance segmentation.
CPU only.

GESD's 21 segmentation slices were defined by experts and their members labelled by hand; each
slice name carries an error type ("merged", "missed", "mis-segmented as", "over-segmented",
"boundary unclear", "under-segmented"). Block 1 (slice_quality/typed_error.py, the thesis typing:
containment claims at 0.5) is run on the same predictions and its type for every GESD unit is
compared with the type the experts' slice name implies.

  ground truth  every COCO val2017 object of the image (instances_val2017.json, crowd regions
                included); where a GESD unit's ground-truth mask matches a COCO object
                (IoU >= 0.5) the expert's mask replaces it, an unmatched one is added
  predictions   the Mask R-CNN masks of the GESD units of the image, identical masks merged;
                class = the unit's category (GESD does not store the predicted class)
  expected      from the slice name, rules fixed before the results (EXPECTED below)

A naive typing from the unit alone (missed if only a ground-truth mask, hallucinated if only a
prediction, boundary/extent otherwise) is reported beside it: what the image context adds.
GESD keeps only the predictions of its annotated units, so the other Mask R-CNN predictions of an
image are missing; the comparison is about the GESD units only.

Run: python slice_quality/examples/gesd_block1.py
Env: GESD (gesd/coco_segmentation.json), COCO_ZIP (coco/annotations_trainval2017.zip), OUT,
     CROWD=keep|drop (COCO crowd regions as ground-truth objects, or left out: sensitivity)
"""
import json
import os
import zipfile
from collections import Counter, defaultdict

import numpy as np
from PIL import Image, ImageDraw

from slice_quality.config import TypedErrorConfig
from slice_quality import typed_error
from slice_quality.typed_error import decompose_image, mask_iou

# without scipy typed_error falls back to a crude distance and every boundary error becomes extent
assert typed_error.distance_transform_edt is not None, "scipy missing: pip install scipy"

GESD = os.environ.get("GESD", "/priv/gesd/coco_segmentation.json")
COCO_ZIP = os.environ.get("COCO_ZIP", "/priv/coco/annotations_trainval2017.zip")
OUT = os.environ.get("OUT", "outputs/gesd/block1_vs_experts")
CROWD = os.environ.get("CROWD", "keep")
# the type each slice name implies, by its words, first rule that applies (fixed in advance)
EXPECTED = [("merged", {"merge"}),
            ("missed", {"fn"}),
            ("mis-segmented as", {"fp", "confusion"}),
            ("over-segmented", {"extent", "merge"}),
            ("expanded", {"extent", "merge"}),
            ("boundary unclear", {"boundary", "extent"}),
            ("under-segmented", {"extent", "boundary", "split"})]
lines = []
say = lambda s="": (lines.append(s), print(s, flush=True))


def expected(name):
    for w, t in EXPECTED:
        if w in name.lower():
            return t
    return set()


def poly_mask(polys, w, h):
    m = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(m)
    for p in polys:
        if len(p) >= 6:
            d.polygon([(p[i], p[i + 1]) for i in range(0, len(p) - 1, 2)], fill=1)
    return np.array(m, bool)


def rle_mask(rle, w, h):
    """uncompressed COCO RLE (column-major runs, starting with zeros)."""
    flat = np.zeros(h * w, np.uint8)
    pos, val = 0, 0
    for c in rle["counts"]:
        flat[pos:pos + c] = val
        pos += c
        val = 1 - val
    return flat.reshape((w, h)).T.astype(bool)


P = lambda v: json.loads(v) if isinstance(v, str) else (v or [])
norm = lambda c: c.replace("_", " ")
d = json.load(open(GESD))
gcat = {str(c["id"]): norm(c["name"]) for c in d["categories"]}
units = defaultdict(list)
for a in d["annotations"]:
    units[str(a["image_id"])].append(a)
with zipfile.ZipFile(COCO_ZIP) as z:
    coco = json.loads(z.read("annotations/instances_val2017.json"))
ccat = {c["id"]: c["name"] for c in coco["categories"]}
cann = defaultdict(list)
for a in coco["annotations"]:
    cann[str(a["image_id"])].append(a)
size = {str(i["id"]): (int(i["width"]), int(i["height"])) for i in d["images"]}
cfg = TypedErrorConfig(claim_cov=0.5)

rows, skipped, compressed = [], Counter(), 0
for img, us in units.items():
    w, h = size[img]
    gts = []
    for a in cann.get(img, []):
        if a.get("iscrowd") and CROWD == "drop":
            continue
        seg = a["segmentation"]
        if isinstance(seg, dict):
            if isinstance(seg.get("counts"), str):
                compressed += 1
                continue
            m = rle_mask(seg, w, h)
        else:
            m = poly_mask(seg, w, h)
        if m.any():
            gts.append({"mask": m, "cls": ccat[a["category_id"]], "src": "coco", "crowd": bool(a.get("iscrowd"))})
    preds, pkey, unit_pred, unit_gt, unit_iou, unit_pm = [], {}, {}, {}, {}, {}
    for u in us:
        inf, gt = P(u.get("inference_segmentation")), P(u.get("gt_segmentation"))
        cls = gcat.get(str(u["gt_category_id"]), "?")
        gm = pm = None
        if gt:
            gm = poly_mask(gt, w, h)
            if gm.any():
                best = max(range(len(gts)), key=lambda j: mask_iou(gm, gts[j]["mask"]), default=None)
                if best is not None and mask_iou(gm, gts[best]["mask"]) >= 0.5:
                    gts[best] = {"mask": gm, "cls": gts[best]["cls"], "src": "gesd", "crowd": False}
                    unit_gt[u["id"]] = best
                else:
                    gts.append({"mask": gm, "cls": cls, "src": "gesd-new", "crowd": False})
                    unit_gt[u["id"]] = len(gts) - 1
        if inf:
            pm = poly_mask(inf, w, h)
            if pm.any():
                k = (pm.tobytes(), cls)
                if k not in pkey:
                    pkey[k] = len(preds)
                    preds.append({"mask": pm, "cls": cls})
                unit_pred[u["id"]] = pkey[k]
        if gm is not None and pm is not None and gm.any() and pm.any():
            unit_iou[u["id"]] = round(float(mask_iou(pm, gm)), 3)
            unit_pm[u["id"]] = (pm, gm)
    if not preds and not gts:
        continue
    errs, refs = decompose_image(img, preds, gts, cfg)
    by_id = {t.instance_id: t for t in errs + refs}
    gt_pair = {t.metadata.get("gt_id"): t for t in errs + refs if t.metadata and t.metadata.get("gt_id")}
    in_merge = {g: t for t in errs if t.metadata for g in t.metadata.get("gt_ids", [])}
    for u in us:
        t = None
        if u["id"] in unit_pred:
            t = by_id.get(f"{img}#p{unit_pred[u['id']]}")
        elif u["id"] in unit_gt:
            gid = f"{img}#g{unit_gt[u['id']]}"
            t = by_id.get(gid) or gt_pair.get(gid) or in_merge.get(gid)
        if t is None:
            skipped["no mask or no typed instance"] += 1
            continue
        inf, gt = P(u.get("inference_segmentation")), P(u.get("gt_segmentation"))
        if inf and gt:
            naive = "boundary/extent"
        elif gt:
            naive = "fn"
        elif inf:
            naive = "fp"
        else:
            naive = "none"
        md = t.metadata or {}
        linked = [md.get("gt_id"), md.get("swallowed_by")] + list(md.get("gt_ids", []))
        crowd = any(g and g.startswith(f"{img}#g") and gts[int(g.split("#g")[1])].get("crowd") for g in linked
                    if isinstance(g, str) and "#g" in g)
        # where the predicted mask goes beyond the unit's object: onto other annotated objects or not,
        # and the largest share of another object it covers (merge needs >= claim_cov of two objects)
        excess_on_gt = max_cov_other = excess_cls = None
        if u["id"] in unit_pm:
            pm, gm = unit_pm[u["id"]]
            others = [g for j, g in enumerate(gts) if j != unit_gt.get(u["id"])]
            excess = pm & ~gm
            if excess.any() and others:
                excess_on_gt = round(float((excess & np.logical_or.reduce([g["mask"] for g in others])).sum() / excess.sum()), 3)
                on = [int((excess & g["mask"]).sum()) for g in others]
                if max(on) > 0:
                    excess_cls = others[int(np.argmax(on))]["cls"] + ("(crowd)" if others[int(np.argmax(on))].get("crowd") else "")
            max_cov_other = round(max([float((pm & g["mask"]).sum() / g["mask"].sum()) for g in others], default=0.0), 3)
        rows.append({"unit": u["id"], "image": img, "slice": u["reason"], "type": t.seg_subtype.value, "naive": naive,
                     "iou": unit_iou.get(u["id"]), "crowd": crowd, "n_gt_covered": md.get("n_gt_covered"),
                     "excess_on_gt": excess_on_gt, "max_cov_other": max_cov_other, "excess_cls": excess_cls})

say("=" * 78)
say("BLOCK 1 AGAINST THE EXPERTS OF GESD (instance segmentation, COCO val2017, Mask R-CNN)")
say("=" * 78)
say(f"units typed: {len(rows)} | skipped: {dict(skipped)} | compressed-RLE COCO objects skipped: {compressed}")
na = [r for r in rows if r["slice"] == "NA"]
say(f"correct units (NA): {len(na)} -> Block 1 types {dict(Counter(r['type'] for r in na).most_common())}")
res = {}
say("\nper expert slice: expected type | Block 1 types | agreement (Block 1, naive)")
agree_b, agree_n, n_tot, per_slice = 0, 0, 0, []
for s in sorted({r["slice"] for r in rows if r["slice"] != "NA"}, key=lambda s: -sum(r["slice"] == s for r in rows)):
    rs = [r for r in rows if r["slice"] == s]
    exp = expected(s)
    nat = {"boundary/extent": {"boundary", "extent"}, "fn": {"fn"}, "fp": {"fp"}, "none": set()}
    ab = sum(r["type"] in exp for r in rs)
    an = sum(bool(nat[r["naive"]] & exp) for r in rs)
    agree_b, agree_n, n_tot = agree_b + ab, agree_n + an, n_tot + len(rs)
    per_slice.append((ab / len(rs), an / len(rs)))
    dist = Counter(r["type"] for r in rs).most_common()
    say(f"   {len(rs):4d}  {s[:58]:58s} exp {'/'.join(sorted(exp)):22s} Block1 {100 * ab / len(rs):5.1f}%  "
        f"naive {100 * an / len(rs):5.1f}%  | {', '.join(f'{k} {v}' for k, v in dist)}")
    res[s] = {"n": len(rs), "expected": sorted(exp), "types": dict(dist), "agree": ab, "agree_naive": an}
say(f"\noverall: Block 1 agrees with the expert type on {agree_b}/{n_tot} units ({100 * agree_b / n_tot:.1f}%), "
    f"mean over slices {100 * np.mean([a for a, _ in per_slice]):.1f}%; naive typing {agree_n}/{n_tot} "
    f"({100 * agree_n / n_tot:.1f}%), mean over slices {100 * np.mean([b for _, b in per_slice]):.1f}%")
say(f"\nwhat the units hold (CROWD={CROWD}): masks | IoU predicted vs expert mask (units with both) | crowd involved")
for s in sorted(res, key=lambda s: -res[s]["n"]):
    rs = [r for r in rows if r["slice"] == s]
    ious = [r["iou"] for r in rs if r["iou"] is not None]
    both = sum(r["naive"] == "boundary/extent" for r in rs)
    q = np.percentile(ious, [25, 50, 75]) if ious else [float("nan")] * 3
    say(f"   {s[:58]:58s} both {both:3d} gt-only {sum(r['naive'] == 'fn' for r in rs):3d} "
        f"pred-only {sum(r['naive'] == 'fp' for r in rs):3d} | IoU q25/50/75 {q[0]:.2f}/{q[1]:.2f}/{q[2]:.2f} "
        f"| crowd {sum(r['crowd'] for r in rs)}")
    res[s]["iou_q"] = [float(x) for x in q]
    res[s]["crowd"] = sum(r["crowd"] for r in rs)
say("\nmerged slices, units Block 1 did not call merge (with a prediction): where the excess of the mask goes")
for s in [s for s in sorted(res, key=lambda s: -res[s]["n"]) if "merged" in s.lower() or "over-segmented" in s.lower()]:
    rs = [r for r in rows if r["slice"] == s and r["type"] != "merge" and r["max_cov_other"] is not None]
    if not rs:
        continue
    eo = [r["excess_on_gt"] for r in rs if r["excess_on_gt"] is not None]
    mc = [r["max_cov_other"] for r in rs]
    say(f"   {s[:58]:58s} n {len(rs):3d} | excess on other objects: median {np.median(eo) if eo else float('nan'):.2f} "
        f"| largest other object covered: median {np.median(mc):.2f}, >= 0.5 in {sum(m >= 0.5 for m in mc)}, "
        f"0.1-0.5 in {sum(0.1 <= m < 0.5 for m in mc)}, < 0.1 in {sum(m < 0.1 for m in mc)}")
    say(f"      excess mostly on: {', '.join(f'{k} {v}' for k, v in Counter(r['excess_cls'] for r in rs).most_common(5))}")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump({"expected_rules": [[w, sorted(t)] for w, t in EXPECTED], "per_slice": res,
           "overall": {"block1": agree_b, "naive": agree_n, "n": n_tot}, "na_types": dict(Counter(r["type"] for r in na)),
           "rows": rows}, open(OUT + ".json", "w"), indent=1)
open(OUT + ".txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("GESD_BLOCK1_DONE")
