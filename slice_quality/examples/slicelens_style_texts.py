"""What SliceLens alone would say about each image: the baseline text of the description comparison.

SliceLens outputs error slices -- a class and a phrase -- and knows which objects are wrong, but not HOW
they are wrong: no error type, no mask measures, no cluster branch. Its most generous per-image report is
therefore the list of the image's wrong objects by class, and for those inside a certified slice, the
slice's phrase. The slices are the ones certified by SliceLens's own slope rule: the language patterns of
the thesis run before 21 October (outputs/slice_quality_inst_slope), with the error type and the
measures removed. Being generous to the baseline is on purpose: the comparison should not be won by
starving it.

Writes <OUT>/image_descriptions.json ({image: {"text": ...}}): the "pattern" text that judge_claims.py and
judge_descriptions.py read, so that both judges score it beside our narrative with the same facts,
the same prompts and the same control pairings.
Run (CPU):  python slice_quality/examples/slicelens_style_texts.py
Env: SLOPE_RUN (outputs/slice_quality_inst_slope), OUT (outputs/slice_quality_slcmp)
"""
import json
import os
import pickle
import re
from collections import Counter, defaultdict

SLOPE_RUN = os.environ.get("SLOPE_RUN", "outputs/slice_quality_inst_slope")
OUT = os.environ.get("OUT", "outputs/slice_quality_slcmp")
os.makedirs(OUT, exist_ok=True)

errors = pickle.load(open(f"{SLOPE_RUN}/result_tagged.pkl", "rb"))["errors"]
wrong = defaultdict(Counter)                       # image -> class -> wrong objects
for e in errors:
    wrong[e.image_id][(e.gt_class or e.pred_class or "object").upper()] += 1

desc = json.load(open(f"{SLOPE_RUN}/image_descriptions.json", encoding="utf-8"))
phrase_re = re.compile(r'^(.*?) \(SliceLens\)')   # 'car — "an object cut off ..." (SliceLens) -> missed'
out, n_with = {}, 0
for img in sorted(set(wrong) | set(desc)):
    pats = [p for p in desc.get(img, {}).get("patterns", []) if p.get("branch") == "language"]
    lines = []
    in_slice = Counter()
    for p in pats:
        m = phrase_re.match(p["description"])
        ph = m.group(1) if m else p["description"].split(" -> ")[0]
        if " — " not in ph:                       # wrong-class members are phrased without the class
            cls = re.sub(r"_(extra|dd|pend)$", "", p["slice_id"].rsplit("_", 1)[0])
            ph = f"{cls} — {ph}"
        n = p.get("n_instances", 1)
        lines.append(f'  - {ph} [{n} {"object" if n == 1 else "objects"}]')
        in_slice[ph.split(" — ")[0].upper()] += n
    w = wrong.get(img, Counter())
    total = sum(w.values())
    txt = [f"Image {img}: {total} wrong {'object' if total == 1 else 'objects'} "
           f"({', '.join(f'{c} {k}' for c, k in sorted(w.items()))})." if total else f"Image {img}: no wrong objects."]
    if lines:
        n_with += 1
        txt += ["Systematic error slices this image belongs to:"] + sorted(set(lines))
    rest = {c: k - in_slice.get(c, 0) for c, k in w.items() if k - in_slice.get(c, 0) > 0}
    if rest and total:
        txt.append("Wrong objects in no slice: " + ", ".join(f"{c} {k}" for c, k in sorted(rest.items())) + ".")
    out[img] = {"text": "\n".join(txt)}
json.dump(out, open(f"{OUT}/image_descriptions.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(f"{len(out)} images, {n_with} with at least one SliceLens slice -> {OUT}/image_descriptions.json")
print(out.get("2007_000332", {}).get("text", ""))
