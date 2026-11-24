"""Per-image Block 1 counts, {image -> {subtype -> n}}, for the comparison page.

CPU, seconds. Reads the val pickle and, if present, the train36 one, so the
counts match the typing the page's texts were generated from. GT-side merge
instances (claim_cov runs) are counted under "merge" like the blob: the page
shows the taxonomy of the run as it is.

Run:  SQ_OUT=outputs/slice_quality_cov05b python slice_quality/examples/typed_per_image.py
"""
import json
import os
import pickle
from collections import Counter, defaultdict

SQ_OUT = os.environ.get("SQ_OUT", "outputs/slice_quality")

per = defaultdict(Counter)
for pkl in (f"{SQ_OUT}/result_tagged.pkl", f"{SQ_OUT}/train36/instances_train36.pkl"):
    if not os.path.exists(pkl):
        print(f"manca {pkl}, salto", flush=True)
        continue
    d = pickle.load(open(pkl, "rb"))
    for e in d["errors"]:
        per[e.image_id][e.seg_subtype.value] += 1
    print(f"{pkl}: {len(d['errors'])} errori", flush=True)

out = {k: dict(v) for k, v in sorted(per.items())}
json.dump(out, open(f"{SQ_OUT}/typed_per_image.json", "w", encoding="utf-8"), indent=1)
print(f"immagini: {len(out)} | totali: {dict(sum((Counter(v) for v in out.values()), Counter()))}")
