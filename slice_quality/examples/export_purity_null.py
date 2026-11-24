"""Export the permutation null of the purity gate for one certified slice.

Feeds the report figure that justifies replacing the absolute purity threshold
with a chance-calibrated test: it shows what attribute agreement random groups of
same-subtype errors reach with THIS tag vocabulary, and where the slice sits.

Run: PYTHONPATH=/workspace python slice_quality/examples/export_purity_null.py
"""
import json
import pickle
import random
from collections import Counter

from slice_quality.metrics.slice_purity import NON_ATTRIBUTES, slice_purity

PKL = "outputs/slice_quality/result_tagged.pkl"
OUT = "outputs/slice_quality/purity_null.json"
N_DRAWS = 600

d = pickle.load(open(PKL, "rb"))
res, errors = d["result"], d["errors"]
kept = sorted(res.kept, key=lambda k: -k.quality.size)
k = kept[0]
sub = k.quality.dominant_error_type.value
members = [m.metadata.get("attribute") for m in k.slice_.members]
informative = [v for v in members if v and v not in NON_ATTRIBUTES]
observed = slice_purity(members)

pool = [e.metadata.get("attribute") for e in errors
        if e.seg_subtype.value == sub and e.metadata.get("attribute")]
rng = random.Random(0)
null = [slice_purity(rng.sample(pool, min(len(members), len(pool))))
        for _ in range(N_DRAWS)]

out = {"slice_id": k.quality.slice_id, "subtype": sub, "size": k.quality.size,
       "n_informative": len(informative), "observed": observed, "null": null,
       "p": k.quality.extras.get("purity_p"),
       "attrs": dict(Counter(informative).most_common()),
       "pool_size": len(pool), "n_draws": N_DRAWS}
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(out, f)
print("slice", out["slice_id"], "size", out["size"],
      "informative", out["n_informative"], "observed", round(observed, 3),
      "p", out["p"])
print("null mean", round(sum(null) / len(null), 3),
      "max", round(max(null), 3))
print("saved", OUT)
