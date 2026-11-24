"""Put a sentence on every NON-certified candidate cluster.

The per-image descriptions name a near-miss by id -- "~ 1 isolated obj near
candidate pattern cluster_fn_8_r2 (DISCARD, not certified)" -- which tells the
reader that something was almost found without saying WHAT, or WHY it was
rejected. Both are recoverable from data already in the pickle, and both matter:
"the pipeline saw nothing here" and "the pipeline saw 59 cluttered aeroplanes
with imprecise boundaries but could not prove the grouping beats chance" are
very different statements about the same image.

Wording is produced by the SAME helpers the certified descriptions use
(describe._class_phrase / _attr_word / _ETYPE), so a candidate reads like a
certified slice and the only difference is the explicit rejection clause -- no
candidate can be mistaken for a finding.

The rejection clause re-derives each Def.1 gate from the stored axes rather than
trusting a remembered threshold:
  severe        mean subtype-normalised severity z >= 0
  big enough    size >= n_min_frac * dataset_size
  pure          >= purity_min_tagged informative tags AND permutation p <= alpha
  significant   survives Benjamini-Hochberg across the candidate batch
  coherent      misprediction coherence AND error-type purity >= threshold

CPU only. Run:
  PYTHONPATH=/workspace python slice_quality/examples/explain_candidates.py
Env: PKL, OUT.
"""
import json
import os
import pickle
from collections import Counter

from slice_quality.config import SliceQualityConfig
from slice_quality.describe import _ETYPE, _attr_word, _class_phrase

PKL = os.environ.get("PKL", "outputs/slice_quality/result_tagged.pkl")
OUT = os.environ.get("OUT", "outputs/slice_quality/candidate_clusters.json")

d = pickle.load(open(PKL, "rb"))
res = d["result"]
dataset_size = len(d["errors"]) + len(d["reference"])
d1 = SliceQualityConfig().def1
n_min = max(1, int(d1.n_min_frac * dataset_size))
print(f"dataset: {dataset_size} instances | N_min = {n_min} "
      f"({d1.n_min_frac:.0%}) | purity alpha = {d1.purity_p_alpha} | "
      f"coherence threshold = {d1.coherence_refine_threshold}", flush=True)

kept_ids = {k.slice_.slice_id for k in res.kept}
out, seen = {}, set()
for s, q in res.scored:
    if q.slice_id in seen:
        continue
    seen.add(q.slice_id)

    # --- what the group looks like, in the certified slices' own wording -----
    attr, clsp = _attr_word(s.members), _class_phrase(s.members)
    parts = [f"{attr} objects" if attr else "objects"]
    if clsp:
        parts.append(clsp)
    if not attr:
        parts.append("grouped by appearance only (no verified attribute)")
    parts.append(_ETYPE.get(q.dominant_error_type.value, q.dominant_error_type.value))
    what = " ".join(parts)

    ex = q.extras or {}
    sev_z = ex.get("mean_sev_z")
    p_pur = ex.get("purity_p")
    n_attr = ex.get("n_attr", 0)
    p_val = ex.get("p_value")

    gates = {
        "severe": (sev_z is not None and sev_z >= 0.0),
        "big_enough": q.size >= n_min,
        "pure": (n_attr >= d1.purity_min_tagged and p_pur is not None
                 and p_pur <= d1.purity_p_alpha),
        "significant": bool(q.significant),
        "coherent": (q.misprediction_coherence >= d1.coherence_refine_threshold
                     and q.error_type_purity >= d1.coherence_refine_threshold),
    }

    why = []
    if not gates["severe"]:
        why.append(f"non abbastanza grave per il suo tipo d'errore "
                   f"(z medio {sev_z:.2f}, serve >= 0)" if sev_z is not None
                   else "severita non calcolabile")
    if not gates["big_enough"]:
        why.append(f"troppo piccolo ({q.size} istanze, servono >= {n_min})")
    if not gates["pure"]:
        if n_attr < d1.purity_min_tagged:
            why.append(f"attributi insufficienti per testare la purezza "
                       f"({n_attr} tag informativi, servono >= {d1.purity_min_tagged})")
        else:
            why.append(f"la purezza non batte il caso "
                       f"(p={p_pur:.2f}, serve <= {d1.purity_p_alpha})")
    if not gates["significant"]:
        extra = f", p={p_val:.2f}" if isinstance(p_val, (int, float)) else ""
        why.append(f"non significativo dopo la correzione per test multipli{extra}")
    if not gates["coherent"]:
        why.append(f"poco coerente (coerenza misprediction {q.misprediction_coherence:.2f}, "
                   f"purezza tipo {q.error_type_purity:.2f}, serve >= "
                   f"{d1.coherence_refine_threshold})")

    classes = Counter((m.gt_class or m.pred_class) for m in s.members if (m.gt_class or m.pred_class))
    attrs = Counter(m.metadata.get("attribute") for m in s.members
                    if m.metadata.get("attribute"))

    out[q.slice_id] = {
        "certified": q.slice_id in kept_ids,
        "verdict": q.verdict,
        "what": what,
        "why_not_certified": why,          # empty when certified
        "size": q.size,
        "n_images": len({m.image_id for m in s.members}),
        "error_type": q.dominant_error_type.value,
        "classes": dict(classes.most_common(4)),
        "attributes": dict(attrs.most_common(5)),
        "gates": gates,
        "axes": {"slice_purity": round(q.slice_purity, 3),
                 "error_type_purity": round(q.error_type_purity, 3),
                 "misprediction_coherence": round(q.misprediction_coherence, 3),
                 "omega_squared": round(q.omega_squared, 4),
                 "manifold_compactness": round(q.manifold_compactness, 3),
                 "mean_sev_z": round(sev_z, 3) if sev_z is not None else None,
                 "purity_p": round(p_pur, 3) if p_pur is not None else None},
    }

cert = sum(1 for v in out.values() if v["certified"])
print(f"slices described: {len(out)} ({cert} certified, {len(out) - cert} candidates)",
      flush=True)
fail = Counter(w.split(" (")[0].split(", p=")[0] for v in out.values()
               if not v["certified"] for w in v["why_not_certified"])
print("\nperche i candidati non passano (un candidato puo fallire su piu gate):", flush=True)
for k, n in fail.most_common():
    print(f"  {n:4d}  {k}", flush=True)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)
print(f"\nsaved {OUT}", flush=True)
print("EXPLAIN_CANDIDATES_DONE", flush=True)
