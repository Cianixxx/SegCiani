"""Recompute TRUE (both-branch union) instance coverage for one or more pickles.

The headline "coverage" figure quoted across the report/slides for weeks was
computed as len({m.instance_id for k in res.kept for m in k.slice_.members}) --
but language slices never enter res.kept (they validate via SliceLens's own
slope-trend test, not the cluster branch's Def.1/refine loop), so that figure was
silently CLUSTER-ONLY coverage every time, understating the true systematic
coverage by omitting every instance the language branch alone accounts for.

This recomputes coverage the way Figure 7 (complementarity) already did it
correctly: the union of instance ids belonging to every slice that actually
ships in the descriptions (build_image_descriptions' own certified pool), from
BOTH branches.

Run: PYTHONPATH=/workspace python slice_quality/examples/recompute_true_coverage.py <pkl1> [pkl2 ...]
"""
import pickle
import sys

from slice_quality.describe import build_image_descriptions

for path in sys.argv[1:]:
    d = pickle.load(open(path, "rb"))
    res, errors = d["result"], d["errors"]
    descs = build_image_descriptions(res)
    used = {p["slice_id"] for v in descs.values() for p in v["patterns"]}
    seen, cluster_ids, lang_ids, all_ids = set(), set(), set(), set()
    for s, q in res.scored:
        if q.slice_id in seen:
            continue
        seen.add(q.slice_id)
        if q.slice_id in used:
            ids = {m.instance_id for m in s.members}
            all_ids |= ids
            (cluster_ids if s.source == "cluster" else lang_ids).update(ids)
    n = len(errors)
    n_cl = sum(1 for k in res.kept)
    n_lang = sum(1 for sid in used if sid not in {k.quality.slice_id for k in res.kept})
    print(f"{path}")
    print(f"  errors={n} | certified slices: cluster={n_cl} language={n_lang}")
    print(f"  cluster-only coverage : {len(cluster_ids)}/{n} = {100*len(cluster_ids)/n:.1f}%")
    print(f"  language-only coverage: {len(lang_ids)}/{n} = {100*len(lang_ids)/n:.1f}%")
    print(f"  TRUE union coverage   : {len(all_ids)}/{n} = {100*len(all_ids)/n:.1f}%")
    print(f"  overlap               : {len(cluster_ids & lang_ids)}")
    print()
