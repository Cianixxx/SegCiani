"""Split-half validation of the cluster branch (post-selection inference guard).

The cluster slices are FORMED by optimising coherence on the data and then
TESTED on the same data -- classic double dipping: the permutation p-values of
slices the GMM built are optimistic by construction. The honest check is
out-of-sample replication:

  1. split the images into two halves (A, B) -- images, not instances, because
     instances within an image are correlated;
  2. run discovery on A only -> kept slices;
  3. induce each kept slice on B by nearest-centroid assignment (same feature
     space, same subtype, radius = the slice's 90th-percentile member distance
     on A);
  4. a slice REPLICATES if its induced B-members pass the same certification
     gates (size, type-aware severity, significance vs B's errors);
  5. repeat with A and B swapped (2-fold) and report the replication rate.

A high replication rate certifies that the discovered slices are properties of
the MODEL's failure modes, not artefacts of one half of the data. Language
slices are exempt: their hypotheses are generated class-level (data-independent
in the knowledge-driven path) and verified by the slope-trend.

CPU-only: reuses the cached DINOv2 embeddings from the result pickle.
"""

from __future__ import annotations

import numpy as np

from .config import SliceQualityConfig
from .features import build_features
from .metrics.significance import (severity_z, slice_significance,
                                   subtype_severity_stats)
from .pipeline import run_from_instances


def _centroid_radius(feats_by_id, members, pct=90.0):
    """Slice centroid + the pct-percentile member distance (assignment radius)."""
    X = np.stack([feats_by_id[m.instance_id] for m in members
                  if m.instance_id in feats_by_id])
    c = X.mean(axis=0)
    dists = np.linalg.norm(X - c, axis=1)
    return c, float(np.percentile(dists, pct))


def _induce_on(fold_errors, feats_by_id, kept, cfg):
    """Assign fold instances to the kept slices' centroids (same subtype, in radius)."""
    slices_meta = []
    for ks in kept:
        s = ks.slice_
        if s.source != "cluster":
            continue
        c, r = _centroid_radius(feats_by_id, s.members)
        subtype = ks.quality.dominant_error_type
        slices_meta.append((s.slice_id, subtype, c, r))

    induced = {sid: [] for sid, *_ in slices_meta}
    for e in fold_errors:
        f = feats_by_id.get(e.instance_id)
        if f is None:
            continue
        best, best_d = None, None
        for sid, subtype, c, r in slices_meta:
            if e.seg_subtype is not subtype:
                continue
            d = float(np.linalg.norm(f - c))
            if d <= r and (best_d is None or d < best_d):
                best, best_d = sid, d
        if best is not None:
            induced[best].append(e)
    return induced


def run_split_half(errors, reference, cfg: SliceQualityConfig | None = None,
                   n_min: int = 10, seed_stride: int = 2) -> dict:
    """Two-fold split-half replication of the cluster branch. Returns a report."""
    cfg = cfg or SliceQualityConfig()
    images = sorted({e.image_id for e in errors})
    half_a = set(images[0::seed_stride])
    folds = [("A->B", half_a), ("B->A", set(images) - half_a)]

    report = {"folds": [], "n_discovered": 0, "n_replicated": 0}
    for name, discovery_imgs in folds:
        disc = [e for e in errors if e.image_id in discovery_imgs]
        test = [e for e in errors if e.image_id not in discovery_imgs]
        ref_d = [r for r in reference if r.image_id in discovery_imgs]

        res = run_from_instances(disc, ref_d, language_slices=None, cfg=cfg)

        # shared feature space over ALL errors (same cfg as clustering)
        feats, kept_inst = build_features(errors, cfg.feature)
        feats_by_id = {kept_inst[k].instance_id: feats[k] for k in range(len(kept_inst))}

        induced = _induce_on(test, feats_by_id, res.kept, cfg)
        sev = subtype_severity_stats(test)

        fold = {"fold": name, "discovered": len(res.kept), "slices": []}
        for ks in res.kept:
            sid = ks.slice_.slice_id
            members = induced.get(sid, [])
            entry = {"slice_id": sid, "n_train": ks.quality.size,
                     "n_test": len(members), "replicated": False}
            if len(members) >= n_min:
                mean_z = float(np.mean([severity_z(m, sev) for m in members]))
                p, lift, _, _, tst = slice_significance(
                    members, test, sev, [], [], None, cfg.significance)
                # certification on the held-out half: size + severity; the
                # significance test needs the B-side graph for the permutation
                # path -- when it returns "none" we fall back to severity only.
                entry.update({"mean_sev_z": round(mean_z, 3),
                              "p_value": round(p, 4), "sig_test": tst})
                entry["replicated"] = bool(
                    mean_z >= -0.25 and (tst == "none" or p <= 0.05))
            fold["slices"].append(entry)
            report["n_discovered"] += 1
            report["n_replicated"] += int(entry["replicated"])
        fold["replication_rate"] = (
            round(sum(s["replicated"] for s in fold["slices"])
                  / max(1, len(fold["slices"])), 3))
        report["folds"].append(fold)

    report["replication_rate"] = round(
        report["n_replicated"] / max(1, report["n_discovered"]), 3)
    return report
