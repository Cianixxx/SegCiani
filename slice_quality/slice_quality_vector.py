"""BLOCK 6 -- Slice-Quality Vector: assemble the 6 axes + the Def.1 verdict.

Source : SEG4SEG (Def. 1 success criterion + omega^2) + the other axes.
Input  : candidate Slices (from BOTH branches) + the reference population +
         the global kNN graph + dataset size.
Output : one SliceQuality per slice, with the routing verdict
         KEEP / REFINE / DISCARD.

Significance needs all p-values together (FDR), so this is a two-pass routine:
score every slice, then apply Benjamini-Hochberg across the batch.

Where to modify
---------------
* Def.1 thresholds : config.Def1Config (purity, perf quantile, N_min, omega^2).
* the verdict logic : ``_verdict`` below (when to KEEP vs REFINE vs DISCARD).
"""

from __future__ import annotations

import numpy as np

from .config import Def1Config, SignificanceConfig, SliceQualityConfig
from .data_types import Slice, SliceQuality, TypedInstance
from .metrics.error_type_purity import error_type_purity
from .metrics.manifold_compactness import manifold_compactness
from .metrics.misprediction_coherence import misprediction_coherence
from .metrics.omega_squared import omega_squared
from .metrics.significance import (benjamini_hochberg, severity_z,
                                   slice_significance, subtype_severity_stats)
from .metrics.slice_purity import (NON_ATTRIBUTES, purity_permutation_p,
                                   slice_purity)


def _verdict(q: SliceQuality, d1: Def1Config) -> str:
    """KEEP a clean real slice; REFINE a severe-but-incoherent one; else DISCARD."""
    coherent = (q.misprediction_coherence >= d1.coherence_refine_threshold
                and q.error_type_purity >= d1.coherence_refine_threshold)
    severe = q.omega_squared >= d1.omega2_threshold
    if q.passes_def1 and q.significant and coherent:
        return "KEEP"
    if severe and q.significant and not coherent:
        return "REFINE"
    return "DISCARD"


def score_slices(
    slices: list[Slice],
    population: list[TypedInstance],
    id_to_index: dict[str, int],
    knn: list[set[int]],
    dataset_size: int,
    cfg: SliceQualityConfig | None = None,
    error_population: list[TypedInstance] | None = None,
) -> list[SliceQuality]:
    """Score every candidate slice and assign a routing verdict.

    ``population``       : the CORRECT reference instances.
    ``error_population`` : ALL error instances (the peers a slice is judged
    against). If None it is reconstructed from the slices' members.
    """
    cfg = cfg or SliceQualityConfig()
    d1 = cfg.def1

    if error_population is None:                # best effort for old callers
        seen_ids: set[str] = set()
        error_population = []
        for s in slices:
            for m in s.members:
                if m.instance_id not in seen_ids:
                    seen_ids.add(m.instance_id)
                    error_population.append(m)

    # Full-dataset performance (errors + reference): omega^2 is the SEG4SEG
    # one-vs-rest disparity over the WHOLE population, not "errors vs correct".
    all_inst = list(error_population) + list(population)
    all_perf_by_id = {i.instance_id: i.iou for i in all_inst}
    sev_stats = subtype_severity_stats(error_population)
    # same-subtype graph-index pools (for the coherence permutation test)
    pools: dict[str, list[int]] = {}
    for m in error_population:
        if m.instance_id in id_to_index:
            pools.setdefault(m.seg_subtype.value, []).append(id_to_index[m.instance_id])
    # same-subtype ATTRIBUTE pools (null distribution for the purity test);
    # global pool as fallback when a subtype has too few tagged errors.
    attr_key = cfg.purity_attribute_key
    attr_pools: dict[str, list] = {}
    global_attr_pool: list = []
    for m in error_population:
        v = m.metadata.get(attr_key)
        if v is not None:
            attr_pools.setdefault(m.seg_subtype.value, []).append(v)
            global_attr_pool.append(v)
    purity_rng = np.random.default_rng(0)

    qualities: list[SliceQuality] = []
    p_values: list[float] = []

    for s in slices:
        slice_perf = np.array([m.iou for m in s.members])
        member_ids = {m.instance_id for m in s.members}
        weights = [s.weight(m) for m in s.members]

        # --- axis 1: omega^2, slice vs REST of the dataset ----------------
        rest_perf = np.array([p for iid, p in all_perf_by_id.items()
                              if iid not in member_ids])
        om = omega_squared(slice_perf, rest_perf)
        # --- axis 2: manifold compactness ---------------------------------
        member_idx = [id_to_index[m.instance_id] for m in s.members
                      if m.instance_id in id_to_index]
        mc = manifold_compactness(member_idx, knn) if knn else 0.0
        # --- axis 3: misprediction coherence ------------------------------
        mpc = misprediction_coherence(s.members, weights)
        # --- axis 4: error-type purity ------------------------------------
        etp, dominant = error_type_purity(s.members)
        # --- axis 5: slice purity (attribute homogeneity vs CHANCE) ---------
        # "normal"/"other"/"error" are non-attributes and are excluded; the
        # gate is a permutation test against random same-subtype draws (fair
        # under VLM tag noise), with NO fallback for untestable slices.
        attr_vals = [m.metadata.get(attr_key) for m in s.members]
        n_tagged = sum(1 for v in attr_vals if v is not None)
        n_attr = sum(1 for v in attr_vals
                     if v is not None and v not in NON_ATTRIBUTES)
        sp = slice_purity(attr_vals)
        if cfg.def1.purity_mode == "permutation":
            pool = attr_pools.get(dominant.value, [])
            if len(pool) < 30:
                pool = global_attr_pool
            p_purity, _, _ = purity_permutation_p(
                attr_vals, pool, n_boot=cfg.significance.n_boot, rng=purity_rng)
            pure_enough_flag = (n_attr >= d1.purity_min_tagged
                                and p_purity <= d1.purity_p_alpha)
        else:                                    # "absolute" (legacy p_theta)
            p_purity = None
            purity_gate = sp if n_tagged >= d1.purity_min_tagged else etp
            pure_enough_flag = purity_gate >= d1.purity_threshold
        # --- axis 6: significance vs the OTHER errors ----------------------
        p_val, lift_z, ci_lo, ci_hi, test = slice_significance(
            s.members, error_population, sev_stats, member_idx,
            pools.get(dominant.value, []), knn, cfg.significance)
        p_values.append(p_val)

        # --- Def.1 gates ----------------------------------------------------
        # severe = at-least-typical severity FOR ITS ERROR TYPE. Two rejected
        # alternatives: raw mean-perf vs population is biased against
        # boundary/extent slices (high IoU by nature); an omega^2 >= 0.14 gate
        # is size-confounded in the one-vs-rest form (a 3%-of-dataset slice can
        # explain little total variance however badly it fails -- with the
        # honest rest-population it zeroed out EVERY slice). omega^2 stays a
        # reported effect-size axis; severity certification against the peers
        # is the significance axis's job (z-boot lift / permutation test).
        mean_z = float(np.mean([severity_z(m, sev_stats) for m in s.members])) \
            if s.members else 0.0
        severe = mean_z >= 0.0
        big_enough = s.size >= max(1, int(d1.n_min_frac * dataset_size))
        passes_def1 = bool(severe and big_enough and pure_enough_flag)

        qualities.append(SliceQuality(
            slice_id=s.slice_id,
            omega_squared=om, manifold_compactness=mc,
            misprediction_coherence=mpc, error_type_purity=etp,
            slice_purity=sp, significant=False,        # set after FDR
            dominant_error_type=dominant, size=s.size,
            passes_def1=passes_def1,
            extras={"lift_z": lift_z, "ci": (ci_lo, ci_hi), "p_value": p_val,
                    "sig_test": test, "mean_sev_z": mean_z, "n_tagged": n_tagged,
                    "n_attr": n_attr if cfg.def1.purity_mode == "permutation" else n_tagged,
                    "purity_p": p_purity,
                    "source": s.source, "hypothesis": s.hypothesis},
        ))

    # --- batch FDR correction (axis 6) ------------------------------------
    sig_mask = benjamini_hochberg(p_values, cfg.significance.fdr_alpha)
    for q, sig in zip(qualities, sig_mask):
        q.significant = bool(sig)
        q.verdict = _verdict(q, d1)

    # rank: KEEP first, then by severity x coherence
    qualities.sort(
        key=lambda q: (q.verdict == "KEEP", q.omega_squared
                       * (q.misprediction_coherence + q.error_type_purity)),
        reverse=True,
    )
    return qualities
