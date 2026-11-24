"""Axis 6/6 -- statistical significance.  Source: new.

The null hypothesis matters. Testing a slice of ERRORS against the CORRECT
reference is vacuous (any group of errors "fails more" than correct instances:
p ~ 0 for everything). The meaningful question is two-fold, and the test
adapts to which one is answerable for the slice at hand:

  A. severity test (``z-boot``): do these instances fail WORSE than the other
     errors, after normalising severity within each error subtype? (Raw 1-IoU
     is incomparable across subtypes: FN/FP sit at 1.0 by construction.)
     BLOCK bootstrap on the z-lift -- resample IMAGES, not instances, because
     instances from the same image are correlated.
  B. coherence test (``perm-compactness``): for slices whose severity signal is
     degenerate (pure FN/FP/confusion strata: every member and every peer has
     z == 0, so A carries no information), test whether the slice is a tighter
     region of the embedding manifold than CHANCE: permutation test of manifold
     compactness against random same-size subsets of the same subtype pool.
     Non-circular for language slices (not formed by embeddings); for cluster
     slices it is post-selection (the GMM optimises this) -- documented limit.

Then Benjamini-Hochberg FDR across the candidate batch.
Output : (p_value, lift, ci_low, ci_high, test_name) per slice; FDR mask.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional, Sequence

import numpy as np

from ..config import SignificanceConfig
from ..data_types import TypedInstance


def subtype_severity_stats(instances: Sequence[TypedInstance]) -> dict:
    """{subtype -> (mean, std) of (1 - IoU)} over the given error population."""
    vals: dict[str, list[float]] = defaultdict(list)
    for m in instances:
        vals[m.seg_subtype.value].append(1.0 - m.iou)
    stats = {}
    for st, v in vals.items():
        mu = float(np.mean(v))
        stats[st] = (mu, float(np.std(v)))
    return stats


def severity_z(inst: TypedInstance, stats: dict) -> float:
    mu, sd = stats.get(inst.seg_subtype.value, (0.0, 0.0))
    return ((1.0 - inst.iou) - mu) / sd if sd > 1e-9 else 0.0


def _errors_by_image(members: Sequence[TypedInstance]) -> dict[str, list[float]]:
    by_img: dict[str, list[float]] = defaultdict(list)
    for m in members:
        by_img[m.image_id].append(1.0 - m.iou)
    return by_img


def _block_resample_mean(by_img: dict[str, list[float]], rng: np.random.Generator) -> float:
    images = list(by_img.keys())
    chosen = rng.choice(len(images), size=len(images), replace=True)
    vals: list[float] = []
    for c in chosen:
        vals.extend(by_img[images[c]])
    return float(np.mean(vals)) if vals else 0.0


def bootstrap_significance(
    slice_members: Sequence[TypedInstance],
    population: Sequence[TypedInstance],
    cfg: SignificanceConfig | None = None,
    seed: int = 0,
) -> tuple[float, float, float, float]:
    """Return (p_value, lift, ci_low, ci_high) for one slice."""
    cfg = cfg or SignificanceConfig()
    rng = np.random.default_rng(seed)

    slice_err = np.mean([1.0 - m.iou for m in slice_members]) if slice_members else 0.0
    pop_err = np.mean([1.0 - m.iou for m in population]) if population else 0.0
    lift = float(slice_err - pop_err)

    s_by_img = _errors_by_image(slice_members)
    p_by_img = _errors_by_image(population)

    lifts = np.empty(cfg.n_boot)
    for b in range(cfg.n_boot):
        if cfg.block_bootstrap:
            sm = _block_resample_mean(s_by_img, rng)
            pm = _block_resample_mean(p_by_img, rng)
        else:                                  # plain instance bootstrap (less safe)
            se = [1.0 - m.iou for m in slice_members]
            pe = [1.0 - m.iou for m in population]
            sm = float(np.mean(rng.choice(se, len(se)))) if se else 0.0
            pm = float(np.mean(rng.choice(pe, len(pe)))) if pe else 0.0
        lifts[b] = sm - pm

    p_value = float(np.mean(lifts <= 0.0))      # H0: slice does not fail more
    ci_low, ci_high = np.percentile(lifts, [2.5, 97.5])
    return p_value, lift, float(ci_low), float(ci_high)


def _z_by_image(members: Sequence[TypedInstance], stats: dict) -> dict[str, list[float]]:
    by_img: dict[str, list[float]] = defaultdict(list)
    for m in members:
        by_img[m.image_id].append(severity_z(m, stats))
    return by_img


def _block_mean(by_img: dict[str, list[float]], rng: np.random.Generator) -> float:
    images = list(by_img.keys())
    if not images:
        return 0.0
    chosen = rng.choice(len(images), size=len(images), replace=True)
    vals: list[float] = []
    for c in chosen:
        vals.extend(by_img[images[c]])
    return float(np.mean(vals)) if vals else 0.0


def slice_significance(
    slice_members: Sequence[TypedInstance],
    error_population: Sequence[TypedInstance],
    sev_stats: dict,
    member_idx: Sequence[int],
    subtype_pool_idx: Sequence[int],
    knn,
    cfg: SignificanceConfig | None = None,
    seed: int = 0,
) -> tuple[float, float, float, float, str]:
    """(p_value, lift_z, ci_low, ci_high, test) for one slice vs the OTHER errors.

    Test A (z-boot) when the severity signal carries information; test B
    (perm-compactness, vs random same-size same-subtype subsets) when it is
    degenerate. ``test == "none"`` when neither applies (e.g. the slice IS the
    whole stratum): p = 1.0, the slice cannot be certified.
    """
    from .manifold_compactness import manifold_compactness

    cfg = cfg or SignificanceConfig()
    rng = np.random.default_rng(seed)
    member_ids = {m.instance_id for m in slice_members}
    rest = [m for m in error_population if m.instance_id not in member_ids]
    if not slice_members or not rest:
        return 1.0, 0.0, 0.0, 0.0, "none"

    # Does the slice's severity carry ANY information? A pure FN/FP/confusion
    # slice cannot deviate from its subtype mean (every peer has the same
    # 1-IoU by construction), so test A would be a coin flip -- use test B.
    informative = any(sev_stats.get(m.seg_subtype.value, (0.0, 0.0))[1] > 1e-9
                      for m in slice_members)

    # --- test A: block bootstrap on the subtype-normalised severity lift -----
    s_by_img = _z_by_image(slice_members, sev_stats)
    r_by_img = _z_by_image(rest, sev_stats)
    lift = (float(np.mean([severity_z(m, sev_stats) for m in slice_members]))
            - float(np.mean([severity_z(m, sev_stats) for m in rest])))
    lifts = np.empty(cfg.n_boot)
    for b in range(cfg.n_boot):
        lifts[b] = _block_mean(s_by_img, rng) - _block_mean(r_by_img, rng)
    ci_low, ci_high = np.percentile(lifts, [2.5, 97.5])

    if informative and float(np.std(lifts)) > 1e-9:
        p = float((1 + np.sum(lifts <= 0.0)) / (cfg.n_boot + 1))
        return p, lift, float(ci_low), float(ci_high), "z-boot"

    # --- test B: coherence-vs-chance permutation (degenerate severity) -------
    n = len(member_idx)
    pool = list(subtype_pool_idx)
    if knn and n >= 2 and len(pool) >= max(2 * n, n + 5):
        obs = manifold_compactness(list(member_idx), knn)
        pool_arr = np.array(pool)
        count = 0
        for _b in range(cfg.n_boot):
            rand = rng.choice(pool_arr, size=n, replace=False)
            if manifold_compactness(list(rand), knn) >= obs:
                count += 1
        p = float((1 + count) / (cfg.n_boot + 1))
        return p, lift, float(ci_low), float(ci_high), "perm-compactness"

    return 1.0, lift, float(ci_low), float(ci_high), "none"


def benjamini_hochberg(p_values: Sequence[float], alpha: float = 0.05) -> np.ndarray:
    """Return a boolean mask of which p-values are significant under BH/FDR."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return np.array([], dtype=bool)
    order = np.argsort(p)
    thresh = (np.arange(1, m + 1) / m) * alpha
    passed = p[order] <= thresh
    sig = np.zeros(m, dtype=bool)
    if passed.any():
        k = np.max(np.where(passed)[0])         # largest rank that passes
        sig[order[: k + 1]] = True
    return sig
