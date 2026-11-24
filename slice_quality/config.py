"""Configuration for the slice_quality pipeline.

Every tunable lives here so the blocks stay parameter-free and easy to modify.
Defaults follow the papers where a value is given; otherwise sensible starting
points are used and flagged with a comment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class TypedErrorConfig:
    """BLOCK 1 -- matching + typed-error decomposition."""

    tau_match: float = 0.5        # class-agnostic IoU to call a match
    tau_iou_good: float = 0.85    # above this (right class) -> CORRECT (reference)
    # Fragment-claim threshold, as a fraction of tau_match. Two disjoint
    # fragments can NEVER both reach IoU >= 0.5 with the same GT (the shares
    # would sum past 1), so with claims at tau_match SPLIT/MERGE never fire and
    # fragments silently become FPs. Claims therefore use tau_match*frag_frac.
    frag_frac: float = 0.5
    # Beyond IoU boundary width: a FP/FN pixel is "boundary" if within 2*d of a
    # TP/TN boundary, else "extent". Relative to the image diagonal (Beyond IoU
    # uses 1% for ADE20K / VOC); int would mean absolute pixels.
    boundary_d_frac: float = 0.01
    # Containment claim (None = pure IoU rule, the behaviour of every run so
    # far). IoU is blind to asymmetric containment: a small GT swallowed at 70%
    # by a blob that also covers a large GT has IoU ~0.1 with it, never claims
    # it, and falls through to FN -- "missed" for a region that is mostly
    # covered (54% of the FNs on pascal_80). Beyond IoU itself calls a GT
    # segment missed only when NO pixel of it is a true positive; this is the
    # instance-level analogue: a GT also claims the prediction holding >= this
    # share of its area, a prediction also claims the GT holding >= this share
    # of its own. At 0.5 the containing region is unique (two disjoint regions
    # cannot each hold more than half of the same mask), the same majority
    # argument behind tau_match. Changes the error population: not a drop-in.
    # Switch by environment (SQ_CLAIM_COV=0.5) so every runner that builds a
    # SliceQualityConfig() picks it up without edits; write to a separate
    # output directory, the two typings are not comparable instance by instance.
    claim_cov: float | None = field(
        default_factory=lambda: float(os.environ["SQ_CLAIM_COV"])
        if os.environ.get("SQ_CLAIM_COV") else None)


@dataclass
class FeatureConfig:
    """BLOCK 3 -- per-instance feature vector emb (+) E (+) subtype."""

    alpha: float = 1.0            # weight of the typed-error block vs embedding
    use_subtype_onehot: bool = True
    normalize: str = "l2"         # "l2" | "zscore" | "none" (per block)


@dataclass
class ClusterConfig:
    """BLOCK 4 -- GMM clustering (SEG4SEG gamma-weighted spirit).

    Improvement #6: by default the clustering is STRATIFIED by error subtype
    (one GMM per subtype -> every cluster slice is error-type pure by
    construction, instead of FN/FP blobs absorbing everything) and k is chosen
    per stratum by BIC (instead of an arbitrary fixed n_slices).
    Set ``stratify_by_subtype=False, k_selection="fixed"`` for the legacy
    single-GMM behaviour.
    """

    n_slices: int = 10            # t_g; used only when k_selection == "fixed"
    gamma: float = 10.0           # weight of the performance term (SEG4SEG: 10)
    covariance_type: str = "diag"
    random_state: int = 0
    # --- improvement #6 -------------------------------------------------------
    stratify_by_subtype: bool = True
    k_selection: str = "bic"      # "fixed" | "bic"
    k_grid: tuple = (2, 3, 4, 6, 8, 10, 14, 18, 24, 30)   # BIC candidates
    min_points_per_component: int = 15   # BIC: only try k <= n / this
    # Strata smaller than this become ONE whole-stratum slice (already pure;
    # let the SQV judge it) instead of being force-split or dropped.
    min_stratum_fit: int = 40


@dataclass
class ManifoldConfig:
    """metrics/manifold_compactness.py -- kNN graph (MCSD Def. 3.1)."""

    k: int = 10                   # neighbours per node
    metric: str = "cosine"        # geodesic approx on the manifold


@dataclass
class SignificanceConfig:
    """metrics/significance.py -- bootstrap + FDR."""

    n_boot: int = 1000            # B
    block_bootstrap: bool = True  # resample images, not instances (correlated)
    fdr_alpha: float = 0.05       # Benjamini-Hochberg level


@dataclass
class Def1Config:
    """BLOCK 6 -- SEG4SEG success criterion (Definition 1)."""

    # Relaxed-gate variants (2026-10-22), set by environment so every runner picks them up; unset = the
    # thesis gates. SQ_PURITY_OFF=1 drops the attribute gate (absolute mode at 0: any slice passes it),
    # SQ_PURITY_ALPHA loosens the permutation test, SQ_COH_MIN the coherence a KEEP needs.
    purity_threshold: float = field(          # p_theta (used only in "absolute" mode)
        default_factory=lambda: 0.0 if os.environ.get("SQ_PURITY_OFF") == "1" else 0.80)
    perf_quantile_threshold: float = 0.40   # q_theta (low-performance gate)
    n_min_frac: float = 0.02            # N_min as a fraction of the dataset
    omega2_threshold: float = 0.14      # "large effect" (Field 2024)
    # Minimum INFORMATIVE (non-"normal"/"other"/"error") tagged members needed
    # before purity is testable at all. No fallback: an untestable slice is not
    # certifiable (the old etp fallback was vacuously 1.0 under stratification
    # and silently passed every under-tagged slice -- exposed 2026-07-18).
    purity_min_tagged: int = 5
    # "permutation" (default): purity must beat CHANCE -- random same-size,
    # same-subtype draws of the SAME noisy VLM tags (self-calibrating to tag
    # noise and vocabulary limits). "absolute": legacy p_theta threshold.
    purity_mode: str = field(
        default_factory=lambda: "absolute" if os.environ.get("SQ_PURITY_OFF") == "1" else "permutation")
    purity_p_alpha: float = field(default_factory=lambda: float(os.environ.get("SQ_PURITY_ALPHA", "0.05")))
    # A slice is REFINE-able (vs DISCARD) when it is severe+significant but its
    # coherence is below this on either axis (mispred. coherence / error purity).
    coherence_refine_threshold: float = field(default_factory=lambda: float(os.environ.get("SQ_COH_MIN", "0.60")))


@dataclass
class RefineConfig:
    """BLOCK 7 -- refine cascade."""

    max_depth: int = 3            # recursion limit (cost guard, bottleneck B3)
    min_split_size: int = 10      # do not split below this many instances
    use_llm_refine: bool = True   # allow the (expensive) LLM re-query for case C


@dataclass
class SliceQualityConfig:
    """Top-level config aggregating every block."""

    typed_error: TypedErrorConfig = field(default_factory=TypedErrorConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    manifold: ManifoldConfig = field(default_factory=ManifoldConfig)
    significance: SignificanceConfig = field(default_factory=SignificanceConfig)
    def1: Def1Config = field(default_factory=Def1Config)
    refine: RefineConfig = field(default_factory=RefineConfig)
    # Which attribute key in TypedInstance.metadata to use for slice_purity.
    # Keep this DIFFERENT from the error type to avoid duplicating axis #4.
    purity_attribute_key: str = "attribute"
