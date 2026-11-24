"""BLOCK 1 -- Typed-error decomposition for instance segmentation.

Source : Beyond IoU (boundary/extent split) + new for instance-seg (the 6-case
         matching: CORRECT / BOUNDARY / EXTENT / CONFUSION / FP / FN / SPLIT / MERGE).
Input  : predicted instances and GT instances of ONE image, each a dict
         {"mask": HxW bool ndarray, "cls": str}. A GT may carry ``n_parts``
         (> 1: a VOC object cut by an occluder, one instance with a
         non-connected mask -- adapters/voc_seg_samples.py, SQ_GT_INSTANCES=
         object); the matching works on masks and does not care.
Output : list[TypedInstance] for the *erroneous* instances (+ the matched
         reference set is returned separately for omega^2 / error-lift).

This block is purely deterministic (masks vs GT, no embeddings, no VLM). It is
the cheapest, most faithful signal -- compute it once and cache it.

Claim rule (config.TypedErrorConfig.claim_cov)
----------------------------------------------
The split/merge claims were pure IoU (>= tau_match * frag_frac). IoU is blind
to asymmetric containment, so a small GT swallowed by a blob fell through to FN
-- "missed" for a region mostly covered (54% of the FNs on pascal_80, 69% on
val). With ``claim_cov`` set, the regions the IoU rule leaves orphan also claim
by containment (see ``match_instances``), an over-extended prediction becomes a
1<->1 pair instead of FP + FN, and every GT swallowed by a merge is KEPT as a
GT-side MERGE instance (``from_gt_mask``, ``swallowed_by``, ``covered_frac``),
so the slices can be about the swallowed objects and not only about the blobs.
A merge is therefore 1 + M instances, and the blob records the GTs it holds
(``gt_ids``) so mask_facts can measure it against their union. Method note,
measured on val: 49 of the 522 CORRECT references (9%) become MERGE because the
blob, at IoU >= 0.85 on its own object, also holds a small GT; the reference
population of omega^2 and of the contour notes shrinks accordingly. ``claim_cov=None`` reproduces every run
made before it (the only change: a GT swallowed by a matched merge is no longer
also counted as FN, which was a bug).

Where to modify
---------------
* ``match_instances``    : swap the greedy matcher for Hungarian if you prefer.
* ``boundary_extent``    : swap for beyond_iou's exact pixel implementation
                           (../../CB-SLICE/beyond-iou-main/beyond_iou).
* ``tau_match`` / ``tau_iou_good`` / ``boundary_d_frac`` : in config.TypedErrorConfig.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

try:
    from scipy.ndimage import distance_transform_edt
except Exception:  # pragma: no cover - scipy is in requirements
    distance_transform_edt = None

from .config import TypedErrorConfig
from .data_types import SegErrorType, TypedInstance


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------
def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def boundary_extent(pred: np.ndarray, gt: np.ndarray, d_pixels: float) -> tuple[float, float, float]:
    """Split the error of a matched pair into (E_bnd, E_ext, iou).

    Faithful to Beyond IoU's idea: an error pixel close to the correctly
    predicted region (TP) is a *boundary* error; one far away is an *extent*
    error. Distances are taken to the TP region; the threshold is ``2*d``.

    Returns fractions over the union (so iou + E_bnd + E_ext ~= 1 for a
    matched-correct instance, mirroring Beyond IoU Eq. 7 at instance level).
    """
    tp = np.logical_and(pred, gt)
    union = np.logical_or(pred, gt)
    n_union = union.sum()
    if n_union == 0:
        return 0.0, 0.0, 0.0
    iou = tp.sum() / n_union

    err = np.logical_xor(pred, gt)            # FP (pred & ~gt) + FN (~pred & gt)
    if err.sum() == 0 or tp.sum() == 0:
        return 0.0, float(err.sum()) / n_union, float(iou)

    if distance_transform_edt is not None:
        dist_to_tp = distance_transform_edt(~tp)   # distance of every pixel to TP
    else:                                          # crude fallback (no scipy)
        dist_to_tp = np.where(tp, 0.0, 1e9)
    bnd = np.logical_and(err, dist_to_tp <= 2.0 * d_pixels)
    n_bnd = bnd.sum()
    n_err = err.sum()
    return float(n_bnd) / n_union, float(n_err - n_bnd) / n_union, float(iou)


# ---------------------------------------------------------------------------
# Stage 1 -- class-agnostic greedy matching (+ cardinality for split/merge)
# ---------------------------------------------------------------------------
def match_instances(
    preds: list[dict], gts: list[dict], tau_match: float, tau_frag: float | None = None,
    claim_cov: float | None = None,
) -> dict[str, Any]:
    """Return matching structures: primary pairs, FP/FN sets, split/merge flags.

    Pragmatic mutual-best greedy matcher. Easy to replace with Hungarian.

    ``tau_frag`` (< tau_match) is the CLAIM threshold used for split/merge
    detection: two disjoint fragments can never both reach IoU >= 0.5 with the
    same GT (their area shares would sum past 1), so claims must use a lower
    bar than the primary match or split/merge never fire.

    ``claim_cov`` adds a CONTAINMENT claim beside the IoU one, for the regions
    the IoU rule leaves orphan. IoU cannot see asymmetric containment: a small
    GT swallowed at 70% by a blob that also covers a large GT has IoU ~0.1 with
    that blob, claims nothing, and falls through to FN -- "missed", for a
    region that is mostly covered. Beyond IoU calls a GT segment missed only
    when NO pixel of it is a true positive; this is the instance-level
    analogue, applied only where the IoU rule found no claim:

        GT j claims pred i   if |P_i & G_j| / |G_j| >= claim_cov   (j inside i)
        pred i claims GT j   if |P_i & G_j| / |P_i| >= claim_cov   (i inside j,
                                                                    same class)

    At claim_cov >= 0.5 the containing region is unique -- two disjoint regions
    cannot each hold more than half of the same mask -- so these claims are as
    well-posed as tau_match = 0.5. None keeps the pure IoU rule.
    """
    tau_frag = tau_match if tau_frag is None else tau_frag
    n_p, n_g = len(preds), len(gts)
    iou = np.zeros((n_p, n_g))
    cov_g = np.zeros((n_p, n_g))          # share of the GT inside the pred
    cov_p = np.zeros((n_p, n_g))          # share of the pred inside the GT
    for i, p in enumerate(preds):
        pm = np.asarray(p["mask"], dtype=bool)
        a_p = float(pm.sum())
        for j, g in enumerate(gts):
            gm = np.asarray(g["mask"], dtype=bool)
            inter = float(np.logical_and(pm, gm).sum())
            if inter == 0.0:
                continue
            a_g = float(gm.sum())
            iou[i, j] = inter / (a_p + a_g - inter)
            cov_g[i, j] = inter / a_g
            cov_p[i, j] = inter / a_p

    pred_best = iou.argmax(axis=1) if n_g else np.full(n_p, -1)
    gt_best = iou.argmax(axis=0) if n_p else np.full(n_g, -1)

    # Which preds claim each gt (>= tau_frag)?  -> split if > 1 claimers.
    # Sorted by IoU desc: claims[0] is the PRIMARY match (gets boundary/extent),
    # the others are SPLIT fragments -- by overlap, not by component order.
    gt_claims: dict[int, list[int]] = {j: [] for j in range(n_g)}
    for i in range(n_p):
        j = int(pred_best[i])
        if j >= 0 and iou[i, j] >= tau_frag:
            gt_claims[j].append(i)
    # Which gts claim each pred (>= tau_frag)?  -> merge if > 1 claimers.
    pred_claims: dict[int, list[int]] = {i: [] for i in range(n_p)}
    for j in range(n_g):
        i = int(gt_best[j])
        if i >= 0 and iou[i, j] >= tau_frag:
            pred_claims[i].append(j)

    if claim_cov is not None and n_p and n_g:
        # IoU rule first; containment only rescues what it left orphan, so a
        # region never claims two things.
        claimed_gts = {j for js in pred_claims.values() for j in js}
        for j in range(n_g):
            if j in claimed_gts:
                continue
            i = int(cov_g[:, j].argmax())
            if cov_g[i, j] >= claim_cov:
                pred_claims[i].append(j)
        claiming_preds = {i for i in range(n_p) if iou[i, pred_best[i]] >= tau_frag}
        for i in range(n_p):
            if i in claiming_preds:
                continue
            j = int(cov_p[i, :].argmax())
            if cov_p[i, j] >= claim_cov and preds[i]["cls"] == gts[j]["cls"]:
                gt_claims[j].append(i)
    for j in gt_claims:
        gt_claims[j].sort(key=lambda i: -iou[i, j])

    return {
        "iou": iou,
        "cov_g": cov_g,
        "cov_p": cov_p,
        "pred_best": pred_best,
        "gt_best": gt_best,
        "gt_claims": gt_claims,
        "pred_claims": pred_claims,
    }


def _primary_pairs(m: dict[str, Any], n_p: int, n_g: int, tau_match: float,
                   claim_cov: float | None) -> dict[int, int]:
    """{gt index -> pred index} of the 1<->1 pairs.

    An IoU match (>= tau_match) is a pair; it is unique both ways. With
    ``claim_cov`` two more kinds of pair exist, both for regions the IoU rule
    left unmatched:
      * a prediction whose only claim is a GT it CONTAINS (>= claim_cov of the
        GT area) is that GT's prediction -- an over-extended mask, which the
        IoU rule counted as one FP plus one FN;
      * a GT whose best-IoU claimer lies mostly INSIDE it is paired with that
        claimer -- an under-extended mask -- and any further claimer is a
        split fragment of it.
    """
    iou, pred_best = m["iou"], m["pred_best"]
    primary: dict[int, int] = {}
    for i in range(n_p):
        j = int(pred_best[i]) if n_g else -1
        if j >= 0 and iou[i, j] >= tau_match:
            primary[j] = i
    if claim_cov is None:
        return primary
    taken = set(primary.values())
    for i in range(n_p):
        js = m["pred_claims"][i]
        if i in taken or len(js) != 1 or js[0] in primary:
            continue
        if m["cov_g"][i, js[0]] >= claim_cov:
            primary[js[0]] = i
            taken.add(i)
    for j in range(n_g):
        if j in primary:
            continue
        cl = [i for i in m["gt_claims"][j] if i not in taken]
        if cl and m["cov_p"][cl[0], j] >= claim_cov:
            primary[j] = cl[0]
            taken.add(cl[0])
    return primary


# ---------------------------------------------------------------------------
# Stage 2 -- classify every prediction / GT into one of the 6 cases
# ---------------------------------------------------------------------------
def decompose_image(
    image_id: str,
    preds: list[dict],
    gts: list[dict],
    cfg: TypedErrorConfig,
) -> tuple[list[TypedInstance], list[TypedInstance]]:
    """Decompose one image -> (errors, reference) lists of TypedInstance.

    ``preds``/``gts`` items: {"mask": HxW bool, "cls": str, "bbox": (...)?}.
    """
    H = preds[0]["mask"].shape[0] if preds else (gts[0]["mask"].shape[0] if gts else 1)
    W = preds[0]["mask"].shape[1] if preds else (gts[0]["mask"].shape[1] if gts else 1)
    d_pixels = (cfg.boundary_d_frac * math.hypot(H, W)
                if isinstance(cfg.boundary_d_frac, float) else float(cfg.boundary_d_frac))

    claim_cov = getattr(cfg, "claim_cov", None)
    tau_frag = cfg.tau_match * cfg.frag_frac
    m = match_instances(preds, gts, cfg.tau_match, tau_frag, claim_cov)
    iou, gt_claims, pred_claims = m["iou"], m["gt_claims"], m["pred_claims"]
    primary = _primary_pairs(m, len(preds), len(gts), cfg.tau_match, claim_cov)
    paired = {i: j for j, i in primary.items()}
    # the GT a prediction claims as a fragment, if any (unique: a pred has one
    # best GT, and a containment claim is only added when it has no IoU claim)
    claimed_gt = {i: j for j, cl in gt_claims.items() for i in cl}

    errors: list[TypedInstance] = []
    reference: list[TypedInstance] = []

    matched_preds: set[int] = set()
    matched_gts: set[int] = set()
    swallowed: dict[int, int] = {}       # gt index -> index of the blob holding it

    def _merge(i: int, p: dict, base: dict) -> None:
        # CASE 6 (merged blob): covers >= 2 GTs -> one under-segmentation
        # error; the swallowed GTs are NOT FNs (whether or not the blob also
        # has an IoU match of its own).
        js = pred_claims[i]
        matched_gts.update(js)
        meta = {"n_gt_covered": len(js),
                # one entry per swallowed GT, so a reader can count the
                # regions of another class the blob relabelled
                "gt_classes": [gts[j]["cls"] for j in js],
                # and which ones: mask_facts measures the blob against THEIR
                # union, the only ground truth a merged mask can be held to
                "gt_ids": [f"{image_id}#g{j}" for j in js]}
        j0 = paired.get(i)
        if claim_cov is not None:
            # keep the swallowed GTs as instances (see the GT loop); the pure
            # IoU rule drops them, as every run before claim_cov did
            for j in js:
                if j != j0:
                    swallowed[j] = i
        if j0 is not None and iou[i, j0] >= cfg.tau_match:
            # the blob also has a proper 1<->1 match: keep that pair's
            # boundary/extent split, which MERGE alone would throw away and
            # which is what the annotators describe ("coarser boundaries")
            e_bnd, e_ext, pair_iou = boundary_extent(p["mask"], gts[j0]["mask"], d_pixels)
            meta.update(primary_gt=f"{image_id}#g{j0}", primary_iou=round(pair_iou, 4),
                        primary_e_bnd=round(e_bnd, 4), primary_e_ext=round(e_ext, 4))
        errors.append(TypedInstance(
            **base, iou=0.0, error_vector=np.array([0., 0., 1.]),
            seg_subtype=SegErrorType.MERGE, pred_class=p["cls"],
            gt_class=gts[js[0]]["cls"], metadata=meta))

    # --- predictions ------------------------------------------------------
    for i, p in enumerate(preds):
        j = paired.get(i)
        iid = f"{image_id}#p{i}"
        base = dict(instance_id=iid, image_id=image_id, bbox=p.get("bbox"))

        if j is None:
            # CASE 5 (fragment): claims a GT whose primary is another pred ->
            # this pred is a split fragment, not a hallucination.
            jc = claimed_gt.get(i)
            if jc is not None and jc in primary and primary[jc] != i:
                errors.append(TypedInstance(
                    **base, iou=0.0, error_vector=np.array([0., 0., 1.]),
                    seg_subtype=SegErrorType.SPLIT, pred_class=p["cls"],
                    gt_class=gts[jc]["cls"]))
                continue
            if len(pred_claims.get(i, [])) > 1:                  # CASE 6: MERGE
                _merge(i, p, base)
                continue
            errors.append(TypedInstance(                     # CASE 4: FP
                **base, iou=0.0, error_vector=np.array([0., 0., 1.]),
                seg_subtype=SegErrorType.FP, pred_class=p["cls"]))
            continue

        matched_preds.add(i)
        matched_gts.add(j)
        if len(pred_claims.get(i, [])) > 1:                  # CASE 6: MERGE (1<->M)
            _merge(i, p, base)
            continue
        # which GT this pair is about, and whether that GT is an object cut into
        # parts (VOC object masks): the narrative says "partly hidden" from it
        base["metadata"] = {"gt_id": f"{image_id}#g{j}"}
        if gts[j].get("n_parts", 1) > 1:
            base["metadata"]["gt_parts"] = int(gts[j]["n_parts"])
        if iou[i, j] < cfg.tau_match:
            base["metadata"].update(paired_by="containment",
                                    cov_gt=round(float(m["cov_g"][i, j]), 4),
                                    cov_pred=round(float(m["cov_p"][i, j]), 4))

        if p["cls"] != gts[j]["cls"]:                        # CASE 3: CONFUSION
            errors.append(TypedInstance(
                **base, iou=0.0, error_vector=np.array([0., 0., 1.]),
                seg_subtype=SegErrorType.CONFUSION, pred_class=p["cls"], gt_class=gts[j]["cls"]))
            continue

        # right class -> boundary/extent decomposition (CASE 1/2)
        e_bnd, e_ext, pair_iou = boundary_extent(p["mask"], gts[j]["mask"], d_pixels)
        if pair_iou >= cfg.tau_iou_good:                     # CASE 1: CORRECT (reference)
            reference.append(TypedInstance(
                **base, iou=pair_iou, error_vector=np.array([e_bnd, e_ext, 0.]),
                seg_subtype=SegErrorType.CORRECT, pred_class=p["cls"], gt_class=gts[j]["cls"]))
        else:                                                # CASE 2: BOUNDARY/EXTENT
            sub = SegErrorType.BOUNDARY if e_bnd >= e_ext else SegErrorType.EXTENT
            errors.append(TypedInstance(
                **base, iou=pair_iou, error_vector=np.array([e_bnd, e_ext, 0.]),
                seg_subtype=sub, pred_class=p["cls"], gt_class=gts[j]["cls"]))

    # --- ground truth side: swallowed by a merge, or not covered -> FN ------
    for j, g in enumerate(gts):
        iid = f"{image_id}#g{j}"
        if j in swallowed:
            # The GT side of a merge, kept as an instance of its own: the blob
            # above carries the prediction's error, this one carries the OBJECT
            # that lost its outline -- pooled on the GT mask, like a FN, so the
            # slices can still be about the swallowed objects (small persons
            # beside large ones) and not only about the blobs that hold them.
            # A merge therefore yields 1 + M instances, declared here; an
            # over-extended prediction already yielded FP + FN under the IoU rule.
            i = swallowed[j]
            meta = {"from_gt_mask": True, "swallowed_by": f"{image_id}#p{i}",
                    "covered_frac": round(float(m["cov_g"][i, j]), 4)}
            if g.get("n_parts", 1) > 1:
                meta["gt_parts"] = int(g["n_parts"])
            errors.append(TypedInstance(
                instance_id=iid, image_id=image_id, bbox=g.get("bbox"),
                iou=0.0, error_vector=np.array([0., 0., 1.]),
                seg_subtype=SegErrorType.MERGE, gt_class=g["cls"],
                pred_class=preds[i]["cls"], metadata=meta))
            continue
        if j in matched_gts:
            continue
        meta = {"from_gt_mask": True}       # BLOCK 2 must pool features on the GT mask
        if g.get("n_parts", 1) > 1:
            meta["gt_parts"] = int(g["n_parts"])
        errors.append(TypedInstance(
            instance_id=iid, image_id=image_id, bbox=g.get("bbox"),
            iou=0.0, error_vector=np.array([0., 0., 1.]),
            seg_subtype=SegErrorType.FN, gt_class=g["cls"], metadata=meta))

    return errors, reference


def decompose_dataset(
    samples: list[dict], cfg: Optional[TypedErrorConfig] = None
) -> tuple[list[TypedInstance], list[TypedInstance]]:
    """Run BLOCK 1 over a dataset.

    ``samples``: list of {"image_id": str, "preds": [...], "gts": [...]}.
    Returns (all_errors, all_reference).
    """
    cfg = cfg or TypedErrorConfig()
    all_err: list[TypedInstance] = []
    all_ref: list[TypedInstance] = []
    for s in samples:
        err, ref = decompose_image(s["image_id"], s["preds"], s["gts"], cfg)
        all_err.extend(err)
        all_ref.extend(ref)
    return all_err, all_ref
