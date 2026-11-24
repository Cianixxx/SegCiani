"""BLOCK 9 -- Repair recommendation (the prescription).

Source : new (the dominant error type IS the routing key, enabled by Beyond IoU).
Input  : a kept Slice's SliceQuality + its ExplanationCard.
Output : a RepairAdvice. This block RECOMMENDS, it does NOT touch the model.

The mechanism (error type) chooses the fix family; the attributes (from the
explanation) say which data to target.

Where to modify
---------------
* ``REPAIR_BY_TYPE`` : the error-type -> fix-family table (extend as needed).
"""

from __future__ import annotations

from dataclasses import dataclass

from .data_types import SegErrorType, SliceQuality
from .explanation import ExplanationCard

# Error mechanism -> repair family (see thesis discussion).
REPAIR_BY_TYPE: dict[SegErrorType, str] = {
    SegErrorType.BOUNDARY: "boundary refinement (boundary loss / higher-res mask head / SAM/CRF post-proc)",
    SegErrorType.EXTENT: "context augmentation (random-erasing/CutMix on non-discriminative parts)",
    SegErrorType.CONFUSION: "classification fix (hard-negative mining on the confused class pair)",
    SegErrorType.FP: "raise score threshold / stronger NMS / add hard-negative backgrounds",
    SegErrorType.FN: "oversample/copy-paste the missed objects / focal-loss reweighting",
    SegErrorType.SPLIT: "mask merging / NMS tuning / instance-unity in the mask head",
    SegErrorType.MERGE: "instance separation (watershed / boundary loss between adjacent instances)",
}


@dataclass
class RepairAdvice:
    slice_id: str
    fix_family: str
    target_data: str
    cost_hint: str
    priority: float
    note: str
    text: str = ""


def recommend(quality: SliceQuality, card: ExplanationCard) -> RepairAdvice:
    et = quality.dominant_error_type
    fix = REPAIR_BY_TYPE.get(et, "manual inspection")
    target = ", ".join(t for t, _, _ in card.over_represented[:3]) or "all members of the slice"

    # cheap fixes for post-processing types, retrain for the rest
    cheap = et in (SegErrorType.FP, SegErrorType.SPLIT, SegErrorType.MERGE)
    cost = "inference-time / post-processing" if cheap else "last-layer retrain or reweighting"
    priority = float(quality.omega_squared
                     * (quality.misprediction_coherence + quality.error_type_purity))

    note = ("verify on the FULL val set, not just this slice -- repairing one "
            "error type can shift the error elsewhere (e.g. FP fix raises FN).")
    text = (f"Slice {quality.slice_id} [{et.value}] -> {fix}; "
            f"target: {target}; cost: {cost}; priority {priority:.2f}.")

    return RepairAdvice(
        slice_id=quality.slice_id, fix_family=fix, target_data=target,
        cost_hint=cost, priority=priority, note=note, text=text,
    )
