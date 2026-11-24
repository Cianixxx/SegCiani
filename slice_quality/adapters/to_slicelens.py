"""Bridge: BLOCK-1 TypedInstances -> SliceLens ``Instance``s (shared substrate).

Motivation (improvement #1). Historically the two branches derived instances
INDEPENDENTLY: BLOCK 1 splits masks into connected components (one instance per
object), while ``slicelens.adapters.voc_seg`` built one instance per
(image, class) -- the union of all objects of that class. Linking them back
required a bbox-IoU >= 0.4 heuristic that loses instances (a bbox spanning two
chairs matches neither chair well) and can mis-pair.

This bridge removes the heuristic at the source: extract instances ONCE with
BLOCK 1, then hand the SAME instances (same ``instance_id``!) to SliceLens for
grounding. Matching the run back onto the typed instances becomes an exact join
by id (see ``build_language_slices_from_run``), with bbox-IoU kept only as a
fallback for legacy run JSONs.

Note the dependency direction: slice_quality -> slicelens (already the case);
the ``slicelens`` package stays a pure, standalone paper implementation --
the branches remain independent discovery paths (open-world constraint).
"""

from __future__ import annotations

from typing import Optional, Sequence

from slicelens.data_types import ErrorType, Instance, Region

from ..data_types import SegErrorType, TypedInstance

# SegErrorType -> the (coarser) SliceLens ErrorType. Mask-quality errors
# (boundary/extent/split/merge) map to FN, mirroring the convention of the old
# class-level VOC adapter ("missed / under-segmented / bad boundary" -> FN).
# The exact subtype is preserved in Instance.metadata["seg_subtype"].
_ETYPE_MAP = {
    SegErrorType.CORRECT: ErrorType.CORRECT,
    SegErrorType.FN: ErrorType.FALSE_NEGATIVE,
    SegErrorType.FP: ErrorType.FALSE_POSITIVE,
    SegErrorType.CONFUSION: ErrorType.MISCLASSIFICATION,
    SegErrorType.BOUNDARY: ErrorType.FALSE_NEGATIVE,
    SegErrorType.EXTENT: ErrorType.FALSE_NEGATIVE,
    SegErrorType.SPLIT: ErrorType.FALSE_NEGATIVE,
    SegErrorType.MERGE: ErrorType.FALSE_NEGATIVE,
}


def typed_to_slicelens_instances(
    errors: Sequence[TypedInstance],
    reference: Sequence[TypedInstance],
    image_paths: dict[str, str],
    *,
    only_label: Optional[str] = None,
) -> list[Instance]:
    """TypedInstances (BLOCK 1) -> SliceLens Instances, ids preserved.

    ``image_paths``: image_id -> path of the SAME crop the masks/bboxes align to
    (the VLM must see the coordinates system the bbox lives in).
    ``only_label``: keep instances of one class (gt_class or pred_class), like
    the ``--label`` filter of the per-class SliceLens runs.

    Slope-trend analysis needs BOTH labels, so pass errors AND reference.
    """
    out: list[Instance] = []
    for t in list(errors) + list(reference):
        label = t.gt_class or t.pred_class
        if only_label and label != only_label:
            continue
        if t.bbox is None or t.image_id not in image_paths:
            continue
        out.append(Instance(
            instance_id=t.instance_id,                      # the shared join key
            image_path=image_paths[t.image_id],
            region=Region(bbox=tuple(t.bbox)),
            label=label,
            is_error=t.seg_subtype is not SegErrorType.CORRECT,
            error_type=_ETYPE_MAP.get(t.seg_subtype, ErrorType.FALSE_NEGATIVE),
            metadata={"name": t.image_id, "iou": round(float(t.iou), 3),
                      "seg_subtype": t.seg_subtype.value},
        ))
    return out
