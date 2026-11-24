"""Load a saved SliceLens run -> TypedInstances + language slices (no masks/GPU).

Lets you test the scoring / language branch / report by REUSING a past SliceLens
run JSON (e.g. voc_broad3_qwen.json), so no LLM/VLM or backbone is re-run.

Expected JSON (SliceLens eval output):
    {"label": ..., "candidates": [
        {"query": "...", "top_instances": [
            {"name": "2007_000042", "label": "train", "iou": 0.86,
             "bbox": [...], "slice_confidence": 0.97, "is_error": false}, ...]},
        ...]}

Caveats (this is a convenience path, not the full pipeline):
* No masks -> no boundary/extent/split/merge split. The typed error is COARSE:
  CORRECT if not is_error, else FN/EXTENT by IoU. For the real typed taxonomy run
  the mask-based BLOCK 1 (typed_error.decompose_dataset).
* No backbone -> we attach a cheap GEOMETRIC embedding (bbox + iou + class), so
  clustering / manifold compactness still produce sensible structure on the side.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Optional

import numpy as np

from ..data_types import SegErrorType, Slice, TypedInstance


def _coarse_subtype(is_error: bool, iou: float) -> tuple[SegErrorType, np.ndarray, float]:
    """Map SliceLens (is_error, iou) -> (subtype, error_vector, perf)."""
    if not is_error:
        return SegErrorType.CORRECT, np.array([0., 0., 0.]), float(iou)
    if iou <= 1e-3:                       # no overlap -> missed / hallucinated
        return SegErrorType.FN, np.array([0., 0., 1.]), 0.0
    return SegErrorType.EXTENT, np.array([0., max(0., 1. - iou), 0.]), float(iou)


def load_slicelens_run(
    json_path: str,
    *,
    conf_threshold: float = 0.5,
    instance_key: str = "top_instances",
) -> tuple[list[TypedInstance], list[TypedInstance], list[Slice]]:
    """Return (errors, reference, language_slices) from a SliceLens run JSON."""
    data = json.load(open(json_path, encoding="utf-8"))
    candidates = data.get("candidates", [])

    # Deduplicate instances across candidates by (name, bbox).
    insts: dict[tuple, TypedInstance] = {}

    def _key(it):
        return (it["name"], tuple(round(float(x), 1) for x in it["bbox"]))

    for cand in candidates:
        for it in cand.get(instance_key, []):
            k = _key(it)
            if k in insts:
                continue
            sub, evec, perf = _coarse_subtype(bool(it.get("is_error")), float(it.get("iou", 0.0)))
            insts[k] = TypedInstance(
                instance_id=f"{it['name']}#{len(insts)}",
                image_id=it["name"],
                iou=perf, error_vector=evec, seg_subtype=sub,
                pred_class=it.get("label"), gt_class=it.get("label"),
                bbox=tuple(it["bbox"]),
                metadata={"attribute": it.get("label")},   # cheap proxy for slice_purity
            )

    by_key = {k: v for k, v in insts.items()}
    all_inst = list(insts.values())
    errors = [t for t in all_inst if t.seg_subtype is not SegErrorType.CORRECT]
    reference = [t for t in all_inst if t.seg_subtype is SegErrorType.CORRECT]

    # Language slices: one per candidate, members = instances above threshold.
    language: list[Slice] = []
    for ci, cand in enumerate(candidates):
        members, membership = [], {}
        for it in cand.get(instance_key, []):
            if float(it.get("slice_confidence", 0.0)) < conf_threshold:
                continue
            t = by_key.get(_key(it))
            if t is None or t.seg_subtype is SegErrorType.CORRECT:
                continue
            members.append(t)
            membership[t.instance_id] = float(it["slice_confidence"])
        if members:
            language.append(Slice(
                slice_id=f"lang_{ci}", members=members, source="language",
                hypothesis=cand.get("query"), membership=membership,
            ))

    return errors, reference, language


def run_image_names(json_path: str, instance_key: str = "top_instances") -> list[str]:
    """The set of image stems referenced by a saved SliceLens run."""
    data = json.load(open(json_path, encoding="utf-8"))
    names: set[str] = set()
    for cand in data.get("candidates", []):
        for it in cand.get(instance_key, []):
            names.add(it["name"])
    return sorted(names)


def build_language_slices_from_run(
    json_path: str,
    typed_instances: list[TypedInstance],
    *,
    conf_threshold: float = 0.5,
    bbox_iou_threshold: float = 0.4,
    instance_key: str = "top_instances",
    prefix: str = "lang",
    id_map: Optional[dict] = None,
) -> list[Slice]:
    """Map a saved SliceLens run onto EXISTING (mask-based, DINOv2) TypedInstances.

    Unlike ``load_slicelens_run`` (which builds coarse instances), this links each
    candidate's instances to the real instances, so the language slices share the
    same embedding space as the cluster branch -- which is what makes their
    manifold-compactness score a genuine (non-circular) test.

    Linking, in order of preference:
    1. EXACT join by ``instance_id`` -- available when the run used the shared
       BLOCK-1 substrate (``run_voc.py --substrate cc``): zero mis-pairs, no
       thresholds.
    2. Fallback (legacy runs without ids): (image stem, bbox IoU >=
       ``bbox_iou_threshold``) heuristic. Coarse when the legacy class-level
       bbox spans several objects.
    """
    from collections import defaultdict

    from ..language_branch import _bbox_iou

    data = json.load(open(json_path, encoding="utf-8"))
    by_id = {t.instance_id: t for t in typed_instances}
    by_img: dict[str, list[TypedInstance]] = defaultdict(list)
    for t in typed_instances:
        by_img[t.image_id].append(t)

    slices: list[Slice] = []
    for ci, cand in enumerate(data.get("candidates", [])):
        # "top_instances" holds only the top-10 by confidence (a display cut in
        # run_voc.py) -- building slices from it CAPS every language slice at 10
        # members and inflates the mean confidence. When the run recorded ids
        # (cc substrate), use the FULL "scored" list instead.
        items = cand.get(instance_key, [])
        scored = cand.get("scored", [])
        if scored and any("instance_id" in it for it in scored):
            items = scored
        members, membership, seen = [], {}, set()
        for it in items:
            conf = float(it.get("slice_confidence", it.get("conf", 0.0)))
            if conf < conf_threshold:
                continue
            if id_map is not None:
                # geometric join (3 November): the substrate object the VLM read -> the final error it is; whether
                # it is an error is decided by the final typing, not by the substrate's label
                best = by_id.get(id_map.get(it.get("instance_id")))
                if best is not None and best.instance_id not in seen:
                    members.append(best)
                    membership[best.instance_id] = conf
                    seen.add(best.instance_id)
                continue
            if not it.get("is_error", True):
                continue        # correct instances feed the slope, not the slice
            best = by_id.get(it.get("instance_id"))      # 1. exact join by id
            if best is None and it.get("bbox"):          # 2. legacy bbox fallback
                box = tuple(it["bbox"])
                best_iou = bbox_iou_threshold
                for t in by_img.get(it["name"], []):
                    if t.bbox is None:
                        continue
                    iou = _bbox_iou(box, t.bbox)
                    if iou >= best_iou:
                        best, best_iou = t, iou
            if best is not None and best.instance_id not in seen:
                members.append(best)
                membership[best.instance_id] = conf
                seen.add(best.instance_id)
        if members:
            slices.append(Slice(
                slice_id=f"{prefix}_{ci}", members=members, source="language",
                hypothesis=cand.get("query"), membership=membership,
                # SliceLens's own verification verdict (slope-trend, Sec. 3.2.2):
                # the natural significance test for language slices -- it uses
                # error AND correct groundings, which our instance-level tests
                # (built for cluster slices) never see.
                metadata={"is_systematic": cand.get("is_systematic")},
            ))
    return slices


def attach_geometric_embeddings(
    instances: list[TypedInstance], image_wh: float = 520.0
) -> list[TypedInstance]:
    """Cheap stand-in for BLOCK 2: embedding = [bbox geometry, iou, class one-hot].

    Enough structure for clustering / manifold compactness to be non-trivial when
    no backbone is available. Replace with DINOv2 features for real runs.
    """
    classes = sorted({i.pred_class or "?" for i in instances})
    cidx = {c: k for k, c in enumerate(classes)}
    for inst in instances:
        if inst.bbox is None:
            continue
        x0, y0, x1, y1 = inst.bbox
        w, h = (x1 - x0) / image_wh, (y1 - y0) / image_wh
        cx, cy = (x0 + x1) / 2 / image_wh, (y0 + y1) / 2 / image_wh
        geom = np.array([cx, cy, w, h, w * h, (w / (h + 1e-6)), inst.iou], dtype=np.float32)
        onehot = np.zeros(len(classes), dtype=np.float32)
        onehot[cidx[inst.pred_class or "?"]] = 1.0
        inst.embedding = np.concatenate([geom, onehot])
    return instances
