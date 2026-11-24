"""Shared helpers for the slice_quality dataset adapters."""

from __future__ import annotations


def make_mask_lookup(samples: list[dict]):
    """Return a callable: TypedInstance -> HxW mask (for BLOCK 2 pooling).

    Parses the instance ids "img#pI" / "img#gJ" produced by
    ``typed_error.decompose_image`` and indexes back into ``samples``.
    """
    by_id = {s["image_id"]: s for s in samples}

    def lookup(inst):
        sid, _, tail = inst.instance_id.partition("#")
        s = by_id.get(sid)
        if s is None or not tail:
            return None
        kind, idx = tail[0], int(tail[1:])
        items = s["preds"] if kind == "p" else s["gts"]
        return items[idx]["mask"] if idx < len(items) else None

    return lookup
