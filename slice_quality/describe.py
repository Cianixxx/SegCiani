"""Per-image textual error descriptions -- the thesis deliverable.

For every image where the segmentation model errs, aggregate the SYSTEMATIC
failure patterns (the KEPT slices) its instances belong to into a natural-language
description: *"in this image the model fails because of pattern A, pattern B"*.

A slice's textual form comes from:
* error type   (deterministic: fn / fp / boundary / extent / confusion / split / merge),
* dominant class (from the members),
* attribute    (Qwen tag, if present: small / dark / occluded / ...),
* hypothesis   (SliceLens query, for language-branch slices).

Output: ``image_descriptions.json`` ({image_id -> description}) + an optional .txt.
This is the per-image deliverable; the slices are the reusable systematic patterns.

Where to modify
---------------
* ``_ETYPE``        : error-type -> phrase (language / wording).
* ``describe_slice`` : how a slice is verbalised.
* ``_render``       : the per-image sentence template (swap for an LLM if wanted).
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict

_ETYPE = {
    "fn": "not detected (missed)",
    "fp": "detected but non-existent (spurious)",
    "boundary": "with imprecise boundaries",
    "extent": "with wrong extent (missing or excess parts)",
    "confusion": "assigned to the wrong class",
    "split": "over-segmented (one object split into several masks)",
    "merge": "under-segmented (several objects merged into one mask)",
}


# Attributes that must never reach the user-facing text. "normal"/"other"/"error"
# assert the ABSENCE of an attribute. "dark" is different: it is a real word the
# tagger emits, but §6.2 measured it as unsupported in BOTH elicitation modes
# (a dark sofa 0.00, a dark sheep 0.00, a dark person 0.25 — at or below the
# matched-chance control), so a description built on it would be a claim the VLM
# itself will not confirm. It still counts towards purity; see the note in §6.2.
_SKIP_ATTR = (None, "normal", "other", "error", "dark")


def _class_phrase(members) -> str | None:
    """Class phrase from THESE members (per image, not the whole slice)."""
    vals = [(m.gt_class or m.pred_class) for m in members
            if (m.gt_class or m.pred_class)]
    if not vals:
        return None
    c, n = Counter(vals).most_common(1)[0]
    if n == len(vals):
        return f"of class '{c}'"
    if n / len(vals) >= 0.6:
        return f"mostly of class '{c}'"
    return None                        # genuinely mixed -> don't claim a class


def _attr_word(members) -> str | None:
    """Attribute from THESE members' Qwen tags (None if untagged/mixed)."""
    vals = [m.metadata.get("attribute") for m in members
            if m.metadata.get("attribute") not in _SKIP_ATTR]
    if not vals:
        return None
    a, n = Counter(vals).most_common(1)[0]
    return a if n / len(vals) >= 0.5 else None


def _position_phrase(members, crop: int = 520) -> str | None:
    """Where in the frame these instances sit, from their bboxes.

    Deterministic geometry, like the class: a FACT about the members, not a
    claim needing VLM confirmation, so it may be stated outright. 47 of the 80
    hand-written annotations locate the error in the frame ("on the left edge",
    "center-bottom"), and until now the pipeline could not say it at all.
    """
    cells = Counter()
    for m in members:
        if not m.bbox:
            continue
        x0, y0, x1, y1 = m.bbox
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        row = "top" if cy < crop / 3 else ("bottom" if cy > 2 * crop / 3 else "middle")
        col = "left" if cx < crop / 3 else ("right" if cx > 2 * crop / 3 else "centre")
        cells[(row, col)] += 1
    if not cells:
        return None
    (row, col), n = cells.most_common(1)[0]
    where = col if row == "middle" else (f"{row} of the frame" if col == "centre"
                                         else f"{row}-{col}")
    return where if n == len(members) else f"mostly {where}"


def _detail_clause(members) -> str | None:
    """Extra deterministic detail about THESE members, or None.

    Everything here comes from Block 1 geometry or from stored mask facts, so it
    is on the same footing as the class name: asserted, never "verified". Kept
    OUT of the main sentence on purpose -- that sentence's wording was settled by
    the §6.2 ablation and is not worth disturbing -- and rendered as a separate
    line instead.
    """
    bits = []

    pairs = Counter(m.confusion_pair for m in members if m.confusion_pair)
    if pairs:
        (gt, pred), n = pairs.most_common(1)[0]
        bits.append(f"{gt} mistaken for {pred}" + (f" ×{n}" if n > 1 else ""))

    be = [m for m in members if m.seg_subtype.value in ("boundary", "extent")]
    if be:
        iou = sum(m.iou for m in be) / len(be)
        e_bnd = sum(float(m.error_vector[0]) for m in be) / len(be)
        e_ext = sum(float(m.error_vector[1]) for m in be) / len(be)
        share = "mostly along the outline" if e_bnd >= e_ext else "mostly whole missing/excess parts"
        bious = [m.metadata["boundary_iou"] for m in be
                 if m.metadata.get("boundary_iou") is not None]
        # boundary IoU beside area IoU: the latter is scale-dependent and hides a
        # wrong outline on large objects (see mask_facts.boundary_iou)
        biou = f", boundary IoU {sum(bious) / len(bious):.2f}" if bious else ""
        bits.append(f"mean IoU {iou:.2f}{biou}, error {share} "
                    f"(E_bnd {e_bnd:.2f} vs E_ext {e_ext:.2f})")

    # FN that a same-class prediction largely covers were not missed but
    # swallowed into a neighbouring blob (mask_facts.covered_frac): say so, or
    # the pattern line contradicts the narrative on the same card
    fn = [m for m in members if m.seg_subtype.value == "fn"
          and m.metadata.get("covered_frac") is not None]
    absorbed = [m for m in fn if m.metadata["covered_frac"] >= 0.5]
    if absorbed:
        cov = sum(m.metadata["covered_frac"] for m in absorbed) / len(absorbed)
        bits.append(f"{len(absorbed)} of {len(fn)} 'missed' object"
                    f"{'' if len(fn) == 1 else 's'} actually absorbed into a "
                    f"neighbouring same-class prediction ({cov:.0%} covered)")

    # the GT side of a merge (claim_cov runs): objects that lost their outline
    sw = [m for m in members if m.seg_subtype.value == "merge"
          and m.metadata.get("from_gt_mask") and m.metadata.get("covered_frac") is not None]
    if sw:
        cov = sum(m.metadata["covered_frac"] for m in sw) / len(sw)
        bits.append(f"{len(sw)} object{'' if len(sw) == 1 else 's'} swallowed by a "
                    f"neighbouring prediction ({cov:.0%} covered)")

    thin = [m for m in members if m.metadata.get("n_thin_lost")]
    if thin:
        nt = sum(m.metadata["n_thin_lost"] for m in thin)
        px = sum(m.metadata.get("thin_lost_px", 0) for m in thin)
        bits.append(f"{nt} thin fragment{'' if nt == 1 else 's'} of the outline lost "
                    f"({px} px)")

    holed = [m for m in members if m.metadata.get("n_holes")]
    if holed:
        nh = sum(m.metadata["n_holes"] for m in holed)
        frac = sum(m.metadata.get("hole_frac", 0.0) for m in holed) / len(holed)
        bits.append(f"{nh} internal hole{'' if nh == 1 else 's'} in the predicted mask "
                    f"of {len(holed)} object{'' if len(holed) == 1 else 's'} "
                    f"({frac:.1%} of the mask area)")

    pos = _position_phrase(members)
    if pos:
        bits.append(pos)

    return "; ".join(bits) if bits else None


def _dominant_error_type(members) -> str:
    vals = [m.seg_subtype.value for m in members]
    return Counter(vals).most_common(1)[0][0] if vals else "error"


def _dominant_class(members) -> str | None:
    """Raw dominant class name of THESE members (>=60% majority), else None."""
    vals = [(m.gt_class or m.pred_class) for m in members
            if (m.gt_class or m.pred_class)]
    if not vals:
        return None
    c, n = Counter(vals).most_common(1)[0]
    return c if n / len(vals) >= 0.6 else None


def _art(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


# The 7 canonical class-agnostic hypotheses. These name no class, so the reader
# cannot tell WHICH object of the image the pattern is about -- we prefix the
# dominant class of THIS image's members. The class is NOT folded into the quoted
# claim: it comes from the ground truth (a fact), while the quoted hypothesis is
# what the pipeline asserts and what was grounded/verified. Measured (§6.2): the
# same instances score 0.784 when the VLM is asked the situation alone and 0.598
# when the class is inside the query -- the conjunctive question shifts the model's
# attention to "is this a cow?" and away from "is it cut off?". Keeping the class
# outside the quotes preserves both readability and honest attribution.
_CANONICAL = (
    "a small object",
    "an object occluded by another object",
    "an object that blends into its background",
    "an object cut off at the image border",
    "an object with thin or elongated parts",
    "several overlapping objects of the same type",
    "an object that looks similar to a different object category",
)

# Extra class-agnostic hypotheses (Oct 2026), aimed at the error types the 7
# canonical ones leave uncovered on val: hallucinations (pictures, reflections,
# toys), boundary/extent (holes, thin occluders) and misses (viewpoint, close-up,
# context). Same rules as above: one visible situation, no class name, nothing
# photometric (dark/blurry collapsed in the precision eval, §6.2). This tuple is
# the single source: examples/ground_extra_hypotheses.py writes its query file
# from it, so the grounded text and the rendered text cannot drift apart.
_GENERIC_EXTRA = (
    "a picture, poster or screen showing an object",
    "a reflection of an object in a mirror, window or water",
    "a toy, statue or model of an object",
    "an object with holes or see-through parts",
    "an object behind a fence, net, bars or branches",
    "an object seen from an unusual viewpoint or in an unusual pose",
    "a very large object seen up close that fills most of the image",
    "an object in an unusual setting for its kind",
)


def _situation_phrase(hypothesis: str, members) -> str:
    """(class prefix, quoted claim) rendered as 'cow -- "an object cut off ..."'."""
    cls = (_dominant_class(members)
           if hypothesis in _CANONICAL or hypothesis in _GENERIC_EXTRA else None)
    return f'{cls} — "{hypothesis}"' if cls else f'"{hypothesis}"'


def describe_instances(members, slice_, quality) -> str:
    """Describe the given instances (an image's members in one slice) accurately.

    Language (SliceLens) slices split the work: SliceLens gives the SITUATION (the
    hypothesis), the cluster/typed-error gives the ERROR (dominant type of THESE
    members) -> 'cow -- "an object cut off at the image border" -> not detected'.
    Cluster slices use only these instances' class/attribute (never claims a class
    not in the image). For the canonical class-agnostic hypotheses the dominant
    class of the image's OWN members is prefixed OUTSIDE the quotes: the quoted
    text is the verified claim, the prefix is ground-truth metadata.
    """
    if slice_.source == "language" and slice_.hypothesis:
        et = _dominant_error_type(members)
        situation = _situation_phrase(slice_.hypothesis, members)
        return f'{situation} (SliceLens) -> {_ETYPE.get(et, et)}'
    phrase = _ETYPE.get(quality.dominant_error_type.value,
                        quality.dominant_error_type.value)
    attr, clsp = _attr_word(members), _class_phrase(members)
    parts = [f"{attr} objects" if attr else "objects"]
    if clsp:
        parts.append(clsp)
    if not attr:
        # No verified attribute: say what the certificate actually supports —
        # these instances recur together in appearance space — instead of a
        # class-only phrase. §6.2 measured "a bottle" at precision 0.536 against
        # a same-class control of 0.557 (lift −0.021): stating the class alone
        # tells the reader nothing the ground truth had not already told them.
        parts.append("grouped by appearance only (no verified attribute)")
    parts.append(phrase)
    return " ".join(parts)


def _render(image_id, n_total, n_sys, n_iso, patterns, near_misses=()) -> str:
    obj = "object" if n_total == 1 else "objects"
    if n_sys == 0:
        lines = [f"Image {image_id}: {n_total} wrong {obj}, "
                 f"no systematic pattern (isolated errors)."]
    else:
        lines = [f"Image {image_id}: {n_total} wrong {obj}, "
                 f"{n_sys} in systematic patterns"
                 + (f" and {n_iso} isolated." if n_iso else "."),
                 "Systematic patterns:"]
        for p in sorted(patterns, key=lambda x: -x["n_instances"]):
            n = p["n_instances"]
            lines.append(f"  - {p['description']} "
                         f"[{p['slice_id']}, {n} obj{'' if n == 1 else 's'}.]")
            if p.get("details"):
                lines.append(f"      · {p['details']}")
    for nm in sorted(near_misses, key=lambda x: (-x["n_instances"], x["slice_id"])):
        n = nm["n_instances"]
        lines.append(f"  ~ {n} isolated obj{'' if n == 1 else 's'} near candidate "
                     f"pattern {nm['slice_id']} ({nm['verdict']}, not certified)")
    return "\n".join(lines)


def _subtype_severity_stats(result) -> dict:
    """{subtype -> (mean, std) of (1 - IoU)} over ALL unique error instances.

    Raw severity (1 - IoU) is NOT comparable across error types: FN/FP sit at
    1.0 by construction, while boundary errors live around 0.25-0.45. A flat
    threshold therefore always admits FN slices and NEVER admits boundary
    slices, however systematic. Normalising within each subtype fixes the bias.
    """
    vals = defaultdict(list)
    seen = set()
    for s, _q in result.scored:
        for m in s.members:
            if m.instance_id in seen:
                continue
            seen.add(m.instance_id)
            vals[m.seg_subtype.value].append(1.0 - m.iou)
    stats = {}
    for st, v in vals.items():
        mu = sum(v) / len(v)
        var = sum((x - mu) ** 2 for x in v) / len(v)
        stats[st] = (mu, var ** 0.5)
    return stats


def _severity_z(members, stats) -> float:
    """Mean subtype-normalised severity z-score of these members."""
    if not members:
        return 0.0
    zs = []
    for m in members:
        mu, sd = stats.get(m.seg_subtype.value, (0.0, 0.0))
        zs.append(((1.0 - m.iou) - mu) / sd if sd > 1e-9 else 0.0)
    return sum(zs) / len(zs)


def build_image_descriptions(result, lang_omega2_min: float = 0.0,
                             lang_conf_min: float = 0.6,
                             lang_min_size: int = 3,
                             lang_sev_z_min: float = -0.25) -> dict:
    """{image_id -> {n_errors, n_systematic, n_isolated, patterns, text}}.

    Cluster patterns = KEPT slices (Def.1). Language (SliceLens) patterns pass an
    adapted gate: subtype-normalised severity z >= ``lang_sev_z_min``, verified
    systematic, mean slice_confidence >= ``lang_conf_min``, size >=
    ``lang_min_size`` -- the axes that fit semantic hypotheses (no geometric
    compactness / error-type purity).

    Verification: a language slice's significance is SliceLens's own slope-trend
    verdict (``is_systematic``, Sec. 3.2.2) when the run recorded it -- it tests
    the error-vs-confidence association on error AND correct groundings, which
    our instance-level tests (built for cluster slices) never see. Fallback for
    old runs: the SQV significance axis. ``lang_omega2_min`` defaults to 0 (the
    one-vs-rest omega^2 is size-confounded for small slices: reported, not
    gated). ``lang_conf_min`` is the mean over ALL members with conf >= 0.5
    (a lower scale than the old top-10 mean; hence 0.6, not 0.7).
    ``lang_sev_z_min`` = -0.25: the slice mean z is a noisy average around 0
    for FN-dominated slices, so an exact >= 0 knife-edge rejects half of them
    by chance; the gate only needs to exclude slices MEANINGFULLY milder than
    typical for their type (e.g. high-IoU boundary groups).
    """
    # Patterns = KEPT cluster slices (Def.1) PLUS language slices that pass an
    # ADAPTED gate. The full Def.1 (with geometric compactness / error-type purity)
    # is wrong for language slices -- those group by semantic hypothesis, not
    # geometry. So we validate them only on axes that fit: severity (type-aware:
    # z-score of 1-IoU within each subtype, so a boundary slice competes against
    # boundary errors, not against FN's automatic 1.0), reliability
    # (significance), VLM agreement (mean slice_confidence) and size.
    sev_stats = _subtype_severity_stats(result)
    pool = {k.slice_.slice_id: (k.slice_, k.quality) for k in result.kept}
    seen_lang = set()
    for s, q in result.scored:
        if s.source != "language" or q.slice_id in pool or q.slice_id in seen_lang:
            continue
        seen_lang.add(q.slice_id)
        conf = (sum(s.membership.values()) / len(s.membership)) if s.membership else 0.0
        sev_z = _severity_z(s.members, sev_stats)
        # SliceLens's slope-trend verdict when recorded; SQV significance otherwise.
        sys_flag = (getattr(s, "metadata", None) or {}).get("is_systematic")
        verified = q.significant if sys_flag is None else bool(sys_flag)
        # valid slice = severe FOR ITS ERROR TYPE (cluster/typed-error side) AND
        # high confidence (SliceLens VLM agreement) AND verified systematic --
        # mirrors SliceLens's error-vs-confidence idea without the
        # flat-threshold FN bias.
        if (sev_z >= lang_sev_z_min and conf >= lang_conf_min
                and q.omega_squared >= lang_omega2_min and verified
                and q.size >= lang_min_size):
            pool[q.slice_id] = (s, q)
    pool_ids = set(pool)

    img_all = defaultdict(set)                          # image -> {instance_id} (all errors)
    img_slice = defaultdict(lambda: defaultdict(list))  # image -> slice_id -> [members here]
    # candidate (non-certified) cluster slice each instance last belonged to:
    # every error is hard-assigned to SOME cluster component each peeling round,
    # so an "isolated" instance always has a near-miss candidate -- the slice
    # that failed certification. Later rounds overwrite (most refined view).
    cand_of: dict[str, tuple] = {}                      # instance_id -> (sid, verdict)
    seen = set()
    for s, q in result.scored:
        if q.slice_id in seen:
            continue
        seen.add(q.slice_id)
        for m in s.members:
            img_all[m.image_id].add(m.instance_id)
            if q.slice_id in pool_ids:
                img_slice[m.image_id][q.slice_id].append(m)
            elif s.source == "cluster":
                cand_of[m.instance_id] = (q.slice_id, getattr(q, "verdict", "?"))

    out = {}
    for image_id, all_ids in img_all.items():
        patterns, sys_ids = [], set()
        for sid, members in img_slice[image_id].items():
            s, q = pool[sid]
            patterns.append({
                "slice_id": sid, "branch": s.source,
                "error_type": q.dominant_error_type.value,
                "description": describe_instances(members, s, q),  # per-image accurate
                "details": _detail_clause(members),
                "n_instances": len(members)})
            sys_ids.update(m.instance_id for m in members)
        n_total, n_sys = len(all_ids), len(sys_ids)
        n_iso = n_total - n_sys
        # near-misses: candidate patterns of THIS image's isolated instances
        nm_count = defaultdict(int)
        for iid in all_ids - sys_ids:
            if iid in cand_of:
                nm_count[cand_of[iid]] += 1
        # ties broken by slice id: the set iteration above follows the string
        # hash seed, and two regenerations must write the same text
        near = [{"slice_id": sid, "verdict": verdict, "n_instances": n}
                for (sid, verdict), n in sorted(nm_count.items(), key=lambda x: (-x[1], x[0][0]))]
        out[image_id] = {
            "n_errors": n_total, "n_systematic": n_sys, "n_isolated": n_iso,
            "patterns": patterns, "near_misses": near,
            "text": _render(image_id, n_total, n_sys, n_iso, patterns, near),
        }
    return out


def export_descriptions(descs, json_path, txt_path=None):
    """Write the per-image descriptions to JSON (+ optional readable .txt)."""
    os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(descs, f, ensure_ascii=False, indent=2)
    if txt_path:
        with open(txt_path, "w", encoding="utf-8") as f:
            for image_id in sorted(descs):
                f.write(descs[image_id]["text"] + "\n\n")
    return json_path


def export_description_index(descs, json_path, txt_path=None):
    """The INVERSE map: textual description -> all images it was assigned to.

    One record per distinct (description, slice) pair, with the branch it comes
    from (cluster or language/SliceLens). Note that one slice can yield slightly
    different descriptions in different images (the wording uses each image's
    OWN members), so a slice may appear with more than one description -- each
    listed with exactly its images.
    """
    groups = {}                                   # (desc, slice_id) -> record
    for image_id in sorted(descs):
        for p in descs[image_id]["patterns"]:
            key = (p["description"], p["slice_id"])
            rec = groups.setdefault(key, {
                "description": p["description"],
                "branch": p["branch"],
                "slice_id": p["slice_id"],
                "error_type": p["error_type"],
                "images": [],
            })
            rec["images"].append(image_id)
    records = sorted(groups.values(),
                     key=lambda r: (r["branch"], -len(r["images"]), r["slice_id"]))
    for r in records:
        r["n_images"] = len(r["images"])

    os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"n_descriptions": len(records), "descriptions": records},
                  f, ensure_ascii=False, indent=2)

    if txt_path:
        lines = [f"{len(records)} descriptions (validated patterns only; "
                 "grouped by branch, then #images)", ""]
        for branch in ("language", "cluster"):
            recs = [r for r in records if r["branch"] == branch]
            src = "SliceLens" if branch == "language" else "cluster"
            lines.append(f"######## {src.upper()} ({len(recs)} descriptions) ########")
            lines.append("")
            for r in recs:
                lines.append(f'== {r["description"]}')
                lines.append(f'   [{src} | {r["slice_id"]} | {r["error_type"]} '
                             f'| {r["n_images"]} images]')
                for img in r["images"]:
                    lines.append(f"   {img}")
                lines.append("")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    return json_path
