"""Per-image error NARRATIVES, written the way the human annotators wrote them.

This is a different deliverable from ``describe.py`` and does not replace it:

* ``describe.py``        answers "which SYSTEMATIC pattern does this image's
  failure belong to" -- it only speaks about instances that fell inside a
  certified slice, and it speaks in the vocabulary of the slices.
* ``describe_errors.py`` answers "what went wrong in THIS image" -- every typed
  error, systematic or isolated, in the register the 80 hand-written annotations
  use, so the two texts can be put side by side and compared.

The reference register, from the annotations themselves:

    "The ground truth AEROPLANE regions have been segmented in a coarser and
     incomplete way, especially regarding the wings, and two tiny AEROPLANE
     patches have been hallucinated on the right edge."

Three things carry that sentence, and all three are already deterministic
outputs of Block 1 -- nothing here is a VLM claim:

  what   the typed error   (fn / fp / boundary / extent / confusion / split / merge)
  which  the class         (gt_class, or pred_class for a hallucination)
  where  the position      (from the bbox, as a 3x3 grid + frame-edge contact)

Mask facts are read from ``metadata`` when a caller has put them there
(``narrate_errors.py`` does): ``frame_frac`` gives the "tiny / small / large"
qualifier; ``missed_px`` / ``excess_px`` give an EXTENT error its direction
(under- vs over-segmentation), which the Beyond IoU vector alone cannot say --
[E_bnd, E_ext, E_seg] gives the magnitude of the extent error, not its sign; and
``boundary_iou`` measures how well the outline is traced. Absent those keys the
sentence is simply less specific, never wrong.

Boundary IoU is carried as a CONTINUOUS FACT, not as a new error type. Area IoU
is scale-dependent -- on the largest objects it sits 0.595 above boundary IoU,
against 0.217 on the smallest -- so a big region can be "correct" at IoU 0.97
with an outline that is essentially wrong. Reporting the number, rather than
reclassifying the instance, fixes what the reader sees without creating
instances, without a GPU pass, and without shrinking the reference population
that Block 6 needs for omega^2. See ``_contour_sentence``.

What the annotators name first (revision of September 2026)
-----------------------------------------------------------
Read card by card against the 79 hand-written notes, the low judge scores had
five causes, and the sentences now carry what those notes say first:

* DIRECTION and SIDE on every pair, merge and correct region, from the missed
  and excess pixels and the sides of the GT box they lie on (``excess_sides`` /
  ``missed_sides``, ``gt_px``): "spilling over the object on the top and right
  (8% of its area in excess)"; silent under ``EXTENT_MIN`` = 3% of the area.
* a sentence for every CORRECT region not already covered by a contour note,
  graded by boundary IoU ("segmented closely", "segmented well, with slightly
  coarse boundaries"); "No segmentation error was found" only at IoU >= 0.99.
* a same-class FP within ``ADJ_PX`` of a GT of its class is the mask SPILLING,
  not a hallucination (``fp_spill``); tiny hallucinated patches (``TINY_FRAC``)
  are gathered in one closing line.
* sub-threshold regions and patches (``subthreshold``, ``subthreshold_pred``)
  are measured but NOT narrated: the sentences that said them spoke about the
  instrument rather than about the image (see the note above those builders).
* an opening VERDICT when the failure is global (``VERDICT_MIN``: over half of
  the annotated area labelled with a wrong class or as background); any group
  covering ``BIG_FRAC`` of the frame is told first whatever its type; "on the
  right half of the scene" when every patch shares a half.
* a merge is told against the UNION of the GTs it swallowed (``union_iou``),
  and a GT object cut by an occluder says so ("partly hidden behind the X").

Which class took the place of which (revision of October 2026)
---------------------------------------------------------------
The facts the hand verdicts found missing most often (20 of 59) were of one kind:
"the DININGTABLE region is often misclassified as CHAIR", "some area is labelled
SOFA instead of CHAIR". The narrative told the two halves apart -- "the CHAIR has
been under-segmented, 46% missing" and "three small SOFA patches have been
hallucinated" -- and never said that the patches lie ON the chair. Now:

* an FP patch lying on a GT object of another class (``over_gt``) is told as part
  of THAT object labelled with the patch's class: "Parts of the ground truth CHAIR
  have been labelled SOFA instead (three small SOFA patches on the top edge)";
* an object whose missing part is predicted with another class (``relabelled_as``,
  at least 3% of its area and 200 px) gets "Part of the DININGTABLE (30% of its
  area) has been labelled CHAIR instead", unless a patch sentence already said
  the same pair.

Where exactly, and the small things (revision v6, October 2026)
---------------------------------------------------------------
The next facts the annotators wrote and the narrative missed: the CORNER of a defect
("incomplete in the top-left corner"), the smaller of two opposite defects ("slight
over-extension on the bottom-right" beside a gap elsewhere), WHERE an outline is
ragged ("coarse boundaries, especially towards the bottom-right"), two classes on one
object ("misclassified with DOG and COW"), small hallucinated patches ("a small PERSON
region on the right") and holes in a region that is otherwise correct. All read from
the masks (mask_facts: cells of a 3x3 grid, ``boundary_where``, ``small_patches``,
``relabelled_as`` as a list, ``n_holes`` on correct regions); thresholds are the
narrative's own (EXTENT_MIN, SIDE_MIN) or stated where they are set. Tried and left
out: saying the smaller of two opposite defects too -- it changed 900 sentences of
1491 narratives for about one fact of the annotations.

Where to modify
---------------
* ``_sentence``  : one template per error type (the wording).
* ``_where``     : the position vocabulary.
* ``narrate``    : grouping, ordering and the cap on how much is said.
"""

from __future__ import annotations

from collections import Counter, defaultdict

# how many (error type, class) groups an image's narrative may mention before
# it stops reading like an annotation and starts reading like a log dump
MAX_GROUPS = 6
EDGE_PX = 10                      # bbox this close to the border touches it

# When an area-correct region gets a remark about its outline. Both are REPORTING
# thresholds, not taxonomy ones: nothing is reclassified as an error, so the
# typed-error counts and the omega^2 reference population are untouched.
#
# A single cut does not work, and the measurement says why. Over the 522 reference
# instances the boundary IoU has median 0.393 and first quartile 0.300, and the
# instances in the six images the hand verdicts marked BASSA sit at 0.037, 0.141,
# 0.238, 0.281, 0.287, 0.327, 0.368 -- inside the bulk, not in the tail. Catching
# all of them with one threshold would mean remarking on HALF the correct regions,
# which says nothing. On this model a mediocre contour is simply the norm.
#
# What actually distinguishes those six is that they have no typed error at all:
# the contour is the only thing there is to say about them. Hence two regimes,
# both anchored to measured quantiles rather than to a chosen constant:
CONTOUR_ALONE = 0.39     # no typed error in the image -> remark below the median
CONTOUR_BESIDE = 0.30    # errors present -> only the worst quartile, or it dilutes

# How many thin missed fragments before the narrative says details were lost.
# Thin loss is present in 90% of images, so its PRESENCE says nothing and only
# the amount can: the count separates best (AUC 0.735 against "the annotation
# mentions details/wings/legs/...", where pixels give 0.712 and the area share
# 0.659). At 8 the rule fires on 31 of 79 images with precision 0.45 against a
# base rate of 0.25 and recall 0.70.
#
# Precision here is NOT accuracy. The label is "did the annotator happen to
# mention it", and an annotation carries 2.1 claims on average, so it cannot
# mention everything. Thin loss is a deterministic mask fact: it is true whether
# or not the human wrote it down. The number measures salience, not correctness.
THIN_LOST_MIN = 8
# Share of the GT area a spill / a gap must reach before the sentence names it:
# below, it is the ordinary raggedness the boundary IoU already grades.
EXTENT_MIN = 0.03
# A group covering this share of the frame is told first whatever its type: an
# error over half the image is the story, not the three tiny patches beside it.
BIG_FRAC = 0.15
# Above this share of the annotated area labelled with one wrong class (or left
# as background) the narrative opens with a verdict on the whole prediction.
VERDICT_MIN = 0.5
TINY_FRAC = 0.005            # patches under this share of the frame are "tiny"

_NUM = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
        6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}


def _n_word(n: int) -> str:
    return _NUM.get(n, str(n))


def _cls(name: str | None) -> str:
    return (name or "unknown").upper()


def _area_frac(m, crop: int):
    """Share of the frame taken by the region the sentence is ABOUT.

    For fn/fp that is the instance's own mask. For a matched pair the sentence
    speaks of the *ground truth* region, so the prediction's area would be the
    wrong thing to qualify: a badly under-segmented mask would make a large
    object read as "tiny". |GT| is recovered exactly from the stored mask facts,
    |GT| = |pred| - excess + missed.
    """
    f = m.metadata.get("frame_frac")
    if f is None:
        return None
    if m.seg_subtype.value in ("boundary", "extent") and "missed_px" in m.metadata:
        f += (m.metadata["missed_px"] - m.metadata["excess_px"]) / float(crop * crop)
    return max(f, 0.0)


def _size_word(members, crop: int) -> str:
    """tiny / small / large, from the share of the frame the region covers."""
    fr = [x for x in (_area_frac(m, crop) for m in members) if x is not None]
    if not fr:
        return ""
    f = sum(fr) / len(fr)
    if f < 0.005:
        return "tiny "
    if f < 0.02:
        return "small "
    if f > 0.25:
        return "large "
    return ""


def _cell(m, crop: int):
    """(row, col, edges) for one instance's bbox."""
    x0, y0, x1, y1 = m.bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    row = "top" if cy < crop / 3 else ("bottom" if cy > 2 * crop / 3 else "center")
    col = "left" if cx < crop / 3 else ("right" if cx > 2 * crop / 3 else "center")
    edges = set()
    if x0 <= EDGE_PX:
        edges.add("left")
    if x1 >= crop - EDGE_PX:
        edges.add("right")
    if y0 <= EDGE_PX:
        edges.add("top")
    if y1 >= crop - EDGE_PX:
        edges.add("bottom")
    return row, col, edges


def _phrase(row: str, col: str, edges: set) -> str:
    """One cell, in the annotators' words."""
    if col in edges and col != "center":
        return f"on the {col} edge"
    if row in edges and row != "center":
        return f"on the {row} edge"
    if row == "center" and col == "center":
        return "in the center of the scene"
    if col == "center":
        return f"on the {row} of the scene"
    if row == "center":
        return f"on the {col} of the scene"
    return f"on the {row}-{col}"


def _where(members, crop: int) -> str:
    """Where these instances sit, aggregated over the group."""
    cells = Counter()
    edges_of = {}
    for m in members:
        if not m.bbox:
            continue
        row, col, edges = _cell(m, crop)
        cells[(row, col)] += 1
        edges_of[(row, col)] = edges_of.get((row, col), set()) | edges
    if not cells:
        return ""
    (row, col), n = cells.most_common(1)[0]
    phrase = _phrase(row, col, edges_of[(row, col)])
    if n == len(members):
        return phrase
    # spread over several cells but all in one half of the frame: say the half
    # ("on the right half of the scene"), which is what the annotators write,
    # rather than "mostly in the center" or "scattered"
    if len(cells) >= 2:
        cx = [(m.bbox[0] + m.bbox[2]) / 2 for m in members if m.bbox]
        cy = [(m.bbox[1] + m.bbox[3]) / 2 for m in members if m.bbox]
        half = crop / 2
        for ok, name in ((all(x >= half for x in cx), "right"), (all(x <= half for x in cx), "left"),
                         (all(y >= half for y in cy), "bottom"), (all(y <= half for y in cy), "top")):
            if ok:
                return f"on the {name} half of the scene"
    if n / len(members) >= 0.6:
        return "mostly " + phrase
    return "scattered across the scene"


def _mid(where: str) -> str:
    """Place the location between subject and verb ("The X <here> has been ...")."""
    if not where:
        return " "
    return f", {where}, " if where.startswith("scattered") else f" {where} "


def _end(where: str) -> str:
    """Place the location at the end of the sentence ("... hallucinated <here>.")."""
    if not where:
        return ""
    return f", {where}" if where.startswith("scattered") else f" {where}"


def _iou_clause(members, iou: float) -> str:
    """Area IoU, and the boundary IoU beside it when it has been measured.

    The two are reported together on purpose: area IoU says how much of the
    region is right, boundary IoU how well its outline is traced, and on large
    objects they diverge badly (IoU 0.968 against boundary IoU 0.042 on
    2011_002685). One number alone would hide the very thing the annotators
    write about most -- 42 of the 80 notes mention the boundaries.
    """
    bs = [m.metadata["boundary_iou"] for m in members
          if m.metadata.get("boundary_iou") is not None]
    if not bs:
        return f"IoU {iou:.2f}"
    return f"IoU {iou:.2f}, boundary IoU {sum(bs) / len(bs):.2f}"


_QUALITY = ("slightly coarser boundaries",
            "coarser and imprecise boundaries",
            "heavily imprecise boundaries, with many details lost")


def _quality(iou: float, members=()) -> str:
    """The adjective for a traced-but-imperfect outline.

    Graded by area IoU, then demoted by the boundary IoU when that has been
    measured: area IoU is scale-dependent and on a large object stays high
    while the outline is wrong (a mask over two adjacent monitors: IoU 0.80,
    boundary IoU 0.08 -- "slightly coarser" would be false). One step down
    below CONTOUR_BESIDE (0.30, the worst quartile of the reference contours),
    two below 0.15, the same cuts the contour notes use.
    """
    level = 0 if iou >= 0.75 else (1 if iou >= 0.5 else 2)
    bs = [m.metadata["boundary_iou"] for m in members
          if m.metadata.get("boundary_iou") is not None]
    if bs:
        b = sum(bs) / len(bs)
        level += 2 if b < 0.15 else (1 if b < CONTOUR_BESIDE else 0)
    return _QUALITY[min(level, 2)]


def _direction(members) -> str:
    """Under- vs over-segmentation, from the mask facts if the caller stored them."""
    miss = sum(m.metadata.get("missed_px", 0) for m in members)
    exc = sum(m.metadata.get("excess_px", 0) for m in members)
    if not miss and not exc:
        return "segmented with the wrong extent, missing or adding whole parts"
    if miss > 1.5 * exc:
        return "under-segmented, with whole parts of the region missing"
    if exc > 1.5 * miss:
        return "over-segmented, spilling over the object's real extent"
    return "segmented with both missing and excess parts"


def _place(cell: str) -> str:
    """'in the top-left corner' for a corner cell, 'on the top' for a side."""
    return f"in the {cell} corner" if "-" in cell else f"on the {cell}"


def _sides(members, key: str) -> str:
    """'in the top-left corner and on the right' for the cell(s) most members agree on."""
    c = Counter(side for m in members for side in (m.metadata.get(key) or [])[:2])
    if not c:
        return ""
    top = [k for k, v in c.most_common(2) if v * 2 >= len(members)]
    if not top:
        top = [c.most_common(1)[0][0]]
    if all("-" not in k for k in top):            # sides only: the v5 wording
        return "on the " + " and ".join(top)
    return " and ".join(_place(k) for k in top)


def _bnd_where(members) -> str:
    """', especially towards the bottom-right' when most members' wrong outline sits in
    one cell of the object (mask_facts.boundary_where)."""
    c = Counter(m.metadata.get("boundary_where") for m in members if m.metadata.get("boundary_where"))
    if not c:
        return ""
    k, v = c.most_common(1)[0]
    return f", especially towards the {k}" if v * 2 >= len(members) else ""


def _extent_clause(members, short: bool = False) -> str:
    """', spilling over the object on the right (8% of its area in excess)'.

    The direction and the side of an extent error are what annotators name
    first ("over-extended on the right") and what Beyond IoU's boundary/extent
    split hides: a 20-px spill along a whole side is "boundary" to the
    taxonomy. Computed from the missed / excess pixels and their sides in
    mask_facts, as shares of the GT area; silent below EXTENT_MIN.
    """
    miss = sum(m.metadata.get("missed_px", 0) for m in members)
    exc = sum(m.metadata.get("excess_px", 0) for m in members)
    area = sum(m.metadata.get("gt_px", 0) for m in members)
    if not area:
        return ""
    fm, fe = miss / area, exc / area
    se, sm = _sides(members, "excess_sides"), _sides(members, "missed_sides")
    if short:                      # the verb already said under / over: add side and share
        if fe >= EXTENT_MIN and exc >= 1.5 * miss:
            return f", {fe:.0%} of its area in excess {se}".rstrip()
        if fm >= EXTENT_MIN and miss >= 1.5 * exc:
            return f", {fm:.0%} of its area missing {sm}".rstrip()
        if fm >= EXTENT_MIN and fe >= EXTENT_MIN:
            return f", {fe:.0%} in excess {se} and {fm:.0%} missing {sm}".replace("  ", " ")
        return ""
    parts = []
    if fe >= EXTENT_MIN and exc >= 1.5 * miss:
        parts.append(f"spilling over the object {se} ({fe:.0%} of its area in excess)".replace("  ", " "))
    elif fm >= EXTENT_MIN and miss >= 1.5 * exc:
        parts.append(f"with parts missing {sm} ({fm:.0%} of its area)".replace("  ", " "))
    elif fm >= EXTENT_MIN and fe >= EXTENT_MIN:
        parts.append(f"spilling over {se} ({fe:.0%} in excess) and missing parts {sm} ({fm:.0%})".replace("  ", " "))
    return (", " + parts[0]) if parts else ""


def _coverage(members):
    """Mean covered_frac over the members, or None if never measured."""
    vals = [m.metadata["covered_frac"] for m in members
            if m.metadata.get("covered_frac") is not None]
    return sum(vals) / len(vals) if vals else None


# Above this coverage a FN is narrated as ABSORBED rather than missed. On
# pascal_80 the distribution is bimodal -- 30% of FNs below 5% coverage, 41%
# above 80% -- so any cut in the middle separates the two populations; 0.5 is
# the natural one and matches the "more than half" a reader would use.
FN_ABSORBED = 0.5


def _merge_sentence(members, cls: str, where, n: int, crop: int = 520,
                    swallowed=()) -> str:
    """One blob (or several) swallowing more than one ground truth region.

    Three facts a reader wants that the bare type does not give: how many
    regions went into the blob (summed over the blobs, not the max), whether
    some of them were of ANOTHER class -- a merge across classes is also a
    confusion, the "table labelled as chair" the annotators write -- and, when
    the blob has a proper 1<->1 match of its own, how well that main region's
    outline is traced, which MERGE alone would silence.
    """
    many = n > 1
    counts = [m.metadata.get("n_gt_covered") or 0 for m in members]
    total = sum(counts)
    subj = f"The {_n_word(n)} {cls} predictions" if many else f"The {cls} prediction"
    if total:
        into = f"into {_n_word(n)} masks" if many else "into a single mask"
        verb = "together merge" if many else "merges"
        core = f"{subj}{_mid(where)}{verb} {_n_word(total)} ground truth regions {into}"
    else:
        core = f"{subj}{_mid(where)}{'merge' if many else 'merges'} several ground truth regions into {'separate masks' if many else 'a single mask'}"
    pred_cls = members[0].pred_class
    foreign = Counter(c for m in members for c in m.metadata.get("gt_classes", [])
                      if c != pred_cls)
    if foreign:
        parts = ", ".join(f"{_n_word(k)} {_cls(c)}" for c, k in foreign.most_common())
        k_all = sum(foreign.values())
        core += (f", among them {parts} {'region' if k_all == 1 else 'regions'} "
                 f"now labelled {_cls(pred_cls)}")
    # the swallowed objects themselves (GT-side merge instances): where they
    # are and how much of them the blob took -- the object-centred sentence the
    # annotators write ("the small boat on the right has been merged into it")
    sw = [x for x in swallowed if x.metadata.get("covered_frac") is not None]
    if sw and len(sw) <= 3:
        bits = [f"the {_size_word([x], crop)}{_cls(x.gt_class)} region"
                f"{_end(_where([x], crop))} ({x.metadata['covered_frac']:.0%} covered)"
                for x in sorted(sw, key=lambda x: -x.metadata.get("frame_frac", 0.0))]
        core += ": " + ", ".join(bits[:-1]) + (" and " if len(bits) > 1 else "") + bits[-1]
    elif sw:
        cov = sum(x.metadata["covered_frac"] for x in sw) / len(sw)
        core += f", covering {cov:.0%} of each of them on average"
    # how the blob fits the regions it merged, TAKEN TOGETHER: the union is the
    # only ground truth a merged mask can be held to. Against the single most
    # similar GT a mask over two adjacent monitors has IoU 0.44 and nothing to
    # say; against their union it has IoU 0.80 and a loose outline -- which is
    # exactly what the annotator wrote. Falls back to the primary pair's numbers
    # on pickles measured before union_iou existed.
    uni = [m for m in members if m.metadata.get("union_iou") is not None]
    if uni:
        iou = sum(m.metadata["union_iou"] for m in uni) / len(uni)
        e_bnd = sum(m.metadata["union_e_bnd"] for m in uni) / len(uni)
        e_ext = sum(m.metadata["union_e_ext"] for m in uni) / len(uni)
        together = "taken together, the merged regions"
        if e_bnd >= e_ext:
            core += (f"; {together} are segmented with {_quality(iou, uni)}{_bnd_where(uni)}"
                     f"{_extent_clause(uni)} ({_iou_clause(uni, iou)})")
        else:
            core += (f"; {together} have been {_direction(uni)}"
                     f"{_extent_clause(uni, short=True)} ({_iou_clause(uni, iou)})")
    else:
        prim = [m.metadata["primary_iou"] for m in members
                if m.metadata.get("primary_iou") is not None]
        if prim:
            iou = sum(prim) / len(prim)
            core += (f"; the main {'region is' if len(prim) == 1 else 'regions are'} "
                     f"otherwise segmented with {_quality(iou, members)} ({_iou_clause(members, iou)})")
    return core + "."


def _hidden(members) -> str:
    """", partly hidden behind the PERSON," when the GT object is cut by one.

    Only with the object adapter (VOC SegmentationObject): a cut object is one
    instance, and mask_facts names the other-class object touching two of its
    parts. Said once per group, for the occluder most of the members share.
    """
    occ = Counter(m.metadata["occluded_by"] for m in members if m.metadata.get("occluded_by"))
    if not occ or occ.most_common(1)[0][1] * 2 < len(members):
        return ""
    return f", partly hidden behind the {_cls(occ.most_common(1)[0][0])},"


def _loc(where: str, hidden: str) -> str:
    """Location between subject and verb, with the occlusion clause spliced in."""
    base = _mid(where)
    if not hidden:
        return base
    base = base.rstrip()
    if base.endswith(","):
        base = base[:-1]
    return base + hidden + " "


def _sentence(subtype: str, members, crop: int, swallowed=()) -> str:
    """One sentence for one (error type, class) group."""
    n = len(members)
    many = n > 1
    size = _size_word(members, crop)
    where = _where(members, crop)
    cls = _cls(members[0].gt_class or members[0].pred_class)
    region = "regions" if many else "region"
    has = "have" if many else "has"
    loc = _loc(where, _hidden(members))

    if subtype == "fn":
        subj = (f"The {_n_word(n)} {size}ground truth {cls} {region}" if many
                else f"The {size}ground truth {cls} {region}")
        cov = _coverage(members)
        if cov is not None and cov >= 0.2:
            return (f"{subj}{loc}{has} been mostly missed: only "
                    f"{cov:.0%} of {'their' if many else 'its'} area is covered by "
                    f"a {cls} prediction.")
        return f"{subj}{loc}{has} been missed entirely."

    if subtype == "fn_absorbed":
        # Block 1 filed these as FN, but a same-class prediction covers most of
        # them: the object was not missed, it was swallowed into a neighbouring
        # blob with no outline of its own. See mask_facts.covered_frac for why
        # the merge test misses this (IoU is blind to asymmetric containment).
        subj = (f"The {_n_word(n)} {size}ground truth {cls} {region}" if many
                else f"The {size}ground truth {cls} {region}")
        cov = _coverage(members)
        return (f"{subj}{loc}{has} been absorbed into a neighbouring {cls} "
                f"prediction: {cov:.0%} covered, but with no outline of "
                f"{'their' if many else 'its'} own.")

    if subtype == "fp":
        patch = "patches" if many else "patch"
        return (f"{_n_word(n).capitalize()} {size}{cls} {patch}"
                f" {has} been hallucinated{_end(where)}.")

    if subtype == "fp_on":
        # a predicted patch lying on an object of ANOTHER class is not a
        # hallucination to the reader: part of that object got this label
        on = Counter(_cls(m.metadata["over_gt"][0]) for m in members).most_common(1)[0][0]
        patch = "patches" if many else "patch"
        return (f"{'Parts' if many else 'Part'} of the ground truth {on} "
                f"{'have' if many else 'has'} been labelled {cls} instead "
                f"({_n_word(n)} {size}{cls} {patch}{_end(where)}).")

    if subtype == "fp_spill":
        # same-class patches touching a GT region of that class: the object's
        # mask spilling beyond its extent, which is how the annotators read it
        patch = "patches" if many else "patch"
        return (f"The {cls} mask also spills beyond the object in {_n_word(n)} "
                f"{size}{patch}{_end(where)}.")

    if subtype == "confusion":
        pred = Counter(_cls(m.pred_class) for m in members).most_common(1)[0][0]
        return (f"The {size}ground truth {cls} {region}{_mid(where)}"
                f"{has} been labelled {pred} instead.")

    if subtype == "split":
        # Block 1 emits ONE instance per SURPLUS fragment (the non-primary
        # claimers of a GT region), so n counts extra masks, not GT regions.
        mask_w = "masks" if many else "mask"
        return (f"{_n_word(n).capitalize()} extra {cls} {mask_w}{_end(where)} "
                f"{'split' if many else 'splits'} a ground truth region that "
                f"should be a single mask.")

    if subtype == "merge":
        return _merge_sentence(members, cls, where, n, crop, swallowed)

    iou = sum(m.iou for m in members) / n
    if subtype == "extent":
        return (f"The {size}ground truth {cls} {region}{loc}"
                f"{has} been {_direction(members)}{_extent_clause(members, short=True)} "
                f"({_iou_clause(members, iou)}).")

    # boundary: the adjective grades the outline, the clause says where it
    # spills or falls short -- the sign and the side the taxonomy hides
    return (f"The {size}ground truth {cls} {region}{loc}"
            f"{has} been segmented with {_quality(iou, members)}{_bnd_where(members)}"
            f"{_extent_clause(members)} ({_iou_clause(members, iou)}).")


def _relabel_sentences(members, told=frozenset(), max_n: int = 1, said=None) -> list:
    """'Part of the DININGTABLE (30% of its area) has been labelled CHAIR instead.'

    From ``relabelled_as`` (mask_facts): the other class predicted on the largest
    part of the object. Pairs (object class, label) already told by a patch
    sentence are skipped, so each confusion is said once.
    """
    by = defaultdict(lambda: defaultdict(list))   # object class -> label -> shares
    n_obj = defaultdict(set)
    said = set() if said is None else said
    for m in members:
        rl = m.metadata.get("relabelled_as")
        if not rl or not m.gt_class:
            continue
        obj = m.metadata.get("relabel_obj") or m.instance_id
        if obj in said:                           # one object, said once
            continue
        if isinstance(rl[0], str):                 # v5 pickles: a single [cls, frac]
            rl = [rl]
        for c, f in rl:
            if (m.gt_class, c) not in told:
                by[m.gt_class][c].append(f)
                n_obj[m.gt_class].add(obj)
        said.add(obj)
    out = []
    for g, labs in sorted(by.items(), key=lambda kv: -sum(sum(v) for v in kv[1].values()))[:max_n]:
        ranked = sorted(labs.items(), key=lambda kv: -sum(kv[1]))[:2]
        n = len(n_obj[g])
        (c1, f1), rest = ranked[0], ranked[1:]
        if n == 1:
            tail = (f", and part ({sum(rest[0][1]):.0%}) {_cls(rest[0][0])}" if rest else "")
            out.append(f"Part of the {_cls(g)} ({sum(f1):.0%} of its area) has been labelled "
                       f"{_cls(c1)} instead{tail}.")
        else:
            tail = (f", and parts {_cls(rest[0][0])}" if rest else "")
            out.append(f"Parts of the {_n_word(n)} {_cls(g)} regions ({sum(f1) / n:.0%} of their area "
                       f"on average) have been labelled {_cls(c1)} instead{tail}.")
    return out


def _details_sentence(members, subject: str | None = None) -> str | None:
    """Thin parts of the region the prediction failed to trace.

    A missed fragment counts as a DETAIL when a morphological opening erases it:
    that is what separates an antenna, a leg or a wing tip from a missing patch,
    which the extent term already covers. Kept separate from E_bnd/E_ext because
    those give the magnitude of the miss, never its shape.
    """
    n = sum(m.metadata.get("n_thin_lost", 0) for m in members)
    if n < THIN_LOST_MIN:
        return None
    px = sum(m.metadata.get("thin_lost_px", 0) for m in members)
    how = "Many fine details" if n >= 15 else "Some fine details"
    if subject is None:
        cls = {m.gt_class or m.pred_class for m in members}
        subject = f"the {_cls(cls.pop())} outline" if len(cls) == 1 else "the outline"
    return (f"{how} of {subject} are lost: {n} thin fragments the prediction "
            f"does not trace, {px} pixels in total.")


# ---------------------------------------------------------------------------
# Sub-threshold regions and patches: measured, no longer narrated
#
# Both sentences below said what the INSTRUMENT does ("below the size the
# pipeline represents as an object"), which is the one thing a description of an
# image must not do: the reader is told about a segmentation, not about the tool
# reading it. Removed from the text on 2026-09-26, by request; the facts are
# still measured (mask_facts.subthreshold / subthreshold_pred) and still counted
# in the narrative record, so nothing downstream lost a number. The two builders
# are kept because the wording is the only thing wrong with them: a caller that
# wants the information back can append them again.
# ---------------------------------------------------------------------------
def _subthreshold_sentence(extras) -> str | None:
    """Ground-truth regions too small to have become instances at all.

    Not "missed" errors: build_voc_samples drops any component under min_pixels
    before Block 1 runs, so nothing downstream can ever see them. Stating this
    is the only way the reader learns that the silence there is a limit of the
    instrument and not a verdict about the image.
    """
    n = (extras or {}).get("subthreshold", 0)
    if not n:
        return None
    px = extras.get("subthreshold_px", 0)
    one = n == 1
    obj = "region" if one else "regions"
    return (f"{_n_word(n).capitalize()} small ground truth {obj} ({px} pixels in "
            f"total) {'is' if one else 'are'} below the size the pipeline "
            f"represents as an object, and {'has' if one else 'have'} no "
            f"counterpart in the prediction.")


def _subthreshold_pred_sentence(extras) -> str | None:
    """Predicted patches too small to be instances, labelled with a class the
    ground truth does not have there -- the 'small area misclassified as
    DININGTABLE' a reader sees and the instrument would otherwise drop."""
    sp = (extras or {}).get("subthreshold_pred") or []
    if not sp:
        return None
    c = Counter((pc, gc) for pc, gc in sp)
    bits = []
    for (pc, gc), k in c.most_common(3):
        on = "the background" if gc == "background" else f"the {_cls(gc)}"
        bits.append(f"{_n_word(k)} tiny {_cls(pc)} patch{'' if k == 1 else 'es'} on {on}")
    more = len(sp) - sum(k for _x, k in c.most_common(3))
    if more > 0:
        bits.append(f"{_n_word(more)} more")
    joined = bits[0] if len(bits) == 1 else ", ".join(bits[:-1]) + " and " + bits[-1]
    return (f"Below the size the pipeline represents as an object there "
            f"{'is' if len(sp) == 1 else 'are'} also {joined}.")


def _verdict_sentence(extras) -> str | None:
    """The opening line when the failure is global."""
    v = (extras or {}).get("verdict")
    if not v:
        return None
    if v["wrong_cls"] and v["wrong_cls_frac"] >= VERDICT_MIN:
        note = "" if v["wrong_cls_in_gt"] else ", a class that is not in the image"
        return (f"The prediction is largely wrong: {v['wrong_cls_frac']:.0%} of the annotated "
                f"area has been labelled {_cls(v['wrong_cls'])}{note}.")
    if v["wrong_frac"] >= VERDICT_MIN:
        return (f"The prediction is largely wrong: {v['wrong_frac']:.0%} of the annotated "
                f"area has been given the wrong class.")
    if v["bg_frac"] >= VERDICT_MIN:
        return (f"The prediction misses most of the annotated area: {v['bg_frac']:.0%} of it "
                f"is predicted as background.")
    return None


def _correct_sentences(reference, noted, crop: int, max_groups: int = 2) -> list:
    """What the annotators always write first about a region that is fine.

    A CORRECT region (area IoU >= tau_iou_good) produced no text unless its
    contour note fired; the hand-written notes say "segmented quite well, the
    boundaries are not perfectly defined" or "very close, with a slight
    under-segmentation" about exactly those. So every correct region not
    already covered by a contour note gets one graded sentence, with the
    extent clause when the spill or gap reaches EXTENT_MIN. Regions that match
    the ground truth exactly (IoU >= 0.99) are the only silent ones.
    """
    rest = [m for m in reference if m.instance_id not in noted and m.iou < 0.99]
    if not rest:
        return []
    groups = defaultdict(list)
    for m in rest:
        groups[m.gt_class or m.pred_class].append(m)
    ordered = sorted(groups.values(),
                     key=lambda ms: -sum(m.metadata.get("frame_frac", 0.0) for m in ms))
    out = []
    for ms in ordered[:max_groups]:
        n = len(ms)
        many = n > 1
        cls = _cls(ms[0].gt_class or ms[0].pred_class)
        iou = sum(m.iou for m in ms) / n
        bs = [m.metadata["boundary_iou"] for m in ms if m.metadata.get("boundary_iou") is not None]
        b = sum(bs) / len(bs) if bs else None
        how = ("closely" if b is None or b >= 0.5
               else "well, with slightly coarse boundaries" + _bnd_where(ms))
        subj = (f"The {_n_word(n)} {_size_word(ms, crop)}ground truth {cls} regions" if many
                else f"The {_size_word(ms, crop)}ground truth {cls} region")
        out.append(f"{subj}{_mid(_where(ms, crop))}{'have' if many else 'has'} been segmented "
                   f"{how}{_extent_clause(ms)} ({_iou_clause(ms, iou)}).")
    return out


class _Box:
    """A bbox with the attributes _where reads, for patches that are not instances."""
    def __init__(self, bbox):
        self.bbox = bbox


def _small_patches_sentence(extras, crop: int, max_groups: int = 2) -> str | None:
    """Predicted patches too small to be instances (mask_facts.small_patches), told as
    what they are in the image: "Besides, a small PERSON patch has been predicted on
    the right edge", or "...on the TRAIN". Up to two groups (class, what lies under)."""
    sp = (extras or {}).get("small_patches") or []
    if not sp:
        return None
    groups = defaultdict(list)
    for x in sp:
        groups[(x["cls"], x["under"])].append(_Box(x["bbox"]))
    out = []
    for (c, u), bx in sorted(groups.items(), key=lambda kv: -len(kv[1]))[:max_groups]:
        n = len(bx)
        where = _end(_where(bx, crop))
        if u == "background":
            out.append(f"{_n_word(n)} small {_cls(c)} patch{'es have' if n > 1 else ' has'} been "
                       f"hallucinated{where}")
        else:
            out.append(f"{'bits' if n > 1 else 'a bit'} of the ground truth {_cls(u)} "
                       f"{'have' if n > 1 else 'has'} been labelled {_cls(c)} instead "
                       f"({_n_word(n)} small patch{'es' if n > 1 else ''}{where})")
    return "Besides, " + "; ".join(out) + "."


def _ref_holes_sentences(reference, max_groups: int = 2) -> list:
    """Holes in the mask of a region that is otherwise correct ("the TRAIN region has
    some holes scattered within it"): until v6 holes were told only on errors."""
    groups = defaultdict(list)
    for m in reference:
        if m.metadata.get("n_true_holes"):
            groups[m.gt_class or m.pred_class].append(m)
    out = []
    for cls, ms in sorted(groups.items(), key=lambda kv: -sum(m.metadata["n_true_holes"] for m in kv[1]))[:max_groups]:
        nh = sum(m.metadata["n_true_holes"] for m in ms)
        frac = sum(m.metadata.get("true_hole_frac", 0.0) for m in ms) / len(ms)
        share = "under 1%" if frac < 0.01 else f"{frac:.0%}"
        out.append(f"The {_cls(cls)} mask has {_n_word(nh)} hole{'' if nh == 1 else 's'} "
                   f"{'inside it' if nh == 1 else 'scattered inside'} ({share} of the mask area).")
    return out


def _holes_sentence(members) -> str | None:
    holed = [m for m in members if m.metadata.get("n_holes")]
    if not holed:
        return None
    nh = sum(m.metadata["n_holes"] for m in holed)
    frac = sum(m.metadata.get("hole_frac", 0.0) for m in holed) / len(holed)
    which = "mask" if len(holed) == 1 else "masks"
    where = "inside it" if nh == 1 else "scattered inside"
    share = "under 1%" if frac < 0.01 else f"{frac:.0%}"
    return (f"The predicted {which} also {'has' if len(holed) == 1 else 'have'} "
            f"{_n_word(nh)} hole{'' if nh == 1 else 's'} {where} "
            f"({share} of the mask area).")


# segment-level failures are what a human notices first; a slightly coarse
# boundary is what they mention last, so the narrative follows that order
_ORDER = {"fn": 0, "fp": 1, "confusion": 2, "fp_on": 2, "merge": 3, "fn_absorbed": 3,
          "fp_spill": 6, "split": 4, "extent": 5, "boundary": 6}


def _contour_sentence(members, crop: int) -> str:
    """A region the model placed right but traced badly.

    These instances are CORRECT for Block 1 -- area IoU above tau_iou_good -- so
    they produce no error and, until now, no text at all. That silence is a
    measured failure: all six images the hand verdicts marked BASSA ("the
    pipeline saw nothing and should have") contain exactly this, an area-correct
    mask over a large object whose outline is wrong. Area IoU cannot see it,
    because it is scale-dependent: on an 85k-260k px object the gap between IoU
    and boundary IoU averages +0.595, against +0.217 on the smallest quartile.

    Nothing here is called an error. The taxonomy is untouched, no instance is
    created, and the reference population Block 6 needs for omega^2 is left
    intact -- this is a REPORTING statement about contour quality.
    """
    n = len(members)
    b = sum(m.metadata["boundary_iou"] for m in members) / n
    where = _where(members, crop)
    cls = _cls(members[0].gt_class or members[0].pred_class)
    quality = ("barely followed at all" if b < 0.15
               else "only roughly followed" if b < 0.25 else "loosely followed")
    if n == 1:
        subj = f"The {_size_word(members, crop)}ground truth {cls} region"
        verb, poss = "is", "its"
    else:
        subj = f"The {_n_word(n)} {_size_word(members, crop)}ground truth {cls} regions"
        verb, poss = "are", "their"
    return (f"{subj}{_mid(where)}{verb} correctly located, but {poss} outline is "
            f"{quality}{_bnd_where(members)}{_extent_clause(members)} (boundary IoU {b:.2f}).")


def _contour_notes(reference, crop: int, has_errors: bool,
                   max_groups: int = 2) -> list:
    """Contour remarks for the area-correct instances that deserve one.

    ``has_errors`` picks the regime: when the image has typed errors the remark
    must earn its place beside them (worst quartile only); when it has none, the
    contour is the only thing that can be said, so the bar drops to the median.
    """
    cut = CONTOUR_BESIDE if has_errors else CONTOUR_ALONE
    poor = [m for m in reference
            if m.metadata.get("boundary_iou") is not None
            and m.metadata["boundary_iou"] < cut]
    if not poor:
        return []
    groups = defaultdict(list)
    for m in poor:
        groups[m.gt_class or m.pred_class].append(m)
    worst = sorted(groups.values(),
                   key=lambda ms: sum(m.metadata["boundary_iou"] for m in ms) / len(ms))
    return [_contour_sentence(ms, crop) for ms in worst[:max_groups]]


def narrate(errors, reference=(), extras=None, *, crop: int = 520,
            max_groups: int = MAX_GROUPS) -> dict:
    """Narrate one image's typed errors, plus contour notes on correct regions."""
    groups = defaultdict(list)
    swallowed_by = defaultdict(list)      # blob instance id -> its GT-side merges
    for e in errors:
        cls = e.gt_class or e.pred_class
        kind = e.seg_subtype.value
        if kind == "merge" and e.metadata.get("from_gt_mask"):
            # the object side of a merge is told inside its blob's sentence
            swallowed_by[e.metadata.get("swallowed_by")].append(e)
            continue
        if kind == "fn" and (e.metadata.get("covered_frac") or 0.0) >= FN_ABSORBED:
            kind = "fn_absorbed"          # narrative only: the taxonomy is untouched
        if kind == "fp" and e.metadata.get("adjacent_gt"):
            kind = "fp_spill"             # the object's mask spilling, not a hallucination
        elif kind == "fp" and e.metadata.get("over_gt"):
            kind = "fp_on"                # part of another object labelled with this class
            cls = f"{cls}>{e.metadata['over_gt'][0]}"
        groups[(kind, cls)].append(e)

    # tiny hallucinated patches are told in one closing line, so that six
    # 200-px specks never come before the object the annotator writes about
    tiny_all = {k: ms for k, ms in groups.items()
                if k[0] in ("fp", "fp_on") and all(m.metadata.get("frame_frac", 1.0) < TINY_FRAC for m in ms)}
    for k in tiny_all:
        del groups[k]
    tiny = {k: ms for k, ms in tiny_all.items() if k[0] == "fp"}
    tiny_on = {k: ms for k, ms in tiny_all.items() if k[0] == "fp_on"}

    def rank(item):
        (subtype, _cls_), members = item
        area = sum(m.metadata.get("frame_frac", 0.0) for m in members)
        order = -1 if area >= BIG_FRAC else _ORDER.get(subtype, 9)
        return (order, -area, -len(members))

    ordered = sorted(groups.items(), key=rank)
    shown, hidden = ordered[:max_groups], ordered[max_groups:]
    # (object class, label) pairs a patch sentence tells: the object's own
    # sentence then does not repeat them
    told = {(m.metadata["over_gt"][0], m.pred_class)
            for (k, _c), ms in list(shown) + list(tiny_on.items()) if k == "fp_on" for m in ms}
    said = set()                          # objects whose relabel is already told

    sentences = []
    v = _verdict_sentence(extras)
    if v:
        sentences.append(v)
    for (subtype, _cls_), members in shown:
        sw = [x for m in members for x in swallowed_by.get(m.instance_id, ())]
        sentences.append(_sentence(subtype, members, crop, sw))
        # the merge sentence already names the classes its blob relabelled ("among them one
        # PERSON region now labelled MOTORBIKE", from gt_classes): those pairs, and only those
        told_here = told | {(c, m.pred_class) for m in members if subtype == "merge"
                            for c in m.metadata.get("gt_classes", []) if c != m.pred_class}
        sentences.extend(_relabel_sentences(list(members) + list(sw), told_here, said=said))
        for extra in (_details_sentence(members), _holes_sentence(members)):
            if extra:
                sentences.append(extra)

    if hidden:
        n = sum(len(m) for _k, m in hidden)
        sentences.append(f"A further {_n_word(n)} smaller error"
                         f"{'' if n == 1 else 's'} on other objects "
                         f"{'is' if n == 1 else 'are'} present.")

    notes = _contour_notes(reference, crop, has_errors=bool(errors))
    sentences.extend(notes)
    noted = {m.instance_id for m in reference
             if m.metadata.get("boundary_iou") is not None
             and m.metadata["boundary_iou"] < (CONTOUR_BESIDE if errors else CONTOUR_ALONE)}
    sentences.extend(_correct_sentences(reference, noted, crop))
    sentences.extend(_relabel_sentences(reference, told, max_n=2, said=said))
    sentences.extend(_ref_holes_sentences(reference))
    # anche le regioni corrette perdono dettagli, e li' nessun errore lo direbbe
    d = _details_sentence(reference)
    if d:
        sentences.append(d)
    if tiny:
        allm = [m for ms in tiny.values() for m in ms]
        where_t = _end(_where(allm, crop))
        if len(allm) == 1:
            sentences.append(f"Besides, one tiny {_cls(allm[0].pred_class)} patch has been "
                             f"hallucinated{where_t}.")
        elif len(tiny) == 1:
            sentences.append(f"Besides, {_n_word(len(allm))} tiny {_cls(allm[0].pred_class)} "
                             f"patches have been hallucinated{where_t}.")
        else:
            bits = [f"{_cls(c)} \u00d7{len(ms)}"
                    for (_k, c), ms in sorted(tiny.items(), key=lambda kv: -len(kv[1]))]
            sentences.append(f"Besides, tiny patches have been hallucinated{where_t}: "
                             f"{', '.join(bits)}.")
    for (_k, key), ms in sorted(tiny_on.items(), key=lambda kv: -len(kv[1]))[:2]:
        on = _cls(ms[0].metadata["over_gt"][0])
        n = len(ms)
        sentences.append(f"Besides, {'bits' if n > 1 else 'a bit'} of the ground truth {on} "
                         f"{'have' if n > 1 else 'has'} been labelled {_cls(ms[0].pred_class)} "
                         f"instead ({_n_word(n)} tiny patch{'es' if n > 1 else ''}"
                         f"{_end(_where(ms, crop))}).")
    sp = _small_patches_sentence(extras, crop)
    if sp:
        sentences.append(sp)
    if not sentences:
        sentences = ["No segmentation error was found in this image."]

    return {"n_errors": len(errors),
            "n_groups": len(groups),
            "n_contour_notes": len(notes),
            "n_detail_notes": sum(1 for x in sentences if "fine details" in x),
            "n_subthreshold": (extras or {}).get("subthreshold", 0),
            "sentences": sentences,
            "text": " ".join(sentences)}


def build_error_narratives(errors, image_ids=None, reference=(), extras=None, *,
                           crop: int = 520, max_groups: int = MAX_GROUPS) -> dict:
    """{image_id -> narrative} for every image in ``image_ids``.

    ``image_ids`` must include the images that have no typed error but do have a
    badly traced correct region: those produce no entry in ``errors`` at all, and
    they are exactly the cases this addition exists for.
    """
    by_img = defaultdict(list)
    for e in errors:
        by_img[e.image_id].append(e)
    ref_img = defaultdict(list)
    for r in reference:
        ref_img[r.image_id].append(r)
    if image_ids is None:
        image_ids = sorted(set(by_img) | set(ref_img))
    extras = extras or {}
    return {n: narrate(by_img.get(n, []), ref_img.get(n, []), extras.get(n),
                       crop=crop, max_groups=max_groups) for n in image_ids}


def export_narratives(narratives: dict, json_path: str, txt_path: str | None = None):
    import json
    import os
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(narratives, f, indent=1, ensure_ascii=False)
    if txt_path:
        with open(txt_path, "w", encoding="utf-8") as f:
            for name in narratives:
                f.write(f"{name}\n{narratives[name]['text']}\n\n")
