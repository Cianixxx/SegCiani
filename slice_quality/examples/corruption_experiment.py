"""Synthetic-corruption recovery experiment (external validation, CPU-only + a
small CPU DINOv2 burst -- no GPU needed).

Idea: deliberately DELETE the model's correct-ish prediction for a chosen,
KNOWN subset of real GT objects -- guaranteeing they become FN through the
normal Block-1 matching logic (no hand-crafted "fake" instances) -- then check
whether the pipeline (a) groups them into a CERTIFIED slice, and (b) whether
the per-image description names the right thing. This tests the pipeline
end-to-end against a KNOWN ground truth, on top of the REAL VOC error corpus
(everything untouched stays untouched -- realistic noise floor).

Two rules, one per branch:
  A. "small <class>"        -> should be found by the CLUSTER branch (a
     dedicated FN cluster, stratified + BIC, should contain the victims).
  B. "<class> cut off at the image border" -> re-uses the ALREADY-VALIDATED
     canonical SliceLens hypothesis of the same name; recovery is checked
     against the ARCHIVED grounding confidence for these exact bboxes (they
     were graded as part of the full-population re-grounding already, since
     Qwen's judgment depends only on appearance, not on correctness) -- zero
     new VLM calls needed.

Robustness trick: predictions are removed from the per-image ``preds`` list,
which SHIFTS the "#pI" instance_id of later predictions in that image. We
never rely on instance_id identity for embedding/tag reuse -- we join on
(image_id, bbox, seg_subtype), which is index-shift-proof (untouched objects
keep their exact geometry).

Run (CPU container, no --gpus):
    PYTHONPATH=/workspace python slice_quality/examples/corruption_experiment.py
"""
from __future__ import annotations

import json
import os
import pickle
import random
from collections import Counter

import numpy as np

from slice_quality.adapters.common import make_mask_lookup
from slice_quality.adapters.voc_seg_samples import build_voc_samples
from slice_quality.config import SliceQualityConfig
from slice_quality.data_types import SegErrorType
from slice_quality.embeddings import attach_embeddings
from slice_quality.pipeline import run_from_instances
from slice_quality.typed_error import decompose_dataset, mask_iou

DATA = "/data/voc2012/VOCdevkit/VOC2012"
CROPS = "/data/voc2012/voc_crops"
# The run to corrupt (its pickle supplies the embeddings and tags to reuse; outputs go beside it). For the
# inst run also set SQ_CLAIM_COV=0.5 SQ_GT_INSTANCES=object so Block 1 types the corrupted images the same way.
OUT_DIR = os.environ.get("SQ_OUT", "outputs/slice_quality")
PKL = f"{OUT_DIR}/result_tagged.pkl"
# How the victims' host slices are tagged: "caption" (the July runs) or "verify" (the channel of the inst run).
TAG_MODE = os.environ.get("TAG_MODE", "caption")

# Injection-strength knobs (env-overridable for dose-response runs). At the
# default 25th percentile only ~13 chairs are deletable (0.2% of all errors) --
# a signal too weak for the strict certification tier by construction; raising
# the percentile raises the dose. CORRUPTION_TAG suffixes the output files so
# multiple doses do not clobber each other.
RULE_A_CLASS = "chair"          # -> "small chair" (cluster branch)
RULE_A_AREA_PCT = int(os.environ.get("RULE_A_AREA_PCT", "25"))
RULE_A_BUDGET = int(os.environ.get("RULE_A_BUDGET", "60"))
RUN_TAG = os.environ.get("CORRUPTION_TAG", "")
TAG_TARGET = int(os.environ.get("TAG_TARGET", "40"))   # tagged members per host slice

RULE_B_CLASS = "car"            # -> "car cut off at the image border" (language)
RULE_B_MARGIN_PX = 3            # bbox within this many px of an image edge
RULE_B_BUDGET = 60
RULE_B_HYPOTHESIS = "an object cut off at the image border"
RULE_B_GROUNDING_FILE = "/data/outputs/voc_paper_car_full.json"
LANG_CONF_MIN = 0.6

SEED = 0


def _round_bbox(bbox, nd=0):
    return tuple(round(float(v), nd) for v in bbox) if bbox else None


def build_corrupted_samples(samples):
    """Return (corrupted_samples, manifest) -- see module docstring for the rule."""
    rng = random.Random(SEED)

    # ---- collect candidate victims per rule, per image -----------------------
    cand_a, cand_b = [], []   # each: (img_idx, gt_idx, best_pred_idx, gt_dict)
    for img_idx, s in enumerate(samples):
        H, W = None, None
        for g_idx, g in enumerate(s["gts"]):
            if g["cls"] not in (RULE_A_CLASS, RULE_B_CLASS):
                continue
            if H is None:
                H, W = g["mask"].shape
            # best-matching prediction of the SAME class (to delete)
            best_p, best_iou = None, 0.4
            for p_idx, p in enumerate(s["preds"]):
                if p["cls"] != g["cls"]:
                    continue
                iou = mask_iou(p["mask"], g["mask"])
                if iou >= best_iou:
                    best_p, best_iou = p_idx, iou
            if best_p is None:
                continue                      # already an FN in reality -> skip
            area = int(g["mask"].sum())
            x0, y0, x1, y1 = g["bbox"]
            touches_border = (x0 <= RULE_B_MARGIN_PX or y0 <= RULE_B_MARGIN_PX
                              or x1 >= W - RULE_B_MARGIN_PX or y1 >= H - RULE_B_MARGIN_PX)
            if g["cls"] == RULE_A_CLASS:
                cand_a.append((img_idx, g_idx, best_p, area, g["bbox"]))
            if g["cls"] == RULE_B_CLASS and touches_border:
                cand_b.append((img_idx, g_idx, best_p, area, g["bbox"]))

    # rule A: bottom-percentile by area
    areas = sorted(c[3] for c in cand_a)
    thresh = areas[max(0, int(len(areas) * RULE_A_AREA_PCT / 100) - 1)] if areas else 0
    victims_a = [c for c in cand_a if c[3] <= thresh]
    rng.shuffle(victims_a)
    victims_a = victims_a[:RULE_A_BUDGET]

    rng.shuffle(cand_b)
    victims_b = cand_b[:RULE_B_BUDGET]

    print(f"rule A ({RULE_A_CLASS}, small, area<={thresh}px): "
          f"{len(cand_a)} eligible -> {len(victims_a)} chosen")
    print(f"rule B ({RULE_B_CLASS}, border-touching): "
          f"{len(cand_b)} eligible -> {len(victims_b)} chosen")

    # ---- apply: remove the chosen pred index from each touched image ---------
    to_remove: dict[int, set] = {}
    manifest = []
    for rule, victims in (("small", victims_a), ("border", victims_b)):
        cls = RULE_A_CLASS if rule == "small" else RULE_B_CLASS
        for img_idx, g_idx, p_idx, area, bbox in victims:
            to_remove.setdefault(img_idx, set()).add(p_idx)
            manifest.append({
                "rule": rule, "cls": cls, "image_id": samples[img_idx]["image_id"],
                "gt_idx": g_idx, "removed_pred_idx": p_idx, "area": area,
                "bbox": list(bbox),
            })

    corrupted = []
    for img_idx, s in enumerate(samples):
        rm = to_remove.get(img_idx)
        if not rm:
            corrupted.append(s)
            continue
        new_preds = [p for i, p in enumerate(s["preds"]) if i not in rm]
        corrupted.append({"image_id": s["image_id"], "preds": new_preds, "gts": s["gts"]})

    return corrupted, manifest


def main():
    print("loading val split + building raw samples (CPU, mask-only)...", flush=True)
    val = [ln.strip() for ln in
           open(f"{DATA}/ImageSets/Segmentation/val.txt", encoding="utf-8") if ln.strip()]
    samples, image_paths = build_voc_samples(DATA, CROPS, val)
    print(f"{len(samples)} images", flush=True)

    corrupted_samples, manifest = build_corrupted_samples(samples)
    json.dump(manifest, open(f"{OUT_DIR}/corruption_manifest{RUN_TAG}.json", "w"), indent=2)
    print(f"manifest: {len(manifest)} victims -> {OUT_DIR}/corruption_manifest{RUN_TAG}.json",
          flush=True)

    print("decomposing corrupted dataset (Block 1, CPU)...", flush=True)
    c_errors, c_reference = decompose_dataset(corrupted_samples)
    # ---- reuse original embeddings/tags by (image_id, bbox, subtype) ---------
    print("loading cached original (tagged) instances for embedding reuse...", flush=True)
    d = pickle.load(open(PKL, "rb"))
    print(f"corrupted: {len(c_errors)} errors, {len(c_reference)} reference "
          f"(original was {len(d['errors'])} / {len(d['reference'])})", flush=True)
    orig_by_key = {}
    for inst in d["errors"] + d["reference"]:
        key = (inst.image_id, _round_bbox(inst.bbox), inst.seg_subtype)
        orig_by_key[key] = inst

    need_embed = []
    reused = 0
    for inst in c_errors + c_reference:
        key = (inst.image_id, _round_bbox(inst.bbox), inst.seg_subtype)
        src = orig_by_key.get(key)
        if src is not None and src.embedding is not None:
            inst.embedding = src.embedding
            inst.metadata.update(src.metadata)
            reused += 1
        else:
            need_embed.append(inst)
    print(f"embeddings reused: {reused} | need fresh DINOv2: {len(need_embed)} "
          f"(touching {len({i.image_id for i in need_embed})} images)", flush=True)

    if need_embed:
        from slice_quality.extractors import DINOv2Extractor
        extractor = DINOv2Extractor(image_paths, device="cpu")
        mask_lookup = make_mask_lookup(corrupted_samples)
        attach_embeddings(need_embed, extractor, mask_lookup)
        n_ok = sum(1 for i in need_embed if i.embedding is not None)
        print(f"fresh embeddings computed: {n_ok}/{len(need_embed)}", flush=True)

    # ---- cluster-branch scoring (rule A recovery) -----------------------------
    print("running cluster branch (Block 3-9, CPU)...", flush=True)
    cfg = SliceQualityConfig()
    cfg.def1.n_min_frac = 0.005
    cfg.significance.n_boot = 200
    res = run_from_instances(c_errors, c_reference, language_slices=None, cfg=cfg)

    kept_member_ids = {m.instance_id for ks in res.kept for m in ks.slice_.members}
    slice_of, verdict_of = {}, {}
    seen_q = set()
    for s, q in res.scored:               # EVERY candidate slice, not just KEPT
        if q.slice_id in seen_q:
            continue
        seen_q.add(q.slice_id)
        for m in s.members:
            slice_of[m.instance_id] = q.slice_id
            verdict_of[m.instance_id] = q.verdict

    victims_a_manifest = [v for v in manifest if v["rule"] == "small"]
    a_ids = []
    by_bbox_a = {(v["image_id"], _round_bbox(v["bbox"])): v for v in victims_a_manifest}
    for inst in c_errors:
        key = (inst.image_id, _round_bbox(inst.bbox))
        if key in by_bbox_a and inst.seg_subtype is SegErrorType.FN:
            a_ids.append(inst.instance_id)
    n_a = len(victims_a_manifest)

    # ---- optional GPU pass: tag the victims' HOST slices ----------------------
    # Design interaction found in the first runs: victims are brand-new FN
    # instances, so they carry no Qwen attribute (their pre-corruption twins were
    # CORRECT and stored the PREDICTION's bbox, so the tag cache never matches).
    # The certification gate is tag-based -> the host slice is *untestable* at any
    # dose, and the certified tier reads 0% for a reason that is about the
    # experiment's setup, not about the pipeline. Tagging the host slices makes
    # that tier measurable. Needs a GPU: CORRUPTION_TAG_HOSTS=1.
    if os.environ.get("CORRUPTION_TAG_HOSTS") == "1":
        host_ids = {slice_of[iid] for iid in a_ids if iid in slice_of}
        to_tag, seen_i = [], set()
        for s, q in res.scored:
            if q.slice_id not in host_ids or s.source != "cluster":
                continue
            n_have = sum(1 for m in s.members if m.metadata.get("attribute"))
            need = max(0, TAG_TARGET - n_have)
            for m in s.members:
                if need <= 0:
                    break
                if m.instance_id in seen_i or m.metadata.get("attribute"):
                    continue
                seen_i.add(m.instance_id)
                to_tag.append(m)
                need -= 1
        print(f"\nhost slices of victims: {len(host_ids)} | instances to tag: "
              f"{len(to_tag)} (target {TAG_TARGET}/slice)", flush=True)
        if to_tag:
            from slice_quality.tagging import tag_instances, tag_instances_verify
            from slicelens.backends.qwen_vlm import QwenVLM
            vlm = QwenVLM(model=os.environ.get("VLM_MODEL", "/models/qwen"),
                          max_new_tokens=10,
                          load_in_4bit=os.environ.get("QWEN_4BIT") == "1")
            (tag_instances_verify if TAG_MODE == "verify" else tag_instances)(to_tag, image_paths, vlm)
            print("ATTRIBUTES (victim hosts):",
                  dict(Counter(m.metadata.get("attribute", "?") for m in to_tag)),
                  flush=True)
            del vlm
            # re-score with the fresh tags: same instances, so purity is now
            # estimable on the slices that actually contain the injected failure.
            print("re-scoring with victim-host tags (CPU)...", flush=True)
            res = run_from_instances(c_errors, c_reference, language_slices=None, cfg=cfg)
            kept_member_ids = {m.instance_id for ks in res.kept
                               for m in ks.slice_.members}
            slice_of, verdict_of, seen_q = {}, {}, set()
            for s, q in res.scored:
                if q.slice_id in seen_q:
                    continue
                seen_q.add(q.slice_id)
                for m in s.members:
                    slice_of[m.instance_id] = q.slice_id
                    verdict_of[m.instance_id] = q.verdict

    n_a_recovered = sum(1 for iid in a_ids if iid in kept_member_ids)
    a_host_slices = Counter(slice_of[iid] for iid in a_ids if iid in slice_of)
    a_verdicts = Counter(verdict_of[iid] for iid in a_ids if iid in verdict_of)
    print(f"\n=== RULE A (small {RULE_A_CLASS}) ===")
    print(f"injected: {n_a} | matched-as-FN: {len(a_ids)} | "
          f"recovered in a KEPT slice: {n_a_recovered} "
          f"({n_a_recovered / max(1, n_a):.1%})")
    print(f"host slice(s) (any verdict, victim count): "
          f"{dict(a_host_slices.most_common(8))}")
    print(f"verdict breakdown of ALL host slices (incl. REFINE/DISCARD): "
          f"{dict(a_verdicts)}")
    print("per-victim detail (instance_id -> host slice, verdict):")
    for iid in a_ids:
        print(f"  {iid:22s} -> {slice_of.get(iid, '(none)'):22s} "
              f"{verdict_of.get(iid, '?')}")

    # why did the DISCARDed host clusters fail Def.1? (which axis)
    disc_hosts = {slice_of[iid] for iid in a_ids
                  if verdict_of.get(iid) == "DISCARD" and iid in slice_of}
    if disc_hosts:
        print("\nSQV of the DISCARDed host slices (why they failed Def.1):")
        printed = set()
        for s, q in res.scored:
            if q.slice_id in disc_hosts and q.slice_id not in printed:
                printed.add(q.slice_id)
                n_victims_here = sum(1 for iid in a_ids if slice_of.get(iid) == q.slice_id)
                print(f"  {q.slice_id:18s} n_victims={n_victims_here} size={q.size:4d} "
                      f"omega2={q.omega_squared:.3f} slice_purity={q.slice_purity:.3f} "
                      f"error_type_purity={q.error_type_purity:.3f} "
                      f"significant={q.significant} "
                      f"mean_sev_z={q.extras.get('mean_sev_z', '?')} "
                      f"n_tagged={q.extras.get('n_tagged', '?')}")

    # ---- language-branch recovery (rule B), via archived grounding -----------
    print(f"\n=== RULE B ({RULE_B_CLASS} cut off at border) ===")
    try:
        grounding = json.load(open(RULE_B_GROUNDING_FILE, encoding="utf-8"))
        cand = next(c for c in grounding["candidates"] if c["query"] == RULE_B_HYPOTHESIS)
        by_id_conf = {s["instance_id"]: s["conf"] for s in cand.get("scored", [])
                      if s.get("instance_id")}
        # Identity join, NOT bbox: a CORRECT/reference TypedInstance stores the
        # PREDICTION's bbox (typed_error.py: base["bbox"]=p["bbox"]), while the
        # manifest records the GT's bbox -- the two rarely round to the same
        # pixels even at IoU>=0.85. removed_pred_idx gives the exact original
        # "{image}#p{idx}" identity instead.
        orig_by_id = {inst.instance_id: inst for inst in d["errors"] + d["reference"]}
        victims_b_manifest = [v for v in manifest if v["rule"] == "border"]
        confs = []
        for v in victims_b_manifest:
            orig_id = f"{v['image_id']}#p{v['removed_pred_idx']}"
            orig = orig_by_id.get(orig_id)
            if orig is None:
                continue
            c = by_id_conf.get(orig.instance_id)
            if c is not None:
                confs.append(c)
        n_b = len(victims_b_manifest)
        n_b_matched = len(confs)
        n_b_pass = sum(1 for c in confs if c >= LANG_CONF_MIN)
        print(f"injected: {n_b} | found in archived grounding: {n_b_matched} | "
              f"pass confidence gate (>= {LANG_CONF_MIN}): {n_b_pass} "
              f"({n_b_pass / max(1, n_b_matched):.1%} of matched, "
              f"{n_b_pass / max(1, n_b):.1%} of injected)")
        if confs:
            print(f"confidence distribution: min={min(confs):.2f} "
                  f"mean={sum(confs) / len(confs):.2f} max={max(confs):.2f}")
    except Exception as exc:
        print(f"rule B evaluation skipped ({exc})")

    summary = {
        "rule_a": {"class": RULE_A_CLASS, "injected": n_a,
                   "matched_as_fn": len(a_ids), "recovered": n_a_recovered,
                   "recovery_rate": n_a_recovered / max(1, n_a),
                   "host_slices": dict(a_host_slices.most_common(5))},
    }
    json.dump(summary, open(f"{OUT_DIR}/corruption_results{RUN_TAG}.json", "w"), indent=2)
    print(f"\nsaved -> {OUT_DIR}/corruption_results{RUN_TAG}.json")
    print("CORRUPTION_EXPERIMENT_DONE", flush=True)


if __name__ == "__main__":
    main()
