"""Data-driven hypothesis generation (SliceLens Sec. 3.1, part 2) for the VOC classes that
never had it. GPU: Gemma (LLM) and Qwen2.5-VL (VLM) co-loaded in 4-bit on one 24 GB card.

The June pilot (examples/_archive/run_voc_datadriven.py) ran it on aeroplane, chair, cow and
person and kept 6 hypotheses per class in combined_<cls>.txt; the other 16 classes have only
the knowledge-driven ones. Same steps here, on the instances of the run (VOC val, the cc
substrate the grounding uses):

  seeds    Gemma's knowledge-driven CLUSTER attributes for the class (Fig. 8), at most
           MAX_ATTRS, the pilot's four generic attributes when Gemma gives fewer than two
  caption  Qwen captions SAMPLE regions (random.Random(0)) grounded on each attribute (Fig. 9)
  extract  Gemma extracts the attribute value from each caption, clusters the values (Fig. 10)
  refine   Gemma refines the clusters into queries (Fig. 11) -- kept in the record
  condense the pilot's last step, written down: the K largest clusters (values that are not
           "none"), each phrased by Gemma as ONE verifiable situation of one object,
           "a <class> ..." -- K = 6 as in the pilot, so the grounding cost is bounded

Classes are claimed from a queue (mkdir lock) so two GPUs share them.
Out: OUT_DIR/dd_<cls>.json (every intermediate) and OUT_DIR/dd_<cls>.txt (the K hypotheses).
Run: OUT_DIR=/data/outputs_dd python examples/gen_datadriven.py
Env: OUT_DIR, CLASSES, SAMPLE (20), MAX_ATTRS (4), K (6), SUBSTRATE.
"""
from __future__ import annotations

import json
import os
import pickle
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from slicelens.data_types import PromptType  # noqa: E402
from slicelens.hypothesis.data_driven import (cluster_keywords, extract_attribute_value,  # noqa: E402
                                              refine_to_queries)
from slicelens.hypothesis.knowledge_driven import generate_knowledge_hypotheses  # noqa: E402
from slice_quality.adapters.to_slicelens import typed_to_slicelens_instances  # noqa: E402

OUT_DIR = Path(os.environ.get("OUT_DIR", "/data/outputs_dd"))
PILOT = ("aeroplane", "chair", "cow", "person")           # already have data-driven hypotheses
CLASSES = os.environ.get("CLASSES", "").split() or [c for c in (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
    "train", "tvmonitor") if c not in PILOT]
SAMPLE = int(os.environ.get("SAMPLE", "20"))
MAX_ATTRS = int(os.environ.get("MAX_ATTRS", "4"))
K = int(os.environ.get("K", "6"))
SUBSTRATE = os.environ.get("SUBSTRATE", "/data/voc2012/substrate_cc.pkl")
FALLBACK_ATTRS = [                                          # the pilot's, unchanged
    "the background or surroundings of the object",
    "the size and scale of the object",
    "the lighting and contrast conditions",
    "occlusion or overlap with other objects",
]
CONDENSE_SYSTEM = ("You write failure hypotheses for an image segmentation model: short phrases "
                   "that describe a visual situation of ONE object, which a person can verify by "
                   "looking at that object in the image.")
CONDENSE_USER = ("Object class: {cls}. Attribute: {attr}. On real {cls} instances this attribute "
                 "took values grouped as '{name}': {values}.\nWrite ONE phrase, starting with "
                 "'{art} {cls}', that describes this situation of a single {cls}, at most 12 words. "
                 "Reply with the phrase only.")


def claim(cls):
    if (OUT_DIR / f"dd_{cls}.txt").exists():
        return False
    try:
        os.mkdir(OUT_DIR / f".claim_{cls}")
        return True
    except FileExistsError:
        return False


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    from slicelens.backends import HFLLM, QwenVLM
    t0 = time.time()
    llm = HFLLM(model="/models/gemma", load_in_4bit=True, max_new_tokens=1024)
    vlm = QwenVLM(model="/models/qwen", grounding="bbox", load_in_4bit=True)
    print(f"Gemma + Qwen loaded in {time.time() - t0:.0f}s", flush=True)
    d = pickle.load(open(SUBSTRATE, "rb"))
    for cls in CLASSES:
        if not claim(cls):
            continue
        t1 = time.time()
        insts = typed_to_slicelens_instances(d["errors"], d["reference"], d["image_paths"], only_label=cls)
        sampled = random.Random(0).sample(insts, SAMPLE) if len(insts) > SAMPLE else list(insts)
        try:
            know = generate_knowledge_hypotheses(llm, task_type="segmentation", label=cls)
        except ValueError as exc:                           # malformed JSON from Gemma (GESD apple,
            print(f"  [{cls}] Gemma's attributes unreadable ({str(exc)[:60]}...)", flush=True)
            know = []                                       # teddy_bear): the fallback attributes below
        attrs = [h.query for h in know if h.prompt_type == PromptType.CLUSTER][:MAX_ATTRS]
        seeds_from = "gemma"
        if len(attrs) < 2:
            attrs, seeds_from = FALLBACK_ATTRS[:MAX_ATTRS], "fallback"
        print(f"\n=== {time.strftime('%H:%M')} {cls}: {len(insts)} instances, {len(sampled)} sampled; "
              f"attributes ({seeds_from}): {attrs}", flush=True)
        rec = {"label": cls, "n_instances": len(insts), "sample": [i.instance_id for i in sampled],
               "seeds_from": seeds_from, "attributes": []}
        groups = []
        for attr in attrs:
            caps = [vlm.caption_region(i.image_path, i.region, attr) for i in sampled]
            vals = [extract_attribute_value(llm, attr, c) for c in caps]
            clusters = cluster_keywords(llm, attr, [v for v in vals if v and v != "none"])
            queries = refine_to_queries(llm, attr, clusters)
            rec["attributes"].append({"attribute": attr, "captions": caps, "values": vals,
                                      "clusters": clusters, "queries": queries})
            for name, members in clusters.items():
                groups.append((len(members), attr, name, members))
            print(f"  [{attr[:40]}] {len(clusters)} clusters: "
                  f"{ {k: len(v) for k, v in clusters.items()} }", flush=True)
        groups.sort(key=lambda g: -g[0])                    # largest clusters first, stable
        hyps = []
        for n, attr, name, members in groups:
            if len(hyps) >= K:
                break
            art = "an" if cls[0] in "aeiou" else "a"
            h = llm.chat(system=CONDENSE_SYSTEM, user=CONDENSE_USER.format(
                cls=cls, attr=attr, name=name, values=", ".join(members[:12]), art=art)).strip()
            h = h.strip().strip('"').splitlines()[0].strip().rstrip(".")
            if h and h.lower() not in {x.lower() for x in hyps}:
                hyps.append(h)
        rec["condensed"] = [{"hypothesis": h} for h in hyps]
        json.dump(rec, open(OUT_DIR / f"dd_{cls}.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        (OUT_DIR / f"dd_{cls}.txt").write_text(
            f"# {cls}: data-driven (SliceLens Sec. 3.1 part 2), {len(hyps)} condensed from the largest clusters\n"
            + "\n".join(hyps) + "\n", encoding="utf-8")
        print(f"=== done {cls} in {(time.time() - t1) / 60:.1f} min: {hyps}", flush=True)
    left = [c for c in CLASSES if not (OUT_DIR / f"dd_{c}.txt").exists()]
    print(f"worker finished; still missing: {left or 'none'}", flush=True)
    print("DD_GEN_DONE" if not left else "DD_WORKER_DONE", flush=True)


if __name__ == "__main__":
    main()
