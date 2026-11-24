"""Data-driven hypothesis generation on Pascal-VOC (SliceLens Section 3.1, part 2).

Bottom-up discovery: Qwen captions the real (error+correct) instances grounded on
each attribute seed (Fig. 9), Gemma extracts the attribute value per caption and
clusters them (Fig. 10), then refines the clusters into query hypotheses (Fig. 11).
Prints the intermediates so we can SEE what the data itself surfaces.

Runs LLM (Gemma) + VLM (Qwen) co-loaded in 4-bit on one 24 GB GPU.

Example (in the container, GPU 5):
    python examples/run_voc_datadriven.py --label person \
        --gemma /models/gemma --qwen /models/qwen \
        --dataset-root /data/dataset --crops-dir /data/voc_crops \
        --sample-size 20 --max-attrs 4 --out /data/outputs/voc_datadriven_person.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slicelens.adapters import load_voc_seg_instances  # noqa: E402
from slicelens.data_types import PromptType  # noqa: E402
from slicelens.hypothesis.data_driven import (  # noqa: E402
    cluster_keywords,
    extract_attribute_value,
    refine_to_queries,
)
from slicelens.hypothesis.knowledge_driven import generate_knowledge_hypotheses  # noqa: E402

FALLBACK_ATTRS = [
    "the background or surroundings of the object",
    "the size and scale of the object",
    "the lighting and contrast conditions",
    "occlusion or overlap with other objects",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="person")
    ap.add_argument("--gemma", default="/models/gemma")
    ap.add_argument("--qwen", default="/models/qwen")
    ap.add_argument("--dataset-root", default="/data/dataset")
    ap.add_argument("--crops-dir", default="/data/voc_crops")
    ap.add_argument("--sample-size", type=int, default=20)
    ap.add_argument("--max-attrs", type=int, default=4)
    ap.add_argument("--names-file", default=None,
                    help="split list (e.g. VOC2012 trainval.txt) for full-dataset mode")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap number of images scanned (0 = all)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from slicelens.backends import HFLLM, QwenVLM
    print("Loading Gemma (LLM, 4-bit) ...", flush=True)
    llm = HFLLM(model=args.gemma, load_in_4bit=True, max_new_tokens=1024)
    print("Loading Qwen2.5-VL (VLM, 4-bit) ...", flush=True)
    vlm = QwenVLM(model=args.qwen, grounding="bbox", load_in_4bit=True)

    instances = load_voc_seg_instances(args.dataset_root, args.crops_dir,
                                        only_label=args.label,
                                        names_file=args.names_file, limit=args.limit)
    n_err = sum(i.is_error for i in instances)
    print(f"\n{len(instances)} '{args.label}' instances ({n_err} errors).")

    # --- knowledge-driven first: gives the cluster ATTRIBUTE SEEDS (Fig. 8) ---
    print("\n=== Knowledge-driven seeds (Gemma) ===", flush=True)
    knowledge = generate_knowledge_hypotheses(llm, task_type="segmentation", label=args.label)
    search = [h for h in knowledge if h.prompt_type == PromptType.SEARCH]
    clusters_seed = [h for h in knowledge if h.prompt_type == PromptType.CLUSTER]
    attrs = [h.query for h in clusters_seed][: args.max_attrs]
    if len(attrs) < 2:
        print(f"  (solo {len(attrs)} attributi cluster da Gemma -> uso fallback)")
        attrs = FALLBACK_ATTRS[: args.max_attrs]
    print(f"  {len(search)} search hypotheses, {len(clusters_seed)} cluster seeds.")
    print("  Attributi (seed) usati per il data-driven:")
    for a in attrs:
        print("    -", a)

    # --- data-driven: caption -> extract -> cluster -> refine, per attribute ---
    rng = random.Random(0)
    sampled = (rng.sample(instances, args.sample_size)
               if len(instances) > args.sample_size else list(instances))
    print(f"\n=== Data-driven mining su {len(sampled)} istanze ===", flush=True)

    out = {"label": args.label, "n_instances": len(instances), "n_errors": int(n_err),
           "attributes": [], "data_driven_hypotheses": []}
    all_queries = []
    for attr in attrs:
        print(f"\n---- attributo: {attr} ----", flush=True)
        captions = [vlm.caption_region(i.image_path, i.region, attr) for i in sampled]
        values = [extract_attribute_value(llm, attr, c) for c in captions]
        print("  caption (esempi):")
        for c in captions[:3]:
            print("     ·", c[:90].replace("\n", " "))
        print("  valori estratti:", [v[:24] for v in values[:12]])
        clusters = cluster_keywords(llm, attr, values)
        print("  cluster:", {k: v[:6] for k, v in clusters.items()})
        queries = refine_to_queries(llm, attr, clusters)
        print("  -> query data-driven:")
        for q in queries:
            print("       *", q)
        out["attributes"].append({"attribute": attr,
                                   "values": values,
                                   "clusters": clusters,
                                   "queries": queries})
        all_queries += queries

    # de-dup
    seen, uniq = set(), []
    for q in all_queries:
        if q.lower() not in seen:
            seen.add(q.lower()); uniq.append(q)
    out["data_driven_hypotheses"] = uniq
    print(f"\n=== {len(uniq)} ipotesi data-driven uniche ===")
    for q in uniq:
        print("  -", q)

    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                  encoding="utf-8")
        # also a plain queries file, ready to feed run_voc.py --queries-file
        qf = Path(args.out).with_suffix(".txt")
        qf.write_text("# data-driven hypotheses for '%s'\n" % args.label
                      + "\n".join(uniq) + "\n", encoding="utf-8")
        print(f"\nWrote {args.out} and {qf}")


if __name__ == "__main__":
    main()
