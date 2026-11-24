"""Full-population re-grounding of ONE class, split in blocks over several GPUs.

ground_ea_hypotheses.py hands out whole classes, so on SBD the person class (8 793
instances x 7 validated hypotheses, more than half of the full phase) would run on
one GPU for a day while the other sits idle. This script computes exactly the same
groundings -- same instances (typed_to_slicelens_instances), same validated hypotheses
(ground_ea_hypotheses.validated), same Qwen call, same reuse of the groundings already
on disk -- but in NBLOCKS blocks of instances claimed from a queue (mkdir lock), so any
number of GPUs share the class. Only the scheduling changes, not one number.

  PHASE=ground    GPU workers: OUT_DIR/split/<cls>_<b>.json = {query: {instance_id: conf}}
  PHASE=merge     CPU, after every block: the blocks go into groundings_<cls>.json, after
                  which ground_ea_hypotheses.py PHASE=full CLASSES=<cls> finds every
                  grounding cached and writes voc_paper_<cls>_full.json without a GPU.

Env: CLS, NBLOCKS (24), plus those of ground_ea_hypotheses.py (OUT_DIR, SUBSTRATE,
SLICES_INDEX, OLD_SUBSTRATE=, EDITED=, CACHE_DIRS=, QWEN_4BIT).
"""
import json
import os
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ground_ea_hypotheses as g  # noqa: E402
from slice_quality.adapters.to_slicelens import typed_to_slicelens_instances  # noqa: E402

CLS = os.environ["CLS"]
NBLOCKS = int(os.environ.get("NBLOCKS", "24"))
PHASE = os.environ.get("PHASE", "ground")
SPLIT = g.OUT_DIR / "split"


def blocks():
    d = pickle.load(open(g.SUBSTRATE, "rb"))
    insts = typed_to_slicelens_instances(d["errors"], d["reference"], d["image_paths"], only_label=CLS)
    insts = sorted(insts, key=lambda i: i.instance_id)
    return [insts[b::NBLOCKS] for b in range(NBLOCKS)]


def ground():
    SPLIT.mkdir(parents=True, exist_ok=True)
    queries = g.validated(CLS)
    bl = blocks()
    cache = g.load_cache(CLS, set())              # this run's own groundings (the 120-sample ones)
    n_all = sum(len(b) for b in bl)
    print(f"[{CLS}] {len(queries)} validated hypotheses x {n_all} instances in {NBLOCKS} blocks; "
          f"{sum(1 for b in bl for i in b for q in queries if (i.instance_id, q) in cache)} already grounded",
          flush=True)
    done_here = 0
    for b, insts in enumerate(bl):
        out = SPLIT / f"{CLS}_{b:03d}.json"
        if out.exists():
            continue
        try:
            os.mkdir(SPLIT / f".claim_{CLS}_{b:03d}")
        except FileExistsError:
            continue
        if os.environ.get("CLAIM_OWNER"):        # who holds the block, so that a stopped worker frees only its own
            (SPLIT / f".owner_{CLS}_{b:03d}").write_text(os.environ["CLAIM_OWNER"])
        t0, new = time.time(), 0
        rec = {q: {} for q in queries}
        for inst in insts:
            for q in queries:
                k = (inst.instance_id, q)
                if k in cache:
                    conf = cache[k]
                else:
                    conf = g.vlm().slice_confidence(inst.image_path, inst.region, q)
                    new += 1
                rec[q][inst.instance_id] = round(float(conf), 4)
        json.dump(rec, open(str(out) + ".tmp", "w"))
        os.replace(str(out) + ".tmp", out)
        done_here += 1
        el = time.time() - t0
        print(f"  {time.strftime('%m-%d %H:%M')} [{CLS} block {b + 1}/{NBLOCKS}] {len(insts)} instances, "
              f"{new} new groundings in {el / 60:.1f} min ({el / max(new, 1):.2f} s each)", flush=True)
    left = [b for b in range(NBLOCKS) if not (SPLIT / f"{CLS}_{b:03d}.json").exists()]
    print(f"worker done ({done_here} blocks here); blocks still missing: {len(left)}", flush=True)
    print("SPLIT_ALL_DONE" if not left else "SPLIT_WORKER_DONE", flush=True)


def merge():
    own_p = g.OUT_DIR / f"groundings_{CLS}.json"
    own = json.loads(own_p.read_text()) if own_p.exists() else {}
    n = 0
    for b in range(NBLOCKS):
        part = json.load(open(SPLIT / f"{CLS}_{b:03d}.json"))     # every block must exist
        for q, m in part.items():
            own.setdefault(q, {}).update(m)
            n += len(m)
    own_p.write_text(json.dumps(own))
    print(f"[{CLS}] merged {n} groundings from {NBLOCKS} blocks into {own_p}", flush=True)
    print("SPLIT_MERGE_DONE", flush=True)


ground() if PHASE == "ground" else merge()
