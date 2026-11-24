"""Ground the extra class-agnostic hypotheses on the SAME sample as the July runs.

The 8 hypotheses come from ``slice_quality.describe._GENERIC_EXTRA`` (single
source: the grounded text is the rendered text). For every VOC class this runs
exactly ``run_voc.py --substrate cc --max-instances 120`` -- same substrate cache,
same deterministic 120-instance sample (``random.Random(0)``), same grounding,
same slope-trend test -- so the output is directly comparable with the
``voc_paper_<cls>_cc.json`` files that discovered the 7 canonical hypotheses.

Two differences, both about cost, none about the result:
  * Qwen is loaded ONCE per worker: ``slicelens.backends.QwenVLM`` is replaced by
    a factory returning the already-loaded model before ``run_voc.main`` runs.
  * classes are claimed from a shared queue (an ``mkdir`` lock per class in
    OUT_DIR), so a second worker on another GPU can join later and split the
    classes that are left; a crashed class leaves its lock and no output, and is
    retried by deleting ``.claim_<cls>``.

Writes OUT_DIR/voc_paper_<cls>_cc.json -- never into /data/outputs, where the
runs of the thesis live.

Run (GPU container, as run_slicelens_classes.sh):
    OUT_DIR=/data/outputs_extra python examples/ground_extra_hypotheses.py
Env: OUT_DIR, CLASSES (default all 20), QWEN_4BIT (0), MAX_INSTANCES (120),
     QUERIES_TEMPLATE (a per-class query file, e.g. /data/outputs_dd/dd_{cls}.txt, in place of
     the 8 extra generic hypotheses; a class without that file is skipped).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import slicelens.backends as backends  # noqa: E402
from slice_quality.describe import _GENERIC_EXTRA  # noqa: E402

DATA = "/data/voc2012/VOCdevkit/VOC2012"
OUT_DIR = Path(os.environ.get("OUT_DIR", "/data/outputs_extra"))
CLASSES = os.environ.get("CLASSES", "").split() or [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
    "train", "tvmonitor"]
MAX_INSTANCES = os.environ.get("MAX_INSTANCES", "120")
QWEN_4BIT = os.environ.get("QWEN_4BIT", "0") == "1"
QUERIES_TEMPLATE = os.environ.get("QUERIES_TEMPLATE", "")


def claim(cls: str) -> bool:
    """True if this worker got the class (atomic: mkdir fails if it exists)."""
    if (OUT_DIR / f"voc_paper_{cls}_cc.json").exists():
        return False
    try:
        os.mkdir(OUT_DIR / f".claim_{cls}")
        return True
    except FileExistsError:
        return False


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    queries = OUT_DIR / "queries_extra.txt"
    queries.write_text("# extra class-agnostic hypotheses (slice_quality.describe._GENERIC_EXTRA)\n"
                       + "\n".join(_GENERIC_EXTRA) + "\n", encoding="utf-8")

    t0 = time.time()
    vlm = backends.QwenVLM(model="/models/qwen", grounding="bbox", load_in_4bit=QWEN_4BIT)
    print(f"Qwen loaded once ({time.time() - t0:.0f}s), 4bit={QWEN_4BIT}", flush=True)
    backends.QwenVLM = lambda *a, **k: vlm          # run_voc.main imports it at call time

    import run_voc
    done = 0
    for cls in CLASSES:
        qf = Path(QUERIES_TEMPLATE.format(cls=cls)) if QUERIES_TEMPLATE else queries
        if not qf.exists():
            continue
        if not claim(cls):
            continue
        t1 = time.time()
        print(f"\n=== {time.strftime('%H:%M')} extra hypotheses: {cls} ===", flush=True)
        sys.argv = ["run_voc.py",
                    "--dataset-root", DATA, "--crops-dir", "/data/voc2012/voc_crops",
                    "--names-file", f"{DATA}/ImageSets/Segmentation/val.txt",
                    "--substrate", "cc", "--substrate-cache", "/data/voc2012/substrate_cc.pkl",
                    "--vlm-backend", "qwen", "--vlm-model", "/models/qwen", "--grounding", "bbox",
                    "--max-instances", MAX_INSTANCES, "--label", cls,
                    "--queries-file", str(qf),
                    "--out", str(OUT_DIR / f"voc_paper_{cls}_cc.json")]
        run_voc.main()
        done += 1
        print(f"=== done {cls} in {(time.time() - t1) / 60:.1f} min ===", flush=True)
    left = [c for c in CLASSES if not (OUT_DIR / f"voc_paper_{c}_cc.json").exists()
            and (not QUERIES_TEMPLATE or Path(QUERIES_TEMPLATE.format(cls=c)).exists())]
    print(f"\nworker finished: {done} classes here; still missing: {left or 'none'}", flush=True)
    print("EXTRA_GROUNDING_DONE" if not left else "EXTRA_WORKER_DONE", flush=True)


if __name__ == "__main__":
    main()
