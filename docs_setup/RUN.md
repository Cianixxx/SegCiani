# Your first real run

Three increasingly complete options. Start with #1 (instant), then do #2 or #3.

---

## 1. Taste it now — real hypotheses, no GPU, no data

Runs Stage 1 (failure-mode hypothesis generation) with the **real LLM**. Needs
only a Gemini key.

```bash
pip install google-genai
export GOOGLE_API_KEY=...                       # PowerShell: $env:GOOGLE_API_KEY="..."
python examples/run_hypotheses.py --label bicycle --task detection
```

You'll see the actual failure hypotheses Gemini proposes (e.g. *"bicycle obscured
by a person"*, *"very small bicycle"*, ...). This is the real Stage 1; the full
pipeline then verifies these against real detector errors (below).

---

## 2. Full pipeline via API — no GPU (laptop/server)

Needs detector predictions (Step A) + a Gemini key. Everything runs through the
API, so **no GPU**.

> ⚠️ The detector (Step A) needs `torch`/`torchvision`, which currently have no
> wheels for Python 3.14. Use Python 3.10–3.12 (a venv/conda), or run Step A
> inside the AIRLab container (option #3), then run Step B anywhere.

**Step A — get the COCO data + make predictions** (once):
```bash
# COCO val2017 (≈780 MB images + ≈240 MB annotations)
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip val2017.zip -d datasets/coco/
unzip annotations_trainval2017.zip -d datasets/coco/

pip install torch torchvision pillow
python scripts/make_coco_predictions.py \
    --images-dir  datasets/coco/val2017 \
    --gt          datasets/coco/annotations/instances_val2017.json \
    --out-results datasets/coco/preds/fasterrcnn.bbox.json \
    --out-gt      datasets/coco/annotations/instances_subset.json \
    --limit 200 --score-threshold 0.3       # 200 imgs = quick first run
```

**Step B — discover error slices (all-API VLM):**
```bash
pip install google-genai
export GOOGLE_API_KEY=...
python examples/run_coco.py \
    --vlm-backend gemini \
    --gt          datasets/coco/annotations/instances_subset.json \
    --results     datasets/coco/preds/fasterrcnn.bbox.json \
    --image-root  datasets/coco/val2017 \
    --label bicycle \
    --out outputs/bicycle_slices.json
```

---

## 3. Full pipeline on AIRLab — local Qwen VLM (GPU)

The proper environment (controlled Python, GPU). See [`AIRLAB.md`](AIRLAB.md) for
build/connect details. In short, inside the running container:

```bash
# (data downloaded to ~/dataset/private/coco, see Step A above)
python scripts/make_coco_predictions.py --images-dir datasets/coco/val2017 \
    --gt datasets/coco/annotations/instances_val2017.json \
    --out-results datasets/coco/preds/fasterrcnn.bbox.json \
    --out-gt datasets/coco/annotations/instances_subset.json --limit 200

python examples/run_coco.py \
    --gt datasets/coco/annotations/instances_subset.json \
    --results datasets/coco/preds/fasterrcnn.bbox.json \
    --image-root datasets/coco/val2017 \
    --label bicycle --out outputs/bicycle_slices.json
    # on westworld (11 GB) add: --load-in-4bit
```

---

### What you get
`outputs/bicycle_slices.json` lists each **systematic error slice** (e.g.
*"bicycle obscured by a person"*) with its slope, the error-rate-vs-confidence
curve, and the top-ranked instances (image path + bbox + slice confidence) — the
material for your thesis figures.

**Scale note:** the VLM is called once per (instance × hypothesis). Keep the
first run small (`--limit`, `--score-threshold`, `--no-data-driven`) before
scaling up. Via API this is the cost/latency bottleneck (paper Table 5).
