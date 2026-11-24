# Running SliceLens on the GPU server

> **On AIRLab (PoliMi DEIB)?** Use the lab-specific Docker guide:
> [`AIRLAB.md`](AIRLAB.md). This file is the generic (conda / non-Docker) path.

The real backends (paper config): **Gemini-2.5-Pro** (hypothesis generation) +
**Qwen2.5-VL-7B-Instruct** (slice confidence). The 7B VLM needs a GPU
(~16–20 GB VRAM in fp16/bf16, or ~8 GB with `--load-in-4bit`).

## 1. Connect & install

Open the server folder in VS Code (Remote-SSH), then:

```bash
# Linux GPU server (one-shot):
bash scripts/setup_server.sh
```

Or manually:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
# Pick the torch build matching your CUDA driver, e.g. CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[gemini,qwen,dev]"
```

Set the Gemini key:

```bash
export GOOGLE_API_KEY=...     # get one at https://aistudio.google.com/apikey
```

## 2. Verify the VLM works

This loads Qwen2.5-VL, builds its own test image, and checks that the **yes/no
logit** slice confidence (Eq. 1) discriminates a matching query — the single
most important thing to confirm:

```bash
python scripts/check_qwen.py
```

Expect `[PASS] slice confidence discriminates the matching query.`

## 3. Run on your detection results (COCO format)

`--results` is a standard COCO detection-results JSON
(`[{image_id, category_id, bbox, score}, ...]`) — Detectron2, mmdetection, YOLO
(via converter), and DETR can all export this.

```bash
python examples/run_coco.py \
    --gt         /data/coco/annotations/instances_val2017.json \
    --results    /data/coco/preds/faster_rcnn.bbox.json \
    --image-root /data/coco/val2017 \
    --label bicycle \
    --out outputs/bicycle_slices.json
```

The script collects FN/FP errors for the class (§4.1), runs the full pipeline,
prints the systematic error slices, and writes a JSON report with each slice's
slope, error-rate curve, and top-ranked instances (image path + bbox +
confidence) — ready to visualise in your thesis.

Useful flags: `--no-data-driven` (faster, knowledge-driven only), `--iou`,
`--score-threshold`, `--grounding point`, `--k`.

## 4. Faster VLM serving (optional)

The paper deploys Qwen with vLLM for throughput. Install `pip install -e ".[vllm]"`.
The current `QwenVLM` uses HuggingFace `transformers` (one forward pass per
instance to read the yes/no logits); a vLLM backend can be added behind the same
`VLMBackend` interface if throughput becomes a bottleneck.

## Notes / gotchas

- **First run downloads** the ~16 GB Qwen weights to the HF cache (`HF_HOME` to
  relocate).
- **Cost/latency:** Gemini is called only during hypothesis generation (a handful
  of calls). The VLM is called once per (instance × verifiable hypothesis) — that
  dominates runtime. Use `--score-threshold` to prune low-score detections and
  `--no-data-driven` for a quick first pass.
- If `transformers` is too old for Qwen2.5-VL, upgrade: `pip install -U "transformers>=4.49"`.
