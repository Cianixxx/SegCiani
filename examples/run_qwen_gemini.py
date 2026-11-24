"""Run SliceLens with the real backends: Gemini-2.5-Pro + Qwen2.5-VL-7B.

This is the configuration from the paper (Sections 5.1.2 / B.1). Run it on your
GPU server after installing the optional dependencies::

    pip install google-genai "transformers>=4.49" accelerate qwen-vl-utils pillow torch
    export GOOGLE_API_KEY=...        # PowerShell: $env:GOOGLE_API_KEY="..."
    python examples/run_qwen_gemini.py

The example shows the end-to-end flow on a *tiny* hand-built set of detection
results. Replace ``build_demo_images()`` with your own detector outputs + ground
truth (e.g. parsed from COCO/KITTI predictions).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slicelens import SliceLens, SliceLensConfig  # noqa: E402
from slicelens.backends import GeminiLLM, QwenVLM  # noqa: E402
from slicelens.errors import (  # noqa: E402
    Detection,
    GroundTruth,
    ImageAnnotation,
    build_detection_error_instances,
)
from slicelens.identification import analyze_slope_trend  # noqa: E402


def build_demo_images() -> list[ImageAnnotation]:
    """Replace with your real detector predictions + ground truth.

    image_path must point at a real image file readable by the VLM.
    """
    return [
        ImageAnnotation(
            image_path="/path/to/coco/000000001.jpg",
            ground_truths=[GroundTruth(box=(120, 80, 260, 300), label="bicycle")],
            detections=[],  # missed -> false negative
        ),
        ImageAnnotation(
            image_path="/path/to/coco/000000002.jpg",
            ground_truths=[GroundTruth(box=(40, 60, 180, 240), label="bicycle")],
            detections=[Detection(box=(42, 62, 182, 238), label="bicycle", score=0.93)],
        ),
        # ... thousands more ...
    ]


def main() -> None:
    images = build_demo_images()
    instances = build_detection_error_instances(
        images, target_label="bicycle", iou_threshold=0.5
    )
    print(f"Built {len(instances)} error/correct instances for 'bicycle'.")

    config = SliceLensConfig(task_type="detection", grounding="bbox", precision_k=10)
    llm = GeminiLLM(model=config.llm_model)
    vlm = QwenVLM(model=config.vlm_model, grounding=config.grounding)

    sl = SliceLens(llm, vlm, config)
    result = sl.run(instances, label="bicycle")

    print(f"\nGenerated {len(result.hypotheses)} hypotheses.")
    print(f"Discovered {len(result.candidate_slices)} candidate slices.")
    print(f"Identified {len(result.error_slices)} systematic error slices:\n")
    for es in result.error_slices:
        trend = analyze_slope_trend(es.candidate, config.slope)
        top = es.candidate.top_k(5)
        print(f"  * {es.hypothesis.query!r}  (slope={es.max_positive_slope:.3f})")
        for s in top:
            print(f"      {s.slice_confidence:0.3f}  {s.instance.image_path}")
        print()


if __name__ == "__main__":
    main()
