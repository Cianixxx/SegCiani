"""Quick real run: generate failure-mode hypotheses with the *real* LLM.

This is the lowest-friction taste of the pipeline -- it runs Stage 1
(knowledge-driven hypothesis generation, Section 3.1) with Gemini and prints the
hypotheses. It needs only ``GOOGLE_API_KEY`` -- **no GPU, no dataset, no VLM** --
so you can run it on your laptop right now::

    pip install google-genai
    export GOOGLE_API_KEY=...        # PowerShell: $env:GOOGLE_API_KEY="..."
    python examples/run_hypotheses.py --label bicycle --task detection

You can also target a named bias dataset (Section A.2) or give free text::

    python examples/run_hypotheses.py --dataset waterbirds
    python examples/run_hypotheses.py --task-description "a stop sign detector"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slicelens import SliceLensConfig  # noqa: E402
from slicelens.backends import GeminiLLM, MockVLM  # noqa: E402
from slicelens.hypothesis import generate_knowledge_hypotheses  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Generate failure hypotheses with the real LLM.")
    p.add_argument("--label", default=None, help="object class, e.g. 'bicycle'")
    p.add_argument("--task", default="detection",
                   choices=["detection", "segmentation", "classification"])
    p.add_argument("--dataset", default=None,
                   help="named dataset prompt: waterbirds | celeba | nico++")
    p.add_argument("--task-description", default=None, help="free-text task context")
    p.add_argument("--llm-model", default="gemini-2.5-pro")
    args = p.parse_args()

    if not (args.label or args.dataset or args.task_description):
        p.error("provide one of --label, --dataset, or --task-description")

    llm = GeminiLLM(model=args.llm_model)
    hyps = generate_knowledge_hypotheses(
        llm,
        task_type=args.task,
        label=args.label,
        dataset=args.dataset,
        task_description=args.task_description,
    )

    search = [h for h in hyps if h.prompt_type.value == "search"]
    cluster = [h for h in hyps if h.prompt_type.value == "cluster"]
    print(f"\nGenerated {len(hyps)} hypotheses "
          f"({len(search)} verifiable 'search', {len(cluster)} 'cluster' seeds):\n")

    by_factor: dict[str, list] = {}
    for h in hyps:
        by_factor.setdefault(h.factor or "(uncategorized)", []).append(h)
    for factor, items in by_factor.items():
        print(f"## {factor}")
        for h in items:
            print(f"  [{h.prompt_type.value:7s}] {h.query}")
            if h.title:
                print(f"            ({h.title})")
        print()

    print("Next: turn these into verified error slices by ranking real instances")
    print("with a VLM (Eq. 1). See examples/run_coco.py once you have detections.")
    # SliceLensConfig / MockVLM imported to show the full objects are available;
    # the VLM is only needed for the verification stage, not here.
    _ = (SliceLensConfig, MockVLM)


if __name__ == "__main__":
    main()
