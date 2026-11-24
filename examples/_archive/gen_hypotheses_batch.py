"""Generate combined_<class>.txt query files for MANY classes in one go.

Loads the LLM (local Gemma) ONCE and loops the classes -- ~2 min per class
instead of ~5 (model load dominates). Each output file = the 7 canonical
class-agnostic queries + up to --max class-specific SEARCH hypotheses from the
knowledge-driven generation (Fig. 8 system + Fig. 13 segmentation user prompt).
A class whose LLM output fails to parse falls back to canonical-only (the batch
never stops).

Run (GPU container, Gemma mounted at /models/gemma):
    python examples/gen_hypotheses_batch.py --llm-model /models/gemma \
        --load-in-4bit --out-dir /data --classes pottedplant sofa ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slicelens import PromptType  # noqa: E402
from slicelens.hypothesis.knowledge_driven import (  # noqa: E402
    build_user_prompt,
    parse_hypotheses,
)
from slicelens.prompts import KNOWLEDGE_DRIVEN_SYSTEM_PROMPT  # noqa: E402
from slicelens.utils import extract_json  # noqa: E402

CANONICAL = [
    "a small object",
    "an object occluded by another object",
    "an object that blends into its background",
    "an object cut off at the image border",
    "an object with thin or elongated parts",
    "several overlapping objects of the same type",
    "an object that looks similar to a different object category",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", nargs="+", required=True)
    ap.add_argument("--llm-model", required=True)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--max", type=int, default=15, help="max class-specific queries")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    from slicelens.backends import HFLLM
    llm = HFLLM(model=args.llm_model, load_in_4bit=args.load_in_4bit,
                max_new_tokens=3072)

    canon_norm = {q.strip().lower() for q in CANONICAL}
    for cls in args.classes:
        out = Path(args.out_dir) / f"combined_{cls}.txt"
        specific: list[str] = []
        try:
            raw = llm.chat(system=KNOWLEDGE_DRIVEN_SYSTEM_PROMPT,
                           user=build_user_prompt("segmentation", label=cls))
            hyps = parse_hypotheses(extract_json(raw))
            search = [h.query.strip() for h in hyps
                      if h.prompt_type == PromptType.SEARCH]
            seen = set(canon_norm)
            for q in search:
                if q.lower() not in seen:
                    seen.add(q.lower())
                    specific.append(q)
                if len(specific) >= args.max:
                    break
        except Exception as exc:                     # keep the batch going
            print(f"[{cls}] generation FAILED ({exc}) -> canonical-only", flush=True)
        header = (f"# {cls}: knowledge-driven (7 canonical + {len(specific)} "
                  f"Gemma-local, batch)\n")
        out.write_text(header + "\n".join(CANONICAL + specific) + "\n",
                       encoding="utf-8")
        print(f"[{cls}] {7 + len(specific)} queries -> {out}", flush=True)

    print("HYPOTHESES_BATCH_DONE", flush=True)


if __name__ == "__main__":
    main()
