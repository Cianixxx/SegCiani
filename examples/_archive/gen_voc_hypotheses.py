"""Generate SliceLens *segmentation* hypotheses (stage 1) with a local or cloud LLM.

Performs the knowledge-driven generation (Fig. 8 system prompt + Fig. 13
segmentation user prompt) and writes:

  * a queries file  -> fed to the server-side Qwen2.5-VL verification
                       (run_voc.py --queries-file), so the hypotheses are
                       LLM-generated, not hand-written;
  * a JSON          -> the full structured hypothesis set, for the report;
  * a raw .txt      -> the LLM's raw response (transparency / debugging).

Backends:
  --llm-backend hf      local model, e.g. Gemma 4 12B on the GPU server
                        (run inside the container with the model mounted)
  --llm-backend gemini  Gemini-2.5-Pro (needs GOOGLE_API_KEY + credits)

Example (local Gemma, in the container):
    python examples/gen_voc_hypotheses.py --llm-backend hf \
        --llm-model /models/gemma --load-in-4bit --label person \
        --out-queries /data/hyp_person.txt --out-json /data/hyp_person.json
"""

from __future__ import annotations

import argparse
import json
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="VOC class, e.g. person")
    ap.add_argument("--llm-backend", default="gemini", choices=["gemini", "hf"])
    ap.add_argument("--llm-model", default="gemini-2.5-pro")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--max", type=int, default=15, help="max SEARCH hypotheses to keep")
    ap.add_argument("--out-queries", required=True)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--raw-out", default=None)
    args = ap.parse_args()

    if args.llm_backend == "hf":
        from slicelens.backends import HFLLM
        llm = HFLLM(model=args.llm_model, load_in_4bit=args.load_in_4bit,
                    max_new_tokens=3072)
    else:
        from slicelens.backends import GeminiLLM
        llm = GeminiLLM(model=args.llm_model)

    user = build_user_prompt("segmentation", label=args.label)
    print(f"Generating segmentation hypotheses for '{args.label}' with "
          f"{args.llm_backend}:{args.llm_model} (Fig. 13 prompt) ...")
    raw = llm.chat(system=KNOWLEDGE_DRIVEN_SYSTEM_PROMPT, user=user)

    if args.raw_out:
        Path(args.raw_out).write_text(raw, encoding="utf-8")
        print(f"raw LLM response -> {args.raw_out}  ({len(raw)} chars)")

    try:
        data = extract_json(raw)
    except ValueError as exc:
        print(f"\n!! Could not parse JSON from the LLM response: {exc}")
        print("   (raw response saved; inspect --raw-out)")
        print("   --- first 600 chars ---")
        print(raw[:600])
        raise SystemExit(1)

    hyps = parse_hypotheses(data)
    search = [h for h in hyps if h.prompt_type == PromptType.SEARCH]
    kept = search[: args.max]

    header = f"# Local-LLM ({args.llm_model}) segmentation hypotheses for '{args.label}'\n"
    Path(args.out_queries).write_text(
        header + "\n".join(h.query for h in kept) + "\n", encoding="utf-8")
    print(f"\n{len(hyps)} hypotheses parsed, {len(search)} SEARCH, {len(kept)} kept "
          f"-> {args.out_queries}")
    for h in kept:
        print("  -", h.query)

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(
            [{"query": h.query,
              "prompt_type": getattr(h.prompt_type, "value", str(h.prompt_type)),
              "title": getattr(h, "title", None),
              "factor": getattr(h, "factor", None),
              "source": getattr(h, "source", None)} for h in hyps],
            indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"full structured set -> {args.out_json}")


if __name__ == "__main__":
    main()
