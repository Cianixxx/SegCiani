"""Evaluate discovered slices against the HUMAN annotations (answer_gts.jsonl).

Upgrades the err@top-k proxy to a human-grounded evaluation, mirroring the
paper's protocol (Precision@k, Eq. 2 + semantic relevance evaluator, A.3).
The paper's judge is Gemini-2.5-Pro; ours is switchable (--judge hf for the
local Gemma, --judge gemini for the paper-exact model when credits allow).

Three levels:
  1. instance-level human Precision@k -- for each systematic slice, does the
     human annotation of each top-k member image report a failure of that class
     consistent with the slice's claim?
  2. pattern extraction -- the judge LLM reads all annotations and lists the
     recurring human-identified failure patterns;
  3. semantic matching (A.3 prompt, verbatim) -- which extracted human pattern
     does each discovered slice correspond to; unmatched patterns = recall gaps.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slicelens.prompts import SEMANTIC_RELEVANCE_PROMPT  # noqa: E402

# Expected Fig.13 failure mode implied by each slice condition.
FIG13_MODE = {
    "an object that blends into its background":
        "a missed region or badly wrong boundary (low contrast with background)",
    "an object cut off at the image border":
        "an incomplete mask for an object truncated by the image border",
    "a small object":
        "a missed / poorly segmented small region",
    "an object occluded by another object":
        "an incomplete mask caused by occlusion",
    "an object with thin or elongated parts":
        "lost thin/elongated structures (legs, wings, poles): detail loss / boundary error",
    "several overlapping objects of the same type":
        "masks of adjacent same-class objects merged or split (under/over-segmentation)",
    "an object that looks similar to a different object category":
        "the object segmented as a different, visually similar class (misclassification)",
}

JUDGE_SYSTEM = (
    "You are a careful evaluator of computer-vision error analyses. "
    "Answer strictly with a single word: yes or no."
)

INSTANCE_JUDGE_PROMPT = """\
A human expert wrote this annotation describing the segmentation errors in an \
image (class names may appear in uppercase or as synonyms, e.g. MONITOR for \
tvmonitor):

"{annotation}"

An automatic system claims that the {label} region in this image belongs to \
the error slice "{slice_query}", i.e. it expects the segmentation of the \
{label} to show: {mode}.

Does the human annotation report a segmentation problem for the {label} class \
consistent with this claim? If the annotation does not mention such a problem \
for this class, answer no. Answer with a single word: yes or no."""

PATTERN_EXTRACT_PROMPT = """\
Below are {n} human annotations describing the segmentation errors of a \
semantic-segmentation model on Pascal-VOC images, one annotation per line.

{annotations}

List the recurring failure patterns across these annotations as a numbered \
list of 5 to 8 items. Each item must be ONE short line: a title plus a brief \
clarification. Respond with ONLY the numbered list."""

# --- Option 1: lift each slice onto the error-TYPE axis --------------------
# A SliceLens slice names a *visual condition* ("a small object"); the human
# patterns name an *error type* ("Under-segmentation"). To compare like with
# like, tag the human annotation of each FAILING member of a slice with the
# patterns it reports -> a slice x pattern matrix grounded in real instances.
ERRTYPE_TAG_PROMPT = """\
A human expert wrote this annotation describing the segmentation errors in an \
image (class names may appear in uppercase or as synonyms, e.g. MONITOR for \
tvmonitor):

"{annotation}"

Below is a numbered list of error-type categories:
{pattern_list}

Which of these categories does the annotation report for the {label} class? \
Consider ONLY problems the annotation describes for the {label}.
End your reply with a single line beginning "ANSWER:" followed by the matching \
category numbers (comma-separated), or "none"."""

# --- Option 2: many-to-many, BIDIRECTIONAL semantic matching ----------------
# One condition can surface as several error types, so let the judge return a
# SET (forward: slice -> patterns) and also ask the reverse (backward: which
# conditions could produce each human pattern). Coverage = union of both.
SEMANTIC_MULTI_PROMPT = """\
I have a description of an error pattern from a computer vision model, and a \
numbered list of ground-truth error patterns. A single condition can manifest \
as several error types. List ALL ground-truth patterns that are semantically \
related to the algorithm output.
End your reply with a single line beginning "ANSWER:" followed by the matching \
numbers (comma-separated), or "-1".

Algorithm output: {algorithm_output}

Ground truth patterns:
{ground_truth_list}"""

SEMANTIC_BACKWARD_PROMPT = """\
I have one human-identified error pattern of a segmentation model, and a \
numbered list of candidate failure CONDITIONS discovered by an automatic \
system. Which of these conditions would plausibly PRODUCE or EXPLAIN this error \
pattern? A pattern may be explained by several conditions, or none.
End your reply with a single line beginning "ANSWER:" followed by the matching \
condition numbers (comma-separated), or "-1".

Human error pattern: {pattern}

Candidate conditions:
{slice_list}"""


def chat_resilient(llm, system: str, user: str, tries: int = 5, base: float = 2.0):
    """llm.chat with retry + exponential backoff. A long run is dozens of
    sequential API calls; a single transient drop (RemoteProtocolError, 5xx,
    timeout) should not throw the whole evaluation away."""
    for i in range(tries):
        try:
            return llm.chat(system=system, user=user)
        except Exception as exc:  # noqa: BLE001 - judge backends raise varied types
            if i == tries - 1:
                raise
            wait = base * (2 ** i)
            print(f"    [judge retry {i + 1}/{tries - 1}: {type(exc).__name__}] "
                  f"waiting {wait:.0f}s", flush=True)
            time.sleep(wait)


def parse_yes(text: str) -> bool:
    t = text.strip().lower()
    return t.startswith("yes") or t.split()[:1] == ["yes"] if t else False


def parse_ints(text: str, lo: int, hi: int) -> list[int]:
    """De-duplicated, order-preserving ints in [lo, hi] mentioned in `text`."""
    seen: dict[int, None] = {}
    for tok in re.findall(r"-?\d+", text or ""):
        n = int(tok)
        if lo <= n <= hi and n not in seen:
            seen[n] = None
    return list(seen)


def parse_label_ints(text: str, lo: int, hi: int) -> list[int]:
    """Robust answer parse: read ints from an 'ANSWER:'/'FINAL:' line if present,
    else from the last non-empty line, else from the whole reply. This avoids the
    classic failure of grabbing a number that appears in the model's reasoning."""
    if not text:
        return []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in reversed(lines):
        m = re.search(r"(?:ANSWER|FINAL|RISPOSTA)\s*[:=]\s*(.+)", ln, re.IGNORECASE)
        if m:
            got = parse_ints(m.group(1), lo, hi)
            if got:
                return got
    if lines:                                   # last line only (skip CoT numbers)
        got = parse_ints(lines[-1], lo, hi)
        if got:
            return got
    return parse_ints(text, lo, hi)             # last resort: scan everything


def pattern_title(p: str) -> str:
    """'3. Under-segmentation: failure to ...' -> 'Under-segmentation'."""
    m = re.match(r"^\s*\d+[.)]\s*([^:]+)", p)
    return (m.group(1) if m else p).strip()


def load_annotations(root: Path):
    matches = json.loads((root / "voc_patch_matches.json").read_text(encoding="utf-8"))
    stem2idx = {Path(v).stem: int(k) for k, v in matches.items()}
    anns = {}
    for line in (root / "answer_gts.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            anns[d["img_idx"]] = d["content"].strip()
    return stem2idx, anns


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--judge", default="hf", choices=["hf", "gemini"])
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--patterns-from", default=None,
                    help="reuse human_patterns from a previous eval JSON so runs "
                         "stay comparable (skips level-2 re-extraction)")
    ap.add_argument("--levels", default="1,2,3,4,5,6",
                    help="comma-separated levels to run, e.g. '4,5,6'")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    levels = {int(x) for x in args.levels.split(",") if x.strip()}

    if args.judge == "hf":
        from slicelens.backends import HFLLM
        llm = HFLLM(model=args.judge_model or "/models/gemma",
                    load_in_4bit=args.load_in_4bit, max_new_tokens=512)
    else:
        from slicelens.backends import GeminiLLM
        llm = GeminiLLM(model=args.judge_model or "gemini-2.5-pro")

    d = json.loads(Path(args.report).read_text(encoding="utf-8"))
    sysset = list(d["systematic_error_slices"])
    cand = {c["query"]: c for c in d["candidates"]}
    stem2idx, anns = load_annotations(Path(args.dataset_root))
    out = {"judge": f"{args.judge}:{args.judge_model}", "slices": [], "k": args.k}

    def annotation_for(member) -> str:
        return anns.get(stem2idx.get(member["name"]), "")

    # ---- level 1: instance-level human Precision@k -----------------------
    if 1 in levels:
        print(f"\n=== Level 1: human-validated Precision@k (judge: {args.judge}) ===")
        for q in sysset:
            mode = FIG13_MODE.get(q, q)
            rows = []
            for t in cand[q]["top_instances"][: args.k]:
                ann = annotation_for(t)
                verdict = False
                if ann:
                    resp = chat_resilient(llm, system=JUDGE_SYSTEM, user=INSTANCE_JUDGE_PROMPT.format(
                        annotation=ann, label=t["label"], slice_query=q, mode=mode))
                    verdict = parse_yes(resp)
                rows.append({"name": t["name"], "label": t["label"],
                             "iou": t["iou"], "is_error": t["is_error"],
                             "conf": round(t["slice_confidence"], 3),
                             "human_match": verdict})
            p5 = sum(r["human_match"] for r in rows[:5]) / max(min(5, len(rows)), 1)
            pk = sum(r["human_match"] for r in rows) / max(len(rows), 1)
            out["slices"].append({"query": q, "mode": mode,
                                  "human_P@5": round(p5, 3),
                                  f"human_P@{args.k}": round(pk, 3), "members": rows})
            print(f"\n  {q}")
            print(f"    human P@5 = {p5:.2f}   human P@{args.k} = {pk:.2f}")
            for r in rows:
                mark = "OK " if r["human_match"] else " - "
                print(f"    [{mark}] {r['name']:14s} {r['label']:11s} "
                      f"iou={r['iou'] if r['iou'] is not None else -1:.2f} "
                      f"err={int(r['is_error'])} conf={r['conf']:.2f}")

    # ---- level 2: extract (or reuse) human failure patterns --------------
    patterns: list[str] = []
    if args.patterns_from:
        src = json.loads(Path(args.patterns_from).read_text(encoding="utf-8"))
        patterns = list(src.get("human_patterns", []))
        print(f"\n=== Level 2: reusing {len(patterns)} human patterns from "
              f"{args.patterns_from} ===")
        for ln in patterns:
            print("  " + ln)
    elif 2 in levels:
        print("\n=== Level 2: recurring human failure patterns ===")
        ann_lines = [anns[i] for i in sorted(anns)]
        resp = chat_resilient(llm, system="", user=PATTERN_EXTRACT_PROMPT.format(
            n=len(ann_lines), annotations="\n".join(ann_lines)))
        patterns = [ln.strip() for ln in resp.splitlines()
                    if re.match(r"^\s*\d+[\.\)]", ln)]
        for ln in patterns:
            print("  " + ln)
    out["human_patterns"] = patterns

    if (levels & {3, 4, 5, 6}) and not patterns:
        raise SystemExit("Levels 3/4/5/6 need human patterns: run level 2 or pass "
                         "--patterns-from.")
    titles = [pattern_title(p) for p in patterns]
    gt_list = "\n".join(patterns)

    # ---- level 3: A.3 semantic matching (single best, paper-verbatim) -----
    if 3 in levels:
        print("\n=== Level 3: semantic match (paper A.3 prompt, verbatim, 1-to-1) ===")
        matched_nums: set[int] = set()
        out["matching"] = []
        for q in sysset:
            algo = f"{q} (expected segmentation failure: {FIG13_MODE.get(q, q)})"
            resp = chat_resilient(llm, system="", user=SEMANTIC_RELEVANCE_PROMPT.format(
                algorithm_output=algo, ground_truth_list=gt_list))
            got = parse_label_ints(resp, 1, len(patterns))
            num = got[0] if got else -1
            if num > 0:
                matched_nums.add(num)
            target = patterns[num - 1] if 0 < num <= len(patterns) else "(none)"
            out["matching"].append({"query": q, "matched_pattern": num,
                                    "pattern_text": target})
            print(f"  {q[:46]:46s} -> {num:3d}  {target[:60]}")
        out["unmatched_human_patterns"] = [
            p for i, p in enumerate(patterns, 1) if i not in matched_nums]
        out["coverage_1to1"] = {"covered": sorted(matched_nums),
                                "n_covered": len(matched_nums),
                                "n_patterns": len(patterns)}
        print(f"\n  1-to-1 coverage: {len(matched_nums)}/{len(patterns)} patterns "
              f"-> {sorted(matched_nums)}")

    # ---- level 4 (Option 1): error-TYPE cross-tab (like-with-like) --------
    if 4 in levels:
        print("\n=== Level 4: error-type cross-tab "
              "(each slice's failing members -> human patterns) ===")
        pat_list = "\n".join(patterns)
        tag_cache: dict[tuple, list[int]] = {}
        crosstab = []
        grounded_cover: set[int] = set()
        for q in sysset:
            counts = [0] * len(patterns)
            members = [t for t in cand[q]["top_instances"][: args.k]
                       if t.get("is_error")]
            n_tagged = 0
            for t in members:
                ann = annotation_for(t)
                if not ann:
                    continue
                key = (t["name"], t["label"])
                if key not in tag_cache:
                    resp = chat_resilient(llm, system="", user=ERRTYPE_TAG_PROMPT.format(
                        annotation=ann, label=t["label"], pattern_list=pat_list))
                    tag_cache[key] = parse_label_ints(resp, 1, len(patterns))
                tags = tag_cache[key]
                n_tagged += 1 if tags else 0
                for n in tags:
                    counts[n - 1] += 1
                    grounded_cover.add(n)
            crosstab.append({"query": q, "n_error_members": len(members),
                             "n_tagged": n_tagged,
                             "pattern_counts": {i + 1: c for i, c in enumerate(counts)}})
            dom = max(range(len(patterns)), key=lambda i: counts[i]) if any(counts) else -1
            print(f"\n  {q}  ({len(members)} error members, {n_tagged} tagged)")
            for i, c in enumerate(counts):
                if c:
                    star = "  <== dominant" if i == dom else ""
                    print(f"      {c:2d}x  [{i + 1}] {titles[i]}{star}")
            if not any(counts):
                print("      (no pattern tagged)")
        out["error_type_crosstab"] = crosstab
        out["coverage_grounded"] = {"covered": sorted(grounded_cover),
                                    "n_covered": len(grounded_cover),
                                    "n_patterns": len(patterns)}
        print(f"\n  Grounded coverage (patterns exhibited by some slice's members): "
              f"{len(grounded_cover)}/{len(patterns)}  -> {sorted(grounded_cover)}")

    # ---- level 5 (Option 2): bidirectional many-to-many semantic match ----
    if 5 in levels:
        print("\n=== Level 5: bidirectional many-to-many semantic match ===")
        fwd: dict[str, set[int]] = {q: set() for q in sysset}
        bwd: dict[str, set[int]] = {q: set() for q in sysset}
        print("\n  forward (slice -> patterns):")
        for q in sysset:
            algo = f"{q} (expected segmentation failure: {FIG13_MODE.get(q, q)})"
            resp = chat_resilient(llm, system="", user=SEMANTIC_MULTI_PROMPT.format(
                algorithm_output=algo, ground_truth_list=gt_list))
            nums = parse_label_ints(resp, 1, len(patterns))
            fwd[q] = set(nums)
            names = ", ".join(titles[n - 1] for n in nums) or "(none)"
            print(f"    {q[:44]:44s} -> {nums}  {names}")
        slice_list = "\n".join(f"{i}. {q}" for i, q in enumerate(sysset, 1))
        print("\n  backward (pattern -> slices):")
        for pi, p in enumerate(patterns, 1):
            resp = chat_resilient(llm, system="", user=SEMANTIC_BACKWARD_PROMPT.format(
                pattern=p, slice_list=slice_list))
            nums = parse_label_ints(resp, 1, len(sysset))
            for sidx in nums:
                bwd[sysset[sidx - 1]].add(pi)
            chosen = ", ".join(sysset[n - 1][:28] for n in nums) or "(none)"
            print(f"    [{pi}] {titles[pi - 1][:24]:24s} <- {chosen}")
        union_cover: set[int] = set()
        matrix = []
        for q in sysset:
            pats = sorted(fwd[q] | bwd[q])
            union_cover |= set(pats)
            matrix.append({"query": q, "forward": sorted(fwd[q]),
                           "backward": sorted(bwd[q]), "union": pats})
        out["bidirectional_matching"] = matrix
        out["coverage_bidirectional"] = {"covered": sorted(union_cover),
                                         "n_covered": len(union_cover),
                                         "n_patterns": len(patterns)}
        print(f"\n  Bidirectional union coverage: {len(union_cover)}/{len(patterns)}"
              f"  -> {sorted(union_cover)}")
        for i in range(1, len(patterns) + 1):
            if i not in union_cover:
                print("    STILL MISSING  " + patterns[i - 1])

    # ---- level 6: fused, calibrated decision matrix (#1 parsing + #3 fusion) --
    # Combine the three signals per (slice, pattern) cell -- grounded instances
    # (Lv4), semantic forward and semantic backward (Lv5) -- but WEIGHTED, since
    # the backward "could this condition cause this error?" is permissive and
    # alone inflates coverage to 7/7. Tiers: 3 = grounded AND semantic-forward
    # (robust), 2 = exactly one strong signal, 1 = backward-only (weak).
    if 6 in levels:
        print("\n=== Level 6: fused decision matrix (calibrated; replaces the union) ===")
        ct = {c["query"]: c for c in out.get("error_type_crosstab", [])}
        bm = {m["query"]: m for m in out.get("bidirectional_matching", [])}
        if not ct or not bm:
            raise SystemExit("Level 6 needs levels 4 and 5 (run e.g. --levels 4,5,6).")
        npat = len(patterns)
        short = [pattern_title(p) for p in patterns]

        def _count(counts, p):
            return counts.get(p, counts.get(str(p), 0)) or 0

        robust: set[int] = set()      # grounded AND semantic-forward
        supported: set[int] = set()   # >= 1 strong signal (grounded or forward)
        weak: set[int] = set()        # only the permissive backward
        fused = []
        print("\n  tiers: 3 = grounded & semantic | 2 = one strong signal | "
              "1 = backward-only | . = none")
        print("    " + "slice \\ pattern".ljust(26)
              + " ".join(f"{i + 1:>2d}" for i in range(npat)))
        for q in sysset:
            counts = ct[q]["pattern_counts"]
            fset, bset = set(bm[q]["forward"]), set(bm[q]["backward"])
            cells, row = {}, []
            for p in range(1, npat + 1):
                g, f, bk = _count(counts, p) >= 1, p in fset, p in bset
                if g and f:
                    t = 3; robust.add(p); supported.add(p)
                elif g or f:
                    t = 2; supported.add(p)
                elif bk:
                    t = 1; weak.add(p)
                else:
                    t = 0
                cells[p] = {"grounded": g, "forward": f, "backward": bk, "tier": t}
                row.append(f"{t if t else '.':>2}")
            fused.append({"query": q, "cells": cells})
            print(f"    {q[:26]:26s}" + " ".join(row))

        weak_only = sorted(weak - supported)
        out["fused_matrix"] = fused
        out["coverage_fused"] = {"robust_core": sorted(robust),
                                 "supported": sorted(supported),
                                 "weak_only": weak_only, "n_patterns": npat}
        print(f"\n  Robust core (grounded AND semantic): {len(robust)}/{npat} -> "
              f"{sorted(robust)}  {[short[p - 1] for p in sorted(robust)]}")
        print(f"  Defensible (>= 1 strong signal)    : {len(supported)}/{npat} -> "
              f"{sorted(supported)}")
        print(f"  Weak-only (backward, discard)      : {weak_only}  "
              f"{[short[p - 1] for p in weak_only]}")
        naive = out.get("coverage_bidirectional", {}).get("n_covered")
        if naive is not None:
            print(f"\n  vs naive union: {naive}/{npat} (inflated)  ->  calibrated: "
                  f"defensible {len(supported)}/{npat}, robust {len(robust)}/{npat}")

    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                  encoding="utf-8")
        print(f"\nWrote evaluation -> {args.out}")


if __name__ == "__main__":
    main()
