"""Agreement between the pipeline's descriptions and hand-written error annotations,
scored by an LLM judge that is CALIBRATED AGAINST CHANCE.

Why the calibration is not optional: the first version of this comparison (2026-07-14)
mapped annotation indices to the wrong images for 57% of the pairs -- i.e. it compared
descriptions of one image with notes about a completely different one -- and still
produced "70% partial agreement". A grading scale that agreeable on unrelated content
measures politeness, not agreement. So every genuine pair is accompanied by a MISMATCHED
pair (the same annotation against another image's description), shuffled together and
indistinguishable to the judge. Whatever the judge scores on the mismatched half is the
noise floor of the instrument; only the gap above it is evidence.

Three layers, cheapest first:
  1. DETERMINISTIC -- object classes and error natures are extracted from both texts by
     rule (VOC class names are upper-cased in the annotations; the pipeline's own
     wording is fixed), giving class precision/recall and nature overlap with no model
     in the loop at all.
  2. JUDGE -- the human note is split into atomic claims and each is marked
     COVERED / PARTIAL / MISSING by the description; symmetrically each pipeline pattern
     is marked SUPPORTED / UNSUPPORTED. Claim recall and pattern precision, not a vote.
  3. VALIDATION OF THE JUDGE ITSELF -- genuine vs mismatched separation, a position-bias
     probe (pass 2 presents the two reports in the opposite order), and Cohen's kappa
     against the human-written verdicts in gt_comparison.md. NOTE on the first run
     (2026-07-29): both passes used the same prompt order, and since HFLLM decodes
     greedily the two passes were identical by construction -- the 1.00 "self-consistency"
     it reported measured nothing. Fixed here; re-run to obtain a real figure.

Judge = Gemma (local), deliberately NOT the Qwen that produced the groundings and tags.

Run (GPU):
  PYTHONPATH=/workspace python slice_quality/examples/eval_gt_agreement.py
Env: JUDGE_MODEL (default /models/gemma), JUDGE_SEED, JUDGE_PASSES (default 2).
"""
import json
import os
import random
import re
from collections import Counter

OUT_JSON = "outputs/slice_quality/gt_agreement.json"
OUT_TXT = "outputs/slice_quality/gt_agreement.txt"
ANN = "/data/dataset/answer_gts.jsonl"
MAP = "/data/dataset/voc_patch_matches.json"
VAL = "/data/voc2012/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt"
DESC = "outputs/slice_quality/image_descriptions.json"
CMP = "outputs/slice_quality/gt_comparison.md"

SEED = int(os.environ.get("JUDGE_SEED", "0"))
PASSES = int(os.environ.get("JUDGE_PASSES", "2"))
MODEL = os.environ.get("JUDGE_MODEL", "/models/gemma")

CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
           "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
           "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
ALIAS = {"bike": "bicycle", "airplane": "aeroplane", "tv": "tvmonitor",
         "plant": "pottedplant", "table": "diningtable", "motorcycle": "motorbike"}

# error nature, in the two vocabularies
NATURE_HUMAN = {
    "fn": ["missed", "not captured", "omitted", "not segmented", "not classified",
           "not predicted", "absent", "completely missing", "left incomplete",
           "under-segmented", "incomplete"],
    "fp": ["hallucinated", "introduced", "over-segmented", "over-extended",
           "over-estimated", "spurious", "not present in the ground"],
    "confusion": ["misclassified", "mispredicted", "instead of", "classified as",
                  "segmented as", "predicted as"],
    "boundary": ["boundaries", "borders", "coarser", "coarse", "imprecise",
                 "irregular", "erratic", "less defined", "not well defined"],
}
NATURE_PIPE = {
    "fn": ["not detected"],
    "fp": ["non-existent"],
    "confusion": ["wrong class"],
    "boundary": ["imprecise boundaries"],
    "extent": ["wrong extent"],
    "merge": ["under-segmented (several"],
    "split": ["split"],
}

JUDGE_SYSTEM = (
    "You compare two reports about mistakes a segmentation model made on ONE image. "
    "Report A was written by a person looking at the image. Report B was produced "
    "automatically. Some pairs in this batch are about the SAME image and some are "
    "about DIFFERENT images; you are not told which, and you must judge only from the "
    "content. Be strict: two reports agree on a point only if they refer to the same "
    "object category AND the same kind of mistake. Different wording for the same "
    "mistake counts as agreement; a mistake mentioned by only one of them does not."
)

JUDGE_USER = """REPORT A (person):
{human}

REPORT B (automatic):
{pipe}

Do this:
1. Split REPORT A into atomic claims. Each claim = one object category + one kind of
   mistake (missed / hallucinated / wrong class / imprecise boundaries / wrong extent).
2. For each claim of A, decide whether REPORT B states it: COVERED, PARTIAL (right
   object but different kind of mistake, or right mistake on a different object), or
   MISSING.
3. For each distinct pattern in REPORT B, decide whether REPORT A supports it:
   SUPPORTED or UNSUPPORTED.
4. Finally judge whether the two reports plausibly describe THE SAME IMAGE: SAME or
   DIFFERENT.

Answer with JSON only, no commentary:
{{"claims": [{{"claim": "...", "verdict": "COVERED|PARTIAL|MISSING"}}],
  "patterns": [{{"pattern": "...", "verdict": "SUPPORTED|UNSUPPORTED"}}],
  "same_image": "SAME|DIFFERENT"}}"""


# Same task, reports presented in the opposite order (position-bias probe).
JUDGE_USER_SWAPPED = """REPORT B (automatic):
{pipe}

REPORT A (person):
{human}

""" + JUDGE_USER.split("Do this:", 1)[1].join(["Do this:", ""])


def human_classes(text):
    ws = {ALIAS.get(w.lower(), w.lower()) for w in re.findall(r"\b[A-Z]{3,}\b", text)}
    return {c for c in ws if c in CLASSES}


def pipe_classes(text):
    out = set()
    out |= {m for m in re.findall(r"of class '([a-z]+)'", text)}
    out |= {m for m in re.findall(r"^\s*-\s+([a-z]+) —", text, flags=re.M)}
    out |= {m for m in re.findall(r'"(?:a|an) ([a-z]+) ', text)}
    return {c for c in out if c in CLASSES}


def natures(text, table):
    low = text.lower()
    return {k for k, kws in table.items() if any(w in low for w in kws)}


def deterministic(human, pipe):
    hc, pc = human_classes(human), pipe_classes(pipe)
    hn, pn = natures(human, NATURE_HUMAN), natures(pipe, NATURE_PIPE)
    inter = hc & pc
    return {
        "human_classes": sorted(hc), "pipe_classes": sorted(pc),
        "class_recall": round(len(inter) / len(hc), 3) if hc else None,
        "class_precision": round(len(inter) / len(pc), 3) if pc else None,
        "human_natures": sorted(hn), "pipe_natures": sorted(pn),
        "nature_overlap": round(len(hn & pn) / len(hn), 3) if hn else None,
    }


def parse_json(raw):
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def load_reference_verdicts():
    """The hand-written ALTA/PARZIALE/BASSA verdicts, for a kappa against the judge."""
    if not os.path.exists(CMP):
        return {}
    out = {}
    for line in open(CMP, encoding="utf-8"):
        m = re.match(r"^## (\S+) - \*\*(ALTA|PARZIALE|BASSA)\*\*", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def kappa(a, b):
    """Cohen's kappa on two aligned label lists."""
    labels = sorted(set(a) | set(b))
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[l] / n) * (cb[l] / n) for l in labels)
    return round((po - pe) / (1 - pe), 3) if pe < 1 else None


def main():
    rng = random.Random(SEED)
    mapping = {int(k): v.replace(".jpg", "")
               for k, v in json.load(open(MAP, encoding="utf-8")).items()}
    val = set(open(VAL, encoding="utf-8").read().split())
    ann = {}
    for ln in open(ANN, encoding="utf-8"):
        o = json.loads(ln)
        ann[o["img_idx"]] = o["content"].strip()
    desc = json.load(open(DESC, encoding="utf-8"))

    pairs = []
    names = [mapping[i] for i in sorted(ann) if mapping[i] in val]
    for i in sorted(ann):
        n = mapping[i]
        if n not in val:
            continue
        pipe = (desc[n]["text"] if n in desc
                else "No errors were detected in this image.")
        pairs.append({"kind": "genuine", "image": n, "human": ann[i], "pipe": pipe})
    # mismatched controls: same annotations, another image's description (derangement)
    shifted = names[1:] + names[:1]
    for p, other in zip(list(pairs), shifted):
        pipe = (desc[other]["text"] if other in desc
                else "No errors were detected in this image.")
        pairs.append({"kind": "control", "image": p["image"], "control_from": other,
                      "human": p["human"], "pipe": pipe})
    rng.shuffle(pairs)
    print(f"pairs: {len(pairs)} ({sum(1 for p in pairs if p['kind']=='genuine')} genuine "
          f"+ {sum(1 for p in pairs if p['kind']=='control')} mismatched controls)",
          flush=True)

    for p in pairs:                      # layer 1, no model involved
        p["deterministic"] = deterministic(p["human"], p["pipe"])

    if os.environ.get("JUDGE_DRYRUN"):
        for p in pairs[:6]:
            print(p["kind"], p["image"], p["deterministic"], flush=True)
        return

    from slicelens.backends.hf_llm import HFLLM
    print(f"loading judge {MODEL} ...", flush=True)
    judge = HFLLM(model=MODEL, max_new_tokens=768,
                  load_in_4bit=os.environ.get("JUDGE_4BIT") == "1")

    # HFLLM decodes greedily (do_sample=False), so re-asking the SAME prompt returns
    # the SAME answer and a "self-consistency" over identical prompts measures nothing
    # (the first run reported 1.00 for exactly that reason). Pass 2 therefore swaps the
    # presentation order of the two reports: same content, different position, which is
    # a real robustness check on format/position bias.
    for k in range(PASSES):
        for i, p in enumerate(pairs, 1):
            tmpl = JUDGE_USER if k % 2 == 0 else JUDGE_USER_SWAPPED
            raw = judge.chat(JUDGE_SYSTEM,
                             tmpl.format(human=p["human"], pipe=p["pipe"]))
            p.setdefault("judge", []).append(parse_json(raw) or {"error": raw[:200]})
            if i % 10 == 0:
                print(f"  pass {k+1}/{PASSES}: {i}/{len(pairs)}", flush=True)

    def summarise(p, k=0):
        j = p["judge"][k]
        cl = j.get("claims") or []
        pt = j.get("patterns") or []
        cov = sum(1 for c in cl if c.get("verdict") == "COVERED")
        par = sum(1 for c in cl if c.get("verdict") == "PARTIAL")
        sup = sum(1 for x in pt if x.get("verdict") == "SUPPORTED")
        return {
            "n_claims": len(cl), "covered": cov, "partial": par,
            "claim_recall": round((cov + 0.5 * par) / len(cl), 3) if cl else None,
            "n_patterns": len(pt), "supported": sup,
            "pattern_precision": round(sup / len(pt), 3) if pt else None,
            "same_image": j.get("same_image"),
        }

    for p in pairs:
        p["scores"] = [summarise(p, k) for k in range(PASSES)]

    def agg(kind, key, k=0):
        vals = [p["scores"][k][key] for p in pairs
                if p["kind"] == kind and p["scores"][k][key] is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    same_rate = {kd: round(sum(1 for p in pairs if p["kind"] == kd
                              and p["scores"][0]["same_image"] == "SAME")
                           / max(1, sum(1 for p in pairs if p["kind"] == kd)), 3)
                 for kd in ("genuine", "control")}
    # self-consistency of the judge across passes
    consist = None
    if PASSES > 1:
        a = [p["scores"][0]["same_image"] for p in pairs]
        b = [p["scores"][1]["same_image"] for p in pairs]
        consist = {"same_image_agreement": round(sum(1 for x, y in zip(a, b)
                                                     if x == y) / len(a), 3),
                   "same_image_kappa": kappa(a, b)}
    # kappa against the hand-written verdicts (genuine pairs only)
    ref = load_reference_verdicts()
    kap = None
    if ref:
        def bucket(s):
            r = s["claim_recall"]
            if r is None:
                return "BASSA"
            return "ALTA" if r >= 0.75 else ("PARZIALE" if r >= 0.25 else "BASSA")
        gen = [p for p in pairs if p["kind"] == "genuine" and p["image"] in ref]
        if gen:
            kap = {"n": len(gen),
                   "kappa_vs_human_written": kappa([bucket(p["scores"][0]) for p in gen],
                                                   [ref[p["image"]] for p in gen]),
                   "judge_distribution": dict(Counter(bucket(p["scores"][0])
                                                      for p in gen)),
                   "reference_distribution": dict(Counter(ref[p["image"]]
                                                          for p in gen))}

    det = {kd: {k: round(sum(p["deterministic"][k] for p in pairs
                             if p["kind"] == kd and p["deterministic"][k] is not None)
                         / max(1, sum(1 for p in pairs if p["kind"] == kd
                                      and p["deterministic"][k] is not None)), 3)
                for k in ("class_recall", "class_precision", "nature_overlap")}
           for kd in ("genuine", "control")}

    summary = {
        "judge_model": MODEL, "seed": SEED, "passes": PASSES,
        "n_pairs": len(pairs),
        "deterministic": det,
        "judge": {kd: {"claim_recall": agg(kd, "claim_recall"),
                       "pattern_precision": agg(kd, "pattern_precision")}
                  for kd in ("genuine", "control")},
        "verdict_same_image": same_rate,
        "self_consistency": consist,
        "vs_human_written": kap,
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "pairs": pairs}, f, indent=2, ensure_ascii=False)
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("GT AGREEMENT, chance-calibrated (genuine vs mismatched pairs)\n\n")
        f.write(json.dumps(summary, indent=2, ensure_ascii=False))
        f.write("\n\nper genuine pair\n")
        for p in sorted((x for x in pairs if x["kind"] == "genuine"),
                        key=lambda x: -(x["scores"][0]["claim_recall"] or 0)):
            s = p["scores"][0]
            f.write(f"  {p['image']}  recall={s['claim_recall']} "
                    f"precision={s['pattern_precision']} same={s['same_image']}\n")
    print("SUMMARY", json.dumps(summary, indent=2), flush=True)
    print("GT_AGREEMENT_DONE", flush=True)


if __name__ == "__main__":
    main()
