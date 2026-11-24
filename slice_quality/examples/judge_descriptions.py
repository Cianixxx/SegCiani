"""Score the pipeline's descriptions against the hand-written ones with a local Gemma.

Rubric (v2): 0-10 = Completeness 4 + Correctness 4 + Specificity 2, weighted towards
what the descriptions are FOR -- saying true things about this image, and saying all
of them. It replaces the five-criterion rubric of PASCAL_BestVLMModel, and the change
is driven by that rubric's own measurements on the first run:

  * Fluency scored 1.00/1 on CONTROL pairs -- a description of a different image is
    just as fluent -- so the criterion was a free point for everyone and measured
    nothing about agreement. Dropped.
  * Conciseness scored 0.67/1 on controls against 0.88 genuine, so it was mostly a
    free point too. Its one useful clause (do not invent discrepancies) belongs to
    Correctness and has been folded into it.
  * The judge was penalising the candidate for stating IoU values and hole counts
    absent from the human note, calling them hallucinations. They are deterministic
    mask facts, verifiably true. The prompt now says extra verifiable detail counts
    against the candidate only when it CONTRADICTS the reference.

Scores from this rubric are NOT comparable with the previous run's numbers; the
older results are kept beside them under their own file name.

Two generated texts are scored per image, because the pipeline now has two and they
answer different questions:
    narrative   error_narratives.json  -- every typed error, annotator register
    pattern     image_descriptions.json / train36 -- only errors inside a certified slice
The pattern text is expected to lose on Completeness by construction; that is a fact
about its purpose, not a defect, and it is reported as such.

CHANCE CALIBRATION IS NOT OPTIONAL. Every genuine pair is accompanied by a control
pair: the same human annotation against ANOTHER image's description, seeded and
deranged so nothing is ever paired with itself. A judge that scores unrelated text
highly is measuring politeness, not agreement -- this project has already been burnt
by exactly that (see eval_gt_agreement.py's note on the 2026-07-14 run). Only the gap
between genuine and control counts as evidence.

Not reported: a "self-consistency" figure. HFLLM decodes greedily, so re-asking the
same prompt returns the same answer by construction and any such number would be 1.00
for trivial reasons.

Run (GPU 1 booked):
  JUDGE_4BIT=1 PYTHONPATH=/workspace python slice_quality/examples/judge_descriptions.py
Env: JUDGE_MODEL (default /models/gemma), JUDGE_4BIT, SEED, LIMIT (0 = all), OUT.
"""
import hashlib
import json
import os
import random
import re
import time
from collections import Counter

OUT = os.environ.get("OUT", "outputs/slice_quality/judge_descriptions_v2")
ANN = "/data/dataset/answer_gts.jsonl"
MAP = "/data/dataset/voc_patch_matches.json"
VAL = "/data/voc2012/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt"
# inputs by env, so a retyped run (outputs/slice_quality_cov05) is judged on
# its own texts; DESC_TR="" skips the train36 patterns (val-only comparison)
NARR = os.environ.get("NARR", "outputs/slice_quality/error_narratives.json")
DESC = os.environ.get("DESC", "outputs/slice_quality/image_descriptions.json")
DESC_TR = os.environ.get("DESC_TR", "outputs/slice_quality/train36/descriptions_train36_full.json")
CMP = "outputs/slice_quality/gt_comparison.md"

MODEL = os.environ.get("JUDGE_MODEL", "/models/gemma")
SEED = int(os.environ.get("SEED", "0"))
LIMIT = int(os.environ.get("LIMIT", "0"))

JUDGE_SYSTEM = """\
You are an expert evaluator of natural-language descriptions of segmentation errors.

You will be given:
1. GROUND TRUTH DESCRIPTION: a reference expert description of how a predicted \
segmentation mask differs from the ground truth mask.
2. CANDIDATE DESCRIPTION: a description generated automatically.

Judge whether the CANDIDATE says the SAME THINGS about the SAME image. Only the \
substance counts: what is asserted, and whether all of it is there.

- Completeness (0-4): does it report every discrepancy the reference reports? \
Take off a point for each one it leaves out.
- Correctness  (0-4): is everything it asserts true of this image -- the object \
classes, where each error is, and the NATURE of each error (missed / hallucinated / \
wrong class / imprecise boundary / wrong extent)? Take off points for each claim \
that contradicts the reference, and for any error it makes up.
- Specificity  (0-2): does it locate and name things precisely, rather than vaguely?

Extra detail the reference simply does not mention -- IoU values, counts of holes, \
exact positions -- is NOT an invention and must NOT be penalised: it counts against \
the candidate only if it CONTRADICTS the reference. Different wording for the same \
fact is fine. Do not reward or punish style, length or grammar. If the candidate \
describes a different image altogether, score 0.

Respond ONLY with valid JSON, keeping the rationale under 40 words:
{"completeness": <0-4>, "correctness": <0-4>, "specificity": <0-2>, \
"score": <float 0-10, one decimal>, "rationale": "<max 40 words>"}"""

JUDGE_USER = """GROUND TRUTH DESCRIPTION:
{gt}

CANDIDATE DESCRIPTION:
{cand}"""

_FIELDS = [("completeness", 4), ("correctness", 4), ("specificity", 2)]


def _hash(text):
    """Short fingerprint of a reference text, stored with the judgement."""
    return hashlib.sha1(" ".join(text.split()).encode("utf-8")).hexdigest()[:12]


def parse_json(raw):
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def clean(rec):
    """Keep BOTH totals: the judge's holistic score and the sum of its components.

    They disagree, and not randomly: on the smoke run the holistic score was on
    average 2 points BELOW the sum of the five sub-scores and never above it. The
    holistic one also separates genuine from control pairs far better, and the
    reason is visible in the components -- Conciseness and Fluency sit at their
    maximum even for a description of a DIFFERENT image, so two of the ten points
    are content-independent by construction. Overriding the judge's own total with
    the sum would therefore inflate every score by that constant and blunt the very
    contrast the control pairs exist to measure. Both are recorded; the report
    shows both and says which is which.
    """
    if not isinstance(rec, dict):
        return None
    out, total = {}, 0.0
    for k, hi in _FIELDS:
        try:
            v = float(rec.get(k))
        except (TypeError, ValueError):
            return None
        v = max(0.0, min(hi, v))
        out[k] = v
        total += v
    try:
        said = float(rec.get("score"))
    except (TypeError, ValueError):
        said = None
    out["score_sum"] = round(total, 1)                   # components added up
    out["score"] = said if said is not None else round(total, 1)   # judge's own total
    out["score_holistic"] = said
    out["gap_sum_holistic"] = None if said is None else round(said - total, 1)
    out["rationale"] = str(rec.get("rationale", ""))[:300]
    return out


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
mp = {int(k): v.replace(".jpg", "") for k, v in
      json.load(open(MAP, encoding="utf-8")).items()}
human = {}
for line in open(ANN, encoding="utf-8"):
    if line.strip():
        o = json.loads(line)
        name = mp.get(o["img_idx"])
        if name and o["content"].strip():
            human[name] = o["content"].strip()

narr = {k: v["text"] for k, v in json.load(open(NARR, encoding="utf-8")).items()}
patt = {k: v.get("text", "") for k, v in json.load(open(DESC, encoding="utf-8")).items()}
if os.path.exists(DESC_TR):
    patt.update({k: v.get("text", "")
                 for k, v in json.load(open(DESC_TR, encoding="utf-8")).items()})

val_split = {l.strip() for l in open(VAL) if l.strip()}

names = sorted(n for n in human if n in narr or n in patt)
if LIMIT:
    names = names[:LIMIT]
print(f"annotated images with a generated description: {len(names)} "
      f"({sum(1 for n in names if n in val_split)} val)", flush=True)

# a seeded derangement: the control pair for image i is another image's description
rng = random.Random(SEED)
shift = rng.randrange(1, len(names))
control_of = {n: names[(i + shift) % len(names)] for i, n in enumerate(names)}

verdict = {}
if os.path.exists(CMP):
    for line in open(CMP, encoding="utf-8"):
        m = re.match(r"^## (\S+) - \*\*(ALTA|PARZIALE|BASSA)\*\*", line)
        if m:
            verdict[m.group(1)] = m.group(2)

jobs = []
for n in names:
    for kind, table in (("narrative", narr), ("pattern", patt)):
        if table.get(n):
            jobs.append({"image": n, "kind": kind, "pairing": "genuine",
                         "gt": human[n], "cand": table[n]})
        c = control_of[n]
        if table.get(c):
            jobs.append({"image": n, "kind": kind, "pairing": "control",
                         "control_from": c, "gt": human[n], "cand": table[c]})
rng.shuffle(jobs)          # genuine and control must be indistinguishable in sequence
print(f"judgements to make: {len(jobs)} "
      f"({Counter((j['kind'], j['pairing']) for j in jobs)})", flush=True)

# --------------------------------------------------------------------------
# judge
# --------------------------------------------------------------------------
from slicelens.backends.hf_llm import HFLLM

t0 = time.time()
print(f"loading judge {MODEL} (4bit={os.environ.get('JUDGE_4BIT') == '1'}) ...", flush=True)
judge = HFLLM(model=MODEL, max_new_tokens=256,
              load_in_4bit=os.environ.get("JUDGE_4BIT") == "1")
print(f"loaded in {time.time() - t0:.0f}s", flush=True)

done_path = OUT + ".json"
done = {}
if os.path.exists(done_path):                       # resume-safe
    for r in json.load(open(done_path, encoding="utf-8"))["judgements"]:
        done[(r["image"], r["kind"], r["pairing"])] = r
    print(f"resuming: {len(done)} judgements already on disk", flush=True)

results, failed, t0 = [], 0, time.time()
for i, j in enumerate(jobs, 1):
    key = (j["image"], j["kind"], j["pairing"])
    # reuse a cached judgement only when BOTH texts are the ones it was given:
    # a key match alone would keep old verdicts after the texts were regenerated,
    # and the candidate alone would keep them after the ANNOTATIONS were corrected
    # (six of them were, on 2026-09-29, and their old scores must not survive)
    if key in done and done[key].get("cand") == j["cand"]             and done[key].get("gt_hash") in (None, _hash(j["gt"])):
        results.append(done[key])
        continue
    raw = judge.chat(JUDGE_SYSTEM, JUDGE_USER.format(gt=j["gt"], cand=j["cand"]))
    rec = clean(parse_json(raw))
    if rec is None:
        failed += 1
        rec = {"error": raw[:200]}
    r = dict(j)
    r.pop("gt")
    r["cand_words"] = len(j["cand"].split())
    r["gt_words"] = len(j["gt"].split())
    r["gt_hash"] = _hash(j["gt"])
    r["split"] = "val" if j["image"] in val_split else "train"
    r["verdict"] = verdict.get(j["image"])
    r.update(rec)
    results.append(r)
    if i % 20 == 0 or i == len(jobs):
        el = time.time() - t0
        print(f"  {i}/{len(jobs)} | {el:.0f}s | {el / max(i, 1):.1f}s per item "
              f"| unparsed: {failed}", flush=True)
        os.makedirs(os.path.dirname(done_path), exist_ok=True)
        json.dump({"judgements": results}, open(done_path, "w", encoding="utf-8"),
                  indent=1, ensure_ascii=False)

ok = [r for r in results if "score" in r]
print(f"\njudged: {len(ok)}/{len(results)} parsed ({failed} unparsed)", flush=True)


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def group(kind, pairing, split=None):
    return [r for r in ok if r["kind"] == kind and r["pairing"] == pairing
            and (split is None or r["split"] == split)]


report = []


def say(line=""):
    print(line, flush=True)
    report.append(line)


say("=" * 74)
say("SCORE 0-10 = Completeness 4 + Correctness 4 + Specificity 2 (Gemma locale)")
say("=" * 74)
say(f"{'':30s}{'genuine':>10s}{'controllo':>11s}{'scarto':>9s}{'n':>6s}")
for label, field in (("score del giudice", "score"),
                     ("somma dei 5 criteri", "score_sum")):
    for kind in ("narrative", "pattern"):
        g, c = group(kind, "genuine"), group(kind, "control")
        say(f"{kind + ' (' + label + ')':30s}{mean(r[field] for r in g):10.2f}"
            f"{mean(r[field] for r in c):11.2f}"
            f"{mean(r[field] for r in g) - mean(r[field] for r in c):9.2f}"
            f"{len(g):6d}")
    say()
say()
say("Lo scarto sopra il controllo e' l'unica parte che vale come evidenza: il")
say("controllo e' la STESSA annotazione umana contro la descrizione di un'ALTRA")
say("immagine, mescolata alle genuine e indistinguibile per il giudice.")

say()
say("-- per criterio: genuine vs controllo " + "-" * 36)
say(f"{'':30s}" + "".join(f"{k[:12]:>13s}" for k, _hi in _FIELDS))
for kind in ("narrative", "pattern"):
    for pairing in ("genuine", "control"):
        g = group(kind, pairing)
        say(f"{kind + ' ' + pairing:30s}"
            + "".join(f"{mean(r[k] for r in g):13.2f}" for k, _hi in _FIELDS))
say(f"{'max':30s}" + "".join(f"{hi:13d}" for _k, hi in _FIELDS))
say()
say("Un criterio che sul controllo resta al massimo non misura l'accordo: quei")
say("punti li prende anche la descrizione di un'altra immagine.")

say()
say("-- per split " + "-" * 60)
for kind in ("narrative", "pattern"):
    for sp in ("val", "train"):
        g, c = group(kind, "genuine", sp), group(kind, "control", sp)
        if g:
            say(f"{kind:12s} {sp:6s} genuine {mean(r['score'] for r in g):5.2f} "
                f"| controllo {mean(r['score'] for r in c):5.2f} | n={len(g)}")

say()
say("-- verdetti scritti a mano (solo val) " + "-" * 36)
for kind in ("narrative", "pattern"):
    for v in ("ALTA", "PARZIALE", "BASSA"):
        g = [r for r in group(kind, "genuine") if r.get("verdict") == v]
        if g:
            say(f"{kind:12s} {v:9s} score medio {mean(r['score'] for r in g):5.2f} "
                f"| n={len(g)}")

# length bias -- the user's own earlier finding, cheap to re-check here
try:
    from scipy import stats
    say()
    say("-- bias di lunghezza (score vs n. parole del candidato) " + "-" * 18)
    for kind in ("narrative", "pattern"):
        g = group(kind, "genuine")
        rho, pv = stats.spearmanr([r["cand_words"] for r in g], [r["score"] for r in g])
        say(f"{kind:12s} Spearman rho={rho:+.2f} (p={pv:.3f}) su n={len(g)} "
            f"| parole medie {mean(r['cand_words'] for r in g):.0f}")
    say("Positivo e significativo = il giudice premia la verbosita', non solo il")
    say("contenuto; da leggere insieme a length_causal_test.py del lavoro sui VLM.")
except ImportError:
    pass

gaps = [r["gap_sum_holistic"] for r in ok if r.get("gap_sum_holistic") is not None]
say()
say(f"Score del giudice meno somma dei criteri: media {mean(gaps):+.2f}, "
    f"negativo in {sum(1 for x in gaps if x < 0)}/{len(gaps)} casi, "
    f"positivo in {sum(1 for x in gaps if x > 0)}.")
say("Non e' un errore di somma casuale: il giudice e' piu' severo quando giudica")
say("il testo nel suo insieme di quanto lo sia criterio per criterio.")

say()
say("-- estremi (narrative, genuine) " + "-" * 42)
g = sorted(group("narrative", "genuine"), key=lambda r: -r["score"])
for tag, rows in (("migliori", g[:3]), ("peggiori", g[-3:])):
    for r in rows:
        say(f"  {tag:9s} {r['image']} {r['score']:4.1f}  {r['rationale'][:88]}")

json.dump({"judgements": results, "report": report},
          open(done_path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
with open(OUT + ".txt", "w", encoding="utf-8") as f:
    f.write("\n".join(report) + "\n")
say()
say(f"saved {done_path} / {OUT}.txt")
print("JUDGE_DONE", flush=True)
