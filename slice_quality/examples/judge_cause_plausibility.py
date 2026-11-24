"""Is the CAUSE the pipeline proposes plausible, given the error it is meant to explain?

The two branches of this pipeline do not describe an error, they NAME the
situation a group of errors shares -- "an object cut off at the image border"
(SliceLens, language branch), "small objects of class 'person' with imprecise
boundaries" (cluster branch). Every measurement so far scored the pipeline's
DESCRIPTION of what went wrong (judge_descriptions.py). This one scores the
other half, the part that is the actual deliverable: given what goes wrong in an
image, how plausible is the situation the system blames for it?

The plausibility is asked TWICE per image, against two independent accounts of
the same errors, and the judge is never told which is which:

    human      the hand-written annotation of that image (the reference)
    pipeline   the generated narrative of every typed error (error_narratives)

Two scores per image therefore, on the same 0-10 rubric. Their agreement is the
result that matters: if the cause scores the same whether the errors are told by
a person or by the pipeline, then the pipeline's own text can stand in for the
human one when judging a failure mode -- which is what the thesis needs it to do.

Rubric, both components auditable beside the total:
    Consistency       0-5   do classes, error types and circumstances agree with
                            the error description? Each contradiction costs.
    Explanatory power 0-5   if the situation were true of this image, would it
                            account for the errors -- all of them or part?
    plausibility = the two added, 0-10.

CHANCE CALIBRATION, as everywhere else in this project: each pair is accompanied
by a control in which the SAME error description is given the cause of ANOTHER
image, seeded and deranged so nothing is paired with itself, shuffled into the
sequence so the judge cannot tell them apart. A judge that finds every cause
plausible is measuring politeness; only the gap above the control counts.

The cause text carries the situation, the class and the error type it claims --
never the mask numbers (mean IoU, boundary IoU, thin fragments). Those are
measurements of the very errors being described, and handing them to the judge
would let it match numbers instead of judging an explanation.

ALL=1 drops the human side and judges EVERY val image that has a certified cause
(582 on the object run) against the pipeline narrative alone. There is no human
annotation out there -- that is the whole point of the pipeline -- so the
measurement becomes one score per image instead of two, and the question it
answers is the coverage-wide one: over the images the system actually speaks
about, how plausible are the situations it blames? The control is drawn on a
seeded sample of CONTROL_N images rather than on all of them: the floor is a
property of the judge and the corpus, not of the single image, and 150 controls
pin it down as well as 582 would while costing four hours less.

SYNTH=1 switches the sources to the synthetic set: its 21 hand-altered masks, of
which 9 fall inside a certified pattern (the branch is INDUCED there, as in
train36 -- no re-certification, 21 images certify nothing). Its annotations are
the declared ones, half of them saying the prediction is perfect, so the same
question is asked where the answer is known: a cause proposed for an image whose
annotation declares no error at all should not look plausible.

Run (GPU booked, 4-bit on a 12 GB card):
  SQ_OUT=outputs/slice_quality_inst JUDGE_4BIT=1 PYTHONPATH=/workspace \
      python slice_quality/examples/judge_cause_plausibility.py
Env: SQ_OUT, NARR, DESC, DESC_TR, OUT, JUDGE_MODEL, JUDGE_4BIT, SEED, LIMIT, SHARD.

SHARD=k/n splits one run over n GPUs: this process judges every n-th job from the
k-th into OUT_shard<k>of<n>.json. A last run without SHARD finds every verdict
cached in the shard files, never loads the judge, and writes the report.
"""
import glob
import hashlib
import json
import os
import random
import re
import time
from collections import Counter

SQ_OUT = os.environ.get("SQ_OUT", "outputs/slice_quality_inst")
SYNTH = os.environ.get("SYNTH") == "1"
SRC = os.environ.get("SRC", "datasets/cropped_synthetic_masks")
if SYNTH:
    NARR = os.environ.get("NARR", f"{SQ_OUT}/synth/narratives_synth.json")
    DESC = os.environ.get("DESC", f"{SQ_OUT}/synth/descriptions_synth.json")
    DESC_TR = os.environ.get("DESC_TR", "")
    OUT = os.environ.get("OUT", f"{SQ_OUT}/synth/judge_cause_synth")
    ANN = os.environ.get("ANN", f"{SRC}/answer_gts.jsonl")
    MAP = os.environ.get("MAP", f"{SRC}/voc_patch__id_matches.json")
else:
    NARR = os.environ.get("NARR", f"{SQ_OUT}/error_narratives.json")
    DESC = os.environ.get("DESC", f"{SQ_OUT}/image_descriptions.json")
    DESC_TR = os.environ.get("DESC_TR", f"{SQ_OUT}/train36/descriptions_train36_full.json")
    OUT = os.environ.get("OUT", f"{SQ_OUT}/judge_cause")
    ANN = os.environ.get("ANN", "/data/dataset/answer_gts.jsonl")
    MAP = os.environ.get("MAP", "/data/dataset/voc_patch_matches.json")
VAL = os.environ.get("VAL", "/data/voc2012/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt")
CMP = os.environ.get("CMP", "outputs/slice_quality/gt_comparison.md")

MODEL = os.environ.get("JUDGE_MODEL", "/models/gemma")
SEED = int(os.environ.get("SEED", "0"))
LIMIT = int(os.environ.get("LIMIT", "0"))
ALL = os.environ.get("ALL") == "1"          # every image with a cause, narrative only
CONTROL_N = int(os.environ.get("CONTROL_N", "150"))
if ALL:
    OUT = os.environ.get("OUT", f"{SQ_OUT}/judge_cause_val")

JUDGE_SYSTEM = """\
You are an expert evaluator of failure analyses of an image segmentation model.

You will be given, about ONE image:
1. ERROR DESCRIPTION: what goes wrong in the predicted segmentation of that image.
2. PROPOSED CAUSE: the situation an automatic analysis blames for those errors. \
It is a recurring failure mode the system certified over many images and then \
attached to this one, naming the object class and the kind of error it claims.

Judge how PLAUSIBLE the proposed cause is as an explanation of the errors \
described. You do not see the image: judge from the two texts only.

- Consistency (0-5): do the object classes, the error types and the \
circumstances named in the cause agree with the error description? Take off \
points for each element the description contradicts -- a class that is not \
mentioned there, an error of a kind the description does not report.
- Explanatory power (0-5): IF the situation named were true of this image, \
would it account for the errors described? Full marks only if it accounts for \
the main error, not a marginal one; take off points if it explains only part of \
what is described, or if it is so generic that it would fit any image.

An error description that reports MORE than the cause covers is normal: the \
cause speaks about one recurring situation, not about everything in the image. \
Do not penalise it for that beyond the Explanatory power scale. Do not reward or \
punish style, length or grammar, and do not guess what the image looks like.

Respond ONLY with valid JSON, keeping the rationale under 40 words:
{"consistency": <0-5>, "explanatory_power": <0-5>, \
"plausibility": <float 0-10, one decimal>, "rationale": "<max 40 words>"}"""

JUDGE_USER = """ERROR DESCRIPTION:
{err}

PROPOSED CAUSE:
{cause}"""

_FIELDS = [("consistency", 5), ("explanatory_power", 5)]


def _hash(text):
    """Short fingerprint of the error description this verdict was given."""
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
    """Keep the judge's own total and the sum of the two components.

    Same discipline as judge_descriptions.py: the two disagree, and which one is
    reported must be visible rather than chosen silently. `score` is the judge's
    own number, `score_sum` the components added up.
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
        said = float(rec.get("plausibility"))
    except (TypeError, ValueError):
        said = None
    out["score_sum"] = round(total, 1)
    out["score"] = said if said is not None else round(total, 1)
    out["score_holistic"] = said
    out["gap_sum_holistic"] = None if said is None else round(said - total, 1)
    out["rationale"] = str(rec.get("rationale", ""))[:300]
    return out


def cause_text(rec):
    """The situation(s) the pipeline attaches to this image, without the numbers.

    ``details`` (mean IoU, boundary IoU, thin fragments, position) is dropped on
    purpose: it measures the errors being described, and a judge given it would
    match measurements instead of weighing an explanation.
    """
    pats = (rec or {}).get("patterns") or []
    if not pats:
        return None
    lines = []
    for p in pats:
        d = (p.get("description") or "").strip()
        if not d:
            continue
        n = p.get("n_instances") or 0
        branch = "SliceLens situation" if p.get("branch") != "cluster" else "visual group"
        lines.append(f"- {d} [{branch}; {n} object{'' if n == 1 else 's'} of this image]")
    return "\n".join(lines) if lines else None


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
mp = {int(k): v.replace(".jpg", "") for k, v in json.load(open(MAP, encoding="utf-8")).items()}
human = {}
for line in open(ANN, encoding="utf-8"):
    if line.strip():
        o = json.loads(line)
        name = mp.get(o["img_idx"])
        if name and o["content"].strip():
            human[name] = o["content"].strip()

narr = {k: v["text"] for k, v in json.load(open(NARR, encoding="utf-8")).items()}
recs = json.load(open(DESC, encoding="utf-8"))
if os.path.exists(DESC_TR):
    recs.update(json.load(open(DESC_TR, encoding="utf-8")))
cause = {k: c for k, c in ((k, cause_text(v)) for k, v in recs.items()) if c}

val_split = set() if SYNTH else {l.strip() for l in open(VAL) if l.strip()}

if ALL:
    names = sorted(n for n in cause if n in narr)
else:
    names = sorted(n for n in human if n in cause and n in narr)
if LIMIT:
    names = names[:LIMIT]
print(f"images with a certified cause and a narrative: {len(names)} "
      f"({sum(1 for n in names if n in val_split)} val)"
      + (", narrative only (no human annotation out there)" if ALL else ""), flush=True)
if len(names) < 4:
    raise SystemExit("too few images to derange: nothing to measure")

rng = random.Random(SEED)
shift = rng.randrange(1, len(names))
control_of = {n: names[(i + shift) % len(names)] for i, n in enumerate(names)}

declared_good = {}
if SYNTH:
    OK_RE = re.compile(r"perfect|no mistake|correctly|accurate|is correct|precisely", re.I)
    BAD_RE = re.compile(r"wrong|miss|hallucinat|unusable|error", re.I)
    declared_good = {n: bool(OK_RE.search(t)) and not bool(BAD_RE.search(t))
                     for n, t in human.items()}
    print(f"synthetic set: {sum(1 for n in names if declared_good.get(n))} of {len(names)} "
          f"judged images are declared PERFECT by their annotation", flush=True)

verdict = {}
if os.path.exists(CMP) and not SYNTH:
    for line in open(CMP, encoding="utf-8"):
        m = re.match(r"^## (\S+) - \*\*(ALTA|PARZIALE|BASSA)\*\*", line)
        if m:
            verdict[m.group(1)] = m.group(2)

kinds = (("pipeline", narr),) if ALL else (("human", human), ("pipeline", narr))
# the control floor is a property of the judge and the corpus: on the full-coverage
# run it is measured on a seeded sample instead of doubling four hours of GPU
ctrl_on = set(names) if not ALL else set(
    rng.sample(names, min(CONTROL_N, len(names))))
jobs = []
for n in names:
    for kind, table in kinds:
        jobs.append({"image": n, "kind": kind, "pairing": "genuine",
                     "err": table[n], "cause": cause[n]})
        if n not in ctrl_on:
            continue
        c = control_of[n]
        jobs.append({"image": n, "kind": kind, "pairing": "control", "control_from": c,
                     "err": table[n], "cause": cause[c]})
rng.shuffle(jobs)
print(f"judgements to make: {len(jobs)} ({Counter((j['kind'], j['pairing']) for j in jobs)})",
      flush=True)
SHARD = os.environ.get("SHARD", "")
tag = ""
if SHARD:
    sk, sn = map(int, SHARD.split("/"))
    jobs = jobs[sk::sn]
    tag = f"_shard{sk}of{sn}"
    print(f"shard {sk}/{sn}: {len(jobs)} of them in this process", flush=True)

# --------------------------------------------------------------------------
# judge
# --------------------------------------------------------------------------
from slicelens.backends.hf_llm import HFLLM

judge = None


def load_judge():
    # loaded on the first verdict that is not cached: a merge run never needs it
    global judge
    t0 = time.time()
    print(f"loading judge {MODEL} (4bit={os.environ.get('JUDGE_4BIT') == '1'}) ...", flush=True)
    judge = HFLLM(model=MODEL, max_new_tokens=256,
                  load_in_4bit=os.environ.get("JUDGE_4BIT") == "1")
    print(f"loaded in {time.time() - t0:.0f}s", flush=True)


done_path = OUT + tag + ".json"
done = {}
for path in [OUT + ".json"] + sorted(glob.glob(glob.escape(OUT) + "_shard*of*.json")):
    if os.path.exists(path):
        for r in json.load(open(path, encoding="utf-8"))["judgements"]:
            done[(r["image"], r["kind"], r["pairing"])] = r
if done:
    print(f"resuming: {len(done)} judgements already on disk", flush=True)

results, failed, t0 = [], 0, time.time()
for i, j in enumerate(jobs, 1):
    key = (j["image"], j["kind"], j["pairing"])
    # a cached verdict is reused only when BOTH texts are the ones it was given:
    # the word count alone survived a corrected annotation of the same length
    if key in done and done[key].get("cause") == j["cause"] \
            and done[key].get("err_hash") in (None, _hash(j["err"])) \
            and done[key].get("err_words") == len(j["err"].split()):
        results.append(done[key])
        continue
    if judge is None:
        load_judge()
    raw = judge.chat(JUDGE_SYSTEM, JUDGE_USER.format(err=j["err"], cause=j["cause"]))
    rec = clean(parse_json(raw))
    if rec is None:
        failed += 1
        rec = {"error": raw[:200]}
    r = dict(j)
    r.pop("err")
    r["err_words"] = len(j["err"].split())
    r["err_hash"] = _hash(j["err"])
    r["split"] = "synth" if SYNTH else ("val" if j["image"] in val_split else "train")
    r["verdict"] = verdict.get(j["image"])
    if SYNTH:
        r["declared_good"] = bool(declared_good.get(j["image"]))
    r.update(rec)
    results.append(r)
    if i % 20 == 0 or i == len(jobs):
        el = time.time() - t0
        print(f"  {i}/{len(jobs)} | {el:.0f}s | {el / max(i, 1):.1f}s per item "
              f"| unparsed: {failed}", flush=True)
        os.makedirs(os.path.dirname(done_path) or ".", exist_ok=True)
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


say("=" * 78)
say("PLAUSIBILITA' DELLA CAUSA 0-10 = Consistency 5 + Explanatory power 5 (Gemma locale)")
say("La stessa causa, giudicata contro due racconti indipendenti dello stesso errore")
say("=" * 78)
say(f"{'':34s}{'genuina':>10s}{'controllo':>11s}{'scarto':>9s}{'n':>6s}")
_KINDS = ("pipeline",) if ALL else ("human", "pipeline")
for label, field in (("voto del giudice", "score"), ("somma dei criteri", "score_sum")):
    for kind in _KINDS:
        g, c = group(kind, "genuine"), group(kind, "control")
        say(f"{('errore raccontato da: ' + kind + ' (' + label + ')')[:34]:34s}"
            f"{mean(r[field] for r in g):10.2f}{mean(r[field] for r in c):11.2f}"
            f"{mean(r[field] for r in g) - mean(r[field] for r in c):9.2f}{len(g):6d}")
    say()

say("Criteri, solo sulle coppie genuine:")
for kind in _KINDS:
    g = group(kind, "genuine")
    say(f"  {kind:9s} consistency {mean(r['consistency'] for r in g):.2f}/5   "
        f"explanatory power {mean(r['explanatory_power'] for r in g):.2f}/5")
say()

# --- l'accordo fra i due racconti: e' il risultato di questa misura ---------
pairs = []
for n in ([] if ALL else sorted({r["image"] for r in ok})):
    h = [r for r in ok if r["image"] == n and r["kind"] == "human" and r["pairing"] == "genuine"]
    p = [r for r in ok if r["image"] == n and r["kind"] == "pipeline" and r["pairing"] == "genuine"]
    if h and p:
        pairs.append((n, h[0]["score"], p[0]["score"]))
if pairs:
    d = [b - a for _n, a, b in pairs]
    ma, mb = mean(a for _n, a, _b in pairs), mean(b for _n, _a, b in pairs)
    sa = (sum((a - ma) ** 2 for _n, a, _b in pairs) / len(pairs)) ** 0.5
    sb = (sum((b - mb) ** 2 for _n, _a, b in pairs) / len(pairs)) ** 0.5
    cov = sum((a - ma) * (b - mb) for _n, a, b in pairs) / len(pairs)
    r_p = cov / (sa * sb) if sa and sb else float("nan")
    say(f"Accordo fra i due giudizi sulla STESSA causa ({len(pairs)} immagini):")
    say(f"  media umano {ma:.2f} vs pipeline {mb:.2f} (differenza media {mean(d):+.2f})")
    say(f"  |differenza| media {mean(abs(x) for x in d):.2f} punti; "
        f"entro 1 punto su {sum(1 for x in d if abs(x) <= 1)}/{len(pairs)} immagini, "
        f"entro 2 su {sum(1 for x in d if abs(x) <= 2)}")
    say(f"  correlazione di Pearson r = {r_p:.2f}")
    say()
    say("Se i due voti coincidono, il testo della pipeline puo' sostituire l'annotazione")
    say("umana quando si giudica la plausibilita' di una causa: e' esattamente cio' che")
    say("serve per usare la pipeline su immagini che nessuno ha annotato.")
    say()
    ext = sorted(pairs, key=lambda x: abs(x[2] - x[1]), reverse=True)[:5]
    say("Dove i due racconti portano a giudizi diversi:")
    for n, a, b in ext:
        say(f"  {n}: umano {a:.1f} / pipeline {b:.1f}")
    say()

if SYNTH:
    for half, lab in ((True, "dichiarate perfette"), (False, "dichiarate errate")):
        for kind in ("human", "pipeline"):
            gg = [r for r in group(kind, "genuine") if bool(r.get("declared_good")) is half]
            if gg:
                say(f"  {lab:20s} {kind:9s} {mean(r['score'] for r in gg):.2f}  (n={len(gg)})")
    say("Sulla meta' dichiarata perfetta una causa non dovrebbe sembrare plausibile:")
    say("l'annotazione dice che non c'e' nulla da spiegare.")
    say()

say("Distribuzione del voto (coppie genuine):")
for kind in _KINDS:
    c = Counter(int(round(r["score"])) for r in group(kind, "genuine"))
    say(f"  {kind:9s} " + "  ".join(f"{k}:{c[k]}" for k in sorted(c)))
say()
say("Per split:")
for kind in _KINDS:
    for sp in ("val", "train"):
        g = group(kind, "genuine", sp)
        if g:
            say(f"  {kind:9s} {sp:5s} {mean(r['score'] for r in g):.2f}  (n={len(g)})")
say()
say("Un'asimmetria da dichiarare: i due racconti non hanno la stessa lunghezza")
for kind in _KINDS:
    g = group(kind, "genuine")
    say(f"  {kind:9s} {mean(r['err_words'] for r in g):.0f} parole in media per descrizione")
say("La narrativa della pipeline dice piu' cose dell'annotazione: offre piu' appigli")
say("alla causa, e insieme piu' occasioni di contraddirla. Il confronto fra i due voti")
say("va letto sapendolo.")
say()
say("Il controllo e' la STESSA descrizione d'errore contro la causa di un'ALTRA")
say("immagine, mescolata alle genuine: e' il pavimento del rumore, e solo lo scarto")
say("sopra di esso vale come evidenza.")

json.dump({"judgements": results, "report": report},
          open(done_path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
open(OUT + tag + ".txt", "w", encoding="utf-8").write("\n".join(report) + "\n")
print(f"\nsalvato {done_path} / {OUT}{tag}.txt", flush=True)
print("JUDGE_CAUSE_DONE", flush=True)
