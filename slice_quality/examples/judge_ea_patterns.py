"""Do the systematic patterns CAPTURE the Pascal-EA modification? Gemma, 1-10.

The planted-cause test (ea_planted_eval.py) counts whether the objects the edit made
the segmenter lose fall inside a certified slice. This asks the other half, in words:
reading what the pipeline says about the image -- its systematic patterns, the
certified situations attached to the image's errors -- would one learn what was done
to it? The truth is known here, which is the point of a planted cause.

Images: the VICTIMS of val-EA (ea_planted_eval.py): images whose edited subject the
segmenter lost (recall drop >= 0.30). An edit that did not hurt the segmenter leaves
nothing for a failure analysis to capture, so those images are not judged.

    MODIFICATION   (the truth, never seen by the pipeline) the subject's class, the
                   material it was re-rendered in, its recall before and after, and
                   what its pixels became
    PATTERNS       the systematic patterns of the image (image_descriptions.json):
                   situation, class and error type of each, without slice ids or
                   mask numbers

Three pairings per image, shuffled together so the judge cannot tell them apart:

    genuine    the val-EA patterns of the same image
    control    the val-EA patterns of ANOTHER victim (seeded derangement): the noise
               floor -- a judge that scores this high is reading the class, not the
               image
    original   the patterns the reference run (inst) attaches to the same image
               BEFORE the edit: what the analysis said when nothing had been changed

An image without any pattern gets 1 without calling the judge ("no pattern
concerns the modified object" is the rubric's own 1) and is marked auto.

Rubric: Coverage 0-5 (a pattern concerns the modified object and the error the edit
caused) + Cause 0-5 (the situation named corresponds to the modification: changed
material / texture / appearance), and the judge's overall score 1-10 with anchors.

Run (GPU booked, 4-bit):  JUDGE_4BIT=1 PYTHONPATH=/workspace \
    python slice_quality/examples/judge_ea_patterns.py
Env: EA_RUN, INST_RUN, EA, CMP, OUT, JUDGE_MODEL, JUDGE_4BIT, SEED, LIMIT, DRY.
"""
import hashlib
import json
import os
import random
import re
import time
from collections import Counter, defaultdict

EA_RUN = os.environ.get("EA_RUN", "outputs/slice_quality_ea")
INST_RUN = os.environ.get("INST_RUN", "outputs/slice_quality_inst")
EAD = os.environ.get("EA", "/ds/pascal_ea_material")
CMP = os.environ.get("CMP", "outputs/pascal_ea/per_image.json")
OUT = os.environ.get("OUT", f"{EA_RUN}/judge_ea_patterns")
MODEL = os.environ.get("JUDGE_MODEL", "/models/gemma")
SEED = int(os.environ.get("SEED", "0"))
LIMIT = int(os.environ.get("LIMIT", "0"))
DRY = os.environ.get("DRY") == "1"

JUDGE_SYSTEM = """\
You are an expert evaluator of failure analyses of an image segmentation model.

An image was deliberately modified, and the modification made the segmentation \
model fail. You will be given:
1. MODIFICATION: what was changed in the image and what the change did to the \
model's segmentation. This is the known truth.
2. SYSTEMATIC PATTERNS: what an automatic failure analysis, which was NOT told \
about any modification, reports for this image: the recurring failure situations \
(each certified over many images) it attaches to the image's errors, each with \
the object class and the kind of error.

Judge how well the systematic patterns CAPTURE the modification, from 1 to 10. \
You do not see the image: judge from the two texts only.

- Coverage (0-5): does a pattern concern the modified object -- its class and \
the error the modification caused (e.g. the object missed, predicted as \
background)? 0 if no pattern is about it.
- Cause (0-5): does the situation the pattern names correspond to what was \
modified -- the object's changed material, texture or appearance, the object \
looking artificial or unlike its usual self? Give little for a generic grouping \
that names no circumstance, and nothing for a situation unrelated to the \
modification (e.g. occlusion, image border, small size), even when it covers the \
right object.

Anchors for the overall score:
1 = no pattern concerns the modified object;
2-4 = a pattern covers the modified object and its error, but the situation it \
names is unrelated to the modification or is a generic label;
5-7 = the right object and error, with a situation that partly reflects the \
modification (an unusual or atypical appearance) without identifying it;
8-10 = a pattern names the changed appearance itself (an unusual material or \
texture, an object that looks like a sculpture, a model or a toy) for the \
modified object.

Patterns about other objects of the image are normal: neither reward nor \
penalise them. Do not reward or punish style or length, and do not guess what \
the image looks like.

Respond ONLY with valid JSON, keeping the rationale under 40 words:
{"coverage": <0-5>, "cause": <0-5>, "score": <integer 1-10>, \
"rationale": "<max 40 words>"}"""

JUDGE_USER = """MODIFICATION:
{mod}

SYSTEMATIC PATTERNS:
{pat}"""

NO_PATTERN = "The analysis reports no systematic pattern for this image."
_FIELDS = [("coverage", 5), ("cause", 5)]


def _hash(text):
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
    """The judge's own 1-10 score, and the two components beside it (their sum is
    reported too, never substituted silently for the judge's number)."""
    if not isinstance(rec, dict):
        return None
    out, total = {}, 0.0
    for k, hi in _FIELDS:
        try:
            v = float(rec.get(k))
        except (TypeError, ValueError):
            return None
        out[k] = max(0.0, min(hi, v))
        total += out[k]
    try:
        said = max(1.0, min(10.0, float(rec.get("score"))))
    except (TypeError, ValueError):
        said = None
    out["score_sum"] = round(total, 1)
    out["score"] = said if said is not None else max(1.0, round(total, 1))
    out["rationale"] = str(rec.get("rationale", ""))[:300]
    return out


def pattern_text(rec):
    """The systematic patterns of an image, as the gallery lists them, without slice
    ids and without ``details`` (position and mask measurements)."""
    lines = []
    for p in (rec or {}).get("patterns") or []:
        d = (p.get("description") or "").strip()
        if not d:
            continue
        n = p.get("n_instances") or 0
        kind = "SliceLens situation" if p.get("branch") != "cluster" else "visual group"
        lines.append(f"- {d} [{kind}; {n} object{'' if n == 1 else 's'} of this image]")
    return "\n".join(lines) if lines else None


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
ev = json.load(open(f"{EA_RUN}/planted_eval.json", encoding="utf-8"))
covered = {v.split("#")[0] for v in ev["covered"]}
names = sorted({v.split("#")[0] for v in ev["victims"]})
rows = {r["stem"]: r for r in json.load(open(CMP))["rows"]}
lab = json.load(open(f"{EAD}/labels.json"))["images"]
pat_ea = {k: pattern_text(v) for k, v in
          json.load(open(f"{EA_RUN}/image_descriptions.json", encoding="utf-8")).items()}
pat_in = {k: pattern_text(v) for k, v in
          json.load(open(f"{INST_RUN}/image_descriptions.json", encoding="utf-8")).items()}
if LIMIT:
    names = names[:LIMIT]


def modification(n):
    l, r = lab[n], rows[n]
    cls, mat = l["voc_class"], l["material_v3"]
    o, e = r["orig"]["rec"], r["v3"]["rec"]
    wl, ws = (r["v3"].get("wrong_top") or [None, 0])[:2]
    t = (f"The {cls} that is the main subject of this image was artificially re-rendered "
         f"as made of {mat}: same shape and position, only its material and texture "
         f"changed; the rest of the image is untouched.\n"
         f"Effect: the segmentation model labelled {o:.0%} of the {cls}'s pixels "
         f"correctly on the original image and {e:.0%} after the modification")
    if wl and ws >= 0.1:
        t += f"; {ws:.0%} of them are now predicted as {wl}"
    return t + "."


rng = random.Random(SEED)
shift = rng.randrange(1, len(names))
control_of = {n: names[(i + shift) % len(names)] for i, n in enumerate(names)}

jobs = []
for n in names:
    mod = modification(n)
    for pairing, src in (("genuine", n), ("control", control_of[n]), ("original", n)):
        pat = (pat_in if pairing == "original" else pat_ea).get(src)
        jobs.append({"image": n, "pairing": pairing, "patterns_of": src,
                     "mod": mod, "pat": pat or NO_PATTERN, "auto": pat is None})
rng.shuffle(jobs)
print(f"victims {len(names)} (covered by a certified slice: {len(covered & set(names))}) | "
      f"judgements {len(jobs)}: {dict(Counter(j['pairing'] for j in jobs))} | "
      f"without any pattern (auto 1): {dict(Counter(j['pairing'] for j in jobs if j['auto']))}",
      flush=True)
if DRY:
    for j in [j for j in jobs if not j["auto"]][:3]:
        print(f"\n--- {j['image']} / {j['pairing']}\n"
              + JUDGE_USER.format(mod=j["mod"], pat=j["pat"]))
    raise SystemExit(0)

# --------------------------------------------------------------------------
# judge
# --------------------------------------------------------------------------
from slicelens.backends.hf_llm import HFLLM

t0 = time.time()
print(f"loading judge {MODEL} (4bit={os.environ.get('JUDGE_4BIT') == '1'}) ...", flush=True)
judge = HFLLM(model=MODEL, max_new_tokens=256, load_in_4bit=os.environ.get("JUDGE_4BIT") == "1")
print(f"loaded in {time.time() - t0:.0f}s", flush=True)

done_path = OUT + ".json"
done = {}
if os.path.exists(done_path):
    for r in json.load(open(done_path, encoding="utf-8"))["judgements"]:
        done[(r["image"], r["pairing"])] = r
    print(f"resuming: {len(done)} judgements already on disk", flush=True)

results, failed, t0 = [], 0, time.time()
for i, j in enumerate(jobs, 1):
    key, h = (j["image"], j["pairing"]), _hash(j["mod"] + "\n" + j["pat"] + JUDGE_SYSTEM)
    if key in done and done[key].get("hash") == h:
        results.append(done[key])
        continue
    r = {k: v for k, v in j.items() if k != "mod"}
    r["hash"] = h
    if j["auto"]:
        r.update({"coverage": 0.0, "cause": 0.0, "score_sum": 0.0, "score": 1.0,
                  "rationale": "no systematic pattern for this image (auto)"})
    else:
        raw = judge.chat(JUDGE_SYSTEM, JUDGE_USER.format(mod=j["mod"], pat=j["pat"]))
        rec = clean(parse_json(raw))
        if rec is None:
            failed += 1
            rec = {"error": raw[:200]}
        r.update(rec)
    results.append(r)
    if i % 20 == 0 or i == len(jobs):
        el = time.time() - t0
        print(f"  {i}/{len(jobs)} | {el:.0f}s | unparsed: {failed}", flush=True)
        json.dump({"judgements": results}, open(done_path, "w", encoding="utf-8"),
                  indent=1, ensure_ascii=False)

# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
ok = [r for r in results if "score" in r]


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def grp(pairing, cond=lambda r: True):
    return [r for r in ok if r["pairing"] == pairing and cond(r)]


report = []


def say(line=""):
    print(line, flush=True)
    report.append(line)


say("=" * 78)
say("I PATTERN SISTEMATICI CATTURANO LA MODIFICA PASCAL-EA? voto 1-10 (Gemma locale)")
say(f"{len(names)} vittime: immagini di val-EA in cui la modifica del materiale ha fatto")
say("perdere il soggetto al segmentatore (recall del soggetto giu' di almeno 0.30)")
say("=" * 78)
say(f"{'':44s}{'voto':>7s}{'copert.':>9s}{'causa':>7s}{'n':>5s}{'auto 1':>8s}")
LBL = {"genuine": "genuina: pattern val-EA della stessa immagine",
       "control": "controllo: pattern val-EA di un'altra vittima",
       "original": "originale: pattern del run inst, prima della modifica"}
for p in ("genuine", "control", "original"):
    g = grp(p)
    say(f"{LBL[p][:44]:44s}{mean(r['score'] for r in g):7.2f}{mean(r['coverage'] for r in g):9.2f}"
        f"{mean(r['cause'] for r in g):7.2f}{len(g):5d}{sum(1 for r in g if r.get('auto')):8d}")
say()
g = grp("genuine")
say("Distribuzione del voto (genuine): " + "  ".join(
    f"{k}:{v}" for k, v in sorted(Counter(int(round(r['score'])) for r in g).items())))
gj = [r for r in g if not r.get("auto")]
say(f"Solo le immagini che hanno almeno un pattern ({len(gj)}): voto {mean(r['score'] for r in gj):.2f}, "
    f"copertura {mean(r['coverage'] for r in gj):.2f}/5, causa {mean(r['cause'] for r in gj):.2f}/5")
for lab_, cond in (("vittime dentro una slice certificata", lambda r: r["image"] in covered),
                   ("vittime fuori da ogni slice certificata", lambda r: r["image"] not in covered)):
    gg = grp("genuine", cond)
    say(f"  {lab_:42s} {mean(r['score'] for r in gg):.2f}  (n={len(gg)})")
say()
say("Per classe del soggetto (genuine):")
by = defaultdict(list)
for r in g:
    by[lab[r["image"]]["voc_class"]].append(r["score"])
for c, xs in sorted(by.items(), key=lambda x: -len(x[1])):
    say(f"  {c:12s} {mean(xs):5.2f}  (n={len(xs)})")
say("Per materiale (genuine):")
by = defaultdict(list)
for r in g:
    by[lab[r["image"]]["material_v3"]].append(r["score"])
for c, xs in sorted(by.items(), key=lambda x: -len(x[1])):
    say(f"  {c:12s} {mean(xs):5.2f}  (n={len(xs)})")
say()
say("Voti piu' alti (genuine):")
for r in sorted(g, key=lambda r: -r["score"])[:6]:
    say(f"  {r['image']} {r['score']:.0f} | {r.get('rationale', '')[:150]}")
say()
say("Il controllo e' il pavimento del rumore: gli stessi fatti della modifica contro i")
say("pattern di un'altra vittima. L'originale dice cosa riportava l'analisi sulla stessa")
say("immagine prima della modifica. Un'immagine senza pattern prende 1 senza giudice.")

json.dump({"judgements": results, "report": report, "victims": names},
          open(done_path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
open(OUT + ".txt", "w", encoding="utf-8").write("\n".join(report) + "\n")
print(f"\nsalvato {done_path} / {OUT}.txt", flush=True)
print("JUDGE_EA_PATTERNS_DONE", flush=True)
