"""A length-neutral agreement score: the annotation's FACTS, checked one by one.

Why a second judge. The rubric judge (judge_descriptions.py) is told not to reward
length, yet on the inst run its score correlates with the candidate's length
(Spearman rho +0.53 on the pattern texts, p < 0.001; +0.27 on the narrative
controls). A sentence in a prompt does not remove the bias: completeness and
specificity are judged on the whole candidate, and a longer text offers more to
match. Here the score is built so that length CANNOT buy points:

  A. facts    Gemma splits each HUMAN annotation into its atomic facts (one object,
              one thing right or wrong about its mask, where). Done once per
              annotation and cached, so genuine, control and duplicate candidates of
              an image are all checked against the SAME list.
  B. verdict  for each fact, Gemma says whether the candidate states it, states it in
              part, contradicts it, or leaves it out. Nothing else is asked: whatever
              the candidate says that the facts do not talk about -- extra objects,
              numbers, repetitions -- is ignored unless it contradicts a fact.
  score       computed HERE, not by the model:
              10 * max(0, stated + 0.5 * partial - contradicted) / n_facts

The denominator comes from the human text, so a longer candidate cannot raise it or
lower it; extra content cannot add a verdict. The price, declared: errors the
candidate invents about objects the annotator did not mention are not penalised,
because the annotation cannot tell whether they are invented or merely unmentioned.

Two checks that length really does not matter, both reported:
  * duplicate   every genuine candidate is also judged written TWICE (the same text
                repeated): double the length, identical content. The score should not
                move. The same duplicates are put through the OLD rubric too (OLD_DUP=1),
                so the two judges' sensitivity is measured on identical inputs.
  * correlation Spearman rho between score and candidate length, genuine and control,
                for the new score and for the old rubric run in the same process.
And, as everywhere in this project, a control pairing: each annotation against
ANOTHER image's text (seeded derangement). Only the gap above it is evidence.

Both annotated sets: the 79 VOC images (val + train36) and the 21 synthetic ones.

Only the NARRATIVE is scored (KINDS). The pattern text says which certified situation
an image belongs to, not what went wrong in it, so checking the annotation's facts
in it measures a different thing (79% of the facts come out absent by construction).
A first run scored both; its pattern rows are left in the shard files and dropped here.

Version 2 (glossary + fact types). Read by hand, the first version's verdicts were
too strict on paraphrase: of the 46 val facts not "stated", about 27 were stated by
the narrative in its own fixed vocabulary ("spilling over the object, 30% in excess"
for "the background around the horse is labelled HORSE"; "heavily imprecise
boundaries, many details lost" was even marked CONTRADICTED against "a coarser
boundary, missing details"). So:
  * the verification prompt carries a GLOSSARY of the narrative's fixed phrases and
    what each one states, the same for genuine, control and duplicate pairings (if
    it only made the judge lenient, the control would rise with the genuine score);
  * each fact gets a TYPE (one call per annotation, cached with the facts):
    "prediction" (an object and what is wrong or right with its mask), "overall" (a
    judgement of the whole mask naming no specific problem) or "ground_truth" (a
    remark on the annotation itself). The narrative reports errors object by object
    and takes the ground truth as the reference, so it cannot state the last two
    kinds: the score counts the "prediction" facts (all facts when an annotation has
    none, e.g. "the prediction is perfect"); the score on every fact is kept too.
The first version's merged file is kept as <OUT>_v1.json and compared in the report.

Run, two GPUs in parallel (each takes half of every stage; the facts of stage A are
exchanged through the shard files, so stage B waits for both):
  SHARD=0 NSHARDS=2 JUDGE_4BIT=1 python slice_quality/examples/judge_claims.py   (GPU a)
  SHARD=1 NSHARDS=2 JUDGE_4BIT=1 python slice_quality/examples/judge_claims.py   (GPU b)
  MERGE=1 python slice_quality/examples/judge_claims.py                           (CPU)
Env: SQ_OUT, OUT (default <SQ_OUT>/judge_claims), SYNTH_SRC, JUDGE_MODEL, JUDGE_4BIT,
     SEED, OLD_DUP (1), LIMIT, KINDS (default narrative), TYPES (1).
"""
import hashlib
import json
import os
import random
import re
import time
from collections import Counter

SQ_OUT = os.environ.get("SQ_OUT", "outputs/slice_quality_inst")
OUT = os.environ.get("OUT", f"{SQ_OUT}/judge_claims")
SHARD = int(os.environ.get("SHARD", "0"))
NSHARDS = int(os.environ.get("NSHARDS", "1"))
MERGE = os.environ.get("MERGE") == "1"
OLD_DUP = os.environ.get("OLD_DUP", "1") == "1"
SEED = int(os.environ.get("SEED", "0"))
LIMIT = int(os.environ.get("LIMIT", "0"))
MODEL = os.environ.get("JUDGE_MODEL", "/models/gemma")
ANN = os.environ.get("ANN", "/data/dataset/answer_gts.jsonl")
MAP = os.environ.get("MAP", "/data/dataset/voc_patch_matches.json")
VAL = "/data/voc2012/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt"
SYNTH_SRC = os.environ.get("SYNTH_SRC", "datasets/cropped_synthetic_masks")
# which generated texts are scored: the narrative only (see the docstring)
KINDS = tuple(os.environ.get("KINDS", "narrative").split(","))
TYPES = os.environ.get("TYPES", "1") == "1"

FACT_SYSTEM = """\
You will read a description, written by an expert, of how a predicted segmentation \
mask differs from the ground-truth mask of one image.

Split it into its atomic FACTS. A fact names one object or region (with its class \
when the text gives it), one thing that is wrong or right about how it was \
segmented, and where it is, if the text says so.
- Keep every fact the text states, and nothing it does not state.
- Do not merge two different objects into one fact.
- Do not split one fact into several just because it is written as a long sentence. \
Synonyms or near-synonyms for the same problem of the same object are ONE fact \
("coarse, imprecise and imperfect boundaries" is one fact), and a detail that only \
says where that same problem is worst ("especially on the wings", "especially the \
bottom and right sides") belongs inside that fact, not in a new one.
- Different problems of the same object (for example a missing part and a spill \
elsewhere) are different facts.
- If the text says the prediction is correct, accurate or free of errors, that is \
one fact.
At most 8 facts. Respond ONLY with valid JSON:
{"facts": ["<fact 1>", "<fact 2>", ...]}"""

VERIFY_SYSTEM = """\
You will be given numbered FACTS taken from an expert's description of the \
segmentation errors of one image, and a CANDIDATE description of the same image \
written by an automatic system.

For EACH fact, decide what the candidate says about it:
- "stated": the candidate states the same fact (different wording is fine; extra \
detail is fine).
- "partial": the candidate mentions the same object and a related problem, but \
only part of the fact or less precisely (for example the same problem placed on \
another side of the object, or a much milder or much stronger version of it).
- "contradicted": the candidate says something about the same object or region \
that is incompatible with the fact (for example missed vs present, wrong class vs \
right class, too large vs too small, correct vs wrong).
- "absent": the candidate does not address the fact.

The candidate is written in a fixed technical vocabulary. Read it with this \
glossary: each phrase on the left STATES what is on the right. When the candidate \
uses the phrase about the same object, a fact worded as on the right is "stated" \
(or "partial"), not "absent" and not "contradicted".
- "spilling over the object (N% of its area in excess)", "over-segmented", \
"spills beyond the object" = the mask is too large, over-extended or \
over-segmented: the background (or a neighbouring object) around it is labelled \
as that class.
- "with parts missing (N%)", "under-segmented, with whole parts of the region \
missing", "mostly missed: only N% covered" = under-segmented, incomplete: parts of \
the object are not predicted.
- "missed entirely" = the object is not segmented at all (predicted as background).
- "the prediction misses most of the annotated area: N% predicted as background" = \
the prediction is largely empty, most objects are missed.
- "segmented with slightly coarser / coarser and imprecise / heavily imprecise \
boundaries", "outline loosely / only roughly / barely followed", "many details \
lost", a low "boundary IoU" = imprecise, coarse, rough, poorly defined or \
inaccurate borders; the prediction is less detailed than the ground truth.
- "fine details of the outline are lost: N thin fragments the prediction does not \
trace" = thin parts or small details (legs, arms, handles, wings, tails, ears, \
edges) are missed or badly segmented.
- "N <CLASS> patches have been hallucinated", "an extra <CLASS> mask", "the \
<CLASS> mask spills beyond the object in N patches" = regions wrongly predicted as \
<CLASS> (false positives): background or another object labelled <CLASS>.
- "labelled <B> instead", "N% of the annotated area has been labelled <B>, a class \
that is not in the image", "given the wrong class" = misclassified as <B>.
- "merges N ground truth regions into a single mask", "absorbed into a neighbouring \
prediction" = separate objects merged, not separated.
- "extra masks split a ground truth region that should be a single mask" = the \
object is fragmented into several pieces.
- "holes inside the mask" = holes or gaps inside the predicted mask.
- "segmented well", "segmented closely", "correctly located" = that object is \
segmented correctly (after "but its outline ..." its borders are still imprecise).
- Class names may differ: BICYCLE = bike, DININGTABLE = table, TVMONITOR = TV, \
monitor or screen, POTTEDPLANT = plant or foliage, AEROPLANE = plane, MOTORBIKE = \
motorcycle, PERSON = person, people, man, woman, child.
- "on the top / bottom / left / right (edge) of the scene" places the object in the \
image; "spilling over on the top and left" places the problem on the object.

Rules:
- Judge each fact on its own.
- Use "contradicted" only when the two descriptions cannot both be true. The same \
problem in other words is never a contradiction.
- The LENGTH of the candidate must not influence any verdict. A fact stated in one \
short sentence is as "stated" as a fact stated at length; a long candidate earns \
nothing for statements that match no fact; repetition changes nothing.
- Ignore everything in the candidate that the facts do not talk about (other \
objects, numbers, positions, repetitions), unless it contradicts a fact.
Respond ONLY with valid JSON, exactly one verdict per fact, in order:
{"verdicts": ["stated" | "partial" | "contradicted" | "absent", ...]}"""

VERIFY_USER = """FACTS:
{facts}

CANDIDATE DESCRIPTION:
{cand}"""

VERDICTS = ("stated", "partial", "contradicted", "absent")
VHASH = None                                    # set below, once VERIFY_SYSTEM is final

TYPE_SYSTEM = """\
You will be given numbered FACTS taken from an expert's description of how a \
predicted segmentation mask differs from the ground-truth mask of one image. \
Label each fact with what it is about:
- "prediction": a specific object, region or part, and something wrong or right \
about how the PREDICTION segmented it (missed, extra, wrong class, too large, too \
small, imprecise borders, merged, split, holes, correct, ...). The experts often \
name an object as "the ground truth X region", or only speak of "its borders" or \
"its boundaries": these facts are still about the prediction ("the ground truth \
CAT region has been over-extended towards the top", "the CAT's boundaries are not \
well defined" are "prediction"). A wrong or absent class ("the subject is actually \
a COW", "the class COW is not in the image") is "prediction". A comparison with the \
ground truth ("the ground truth has a finer boundary than the prediction") is \
"prediction". A judgement on one named object ("the PERSON is segmented quite \
well") is "prediction".
- "overall": a judgement of the whole prediction that names no specific object \
("the prediction is generally good", "the mask is chaotic", "most regions were \
captured", "the segmentation is perfect", "the prediction is totally wrong").
- "ground_truth": ONLY a statement that the ground-truth annotation ITSELF is \
imprecise or wrong ("the ground truth is not segmented correctly", "the borders \
are not precise in the ground truth"). This is rare.
Respond ONLY with valid JSON, exactly one label per fact, in order:
{"types": ["prediction" | "overall" | "ground_truth", ...]}"""
FTYPES = ("prediction", "overall", "ground_truth")


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


def score_of(verdicts):
    c = Counter(verdicts)
    n = len(verdicts)
    return round(10.0 * max(0.0, c["stated"] + 0.5 * c["partial"] - c["contradicted"]) / n, 2)


VHASH = _hash(VERIFY_SYSTEM)
THASH = _hash(TYPE_SYSTEM)


def counted(verdicts, types):
    """The verdicts the score counts: the facts about the prediction; every fact when
    the annotation has none (e.g. "the prediction is perfect") or was not typed."""
    if not types or len(types) != len(verdicts):
        return list(verdicts)
    c = [v for v, t in zip(verdicts, types) if t == "prediction"]
    return c or list(verdicts)


def type_facts(judge, fl):
    user = "FACTS:\n" + "\n".join(f"{k + 1}. {f}" for k, f in enumerate(fl))
    for attempt in range(2):
        raw = judge.chat(TYPE_SYSTEM, user if attempt == 0 else
                         user + f"\n\nThere are exactly {len(fl)} facts: give exactly {len(fl)} labels.")
        t = [str(x).strip().lower() for x in ((parse_json(raw) or {}).get("types") or [])]
        if len(t) == len(fl) and all(x in FTYPES for x in t):
            return t, False
    return ["prediction"] * len(fl), True


# --------------------------------------------------------------------------
# data: the two annotated sets
# --------------------------------------------------------------------------
def load_sets():
    items = []                      # (set, image, human, {kind: text}, extra)
    mp = {int(k): v.replace(".jpg", "") for k, v in json.load(open(MAP, encoding="utf-8")).items()}
    human = {}
    for line in open(ANN, encoding="utf-8"):
        if line.strip():
            o = json.loads(line)
            n = mp.get(o["img_idx"])
            if n and o["content"].strip():
                human[n] = o["content"].strip()
    narr = {k: v["text"] for k, v in json.load(open(f"{SQ_OUT}/error_narratives.json", encoding="utf-8")).items()}
    patt = {k: v.get("text", "") for k, v in json.load(open(f"{SQ_OUT}/image_descriptions.json", encoding="utf-8")).items()}
    tr = f"{SQ_OUT}/train36/descriptions_train36_full.json"
    if os.path.exists(tr):
        patt.update({k: v.get("text", "") for k, v in json.load(open(tr, encoding="utf-8")).items()})
    val = {l.strip() for l in open(VAL) if l.strip()} if os.path.exists(VAL) else set()
    for n in sorted(human):
        if n in narr or n in patt:
            items.append(("voc", n, human[n], {"narrative": narr.get(n, ""), "pattern": patt.get(n, "")},
                          {"split": "val" if n in val else "train"}))
    smp = {int(k): v.replace(".jpg", "") for k, v in json.load(open(
        os.path.join(SYNTH_SRC, "voc_patch__id_matches.json"), encoding="utf-8")).items()}
    snarr = {k: v["text"] for k, v in json.load(open(f"{SQ_OUT}/synth/narratives_synth.json", encoding="utf-8")).items()}
    spatt = {k: v.get("text", "") for k, v in json.load(open(f"{SQ_OUT}/synth/descriptions_synth.json", encoding="utf-8")).items()}
    ok_re = re.compile(r"perfect|no mistake|correctly|accurate|is correct|precisely", re.I)
    bad_re = re.compile(r"wrong|miss|hallucinat|unusable|error", re.I)
    for line in open(os.path.join(SYNTH_SRC, "answer_gts.jsonl"), encoding="utf-8"):
        if line.strip():
            o = json.loads(line)
            n, t = smp[o["img_idx"]], o["content"].strip()
            items.append(("synth", n, t, {"narrative": snarr.get(n, ""), "pattern": spatt.get(n, "")},
                          {"split": "synth",
                           "declared_good": bool(ok_re.search(t)) and not bool(bad_re.search(t))}))
    return items[:LIMIT] if LIMIT else items


def build_jobs(items):
    """Genuine, control (seeded derangement within each set), duplicate; plus the
    old rubric on genuine and duplicate (VOC + synth). Same order in every shard."""
    rng = random.Random(SEED)
    jobs = []
    for sname in ("voc", "synth"):
        its = [it for it in items if it[0] == sname]
        if len(its) < 2:
            continue
        shift = rng.randrange(1, len(its))
        for i, (s, n, h, texts, extra) in enumerate(its):
            other = its[(i + shift) % len(its)]
            for kind in KINDS:
                cand = texts.get(kind) or ""
                if not cand:
                    continue
                base = {"set": s, "image": n, "kind": kind, **extra}
                jobs.append(dict(base, pairing="genuine", rubric="facts", cand=cand))
                jobs.append(dict(base, pairing="duplicate", rubric="facts", cand=cand + "\n\n" + cand))
                oc = other[3].get(kind) or ""
                if oc:
                    jobs.append(dict(base, pairing="control", rubric="facts", cand=oc, control_from=other[1]))
                if OLD_DUP:
                    jobs.append(dict(base, pairing="genuine", rubric="old", cand=cand))
                    jobs.append(dict(base, pairing="duplicate", rubric="old", cand=cand + "\n\n" + cand))
    rng.shuffle(jobs)
    return jobs


def key(j):
    return (j["set"], j["image"], j["kind"], j["pairing"], j["rubric"])


# --------------------------------------------------------------------------
# GPU worker
# --------------------------------------------------------------------------
def worker():
    items = load_sets()
    jobs = build_jobs(items)
    print(f"shard {SHARD}/{NSHARDS}: annotations {len(items)}, jobs {len(jobs)} "
          f"({Counter((j['rubric'], j['pairing']) for j in jobs)})", flush=True)
    from slicelens.backends.hf_llm import HFLLM
    t0 = time.time()
    judge = HFLLM(model=MODEL, max_new_tokens=400, load_in_4bit=os.environ.get("JUDGE_4BIT") == "1")
    print(f"judge loaded in {time.time() - t0:.0f}s", flush=True)

    # ---- A. the facts of each annotation (this shard's half), then a barrier ----
    fpath = f"{OUT}_facts_shard{SHARD}.json"
    mine = {}
    if os.path.exists(fpath):
        mine = json.load(open(fpath, encoding="utf-8")).get("facts", {})
    todo = [it for i, it in enumerate(items) if i % NSHARDS == SHARD]
    for i, (s, n, h, _t, _e) in enumerate(todo, 1):
        k = f"{s}|{n}"
        changed = False
        if not (k in mine and mine[k].get("gt_hash") == _hash(h)):
            rec = parse_json(judge.chat(FACT_SYSTEM, h)) or {}
            facts = [str(f).strip() for f in (rec.get("facts") or []) if str(f).strip()][:8]
            mine[k] = {"facts": facts or [h], "gt_hash": _hash(h), "fallback": not facts}
            changed = True
        if TYPES and (len(mine[k].get("types") or []) != len(mine[k]["facts"])
                      or mine[k].get("types_hash") != THASH):
            mine[k]["types"], mine[k]["types_fallback"] = type_facts(judge, mine[k]["facts"])
            mine[k]["types_hash"] = THASH
            changed = True
        if changed and (i % 10 == 0 or i == len(todo)):
            json.dump({"facts": mine, "complete": False}, open(fpath, "w", encoding="utf-8"),
                      indent=1, ensure_ascii=False)
            print(f"  facts {i}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    json.dump({"facts": mine, "complete": True}, open(fpath, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    facts = {}
    while True:                                   # barrier: every shard's facts on disk
        parts = [f"{OUT}_facts_shard{k}.json" for k in range(NSHARDS)]
        try:                                      # another shard may be mid-write
            loaded = [json.load(open(p, encoding="utf-8")) for p in parts if os.path.exists(p)]
        except ValueError:
            loaded = []
        if (len(loaded) == NSHARDS and all(x.get("complete") for x in loaded)
                and (not TYPES or all(len(v.get("types") or []) == len(v["facts"])
                                      and v.get("types_hash") == THASH
                                      for x in loaded for v in x["facts"].values()))):
            for x in loaded:
                facts.update(x["facts"])
            break
        time.sleep(20)
    print(f"facts for {len(facts)} annotations "
          f"(mean {sum(len(v['facts']) for v in facts.values()) / len(facts):.1f} per annotation)", flush=True)

    # ---- B. the verdicts (this shard's half of the jobs) ----
    src = open("slice_quality/examples/judge_descriptions.py", encoding="utf-8").read()
    ns = {}
    exec(re.search(r'JUDGE_SYSTEM = """.*?"""', src, re.S).group(0), ns)
    exec(re.search(r'JUDGE_USER = """.*?"""', src, re.S).group(0), ns)
    gt_text = {f"{s}|{n}": h for s, n, h, _t, _e in items}
    opath = f"{OUT}_shard{SHARD}.json"
    done = {}
    if os.path.exists(opath):
        for r in json.load(open(opath, encoding="utf-8"))["judgements"]:
            done[key(r)] = r
    mine_jobs = [j for i, j in enumerate(jobs) if i % NSHARDS == SHARD]
    out, failed, t1 = [], 0, time.time()
    for i, j in enumerate(mine_jobs, 1):
        fk = f"{j['set']}|{j['image']}"
        gh = facts[fk]["gt_hash"]
        if (key(j) in done and done[key(j)].get("cand_hash") == _hash(j["cand"])
                and done[key(j)].get("gt_hash") == gh and "score" in done[key(j)]
                and (j["rubric"] != "facts" or done[key(j)].get("verify_hash") == VHASH)):
            out.append(done[key(j)])
            continue
        r = {k: v for k, v in j.items() if k != "cand"}
        r.update(cand_hash=_hash(j["cand"]), gt_hash=gh, cand_words=len(j["cand"].split()),
                 gt_words=len(gt_text[fk].split()))
        if j["rubric"] == "facts":
            fl = facts[fk]["facts"]
            user = VERIFY_USER.format(facts="\n".join(f"{k + 1}. {f}" for k, f in enumerate(fl)),
                                      cand=j["cand"])
            for attempt in range(2):              # one retry if the count is wrong
                raw = judge.chat(VERIFY_SYSTEM, user if attempt == 0 else
                                 user + f"\n\nThere are exactly {len(fl)} facts: give exactly "
                                        f"{len(fl)} verdicts.")
                v = [str(x).strip().lower() for x in ((parse_json(raw) or {}).get("verdicts") or [])]
                if len(v) == len(fl) and all(x in VERDICTS for x in v):
                    break
            if len(v) != len(fl) or any(x not in VERDICTS for x in v):
                failed += 1
                r["error"] = raw[:200]
            else:
                cv = counted(v, facts[fk].get("types"))
                r.update(verdicts=v, n_facts=len(fl), n_counted=len(cv), score=score_of(cv),
                         score_all=score_of(v), verify_hash=VHASH)
        else:
            raw = judge.chat(ns["JUDGE_SYSTEM"], ns["JUDGE_USER"].format(gt=gt_text[fk], cand=j["cand"]))
            rec = parse_json(raw) or {}
            try:
                parts = [max(0.0, min(hi, float(rec[k]))) for k, hi in
                         (("completeness", 4), ("correctness", 4), ("specificity", 2))]
                r.update(score=round(sum(parts), 1), completeness=parts[0], correctness=parts[1],
                         specificity=parts[2])
            except Exception:
                failed += 1
                r["error"] = raw[:200]
        out.append(r)
        if i % 10 == 0 or i == len(mine_jobs):
            el = time.time() - t1
            print(f"  verdicts {i}/{len(mine_jobs)} | {el:.0f}s | {el / i:.1f}s per item | unparsed {failed}",
                  flush=True)
            json.dump({"judgements": out}, open(opath, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    json.dump({"judgements": out, "complete": True}, open(opath, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"CLAIMS_SHARD_DONE {SHARD}", flush=True)


# --------------------------------------------------------------------------
# merge + report (CPU)
# --------------------------------------------------------------------------
def merge():
    import statistics as st
    rows, facts = [], {}
    for k in range(NSHARDS):
        rows += json.load(open(f"{OUT}_shard{k}.json", encoding="utf-8"))["judgements"]
        facts.update(json.load(open(f"{OUT}_facts_shard{k}.json", encoding="utf-8"))["facts"])
    rows = [r for r in rows if r["kind"] in KINDS]
    for r in rows:                                # the score from verdicts + types, here
        if r.get("rubric") == "facts" and r.get("verdicts"):
            cv = counted(r["verdicts"], facts.get(f"{r['set']}|{r['image']}", {}).get("types"))
            r.update(n_counted=len(cv), score=score_of(cv), score_all=score_of(r["verdicts"]))
    ok = [r for r in rows if "score" in r]
    lines = []
    say = lambda s="": (lines.append(s), print(s))
    m = lambda xs: st.mean(xs) if xs else float("nan")
    G = lambda **kw: [r for r in ok if all(r.get(k) == v for k, v in kw.items())]

    say("=" * 78)
    say("VOTO PER FATTI (0-10): i fatti dell'annotazione, verificati uno per uno")
    say("10 * (dichiarati + 0.5 * in parte - contraddetti) / n. fatti   -- calcolato, non chiesto")
    say("=" * 78)
    nf = [len(v["facts"]) for v in facts.values()]
    say(f"annotazioni: {len(facts)} | fatti per annotazione: media {m(nf):.1f}, "
        f"min {min(nf)}, max {max(nf)} | estrazione fallita (testo intero usato): "
        f"{sum(1 for v in facts.values() if v.get('fallback'))}")
    say(f"giudizi validi: {len(ok)}/{len(rows)}")
    ft = Counter(t for v in facts.values() for t in (v.get("types") or []))
    if ft:
        say(f"tipo dei fatti: " + ", ".join(f"{t} {ft[t]}" for t in FTYPES) +
            f" | tipizzazione fallita: {sum(1 for v in facts.values() if v.get('types_fallback'))}"
            f" | annotazioni senza fatti sulla predizione (contano tutti): "
            f"{sum(1 for v in facts.values() if v.get('types') and 'prediction' not in v['types'])}")
        say("il voto conta i fatti sulla predizione; 'tutti i fatti' conta anche giudizi complessivi "
            "e osservazioni sulla ground truth")
    say()
    say(f"{'':28s}{'genuina':>9s}{'controllo':>11s}{'scarto':>9s}{'n':>5s}")
    summary = {}
    for s in ("voc", "synth"):
        for kind in KINDS:
            g, c = G(set=s, kind=kind, pairing="genuine", rubric="facts"), G(set=s, kind=kind, pairing="control", rubric="facts")
            if not g:
                continue
            summary[f"{s}_{kind}"] = dict(genuine=m([r["score"] for r in g]), control=m([r["score"] for r in c]), n=len(g))
            say(f"{s + ' ' + kind:28s}{m([r['score'] for r in g]):9.2f}{m([r['score'] for r in c]):11.2f}"
                f"{m([r['score'] for r in g]) - m([r['score'] for r in c]):9.2f}{len(g):5d}")
    # every annotated image, wherever it comes from (VOC val, VOC train36, synthetic)
    for kind in KINDS:
        g, c = G(kind=kind, pairing="genuine", rubric="facts"), G(kind=kind, pairing="control", rubric="facts")
        if g:
            summary[f"all_{kind}"] = dict(genuine=m([r["score"] for r in g]), control=m([r["score"] for r in c]), n=len(g))
            say(f"{'tutte ' + kind:28s}{m([r['score'] for r in g]):9.2f}{m([r['score'] for r in c]):11.2f}"
                f"{m([r['score'] for r in g]) - m([r['score'] for r in c]):9.2f}{len(g):5d}")
    summary_all = {}
    for s in ("voc", "synth", None):
        for kind in KINDS:
            kw = dict(kind=kind, pairing="genuine", rubric="facts", **({"set": s} if s else {}))
            g = G(**kw)
            c = G(**dict(kw, pairing="control"))
            if g:
                summary_all[f"{s or 'all'}_{kind}"] = dict(genuine=m([r["score_all"] for r in g]),
                                                         control=m([r["score_all"] for r in c]), n=len(g))
                say(f"{(s or 'tutte') + ' tutti i fatti':28s}"
                    f"{m([r['score_all'] for r in g]):9.2f}{m([r['score_all'] for r in c]):11.2f}"
                    f"{m([r['score_all'] for r in g]) - m([r['score_all'] for r in c]):9.2f}{len(g):5d}")
    v1 = {}
    pv = f"{OUT}_v1.json"
    if os.path.exists(pv):
        say()
        say("-- prima del glossario (v1, tutti i fatti) -> ora " + "-" * 27)
        old = [r for r in json.load(open(pv, encoding="utf-8"))["judgements"]
               if r.get("rubric") == "facts" and "score" in r and r["kind"] in KINDS]
        O = lambda **kw: [r for r in old if all(r.get(k) == v for k, v in kw.items())]
        for s in ("voc", "synth", None):
            for pairing in ("genuine", "control", "duplicate"):
                kw = dict(pairing=pairing, **({"set": s} if s else {}))
                a, b = O(**kw), G(rubric="facts", **kw)
                if a and b:
                    v1[f"{s or 'all'}_{pairing}"] = dict(v1=m([r["score"] for r in a]),
                                                         all_facts=m([r["score_all"] for r in b]),
                                                         now=m([r["score"] for r in b]), n=len(b))
                    say(f"   {(s or 'tutte'):6s} {pairing:9s} v1 {m([r['score'] for r in a]):5.2f}  ->  "
                        f"glossario, tutti i fatti {m([r['score_all'] for r in b]):5.2f}  ->  "
                        f"+ solo fatti sulla predizione {m([r['score'] for r in b]):5.2f}  (n={len(b)})")
        for s in ("voc", "synth"):
            a = Counter(v for r in O(set=s, pairing="genuine") for v in r["verdicts"])
            b = Counter(v for r in G(set=s, pairing="genuine", rubric="facts") for v in r["verdicts"])
            ta, tb = sum(a.values()), sum(b.values())
            if ta and tb:
                v1[f"{s}_verdicts"] = dict(v1={k: a[k] / ta for k in VERDICTS}, now={k: b[k] / tb for k in VERDICTS})
                say(f"   {s:6s} verdetti v1:  " + "  ".join(f"{k} {100 * a[k] / ta:3.0f}%" for k in VERDICTS))
                say(f"   {s:6s} verdetti ora: " + "  ".join(f"{k} {100 * b[k] / tb:3.0f}%" for k in VERDICTS))
    say()
    say("-- verdetti sui fatti (coppie genuine) " + "-" * 38)
    for s in ("voc", "synth"):
        for kind in KINDS:
            g = G(set=s, kind=kind, pairing="genuine", rubric="facts")
            if g:
                c = Counter(v for r in g for v in r["verdicts"])
                tot = sum(c.values())
                say(f"{s + ' ' + kind:20s} " + "  ".join(f"{v} {100 * c[v] / tot:4.0f}%" for v in VERDICTS))
    say()
    say("-- la lunghezza conta? " + "-" * 55)
    say("1) stesso testo scritto due volte (lunghezza doppia, contenuto identico):")
    dup = {}
    for rub in ("facts", "old"):
        d = []
        for r in G(pairing="genuine", rubric=rub):
            twin = [x for x in G(pairing="duplicate", rubric=rub)
                    if (x["set"], x["image"], x["kind"]) == (r["set"], r["image"], r["kind"])]
            if twin:
                d.append(twin[0]["score"] - r["score"])
        if d:
            dup[rub] = dict(mean=m(d), absmean=m([abs(x) for x in d]), same=sum(1 for x in d if abs(x) < 1e-9),
                            n=len(d), up=sum(1 for x in d if x > 0), down=sum(1 for x in d if x < 0))
            say(f"   {'voto per fatti' if rub == 'facts' else 'rubrica attuale':18s} variazione media "
                f"{dup[rub]['mean']:+.2f}, media assoluta {dup[rub]['absmean']:.2f}, invariato in "
                f"{dup[rub]['same']}/{len(d)} (su {dup[rub]['up']}, giu' {dup[rub]['down']})")
    say("2) correlazione voto / n. parole del testo (Spearman):")
    corr = {}
    try:
        from scipy import stats
        for rub in ("facts", "old"):
            for pairing in ("genuine", "control"):
                for kind in KINDS:
                    g = G(set="voc", kind=kind, pairing=pairing, rubric=rub)
                    if len(g) > 5:
                        rho, p = stats.spearmanr([r["cand_words"] for r in g], [r["score"] for r in g])
                        corr[f"{rub}_{pairing}_{kind}"] = (rho, p, len(g))
                        say(f"   {'voto per fatti' if rub == 'facts' else 'rubrica attuale':18s} "
                            f"{kind:9s} {pairing:8s} rho {rho:+.2f} (p={p:.3f}, n={len(g)})")
    except ImportError:
        pass
    say()
    say("-- VOC per split e sintetiche per meta' " + "-" * 38)
    split = {}
    for kind in KINDS:
        for sp in ("val", "train"):
            g = G(set="voc", kind=kind, pairing="genuine", rubric="facts", split=sp)
            if g:
                split[f"{kind}_{sp}"] = (m([r["score"] for r in g]), len(g))
                say(f"   voc {kind:9s} {sp:6s} {m([r['score'] for r in g]):5.2f}  n={len(g)}")
        for good in (True, False):
            g = [r for r in G(set="synth", kind=kind, pairing="genuine", rubric="facts") if r.get("declared_good") == good]
            if g:
                split[f"synth_{kind}_{'good' if good else 'bad'}"] = (m([r["score"] for r in g]), len(g))
                say(f"   synth {kind:9s} {'dichiarate corrette' if good else 'dichiarate sbagliate':20s} "
                    f"{m([r['score'] for r in g]):5.2f}  n={len(g)}")
    json.dump({"judgements": rows, "facts": facts, "summary": summary, "summary_all": summary_all,
               "v1": v1, "fact_types": dict(ft), "duplicate": dup,
               "correlation": {k: list(v) for k, v in corr.items()}, "split": split, "report": lines},
              open(f"{OUT}.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    open(f"{OUT}.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"\nscritto {OUT}.json / .txt\nCLAIMS_MERGE_DONE", flush=True)


if __name__ == "__main__":
    merge() if MERGE else worker()
