"""Giudica le 21 narrative del set sintetico contro le annotazioni umane. GPU.

Stessa rubrica e stesso identico prompt di judge_descriptions.py -- letti da
quel file invece che ricopiati, cosi' non possono divergere.

Cosa rende questo set diverso da VOC: dieci delle ventuno annotazioni dichiarano
la predizione PERFETTA. Su VOC ogni annotazione descriveva un difetto, quindi si
poteva misurare solo quanto la pipeline vede; qui si misura anche l'accordo su
"non c'e' niente da dire", che e' meta' del lavoro di un sistema del genere.

Il controllo contro il caso resta obbligatorio, e su questo set e' piu' severo
del solito: accoppiando un'annotazione all'immagine sbagliata si mette spesso un
"la predizione e' perfetta" contro un testo che elenca errori catastrofici, o
viceversa. Se il giudice non distingue quello, non distingue nulla.

Run (GPU 5):
  JUDGE_4BIT=1 PYTHONPATH=/workspace python slice_quality/examples/judge_synth.py
Env: JUDGE_MODEL, JUDGE_4BIT, SEED, SRC, NARR, OUT.
"""
import json
import os
import random
import re
import time
from collections import Counter

SRC = os.environ.get("SRC", "datasets/cropped_synthetic_masks")
NARR = os.environ.get("NARR", "outputs/slice_quality/synth/narratives_synth.json")
DESC = os.environ.get("DESC", "outputs/slice_quality/synth/descriptions_synth.json")
OUT = os.environ.get("OUT", "outputs/slice_quality/synth/judge_synth")
MODEL = os.environ.get("JUDGE_MODEL", "/models/gemma")
SEED = int(os.environ.get("SEED", "0"))

# lo stesso prompt del giudice principale, letto dal suo file
_src = open("slice_quality/examples/judge_descriptions.py", encoding="utf-8").read()
_ns = {}
exec(re.search(r'JUDGE_SYSTEM = """.*?"""', _src, re.S).group(0), _ns)
exec(re.search(r'JUDGE_USER = """.*?"""', _src, re.S).group(0), _ns)
JUDGE_SYSTEM, JUDGE_USER = _ns["JUDGE_SYSTEM"], _ns["JUDGE_USER"]

_FIELDS = [("completeness", 4), ("correctness", 4), ("specificity", 2)]


def parse_json(raw):
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def clean(rec):
    if not isinstance(rec, dict):
        return None
    out, total = {}, 0.0
    for k, hi in _FIELDS:
        try:
            v = max(0.0, min(hi, float(rec.get(k))))
        except (TypeError, ValueError):
            return None
        out[k] = v
        total += v
    out["score"] = round(total, 1)          # la somma, come in pagina
    try:
        out["score_holistic"] = float(rec.get("score"))
    except (TypeError, ValueError):
        out["score_holistic"] = None
    out["rationale"] = str(rec.get("rationale", ""))[:300]
    return out


mp = {int(k): v.replace(".jpg", "") for k, v in
      json.load(open(os.path.join(SRC, "voc_patch__id_matches.json"),
                     encoding="utf-8")).items()}
human, order = {}, []
for line in open(os.path.join(SRC, "answer_gts.jsonl"), encoding="utf-8"):
    if line.strip():
        o = json.loads(line)
        nm = mp[o["img_idx"]]
        human[nm] = o["content"].strip()
        order.append(nm)
nar = {k: v["text"] for k, v in json.load(open(NARR, encoding="utf-8")).items()}
# il ramo dei pattern (cluster + SliceLens indotti), se e' stato generato
patt = {}
if os.path.exists(DESC):
    patt = {k: v.get("text", "") for k, v in json.load(open(DESC, encoding="utf-8")).items()}
texts = {"narrative": nar, "pattern": patt}

# quale meta': l'annotazione dichiara la predizione corretta?
OK_RE = re.compile(r"perfect|no mistake|correctly|accurate|is correct|precisely", re.I)
BAD_RE = re.compile(r"wrong|miss|hallucinat|unusable|error", re.I)
good = {n: bool(OK_RE.search(t)) and not bool(BAD_RE.search(t))
        for n, t in human.items()}
print(f"immagini: {len(order)} | dichiarate corrette: {sum(good.values())} | "
      f"dichiarate sbagliate: {len(order) - sum(good.values())}", flush=True)

rng = random.Random(SEED)
shift = rng.randrange(1, len(order))
control_of = {n: order[(i + shift) % len(order)] for i, n in enumerate(order)}

jobs = []
for n in order:
    for kind, table in texts.items():
        if not table.get(n):
            continue
        jobs.append({"image": n, "kind": kind, "pairing": "genuine",
                     "gt": human[n], "cand": table[n]})
        c = control_of[n]
        if table.get(c):
            jobs.append({"image": n, "kind": kind, "pairing": "control",
                         "control_from": c, "gt": human[n], "cand": table[c]})
rng.shuffle(jobs)
print(f"giudizi da fare: {len(jobs)}", flush=True)

from slicelens.backends.hf_llm import HFLLM

t0 = time.time()
print(f"carico {MODEL} ...", flush=True)
judge = HFLLM(model=MODEL, max_new_tokens=256,
              load_in_4bit=os.environ.get("JUDGE_4BIT") == "1")
print(f"caricato in {time.time() - t0:.0f}s", flush=True)

results, failed = [], 0
for i, j in enumerate(jobs, 1):
    raw = judge.chat(JUDGE_SYSTEM, JUDGE_USER.format(gt=j["gt"], cand=j["cand"]))
    rec = clean(parse_json(raw))
    if rec is None:
        failed += 1
        rec = {"error": raw[:200]}
    r = dict(j)
    r.pop("gt")
    r["declared_good"] = good[j["image"]]
    r.update(rec)
    results.append(r)
    if i % 10 == 0 or i == len(jobs):
        print(f"  {i}/{len(jobs)} | {time.time() - t0:.0f}s | non parsati: {failed}",
              flush=True)

ok = [r for r in results if "score" in r]
report = []


def say(s=""):
    print(s, flush=True)
    report.append(s)


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def sel(pairing, declared=None, kind="narrative"):
    return [r for r in ok if r["pairing"] == pairing and r["kind"] == kind
            and (declared is None or r["declared_good"] == declared)]


say("=" * 74)
say("SET SINTETICO -- 0-10 = Completezza 4 + Correttezza 4 + Specificita 2")
say("=" * 74)
say(f"giudicati: {len(ok)}/{len(results)} ({failed} non parsati)")
say()
say(f"{'':34s}{'genuine':>10s}{'controllo':>11s}{'scarto':>9s}{'n':>5s}")
for kind in ("narrative", "pattern"):
    for lbl, dec in (("tutte", None), ("dichiarate corrette", True),
                     ("dichiarate sbagliate", False)):
        g, c = sel("genuine", dec, kind), sel("control", dec, kind)
        if not g:
            continue
        say(f"{kind + ' ' + lbl:34s}{mean(r['score'] for r in g):10.2f}"
            f"{mean(r['score'] for r in c):11.2f}"
            f"{mean(r['score'] for r in g) - mean(r['score'] for r in c):9.2f}{len(g):5d}")
    say()
say()
say("-- per criterio (coppie genuine) " + "-" * 40)
say(f"{'':28s}" + "".join(f"{k[:12]:>14s}" for k, _h in _FIELDS))
for lbl, dec in (("dichiarate corrette", True), ("dichiarate sbagliate", False)):
    g = sel("genuine", dec)
    say(f"{lbl:28s}" + "".join(f"{mean(r[k] for r in g):14.2f}" for k, _h in _FIELDS))
say(f"{'massimo':28s}" + "".join(f"{h:14d}" for _k, h in _FIELDS))

say()
say("-- ogni immagine " + "-" * 56)
for n in order:
    g = [r for r in ok if r["image"] == n and r["pairing"] == "genuine"
         and r["kind"] == "narrative"]
    pt = [r for r in ok if r["image"] == n and r["pairing"] == "genuine"
          and r["kind"] == "pattern"]
    if g:
        say(f"  {n}  {'corretta ' if good[n] else 'sbagliata'}  narr {g[0]['score']:4.1f}"
            f"{'  patt ' + format(pt[0]['score'], '4.1f') if pt else ''}   "
            f"{g[0]['rationale'][:70]}")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump({"judgements": results, "report": report},
          open(OUT + ".json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
open(OUT + ".txt", "w", encoding="utf-8").write("\n".join(report) + "\n")
say(f"\nsalvato {OUT}.json / .txt")
print("JUDGE_SYNTH_DONE", flush=True)
