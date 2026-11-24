"""The facts score with verdicts written by hand, on the facts of judge_claims.py.

Why. The automatic judge (judge_claims.py, Gemma-4-12B in 4-bit) misses paraphrases
even with the glossary in its prompt: read by hand, the narrative states about 8.3
on the 43 val images where the judge gives 7.5. Here the same facts (and the same
fact types) of every annotation are checked against the narrative by a careful
reader instead -- in this project Claude, the assistant the pipeline was built with,
declared as such -- so that only the verdicts change, and the automatic judge's
agreement with them is measured fact by fact.

Bias control. The reader is not blind to the pipeline, so:
  * every genuine pairing is mixed with its control pairing (the narrative of
    ANOTHER image, the same derangement judge_claims.py used), shuffled, with no
    image name and no pairing label on the item;
  * the verdict scale and its rules are those of judge_claims.py VERIFY_SYSTEM
    (stated / partial / contradicted / absent; extra content ignored unless it
    contradicts; "contradicted" only when both cannot be true; length irrelevant);
  * the score is computed here, with the same formula and the same counted facts.
The duplicate test is not repeated: a reader gives a text written twice the same
verdicts by construction, so it would measure nothing.

Run (CPU, local files):
  STEP=export python slice_quality/examples/judge_claims_manual.py
      -> <OUT_DIR>/items.jsonl (blinded, shuffled), <OUT_DIR>/key.json
  ... verdicts written to <OUT_DIR>/verdicts.jsonl: {"item": n, "verdicts": [...]}
  STEP=score python slice_quality/examples/judge_claims_manual.py
      -> <SQ_OUT>/judge_claims_manual.{json,txt}
"""
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from judge_claims import VERDICTS, _hash, counted, score_of  # noqa: E402

SQ_OUT = os.environ.get("SQ_OUT", "outputs/slice_quality_inst")
OUT_DIR = os.environ.get("OUT_DIR", f"{SQ_OUT}/claims_manual")
STEP = os.environ.get("STEP", "export")
SEED = int(os.environ.get("SEED", "7"))


def _narratives():
    n = {("voc", k): v["text"] for k, v in json.load(open(f"{SQ_OUT}/error_narratives.json", encoding="utf-8")).items()}
    n.update({("synth", k): v["text"] for k, v in
              json.load(open(f"{SQ_OUT}/synth/narratives_synth.json", encoding="utf-8")).items()})
    return n


def export():
    d = json.load(open(f"{SQ_OUT}/judge_claims.json", encoding="utf-8"))
    narr = _narratives()
    rows = [r for r in d["judgements"] if r.get("rubric") == "facts" and r["kind"] == "narrative"
            and r["pairing"] in ("genuine", "control")]
    items = []
    for r in rows:
        src = r["control_from"] if r["pairing"] == "control" else r["image"]
        items.append(dict(set=r["set"], image=r["image"], pairing=r["pairing"], cand_from=src,
                          facts=d["facts"][f"{r['set']}|{r['image']}"]["facts"],
                          cand=narr[(r["set"], src)]))
    random.Random(SEED).shuffle(items)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/items.jsonl", "w", encoding="utf-8") as f:
        for i, it in enumerate(items):
            f.write(json.dumps({"item": i, "facts": it["facts"], "cand": it["cand"]}, ensure_ascii=False) + "\n")
    json.dump({str(i): {k: it[k] for k in ("set", "image", "pairing", "cand_from")} for i, it in enumerate(items)},
              open(f"{OUT_DIR}/key.json", "w", encoding="utf-8"), indent=1)
    print(f"{len(items)} items -> {OUT_DIR}/items.jsonl")


def score():
    import statistics as st
    d = json.load(open(f"{SQ_OUT}/judge_claims.json", encoding="utf-8"))
    key = json.load(open(f"{OUT_DIR}/key.json", encoding="utf-8"))
    items = {json.loads(l)["item"]: json.loads(l) for l in open(f"{OUT_DIR}/items.jsonl", encoding="utf-8")}
    ver = {}
    for l in open(f"{OUT_DIR}/verdicts.jsonl", encoding="utf-8"):
        if l.strip():
            o = json.loads(l)
            ver[o["item"]] = o
    gem = {(r["set"], r["image"], r["pairing"]): r for r in d["judgements"]
           if r.get("rubric") == "facts" and r["kind"] == "narrative"}
    lines = []
    say = lambda s="": (lines.append(s), print(s))
    m = lambda xs: st.mean(xs) if xs else float("nan")
    rows, bad = [], []
    for i, it in items.items():
        v = ver.get(i)
        if not v or len(v["verdicts"]) != len(it["facts"]) or any(x not in VERDICTS for x in v["verdicts"]):
            bad.append(i)
            continue
        k = key[str(i)]
        ty = d["facts"][f"{k['set']}|{k['image']}"].get("types")
        cv = counted(v["verdicts"], ty)
        g = gem.get((k["set"], k["image"], k["pairing"]), {})
        # Gemma judged the narrative of its own run: when the narrative has been
        # regenerated since (v5), its verdicts are about another text
        same = g.get("cand_hash") == _hash(it["cand"]) if g.get("cand_hash") else None
        rows.append(dict(k, item=i, verdicts=v["verdicts"], note=v.get("note", ""), n_facts=len(v["verdicts"]),
                         n_counted=len(cv), score=score_of(cv), score_all=score_of(v["verdicts"]),
                         gemma_verdicts=g.get("verdicts"), gemma_score=g.get("score"), gemma_same_text=same,
                         split=g.get("split"), declared_good=g.get("declared_good")))
    say("=" * 78)
    say("VOTO PER FATTI, verdetti scritti a mano (Claude) sui fatti di judge_claims.py")
    say("stessi fatti, stessi tipi, stessa formula; coppie vere e di controllo mescolate, senza nome")
    say("=" * 78)
    say(f"elementi giudicati: {len(rows)}/{len(items)}" + (f" | mancanti o non validi: {bad[:10]}" if bad else ""))
    say()
    say(f"{'':26s}{'a mano':>8s}{'controllo':>11s}{'scarto':>8s}{'Gemma v2':>10s}{'n':>5s}")
    G = lambda **kw: [r for r in rows if all(r.get(k) == v for k, v in kw.items())]
    summary = {}
    for lab, kw in (("voc val", dict(set="voc", split="val")), ("voc train", dict(set="voc", split="train")),
                    ("voc", dict(set="voc")), ("synth", dict(set="synth")), ("tutte", {})):
        g, c = G(pairing="genuine", **kw), G(pairing="control", **kw)
        if g:
            summary[lab.replace(" ", "_")] = dict(genuine=m([r["score"] for r in g]),
                                                  control=m([r["score"] for r in c]),
                                                  gemma=m([r["gemma_score"] for r in g if r["gemma_score"] is not None]),
                                                  gemma_control=m([r["gemma_score"] for r in c if r["gemma_score"] is not None]),
                                                  n=len(g))
            s_ = summary[lab.replace(" ", "_")]
            say(f"{lab:26s}{s_['genuine']:8.2f}{s_['control']:11.2f}{s_['genuine'] - s_['control']:8.2f}"
                f"{s_['gemma']:10.2f}{len(g):5d}")
    say()
    say("-- verdetti (coppie vere) " + "-" * 52)
    verd = {}
    for s in ("voc", "synth"):
        c = Counter(v for r in G(pairing="genuine", set=s) for v in r["verdicts"])
        tot = sum(c.values())
        if tot:
            verd[s] = {k: c[k] / tot for k in VERDICTS}
            say(f"   {s:6s} " + "  ".join(f"{k} {100 * c[k] / tot:3.0f}%" for k in VERDICTS))
    say()
    stale = sum(1 for r in rows if r["gemma_same_text"] is False)
    say("-- accordo fatto per fatto con Gemma v2 " + "-" * 38)
    if stale:
        say(f"   solo sugli elementi in cui Gemma ha giudicato lo STESSO testo: {stale} di {len(rows)} "
            f"hanno una narrativa rigenerata dopo il suo giudizio e sono esclusi")
    agree = {}
    for pairing in ("genuine", "control"):
        pairs = [(a, b) for r in G(pairing=pairing) if r["gemma_verdicts"] and r["gemma_same_text"] is not False
                 for a, b in zip(r["verdicts"], r["gemma_verdicts"])]
        if not pairs:
            continue
        po = sum(a == b for a, b in pairs) / len(pairs)
        ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
        pe = sum(ca[k] * cb[k] for k in VERDICTS) / len(pairs) ** 2
        kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
        conf = Counter(pairs)
        agree[pairing] = dict(n=len(pairs), agreement=po, kappa=kappa,
                              confusion={f"{a}|{b}": conf[(a, b)] for a in VERDICTS for b in VERDICTS if conf[(a, b)]})
        say(f"   {pairing:8s} fatti {len(pairs):4d} | accordo {100 * po:4.1f}% | kappa di Cohen {kappa:.2f}")
        if pairing == "genuine":
            say("   (righe: a mano, colonne: Gemma)" + "".join(f"{k[:6]:>9s}" for k in VERDICTS))
            for a in VERDICTS:
                say(f"   {a:31s}" + "".join(f"{conf[(a, b)]:9d}" for b in VERDICTS))
    json.dump({"rows": rows, "summary": summary, "verdicts": verd, "agreement": agree,
               "gemma_stale_items": stale, "report": lines},
              open(f"{SQ_OUT}/judge_claims_manual.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    open(f"{SQ_OUT}/judge_claims_manual.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"\nscritto {SQ_OUT}/judge_claims_manual.json / .txt")


if __name__ == "__main__":
    export() if STEP == "export" else score()
