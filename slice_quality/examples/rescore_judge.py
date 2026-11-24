"""Assemble the scored file the HTML reads. CPU only, no GPU pass.

The score is Completeness + Correctness + Specificity, which is already a 0-10
scale because the rubric is 4 + 4 + 2. Using the sum rather than the judge's own
holistic verdict is a deliberate choice about what a reader can check:

  * the two disagree, systematically. The holistic verdict sat on average 1.02
    points BELOW the sum and matched it on only 30% of the cards. Both are real
    judgements -- the judge is harder on a text taken as a whole than on its parts
    -- but a card showing "1.0/10" beside "completeness 1 + correctness 1 +
    specificity 2" reads as a bug, whatever the footnote says.
  * the sum is auditable. Every point in it is attached to a criterion the reader
    can see and argue with, which is what a score in a thesis has to be.

The holistic verdict is kept in every record as ``score_holistic`` and summarised
below, so nothing is lost; it separates genuine from control pairs about as well
(gap 4.27 against 4.05), so little is given up by not leading with it.

Run:  python slice_quality/examples/rescore_judge.py
Env: SRC (default judge_descriptions_v2), OUT (default judge_descriptions_scored).
"""
import json
import os
from collections import Counter

SRC = os.environ.get("SRC", "outputs/slice_quality/judge_descriptions_v2")
OUT = os.environ.get("OUT", "outputs/slice_quality/judge_descriptions_scored")

rows = json.load(open(SRC + ".json", encoding="utf-8"))["judgements"]
for r in rows:
    if "completeness" in r:
        r["score"] = round(r["completeness"] + r["correctness"] + r["specificity"], 1)

ok = [r for r in rows if "score" in r and "completeness" in r]
report = []


def say(line=""):
    print(line, flush=True)
    report.append(line)


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def sel(kind, pairing, split=None):
    return [r for r in ok if r["kind"] == kind and r["pairing"] == pairing
            and (split is None or r["split"] == split)]


say("=" * 74)
say("SCORE 0-10 = Completezza 4 + Correttezza 4 + Specificita 2 (Gemma locale)")
say("=" * 74)
say(f"{'':22s}{'genuine':>10s}{'controllo':>11s}{'scarto':>9s}{'n':>6s}")
for kind in ("narrative", "pattern"):
    g, c = sel(kind, "genuine"), sel(kind, "control")
    say(f"{kind:22s}{mean(r['score'] for r in g):10.2f}"
        f"{mean(r['score'] for r in c):11.2f}"
        f"{mean(r['score'] for r in g) - mean(r['score'] for r in c):9.2f}{len(g):6d}")
say()
say("Il controllo e' la STESSA annotazione umana contro la descrizione di un'ALTRA")
say("immagine: e' il pavimento del rumore, e solo lo scarto vale come evidenza.")

say()
say("-- i tre criteri che il giudice ha pesato " + "-" * 31)
say(f"{'':30s}{'completezza':>14s}{'correttezza':>14s}{'specificita':>14s}")
for kind in ("narrative", "pattern"):
    for pairing in ("genuine", "control"):
        g = sel(kind, pairing)
        say(f"{kind + ' ' + pairing:30s}{mean(r['completeness'] for r in g):14.2f}"
            f"{mean(r['correctness'] for r in g):14.2f}"
            f"{mean(r['specificity'] for r in g):14.2f}")
say(f"{'max':30s}{4:14d}{4:14d}{2:14d}")
say("La specificita' paga 1.26/2 anche sui testi di un'altra immagine: in parte")
say("misura lo stile, non l'accordo. Completezza e correttezza no (0.17 e 0.21).")

say()
say("-- il voto d'insieme del giudice, NON usato come punteggio " + "-" * 14)
for kind in ("narrative", "pattern"):
    g, c = sel(kind, "genuine"), sel(kind, "control")
    say(f"{kind:14s} genuine {mean(r['score_holistic'] for r in g):5.2f} | "
        f"controllo {mean(r['score_holistic'] for r in c):5.2f}")
gaps = [r["score_holistic"] - r["score"] for r in ok]
say(f"scarto rispetto alla somma: media {mean(gaps):+.2f} | sotto in "
    f"{sum(1 for x in gaps if x < 0)} casi su {len(gaps)}, sopra in "
    f"{sum(1 for x in gaps if x > 0)}, uguale in {sum(1 for x in gaps if x == 0)}")
say("E' un secondo giudizio, non un errore di somma: il giudice e' piu' severo sul")
say("testo intero che criterio per criterio. In pagina si usa la somma, perche' ogni")
say("suo punto e' verificabile guardando i criteri accanto.")

say()
say("-- per split " + "-" * 60)
for kind in ("narrative", "pattern"):
    for sp in ("val", "train"):
        g, c = sel(kind, "genuine", sp), sel(kind, "control", sp)
        if g:
            say(f"{kind:12s} {sp:6s} genuine {mean(r['score'] for r in g):5.2f} | "
                f"controllo {mean(r['score'] for r in c):5.2f} | n={len(g)}")

say()
say("-- verdetti scritti a mano (solo val) " + "-" * 36)
for kind in ("narrative", "pattern"):
    for v in ("ALTA", "PARZIALE", "BASSA"):
        g = [r for r in sel(kind, "genuine") if r.get("verdict") == v]
        if g:
            say(f"{kind:12s} {v:9s} score medio {mean(r['score'] for r in g):5.2f} | n={len(g)}")

say()
say("-- distribuzione (narrative, genuine) " + "-" * 36)
c = Counter(int(r["score"]) for r in sel("narrative", "genuine"))
for k in sorted(c):
    say(f"  {k:2d}/10 {'#' * c[k]} {c[k]}")

json.dump({"judgements": rows, "report": report},
          open(OUT + ".json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
with open(OUT + ".txt", "w", encoding="utf-8") as f:
    f.write("\n".join(report) + "\n")
say()
say(f"saved {OUT}.json / {OUT}.txt")
