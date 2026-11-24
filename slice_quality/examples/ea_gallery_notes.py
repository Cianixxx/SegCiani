"""Notes and image order for the val-EA gallery: what was changed in each edited image
and what the change did to the segmenter. CPU only, after ea_planted_eval.py.

For every image carrying a Pascal-EA material edit, one line: the subject's class, the
material it was rendered in, the subject's recall on the original and on the edited
image (pascal_ea_compare.py) and, for a victim (recall drop >= DROP), whether the
pipeline put it in a certified slice (planted_eval.json).

The order makes the capped gallery REPRESENTATIVE, not a list of failures: first the
victims (largest drop first), then N_UNCHANGED edited images drawn at random (seed 0)
among those whose subject recall moved by less than 0.05 -- most edits do not hurt the
segmenter, and a gallery of victims alone hides that --, then the other edited images.
Each note's label says which section the card is in, and the header line says how
the gallery is composed. MAX_IMAGES of regen_gallery.py must cover both sections.

JUDGE=<judge_ea_patterns.json> (off by default) adds the Gemma "do the patterns capture
the edit?" badges to the victims and their means to the header.

Run:  python slice_quality/examples/ea_gallery_notes.py
Out:  outputs/slice_quality_ea/gallery_notes.json
      {"order": [...], "notes": {stem: [label, text]}, "badges": {stem: [text, tip]}, "header": html}
"""
import html
import json
import os
import random

EA_RUN = os.environ.get("EA_RUN", "outputs/slice_quality_ea")
EAD = os.environ.get("EA", "/ds/pascal_ea_material")
CMP = os.environ.get("CMP", "outputs/pascal_ea/per_image.json")
DROP = float(os.environ.get("DROP", "0.30"))
N_UNCHANGED = int(os.environ.get("N_UNCHANGED", "100"))

edited = [l.strip() for l in open("/data/voc2012_ea/edited_stems.txt") if l.strip()]
rows = {r["stem"]: r for r in json.load(open(CMP))["rows"]}
lab = json.load(open(f"{EAD}/labels.json"))["images"]
ev = json.load(open(f"{EA_RUN}/planted_eval.json"))
victim = {v.split("#")[0] for v in ev["victims"]}
covered = {v.split("#")[0] for v in ev["covered"]}

notes, drops = {}, {}
for s in edited:
    l, r = lab.get(s, {}), rows.get(s, {})
    cls, mat = l.get("voc_class", "?"), l.get("material_v3") or "?"
    o, e = r.get("orig", {}).get("rec"), r.get("v3", {}).get("rec")
    t = f"il soggetto ({cls}) è stato reso in «{mat}»"
    if o is not None and e is not None:
        drops[s] = e - o
        t += f"; recall del soggetto {o:.2f} sull'originale → {e:.2f} sulla modificata"
    if s in victim:
        t += (". VITTIMA: l'edit ha fatto perdere il soggetto al segmentatore; la pipeline "
              + ("lo mette in una slice certificata" if s in covered
                 else "NON lo mette in nessuna slice certificata"))
    elif s in drops and abs(drops[s]) < 0.05:
        t += ". L'edit non ha cambiato la segmentazione del soggetto"
    notes[s] = ["modifica Pascal-EA", t]

# ---- Gemma: do the systematic patterns capture the modification? ----------
badges, header = {}, None
JUDGE = os.environ.get("JUDGE", "")
if JUDGE and os.path.exists(JUDGE):
    js = [r for r in json.load(open(JUDGE, encoding="utf-8"))["judgements"] if "score" in r]
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    by = {p: [r for r in js if r["pairing"] == p] for p in ("genuine", "control", "original")}
    for r in by["genuine"]:
        c, k, g = r["coverage"], r["cause"], r["score"]
        badges[r["image"]] = [
            f"modifica: copertura {c:.0f}/5 · causa {k:.0f}/5 · Gemma {g:.0f}/10",
            "Giudizio di Gemma su quanto i pattern sistematici catturano la modifica Pascal-EA. "
            "Copertura: un pattern riguarda l'oggetto modificato e l'errore che la modifica ha "
            "causato. Causa: la situazione nominata corrisponde al cambio di materiale. "
            f"Copertura + causa = {c + k:.0f}/10; voto complessivo di Gemma {g:.0f}/10."
            + (" Nessun pattern sull'immagine: 1 d'ufficio." if r.get("auto") else "")]
        why = "nessun pattern sull'immagine, 1 d'ufficio" if r.get("auto") else r.get("rationale", "")
        notes[r["image"]][1] += (f". Giudizio di Gemma: copertura {c:.0f}/5, causa {k:.0f}/5 "
                                 f"(somma {c + k:.0f}/10), voto complessivo {g:.0f}/10 — {why}")
    gj = [r for r in by["genuine"] if not r.get("auto")]
    sm = lambda p: mean([r["coverage"] + r["cause"] for r in by[p]])
    header = (
        f"<b>Cattura della modifica</b> (Gemma, {len(by['genuine'])} vittime): copertura + causa "
        f"<b>{sm('genuine'):.2f}/10</b> (controllo con i pattern di un'altra vittima {sm('control'):.2f}; "
        f"stesse immagini prima della modifica {sm('original'):.2f}) &middot; copertura "
        f"<b>{mean([r['coverage'] for r in by['genuine']]):.2f}/5</b> &middot; causa "
        f"<b>{mean([r['cause'] for r in by['genuine']]):.2f}/5</b> &middot; voto complessivo di Gemma "
        f"{mean([r['score'] for r in by['genuine']]):.2f}/10 (controllo "
        f"{mean([r['score'] for r in by['control']]):.2f}) &middot; sulle {len(gj)} vittime con almeno "
        f"un pattern: copertura {mean([r['coverage'] for r in gj]):.2f}/5, causa "
        f"{mean([r['cause'] for r in gj]):.2f}/5. "
        + html.escape("Copertura: un pattern riguarda l'oggetto modificato e il suo errore; causa: la "
                      "situazione nominata corrisponde al cambio di materiale."))

vict = sorted((s for s in edited if s in victim), key=lambda s: drops.get(s, 0))
unchanged = sorted(s for s in edited if s not in victim and s in drops and abs(drops[s]) < 0.05)
pick = sorted(random.Random(0).sample(unchanged, min(N_UNCHANGED, len(unchanged))))
for i, s in enumerate(vict, 1):
    notes[s][0] = f"modifica Pascal-EA · vittima {i}/{len(vict)}"
for i, s in enumerate(pick, 1):
    notes[s][0] = f"modifica Pascal-EA · segmentazione invariata, campione casuale {i}/{len(pick)}"
order = vict + pick + sorted((s for s in edited if s not in set(vict) | set(pick)), key=lambda s: drops.get(s, 0))
n_hurt = sum(1 for s in edited if s in drops and drops[s] <= -0.10)
sections = html.escape(
    f"Vista per immagine: le prime {len(vict)} schede sono le vittime, cioè le immagini in cui la "
    f"modifica del materiale ha fatto perdere il soggetto al segmentatore (recall giù di almeno "
    f"{DROP:.2f}; dalla perdita più forte). Le {len(pick)} schede successive sono immagini modificate "
    f"prese a caso (seme 0) tra le {len(unchanged)} in cui la recall del soggetto è cambiata meno di "
    f"0.05. Su tutte le {len(edited)} immagini modificate: {len(vict)} vittime "
    f"({100 * len(vict) / len(edited):.0f}%), {n_hurt} con un calo di almeno 0.10 "
    f"({100 * n_hurt / len(edited):.0f}%), {len(unchanged)} praticamente invariate "
    f"({100 * len(unchanged) / len(edited):.0f}%).")
header = sections if header is None else header + "<br>" + sections
json.dump({"order": order, "notes": notes, "badges": badges, "header": header}, open(f"{EA_RUN}/gallery_notes.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print(f"notes {len(notes)} | victims {len(vict)} (covered {len(covered)}) + {len(pick)} unchanged of "
      f"{len(unchanged)} first | judge badges {len(badges)} | written {EA_RUN}/gallery_notes.json")
