# -*- coding: utf-8 -*-
"""Build a self-contained HTML comparing, per image: picture | ground truth |
prediction, the hand-written annotation, and the pipeline's own description.

Covers the 79 hand-annotated images that have a ground-truth mask (the 80th,
2011_003554, sits only in VOC's Action splits and has no segmentation GT, so
Block 1 cannot run on it at all). Images with no typed error are kept and
labelled as such: "the pipeline had nothing to say" is a real outcome, not a
gap in the data.

Each card also carries the FULL Block-1 (Beyond IoU) taxonomy of the image --
every typed error, not only the ones a certified slice covers. That row is what
explains most human/pipeline disagreements: a `boundary` error is detected and
typed by Block 1 even when no certified slice names it, so the description stays
silent about something the annotator wrote a sentence on.

Two generated texts sit on each card, and they answer different questions:
`error_narratives.json` (describe_errors.py) narrates EVERY typed error in the
annotators' own register -- that is the text the hand-written annotation should
be read against -- while the systematic-pattern block below it speaks only about
errors that fell inside a certified slice.

Inputs (all produced elsewhere):
  outputs/slice_quality/error_narratives.json              (val + train36)
  outputs/slice_quality/image_descriptions.json            (val)
  outputs/slice_quality/train36/descriptions_train36_full.json
  outputs/slice_quality/typed_per_image.json               (Block 1 per image)
  outputs/slice_quality/gt_comparison.md                   (hand verdicts, val only)
  _evaldata/dataset/{answer_gts.jsonl,voc_patch_matches.json}
"""
import base64
from collections import Counter
import io
import json
import os
import sys
import hashlib
import re
from html import escape

ROOT = r"<project-root>\SlicelenceCode"
B = r"<datasets>\pascal_80"
# SQ_OUT: the run to render (a retyped run lives in its own directory); the
# hand verdicts and the human annotations do not belong to a run and stay put.
# The reference run is also what a retyped run is compared against in the legend.
sys.path.insert(0, os.path.join(ROOT, "docs"))
from _revision_numbers import out_path as _out_path  # noqa: E402  (RUN_UFFICIALI, RUN_VECCHI or outputs/)
REF = _out_path("slice_quality")
# SQ_OUT: a run by its name, as on the server ("outputs/slice_quality_inst" or "slice_quality_inst")
SQ = _out_path(os.path.basename(os.path.normpath(os.environ.get("SQ_OUT", "slice_quality"))))
IS_REF = os.path.normcase(SQ) == os.path.normcase(REF)
JUDGE_FILE = os.environ.get("JUDGE_FILE", "judge_descriptions_scored.json")
SYNTH_JUDGE = os.environ.get("SYNTH_JUDGE", "judge_synth_v2.json")
# page furniture: the judge's average in the header and the introductory block of
# the synthetic set (its warning, the two declared halves, its four numbers and the
# caveats). Both are commentary rather than evidence, and a page meant to be read
# card by card can drop either (SHOW_AVG=0 / SHOW_SYNTH_INTRO=0). The 21 synthetic
# cards keep their place at the head of the list either way: each carries its own
# badges -- "sintetica", "dichiarata corretta/errata" and, on the eleven whose
# prediction belongs to another image, the orange one.
SHOW_AVG = os.environ.get("SHOW_AVG", "1") != "0"
SHOW_SYNTH_INTRO = os.environ.get("SHOW_SYNTH_INTRO", "1") != "0"
OUT = os.path.join(SQ, os.environ.get("OUT_HTML", "confronto_umano_pipeline.html"))
# short names of the cluster slices (slice_quality/naming.py, written next to the run by
# slice_quality/examples/write_cluster_names.py), shown instead of the id as in the gallery;
# without the file the ids stay
_CN = os.path.join(SQ, "cluster_names.json")
CNAMES = json.load(open(_CN, encoding="utf-8")) if os.path.exists(_CN) else {}
_CID = re.compile(r"\bcluster_[a-z]+_\d+_r\d+\b")


def named(text_html):
    return _CID.sub(lambda m: ('<span class="cname" title="%s">%s</span>'
                               % (m.group(0), escape(CNAMES[m.group(0)])))
                    if m.group(0) in CNAMES else m.group(0), text_html)
SIDE, QUALITY = 330, 72

import numpy as np
from PIL import Image

mp = {int(k): v.replace(".jpg", "") for k, v in
      json.load(open(os.path.join(ROOT, r"_evaldata\dataset\voc_patch_matches.json"),
                     encoding="utf-8")).items()}
ann = {}
for line in open(os.path.join(ROOT, r"_evaldata\dataset\answer_gts.jsonl"), encoding="utf-8"):
    if line.strip():
        o = json.loads(line)
        ann[mp[o["img_idx"]]] = (o["img_idx"], o["content"].strip())

desc_val = json.load(open(os.path.join(SQ, "image_descriptions.json"), encoding="utf-8"))
desc_tr = json.load(open(os.path.join(SQ, r"train36\descriptions_train36_full.json"),
                         encoding="utf-8"))
typed = json.load(open(os.path.join(SQ, "typed_per_image.json"), encoding="utf-8"))

cand = json.load(open(os.path.join(SQ, "candidate_clusters.json"), encoding="utf-8"))

# the human-style narrative of EVERY typed error (describe_errors.py), which is
# what the hand-written annotation should be compared against -- the systematic
# patterns below speak only about errors that fell inside a certified slice
narr = json.load(open(os.path.join(SQ, "error_narratives.json"), encoding="utf-8"))

# plausibility of the CAUSE (judge_cause_plausibility.py): the same certified
# situation judged against two independent accounts of the same errors, the human
# annotation and the pipeline narrative. Absent file -> the block is simply not drawn.
CAUSE_FILE = os.environ.get("CAUSE_FILE", "judge_cause.json")
CAUSE_SYNTH = os.environ.get("CAUSE_SYNTH", os.path.join("synth", "judge_cause_synth.json"))
cause_j, _cctrl = {}, []
for _cp in (os.path.join(SQ, CAUSE_FILE), os.path.join(SQ, CAUSE_SYNTH)):
    if not os.path.exists(_cp):
        continue
    for r in json.load(open(_cp, encoding="utf-8"))["judgements"]:
        if "score" not in r:
            continue
        if r.get("pairing") == "genuine":
            cause_j.setdefault(r["image"], {})[r["kind"]] = r
        elif r.get("pairing") == "control":
            _cctrl.append(r["score"])
CAUSE_FLOOR = sum(_cctrl) / len(_cctrl) if _cctrl else float("nan")
CAUSE_N = len(cause_j)
# every image that HAS a certified pattern, judged or not: on those the absence
# of a score means "not measured", not "nothing was blamed"
with_cause = {k for k, v in desc_val.items() if v.get("patterns")}
with_cause |= {k for k, v in desc_tr.items() if v.get("patterns")}


# the LLM judge's score for THIS image: how well the generated text matches the
# hand-written one, the human annotation being the reference. Only the genuine
# pairings go on a card; the control pairings (same annotation vs another image's
# text) are what set the 0.3/10 floor quoted in the legend.
judge = {}
_jall = json.load(open(os.path.join(SQ, JUDGE_FILE), encoding="utf-8"))["judgements"]
for r in _jall:
    if r.get("pairing") == "genuine" and "score" in r:
        judge.setdefault(r["image"], {})[r["kind"]] = r


# the length-neutral FACTS score (judge_claims.py): the annotation split into its
# atomic facts once, each fact checked in the generated text; the score is computed
# from the verdicts, so the length of the generated text cannot buy points
CLAIMS_FILE = os.environ.get("CLAIMS_FILE", "judge_claims.json")
claims, claim_facts, CL = {}, {}, None
if os.path.exists(os.path.join(SQ, CLAIMS_FILE)):
    _cj = json.load(open(os.path.join(SQ, CLAIMS_FILE), encoding="utf-8"))
    claim_facts = _cj.get("facts", {})
    for r in _cj["judgements"]:
        if r.get("rubric") == "facts" and r.get("pairing") == "genuine" and "score" in r:
            claims[(r["set"], r["image"], r["kind"])] = r
    CL = dict(summary=_cj.get("summary", {}), duplicate=_cj.get("duplicate", {}),
              correlation=_cj.get("correlation", {}), v1=_cj.get("v1", {}),
              fact_types=_cj.get("fact_types", {}))
# the same facts with verdicts written by hand (judge_claims_manual.py): when present
# they are the score shown, and Gemma's verdict is kept beside each fact it disagrees on
MANUAL_FILE = os.environ.get("MANUAL_FILE", "judge_claims_manual.json")
CLM = None
if claims and os.path.exists(os.path.join(SQ, MANUAL_FILE)):
    _mj = json.load(open(os.path.join(SQ, MANUAL_FILE), encoding="utf-8"))
    for r in _mj["rows"]:
        if r["pairing"] == "genuine":
            claims[(r["set"], r["image"], "narrative")] = dict(r, kind="narrative")
    CLM = dict(summary=_mj["summary"], agreement=_mj.get("agreement", {}), verdicts=_mj.get("verdicts", {}))

_VSYM = {"stated": ("&#10003;", "riportato"), "partial": ("&frac12;", "in parte"),
         "contradicted": ("&#10007;", "contraddetto"), "absent": ("&ndash;", "assente")}


def facts_html(sname, name, kind):
    """The facts score of one generated text, with the annotation's facts and the
    verdict on each, folded; empty when the text was not judged."""
    from collections import Counter as _C
    r = claims.get((sname, name, kind))
    fe = claim_facts.get("%s|%s" % (sname, name)) or {}
    f = fe.get("facts")
    if not r or not f:
        return ""
    # the score counts the facts about the prediction (all of them when there are
    # none); overall judgements and remarks on the ground truth are shown, not counted
    ty = fe.get("types") if len(fe.get("types") or []) == len(f) else None
    cnt = [t == "prediction" for t in ty] if ty and "prediction" in ty else [True] * len(f)
    c = _C(v for v, k in zip(r["verdicts"], cnt) if k)
    sc = r["score"]
    band = "hi" if sc >= 7 else ("lo" if sc <= 2 else "mid")
    _nc = {"overall": "giudizio complessivo", "ground_truth": "osservazione sulla ground truth"}
    gv = r.get("gemma_verdicts") or []
    gdiff = lambda i, v: (' <i class="fnt">(Gemma: %s)</i>' % _VSYM[gv[i]][1]
                          if i < len(gv) and gv[i] != v else "")
    rows = "".join('<li class="fv-%s%s"><span class="fs" title="%s">%s</span>%s%s%s</li>'
                   % (v, "" if k else " fnc", _VSYM[v][1], _VSYM[v][0], escape(t),
                      "" if k else ' <i class="fnt">(non contato: %s)</i>' % _nc.get(ty[i], ty[i]),
                      gdiff(i, v))
                   for i, (t, v, k) in enumerate(zip(f, r["verdicts"], cnt)))
    nn = len(f) - sum(cnt)
    who = (" (verdetti di Claude; Gemma %.1f%s)" % (r["gemma_score"], ", data alla narrativa precedente"
                                                   if r.get("gemma_same_text") is False else "")
           if r.get("gemma_score") is not None else "")
    return ('<details class="fct j-%s"><summary><span class="fsc">%.1f<small>/10</small></span>'
            '<span class="fl">voto per fatti%s &middot; %d riportati, %d in parte, %d contraddetti, '
            '%d assenti su %d fatti dell&#39;annotazione%s</span></summary><ul>%s</ul></details>'
            % (band, sc, who, c["stated"], c["partial"], c["contradicted"], c["absent"], sum(cnt),
               (" (%d non contat%s)" % (nn, "o" if nn == 1 else "i")) if nn else "", rows))


def _jmean(rows, kind, pairing, key="score"):
    v = [r[key] for r in rows if r.get("kind") == kind and r.get("pairing") == pairing
         and r.get(key) is not None]
    return sum(v) / len(v) if v else float("nan")


# the numbers the legend quotes, taken from the very file the cards use
LEG = dict(gn=_jmean(_jall, "narrative", "genuine"), gp=_jmean(_jall, "pattern", "genuine"),
           cn=_jmean(_jall, "narrative", "control"), cp=_jmean(_jall, "pattern", "control"),
           spec_c=_jmean(_jall, "narrative", "control", "specificity"),
           comp_c=_jmean(_jall, "narrative", "control", "completeness"),
           corr_c=_jmean(_jall, "narrative", "control", "correctness"))
_sidx = json.load(open(os.path.join(SQ, "slices_index.json"), encoding="utf-8"))["slices"]
LEG["n_cluster"] = sum(1 for x in _sidx if x.get("in_descriptions") and x.get("source") == "cluster")
LEG["n_lang"] = sum(1 for x in _sidx if x.get("in_descriptions") and x.get("source") != "cluster")
# images whose narrative carries a contour note on a CORRECT region, and how many of
# them have no typed error at all (before the notes, those produced no text)
_contour = [k for k, v in narr.items()
            if "correctly located" in v["text"] or "outline is" in v["text"]]
LEG["n_contour"] = len(_contour)
LEG["n_contour_noerr"] = sum(1 for k in _contour if not sum(typed.get(k, {}).values()))

# a retyped run is compared, image by image, against the reference run
RUN_NOTE = ""
if not IS_REF:
    import pickle
    import sys
    sys.path.insert(0, ROOT)          # the pickle holds slice_quality dataclasses
    _ref = {(r["image"], r["kind"]): r["score"] for r in json.load(open(
        os.path.join(REF, "judge_descriptions_scored.json"), encoding="utf-8"))["judgements"]
        if r.get("pairing") == "genuine" and "score" in r}
    _new = {(r["image"], r["kind"]): r["score"] for r in _jall
            if r.get("pairing") == "genuine" and "score" in r}
    _cmp = {}
    for kind in ("narrative", "pattern"):
        pairs = [(_ref[k], _new[k]) for k in _new if k[1] == kind and k in _ref]
        d = [b - a for a, b in pairs]
        _cmp[kind] = dict(n=len(pairs), a=sum(a for a, _ in pairs) / len(pairs),
                          b=sum(b for _, b in pairs) / len(pairs),
                          up=sum(1 for x in d if x > 0), dn=sum(1 for x in d if x < 0),
                          eq=sum(1 for x in d if x == 0))
    _pk = pickle.load(open(os.path.join(SQ, "result_tagged.pkl"), "rb"))
    _err, _refset = _pk["errors"], _pk["reference"]
    _moved = sum(1 for e in _err if e.seg_subtype.value == "merge"
                 and (e.metadata.get("primary_iou") or 0.0) >= 0.85)
    _sw = sum(1 for e in _err if e.seg_subtype.value == "merge" and e.metadata.get("from_gt_mask"))
    _pairs = sum(1 for e in _err + _refset if e.metadata.get("paired_by") == "containment")
    _fn = sum(1 for e in _err if e.seg_subtype.value == "fn")
    # the object adapter (SegmentationObject) leaves gt_parts / occluded_by on
    # the instances: say so, with counts, only when it was on for this run
    _cut = [e for e in _err + _refset if (e.metadata.get("gt_parts") or 1) > 1]
    _INST_NOTE = ""
    if _cut:
        _occ = Counter(e.metadata.get("occluded_by") for e in _cut if e.metadata.get("occluded_by"))
        _INST_NOTE = (
            ' <b>Istanze GT = oggetti (SegmentationObject di VOC):</b> in questo run un&#39;istanza di '
            'ground truth &egrave; un oggetto, non una componente connessa della maschera di classe. '
            'Un oggetto tagliato da un occlusore &egrave; una sola istanza (maschera non connessa), '
            'due oggetti che si toccano sono due; le predizioni restano componenti, perch&eacute; il '
            'modello &egrave; semantico. Sul val la regola delle componenti spezzava 699 oggetti su '
            '3085 in pi&ugrave; pezzi (35%% delle immagini) e fondeva 264 oggetti in 118 componenti; '
            'fra i merge del run precedente 170 su 559 erano un solo oggetto occluso. Qui '
            '<b>%d</b> istanze sono oggetti tagliati, e per <b>%d</b> di esse l&#39;occlusore &egrave; '
            'stato individuato (l&#39;oggetto di un&#39;altra classe che tocca almeno due parti: %s), '
            'da cui la frase &laquo;partly hidden behind the &hellip;&raquo;. Le sintetiche non hanno '
            'maschere di oggetto e restano a componenti.'
            % (len(_cut), sum(_occ.values()),
               ", ".join(f"{c.upper()} {k}" for c, k in _occ.most_common(4))))
    RUN_NOTE = (
        '<br><b>Regola di claim di questo run (contenimento, claim_cov = 0.5):</b> il Blocco 1 '
        'decideva merge e split con un test di IoU, cieco al contenimento asimmetrico: una regione '
        'piccola inglobata al 70%% in un blob che copre anche una regione grande ha IoU ~0.1 con '
        'quel blob, non lo rivendica e finisce in FN &mdash; &laquo;mancata&raquo; per un oggetto '
        'coperto per due terzi (il 54%% dei FN sulle 80). Beyond IoU stesso chiama <i>mancato</i> un '
        'segmento solo se nessun suo pixel &egrave; un vero positivo; qui la regola analoga a '
        'livello di istanza: una GT rivendica anche la predizione che ne contiene almeno met&agrave;, '
        'una predizione contenuta per almeno met&agrave; in una GT della sua classe la rivendica. '
        'A 0.5 il contenitore &egrave; unico (due regioni disgiunte non possono tenere pi&ugrave; '
        'di met&agrave; della stessa maschera). Effetto sul val: FN <b>2651 &rarr; %d</b>; '
        '<b>%d</b> coppie 1:1 nate per contenimento (prima un FP <i>e</i> un FN ciascuna); '
        '<b>%d</b> oggetti inglobati tenuti come istanze proprie lato GT (sottotipo MERGE, '
        'embedding sulla maschera GT), cos&igrave; le slice possono ancora riguardare gli oggetti '
        'inglobati e non solo i blob che li contengono: un merge produce 1 + M istanze, dichiarato. '
        '<b>Nota di metodo:</b> <b>%d</b> riferimenti CORRECT su %d (%.0f%%) diventano MERGE perch&eacute; '
        'il blob, pur a IoU &ge; 0.85 sul proprio oggetto, ingloba anche una GT piccola: la '
        'popolazione di riferimento di &omega;&sup2; e delle note di contorno scende a %d regioni. '
        '<b>Confronto con il run di riferimento</b> (regola IoU), sulle stesse immagini: narrativa '
        '<b>%.2f &rarr; %.2f</b> su %d (%d migliorate, %d peggiorate, %d uguali); pattern '
        '<b>%.2f &rarr; %.2f</b> su %d (%d, %d, %d). Slice cluster certificate: 14 &rarr; %d, di cui '
        'quelle di tipo MERGE sono fatte in maggioranza di oggetti inglobati.'
        % (_fn, _pairs, _sw, _moved, _moved + len(_refset), 100.0 * _moved / (_moved + len(_refset)),
           len(_refset),
           _cmp["narrative"]["a"], _cmp["narrative"]["b"], _cmp["narrative"]["n"],
           _cmp["narrative"]["up"], _cmp["narrative"]["dn"], _cmp["narrative"]["eq"],
           _cmp["pattern"]["a"], _cmp["pattern"]["b"], _cmp["pattern"]["n"],
           _cmp["pattern"]["up"], _cmp["pattern"]["dn"], _cmp["pattern"]["eq"],
           LEG["n_cluster"]))
    RUN_NOTE += _INST_NOTE
    # the narrative revision (direction + side, correct regions, spills, tiny
    # patches, verdict, ordering): paired against the judge file frozen before it
    _objf = os.path.join(SQ, "judge_descriptions_scored_objects.json")
    if os.path.exists(_objf):
        _old = {(r["image"], r["kind"]): r["score"] for r in json.load(open(_objf, encoding="utf-8"))["judgements"]
                if r.get("pairing") == "genuine" and "score" in r}
        pairs = [(_old[k], _new[k]) for k in _new if k[1] == "narrative" and k in _old]
        d = [b - a for a, b in pairs]
        _txts = [v["text"] for v in narr.values()]
        RUN_NOTE += (
            ' <b>Narrativa rivista (direzione e lato, regioni corrette, sconfinamenti, toppe sotto '
            'soglia, verdetto in apertura, ordine):</b> letta scheda per scheda, la narrativa diceva '
            '<i>quanto</i> era sbagliato un contorno dove l&#39;annotatore dice <i>in che direzione e '
            'dove</i>. Ora ogni coppia, merge e regione corretta porta il verso e il lato '
            '(&laquo;spilling over the object on the top and right (8%% of its area in excess)&raquo;), '
            'ogni regione corretta ha una frase (&laquo;segmented closely&raquo;; &laquo;No segmentation '
            'error&raquo; solo a IoU &ge; 0.99), una toppa della stessa classe attaccata all&#39;oggetto '
            '&egrave; la maschera che sborda e non un&#39;allucinazione, le toppe predette sotto i 200 px '
            'di classe sbagliata vengono contate, e un fallimento globale apre la frase (&laquo;The '
            'prediction is largely wrong: 76%% of the annotated area has been labelled DOG&raquo;). '
            'Sul val: %d clausole di sconfinamento, %d frasi sulle regioni corrette, %d verdetti. '
            '<b>Giudice, stesse %d immagini: %.2f &rarr; %.2f</b> (%d migliorate, %d peggiorate, %d uguali); '
            'il pavimento del controllo non si muove.'
            % (sum(1 for t in _txts if "spilling over" in t),
               sum(1 for t in _txts if "segmented closely" in t or "segmented well" in t),
               sum(1 for t in _txts if t.startswith("The prediction is largely wrong") or t.startswith("The prediction misses most")),
               len(pairs), sum(a for a, _ in pairs) / len(pairs), sum(b for _, b in pairs) / len(pairs),
               sum(1 for x in d if x > 0), sum(1 for x in d if x < 0), sum(1 for x in d if x == 0)))

STALE_NOTE = ""   # filled after the cards are built, see below

# the legend of the facts score, with the two length checks read from its file
CLAIMS_NOTE = ""
if CL:
    _sm, _dp, _co = CL["summary"], CL["duplicate"], CL["correlation"]
    _f = lambda k: _sm.get(k, {})
    CLAIMS_NOTE = (
        '<br><b>Voto per fatti (indipendente dalla lunghezza):</b> confronta l&#39;annotazione '
        'umana con la <i>narrativa</i> della pipeline (il testo dei pattern non &egrave; votato: '
        'dice in quale situazione certificata ricade l&#39;immagine, non che cosa &egrave; andato '
        'storto). Gemma divide ogni annotazione nei suoi fatti atomici (un oggetto, che cosa '
        'c&#39;&egrave; di giusto o sbagliato nella sua maschera, dove), una volta sola; poi, fatto '
        'per fatto, dice se la narrativa lo <i>riporta</i>, lo riporta <i>in parte</i>, lo '
        '<i>contraddice</i> o non ne parla. Il voto lo calcola il codice: 10 &times; (riportati + '
        '&frac12; in parte &minus; contraddetti) / numero di fatti. Il denominatore viene dal testo '
        'umano e ci&ograve; che la narrativa dice in pi&ugrave; &egrave; ignorato, salvo '
        'contraddizioni: la lunghezza non pu&ograve; comprare punti (il prezzo: errori inventati su '
        'oggetti che l&#39;annotatore non cita non sono penalizzati). Media <b>%.2f</b>/10 su tutte '
        'le %d immagini annotate, da qualunque parte vengano (VOC %.2f su %d tra val e train, '
        'sintetiche %.2f su %d). Come riferimento, la stessa annotazione confrontata con la '
        'narrativa di un&#39;<i>altra</i> immagine prende %.2f: &egrave; il voto che si ottiene '
        'per caso. '
        % (_f("all_narrative").get("genuine", float("nan")), _f("all_narrative").get("n", 0),
           _f("voc_narrative").get("genuine", float("nan")), _f("voc_narrative").get("n", 0),
           _f("synth_narrative").get("genuine", float("nan")), _f("synth_narrative").get("n", 0),
           _f("all_narrative").get("control", float("nan"))))
    _v1, _ft = CL.get("v1", {}), CL.get("fact_types", {})
    if _v1.get("all_genuine") and _ft:
        _g, _c = _v1["all_genuine"], _v1.get("all_control", {})
        CLAIMS_NOTE += (
            '<b>Glossario e tipo dei fatti:</b> la narrativa usa un vocabolario tecnico fisso '
            '(&laquo;spilling over the object, 30%% in excess&raquo; per &laquo;lo sfondo intorno '
            '&egrave; etichettato come cavallo&raquo;), che la prima versione del giudice non '
            'riconosceva. Ora il prompt dei verdetti contiene un glossario di queste frasi, uguale '
            'per coppie vere, di controllo e duplicate. Inoltre ogni fatto ha un tipo: errore della '
            'predizione (%d), giudizio complessivo (%d), osservazione sulla ground truth (%d). La '
            'narrativa racconta gli errori oggetto per oggetto e prende la ground truth come '
            'riferimento, quindi il voto conta solo i fatti sulla predizione (tutti, se '
            'un&#39;annotazione non ne ha); gli altri restano visibili, in grigio. Sulle %d immagini: '
            'prima del glossario %.2f, con il glossario su tutti i fatti %.2f, contando solo i '
            'fatti sulla predizione %.2f; il controllo passa da %.2f a %.2f. '
            % (_ft.get("prediction", 0), _ft.get("overall", 0), _ft.get("ground_truth", 0),
               _g["n"], _g["v1"], _g["all_facts"], _g["now"], _c.get("v1", float("nan")),
               _c.get("now", float("nan"))))
    if CLM and CLM["summary"].get("tutte"):
        _t, _ag = CLM["summary"]["tutte"], CLM["agreement"].get("genuine", {})
        CLAIMS_NOTE += (
            '<b>Verdetti scritti a mano:</b> su richiesta, gli stessi fatti sono stati verificati anche '
            'uno per uno da Claude, l&#39;assistente con cui &egrave; stata costruita la pipeline (non '
            '&egrave; quindi un giudice indipendente: va dichiarato). Le coppie vere e quelle di '
            'controllo sono state mescolate e presentate senza nome dell&#39;immagine, con le stesse '
            'regole e la stessa formula. Risultato sulle %d immagini annotate: <b>%.2f</b> contro un '
            'controllo di %.2f (Gemma: %.2f contro %.2f). Accordo fatto per fatto con Gemma: %.0f%% '
            '(kappa di Cohen %.2f); quasi tutto il disaccordo &egrave; Gemma che segna &laquo;assente&raquo; '
            'un fatto che la narrativa dice con altre parole. Il voto mostrato nelle schede &egrave; '
            'questo; accanto a ogni fatto su cui Gemma dava un verdetto diverso c&#39;&egrave; il suo. '
            % (_t["n"], _t["genuine"], _t["control"], _t["gemma"], _t["gemma_control"],
               100 * _ag.get("agreement", float("nan")), _ag.get("kappa", float("nan"))))
    if _dp.get("facts") and _dp.get("old"):
        CLAIMS_NOTE += (
            '<b>Prova:</b> la stessa narrativa scritta due volte (lunghezza doppia, contenuto '
            'identico) non cambia il voto per fatti in %d casi su %d (variazione media %+.2f), '
            'contro %d su %d per la rubrica completezza-correttezza-specificit&agrave; (%+.2f). '
            % (_dp["facts"]["same"], _dp["facts"]["n"], _dp["facts"]["mean"],
               _dp["old"]["same"], _dp["old"]["n"], _dp["old"]["mean"]))
    if "facts_genuine_narrative" in _co:
        CLAIMS_NOTE += (
            'Correlazione voto / lunghezza della narrativa: %+.2f (rubrica: %+.2f). '
            'Clic sul voto per vedere i fatti e il verdetto su ciascuno.'
            % (_co["facts_genuine_narrative"][0], _co.get("old_genuine_narrative", [float("nan")])[0]))
def _cause_mean(kind):
    v = [r[kind]["score"] for r in cause_j.values() if kind in r]
    return (sum(v) / len(v)) if v else float("nan"), len(v)


CAUSE_MEAN_NAR, CAUSE_N_NAR = _cause_mean("pipeline")
CAUSE_MEAN_HUM, CAUSE_N_HUM = _cause_mean("human")

CAUSE_NOTE = (
    '<br><b>Plausibilit&agrave; della causa:</b> sotto i pattern certificati, dove ci sono, '
    'due voti di Gemma sulla stessa domanda &mdash; la situazione che la pipeline incolpa '
    'spiega l&#39;errore? La stessa causa &egrave; giudicata contro due racconti indipendenti '
    'dello stesso errore, la narrativa generata e l&#39;annotazione umana, senza che il giudice '
    'sappia quale sta leggendo; la rubrica &egrave; <b>coerenza 5 + forza esplicativa 5</b> e i '
    'numeri di maschera (IoU, frammenti) sono tolti dalla causa, altrimenti il giudice '
    'combacerebbe misure invece di pesare una spiegazione. Su %d immagini con una causa '
    'certificata: %.2f/10 con la narrativa, %.2f/10 con l&#39;annotazione, contro <b>%.2f</b> '
    'del controllo (la stessa descrizione d&#39;errore accoppiata alla causa di un&#39;ALTRA '
    'immagine).' % (CAUSE_N, CAUSE_MEAN_NAR, CAUSE_MEAN_HUM, CAUSE_FLOOR)) if cause_j else ""

verdict = {}
for line in open(os.path.join(REF, "gt_comparison.md"), encoding="utf-8"):
    m = re.match(r"^## (\S+) - \*\*(ALTA|PARZIALE|BASSA)\*\*", line)
    if m:
        verdict[m.group(1)] = m.group(2)
_bassa = [k for k, v in verdict.items() if v == "BASSA"]
_nb = sum(1 for k in _bassa if k in _contour)
LEG["bassa_contour"] = ("tutte e sei quelle giudicate BASSA a mano" if _nb == len(_bassa) else
                        "%d delle sei giudicate BASSA a mano (le altre sono ora errori tipati)" % _nb)

PAL = Image.open(os.path.join(B, "SegmentationClass", "2007_000033.png")).getpalette()


def rcc(img, size, resample):
    """Short side to `size`, then centre-crop -- the same alignment the pipeline
    applies, so the three panels are comparable pixel for pixel."""
    w, h = img.size
    if w < h:
        nw, nh = size, int(round(h * size / w))
    else:
        nh, nw = size, int(round(w * size / h))
    img = img.resize((nw, nh), resample)
    l, t = (nw - size) // 2, (nh - size) // 2
    return img.crop((l, t, l + size, t + size))


def b64(img):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=QUALITY)
    return base64.b64encode(buf.getvalue()).decode()


# il prompt lo leggo dal file del giudice invece di ricopiarlo, cosi non puo
# divergere da quello effettivamente inviato a Gemma
_js = io.open(os.path.join(ROOT, r"slice_quality\examples\judge_descriptions.py"),
              encoding="utf-8").read()
_ns = {}
exec(re.search(r'JUDGE_SYSTEM = """.*?"""', _js, re.S).group(0), _ns)
exec(re.search(r'JUDGE_USER = """.*?"""', _js, re.S).group(0), _ns)
PROMPT = ('<p>Il giudice <b>non vede l&#39;immagine</b>: riceve due '
          'testi e basta. Misura quindi l&#39;<i>accordo</i> con l&#39;annotazione umana, '
          'che fa da riferimento &mdash; se l&#39;annotatore sbaglia, la pipeline paga.</p>'
          '<p><b>System</b></p><pre>%s</pre>'
          '<p><b>User</b> &mdash; solo i due testi, nessun&#39;altra istruzione</p><pre>%s</pre>'
          '<p>Le coppie di <b>controllo</b> usano lo stesso identico prompt: il giudice non '
          'sa se la coppia che sta valutando &egrave; vera o disallineata.</p>'
          % (escape(_ns["JUDGE_SYSTEM"]), escape(_ns["JUDGE_USER"])))

NO_CAUSE = ('<div class="cause"><div class="nocause">Nessuna causa certificata per '
            'questa immagine: nessuno dei suoi errori ricade in una slice certificata, '
            'quindi non c&#39;&egrave; una situazione da giudicare.</div></div>')


def cause_html(name):
    """Plausibility of this image's cause, told by the two accounts.

    No judgement on this image means one of two different things, and the page
    must not leave them looking alike: either the image has no certified pattern
    at all -- nothing was blamed, so there is nothing to judge -- or a pattern
    exists and the judge has not been run on it yet.
    """
    rec = cause_j.get(name)
    if not rec:
        return NO_CAUSE if name not in with_cause else ""
    bits = []
    for kind, label in (("pipeline", "narrativa"), ("human", "annotazione")):
        r = rec.get(kind)
        if not r:
            continue
        sc = r["score"]
        band = "hi" if sc >= 7 else ("lo" if sc <= 4 else "mid")
        bits.append(
            '<div class="cs j-%s"><span class="sc">%.1f<small>/10</small></span>'
            '<span class="cl">errore raccontato<br>dalla %s</span>'
            '<span class="cq">coerenza %g/5 &middot; forza esplicativa %g/5<br>'
            '<i>%s</i></span></div>'
            % (band, sc, label, r["consistency"], r["explanatory_power"],
               escape(r.get("rationale", ""))))
    if not bits:
        return ""
    return ('<div class="cause"><div class="ch">Plausibilit&agrave; della causa '
            '<span>(Gemma, coerenza 5 + forza esplicativa 5; pavimento del '
            'controllo %.1f)</span></div>%s</div>' % (CAUSE_FLOOR, "".join(bits)))


cards, page_scores = [], []
# images whose narrative has been regenerated after the judgement on the card,
# or whose hand-written annotation was corrected after it
STALE = set()


def _gt_changed(name, rec):
    """Was this verdict given a reference text that is no longer the one on file?

    Newer judgements carry gt_hash and the answer is exact; older ones carry only
    the word count, which catches a correction that changes the length and misses
    one that does not -- stated here rather than silently assumed.
    """
    txt = ann.get(name, (None, ""))[1]
    if not txt:
        return False
    h = rec.get("gt_hash")
    if h:
        return h != hashlib.sha1(" ".join(txt.split()).encode("utf-8")).hexdigest()[:12]
    w = rec.get("gt_words")
    return w is not None and w != len(txt.split())
stats = {"val": 0, "train": 0, "noerr": 0}
for name in sorted(ann, key=lambda n: ann[n][0]):
    idx, human = ann[name]
    jp = os.path.join(B, "JPEGImages", name + ".jpg")
    gp = os.path.join(B, "SegmentationClass", name + ".png")
    pp = os.path.join(B, "mobilenet_pred", "mask_pr_" + name + ".png")
    if not (os.path.exists(jp) and os.path.exists(gp) and os.path.exists(pp)):
        continue                       # 2011_003554: no segmentation GT at all

    if name in desc_val:
        split, rec = "val", desc_val[name]
    elif name in desc_tr:
        split, rec = "train", desc_tr[name]
    else:
        split, rec = "val", None       # in val, but the model made no typed error
    stats[split] += 1

    tx = typed.get(name, {})
    n_err = sum(tx.values())
    n_cov = rec["n_systematic"] if rec else 0
    empty = n_err == 0
    if empty:
        stats["noerr"] += 1

    o = rcc(Image.open(jp).convert("RGB"), SIDE, Image.LANCZOS)
    g = rcc(Image.open(gp).convert("RGB"), SIDE, Image.NEAREST)
    pr = Image.open(pp).split()[0]
    pr.putpalette(PAL)
    p = rcc(pr.convert("RGB"), SIDE, Image.NEAREST)

    v = verdict.get(name, "")
    badges = ['<span class="b split-%s">%s</span>' % (split, split)]
    if v:
        badges.append('<span class="b v-%s">%s</span>' % (v, v))
    if empty:
        badges.append('<span class="b noerr">nessun errore tipato</span>')

    if empty:
        pipe_html = ('<em class="none">La pipeline non produce descrizione: su questa '
                     'immagine il Blocco 1 non trova alcun errore tipato.</em>')
        tax_html = ('<div class="tax"><span class="tl">Blocco 1 (Beyond IoU): '
                    'nessun errore tipato</span></div>')
    else:
        main, nms = [], []
        for line in rec["text"].split("\n"):
            m = re.match(r"^\s*~\s*(\d+)\s+isolated\s+objs?\s+near candidate "
                         r"pattern\s+(\S+)\s+\(([^)]+)\)", line)
            if m:
                nms.append((int(m.group(1)), m.group(2)))
            else:
                main.append(line)
        pipe_html = ("<pre>%s</pre>" % named(escape("\n".join(main)))
                     + cause_html(name))
        if nms:
            items = []
            for n, sid in nms:
                c = cand.get(sid)
                if not c:
                    items.append('<div class="nm-item"><span class="nm-id" title="%s">%s</span></div>'
                                 % (sid, escape(CNAMES.get(sid, sid))))
                    continue
                cls = ", ".join(list(c["classes"])[:3])
                items.append(
                    '<div class="nm-item"><span class="nm-id" title="%s">%s</span>'
                    '<span class="nm-n">%d obj isolat%s</span>'
                    '<div class="nm-what">%s &mdash; %d istanze in %d immagini%s</div>'
                    '<div class="nm-why"><b>Non certificato:</b> %s</div></div>'
                    % (sid, escape(CNAMES.get(sid, sid)), n, "o" if n == 1 else "i", escape(c["what"]),
                       c["size"], c["n_images"],
                       (", classi: " + escape(cls)) if cls else "",
                       escape("; ".join(c["why_not_certified"]))))
            pipe_html += ('<div class="nearmiss"><div class="nm-title">Pattern candidati '
                          'vicini, non certificati</div>%s</div>' % "".join(items))
        chips = "".join('<span class="t t-%s">%s %d</span>' % (k, k, n)
                        for k, n in sorted(tx.items(), key=lambda kv: -kv[1]))
        tax_html = ('<div class="tax"><span class="tl">Blocco 1 (Beyond IoU):</span>%s'
                    '<span class="cov">%d/%d in pattern certificati</span></div>'
                    % (chips, n_cov, n_err))

    def score_html(kind):
        """Score + the five sub-scores + the judge's own reason, for one text."""
        r = judge.get(name, {}).get(kind)
        if not r:
            return "", "na"
        sc = r["score"]
        stale = ""
        why = []
        if kind == "narrative" and narr.get(name) and r.get("cand")                 and r["cand"].strip() != narr[name]["text"].strip():
            why.append("alla narrativa precedente")
        if _gt_changed(name, r):
            why.append("all&#39;annotazione precedente, poi corretta")
        if why:
            STALE.add(name)
            stale = ('<div class="stalew">voto dato %s, non al testo mostrato</div>'
                     % " e ".join(why))
        band = "hi" if sc >= 7 else ("lo" if sc <= 2 else "mid")
        crit = " ".join(
            '<span class="c"><b>%s</b>%g/%d</span>' % (lab, r[k], hi)
            for k, hi, lab in (("completeness", 4, "completezza "),
                               ("correctness", 4, "correttezza "),
                               ("specificity", 2, "specificit&agrave; ")))
        return ('<div class="jud j-%s"><span class="sc">%.1f<small>/10</small></span>'
                '<div class="jd"><div class="crit">%s<span class="eq">= %g</span></div>'
                '<div class="why">%s</div>%s</div></div>'
                % (band, sc, crit, sc, escape(r.get("rationale", "")), stale)), band

    j_nar, band_nar = score_html("narrative")
    if band_nar != "na":                 # media di pagina, sulle sole narrative
        page_scores.append(judge[name]["narrative"]["score"])

    nrec = narr.get(name)
    nar_html = ('<p>%s</p>' % escape(nrec["text"]) if nrec
                else '<em class="none">Nessun errore tipato in questa immagine.</em>')

    cards.append('''
<article class="card" data-split="{split}" data-verdict="{v}" data-noerr="{ne}" data-band="{band}">
  <header><span class="idx">{idx}</span><span class="nm">{name}</span>{badges}</header>
  {tax}
  <div class="panels">
    <figure><img src="data:image/jpeg;base64,{a}" alt="immagine"><figcaption>immagine</figcaption></figure>
    <figure><img src="data:image/jpeg;base64,{b}" alt="ground truth"><figcaption>ground truth</figcaption></figure>
    <figure><img src="data:image/jpeg;base64,{c}" alt="predizione"><figcaption>predizione (MobileNetV3)</figcaption></figure>
  </div>
  <div class="texts">
    <section><h4>Annotazione umana</h4><p>{human}</p></section>
    <section class="gen"><h4>Narrativa generata dalla pipeline</h4>{jnar}{fnar}{nar}</section>
  </div>
  <div class="patt"><h4>Pattern sistematici (slice certificate)</h4>{pipe}</div>
</article>'''.format(split=split, v=v, ne=int(empty), idx=idx, name=name,
                     badges="".join(badges), tax=tax_html,
                     a=b64(o), b=b64(g), c=b64(p),
                     human=escape(human), nar=nar_html, pipe=pipe_html,
                     jnar=j_nar, fnar=facts_html("voc", name, "narrative"), band=band_nar))

CSS = """
:root{--bg:#0f1115;--card:#171a21;--line:#2a2f3a;--ink:#e6e6e6;--mut:#9aa3b2;
--val:#1f5f9e;--train:#7a4fbf;--alta:#1f7a3f;--parz:#a8760f;--bassa:#a83232;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,Segoe UI,sans-serif}
header.top{position:sticky;top:0;background:#171a21;border-bottom:1px solid var(--line);
padding:14px 20px;z-index:5}
h1{margin:0 0 4px;font-size:17px}
.sub{color:var(--mut);font-size:13px;display:inline-block;vertical-align:middle}
.sub summary{cursor:pointer;list-style:none;display:inline-flex;align-items:center;gap:7px;
color:#cfd6e4;background:#222734;border:1px solid #313847;border-radius:6px;
padding:5px 11px;font-size:12px;user-select:none}
.sub summary::-webkit-details-marker{display:none}
.sub summary::before{content:"▸";transition:transform .15s;font-size:11px;color:var(--mut)}
.sub[open] summary::before{transform:rotate(90deg)}
.sub summary:hover{background:#2a3040;border-color:#3d4657}
.sub .subtext{position:fixed;left:20px;right:20px;top:var(--panel-top,110px);z-index:30;
max-height:calc(100vh - var(--panel-top,110px) - 20px);overflow-y:auto;padding:16px 20px;background:#171a21;
border:1px solid #3d4657;border-radius:10px;line-height:1.65;max-width:110ch;
box-shadow:0 18px 46px rgba(0,0,0,.55)}
.sub.prompt pre{white-space:pre-wrap;background:#12151b;border:1px solid var(--line);
border-radius:6px;padding:10px 12px;font:12px/1.5 ui-monospace,Consolas,monospace;
color:#c3cbd8;overflow-x:auto;max-width:100%}
.sub.prompt p{margin:12px 0 6px}
.sub code{background:#222734;border-radius:3px;padding:0 4px;font-size:12px}
.filters button{background:#222734;color:#cfd6e4;border:1px solid #313847;border-radius:6px;
padding:5px 11px;margin-right:6px;cursor:pointer;font-size:13px}
.filters button.on{background:#2f4a6b;border-color:#4a6b95;color:#fff}
main{padding:18px 20px;display:flex;flex-direction:column;gap:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px 16px}
.card header{display:flex;align-items:center;gap:10px;margin-bottom:9px;flex-wrap:wrap}
.idx{font:600 13px ui-monospace,Consolas,monospace;color:var(--mut);background:#222734;
border-radius:4px;padding:2px 7px}
.nm{font:600 14px ui-monospace,Consolas,monospace}
.b{font-size:11px;font-weight:600;padding:2px 8px;border-radius:20px;text-transform:uppercase}
.split-val{background:var(--val);color:#fff} .split-train{background:var(--train);color:#fff}
.v-ALTA{background:var(--alta);color:#fff} .v-PARZIALE{background:var(--parz);color:#fff}
.v-BASSA{background:var(--bassa);color:#fff} .noerr{background:#3a3f4a;color:#cfd6e4}
.tax{display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin:0 0 11px}
.tl{font-size:11.5px;color:var(--mut);margin-right:2px}
.t{font:600 11px ui-monospace,Consolas,monospace;padding:2px 8px;border-radius:4px;
background:#222734;color:#cfd6e4;border:1px solid #313847}
.t-fn{border-color:#3b5f96;color:#9dc0f0} .t-fp{border-color:#96453b;color:#f0a49d}
.t-boundary{border-color:#2f7d86;color:#8fd8e0} .t-extent{border-color:#7a5f2f;color:#e0c48f}
.t-merge{border-color:#6b4b96;color:#c6a9f0} .t-split{border-color:#2f7d55;color:#95dfb8}
.t-confusion{border-color:#96773b;color:#f0d79d}
.cov{font-size:11.5px;color:var(--mut);margin-left:4px}
.panels{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px}
.panels figure{margin:0} .panels img{display:block;border-radius:5px;max-width:100%}
figcaption{font-size:11.5px;color:var(--mut);margin-top:4px}
.texts{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:900px){.texts{grid-template-columns:1fr}}
.texts h4{margin:0 0 6px;font-size:12px;letter-spacing:.06em;text-transform:uppercase;
color:var(--mut)}
.texts p{margin:0}
.texts .gen{border-left:2px solid #2f4a6b;padding-left:12px}
.hrow{display:flex;align-items:center;justify-content:space-between;gap:20px;flex-wrap:wrap}
.avg{display:flex;align-items:center;gap:11px;padding:7px 14px;border-radius:9px;
background:#1c2129;border:1px solid var(--line)}
.avgn{font-size:27px;font-weight:700;color:#5fd08a;line-height:1}
.avgn small{font-size:12px;font-weight:400;color:var(--mut)}
.avgl{font-size:11px;color:var(--mut);line-height:1.35}
.avgl span{color:#7d8798}
.avg-c .avgn{font-size:22px;color:#8fb8f0}
.avgs{display:flex;gap:10px;flex-wrap:wrap;justify-content:flex-end}
.jud{display:flex;gap:11px;align-items:flex-start;margin:0 0 9px;padding:8px 10px;
border-radius:7px;background:#1c2129;border:1px solid var(--line)}
.jud .sc{font-size:21px;font-weight:700;line-height:1.1;white-space:nowrap}
.jud .sc small{font-size:11px;font-weight:400;color:var(--mut)}
.j-hi .sc{color:#5fd08a} .j-mid .sc{color:#d9a441} .j-lo .sc{color:#e06666}
.jud .jd{min-width:0}
.jud .crit{font-size:11px;color:var(--mut);margin-bottom:3px}
.jud .crit .c{margin-right:9px;white-space:nowrap}
.jud .crit b{font-weight:600;color:#8f9bb0}
.jud .crit .eq{color:#8f9bb0;font-weight:600}
.synthblock{margin:26px 18px 8px;padding-top:18px;border-top:2px solid #3d4657}
.synthblock h2{margin:0 0 10px;font-size:19px}
.synthintro{max-width:100ch;color:var(--mut);font-size:13px;line-height:1.6;margin-bottom:16px}
.synthintro p{margin:0 0 10px}
.synthintro .cav{font-size:12.5px;color:#8f9bb0}
.synthnums{display:flex;gap:12px;flex-wrap:wrap;margin:12px 0}
.sn{background:#171a21;border:1px solid var(--line);border-radius:9px;padding:10px 14px;min-width:152px}
.sn b{display:block;font-size:23px;color:#5fd08a;line-height:1.15}
.sn span{font-size:11px;color:var(--mut)}
.b.split-synth{background:#6b4a2f;color:#fff}
.b.orphan{background:#8a4b12;color:#fff}
.stalew{margin-top:6px;font-size:11.5px;color:#d7a45f}
.warn{background:#2a1c12;border:1px solid #8a4b12;border-left:4px solid #d98324;
border-radius:9px;padding:14px 16px;margin:0 0 16px;max-width:100ch;
color:#e8d5c4;font-size:13px;line-height:1.6}
.warn b{color:#ffb765}
.jud .why{font-size:12px;color:#c3cbd8;font-style:italic}
.fct{margin:0 0 9px;padding:6px 10px;border-radius:6px;background:#1b1f28;border:1px solid #2a3140}
.fct summary{cursor:pointer;list-style:none;display:flex;align-items:baseline;gap:10px}
.fct summary::-webkit-details-marker{display:none}
.fct .fsc{font-size:17px;font-weight:700;white-space:nowrap}
.fct .fsc small{font-size:10px;font-weight:400;color:var(--mut)}
.fct .fl{font-size:11.5px;color:var(--mut)}
.fct.j-hi .fsc{color:#7fd39a} .fct.j-mid .fsc{color:#e3c37a} .fct.j-lo .fsc{color:#e88a8a}
.fct ul{margin:7px 0 2px;padding-left:4px;list-style:none;font-size:12.5px}
.fct li{margin:3px 0;display:flex;gap:8px}
.fct .fs{display:inline-block;min-width:16px;text-align:center;font-weight:700}
.fct li.fnc{opacity:.55} .fct .fnt{color:var(--mut);font-size:11px}
.fv-stated .fs{color:#7fd39a} .fv-partial .fs{color:#e3c37a}
.fv-contradicted .fs{color:#e88a8a} .fv-absent .fs{color:#8f9bb0}
.avg-f{border-color:#3b6b52}
.cause{margin-top:10px;display:flex;gap:10px;flex-wrap:wrap}
.cause .ch{flex:1 0 100%;font-size:11px;letter-spacing:.05em;text-transform:uppercase;
color:var(--mut);margin-bottom:2px}
.cause .ch span{text-transform:none;letter-spacing:0;color:#7d8798}
.cs{flex:1 1 300px;display:flex;gap:9px;align-items:flex-start;padding:7px 10px;
border-radius:7px;background:#1c2129;border:1px solid var(--line)}
.cs .sc{font-size:19px;font-weight:700;line-height:1.15;white-space:nowrap}
.cs .sc small{font-size:10px;font-weight:400;color:var(--mut)}
.cs .cl{font-size:10.5px;color:var(--mut);white-space:nowrap;line-height:1.3}
.cs .cq{font-size:11px;color:#8f9bb0;min-width:0}
.cs .cq i{color:#c3cbd8}
.cause .nocause{flex:1 1 100%;font-size:11.5px;color:#7d8798;font-style:italic;
padding:6px 10px;border:1px dashed var(--line);border-radius:7px}
.patt{margin-top:14px;padding-top:12px;border-top:1px dashed var(--line)}
.patt h4{margin:0 0 6px;font-size:12px;letter-spacing:.06em;text-transform:uppercase;
color:var(--mut)}
pre{margin:0;white-space:pre-wrap;font:12.5px/1.5 ui-monospace,Consolas,monospace;
background:#12141a;border:1px solid var(--line);border-radius:5px;padding:9px 11px;
overflow-x:auto}
.none{color:var(--mut)}
.nearmiss{margin-top:9px;border-top:1px dashed var(--line);padding-top:8px}
.nm-title{font-size:11px;letter-spacing:.05em;text-transform:uppercase;color:var(--mut);
margin-bottom:6px}
.nm-item{background:#12141a;border:1px solid var(--line);border-left:2px solid #4a4f5c;
border-radius:5px;padding:7px 10px;margin-bottom:6px}
.nm-id{font:600 11.5px ui-monospace,Consolas,monospace;color:#b8c2d4}
.cname{font-weight:700}
.nm-n{font-size:11px;color:var(--mut);margin-left:8px}
.nm-what{font-size:12.5px;margin-top:4px}
.nm-why{font-size:12px;color:#d0a68a;margin-top:3px}
"""

JS = """
// il pannello aperto e' fissato alla viewport: deve partire sotto la testata,
// la cui altezza cambia se i filtri vanno a capo -> la misuro invece di indovinarla
const hdr=document.querySelector('header.top');
function placePanels(){
  document.documentElement.style.setProperty('--panel-top',
    (hdr.getBoundingClientRect().height + 8) + 'px');
}
placePanels();
addEventListener('resize', placePanels);
document.querySelectorAll('details.sub').forEach(d=>d.addEventListener('toggle', ()=>{
  placePanels();
  if(d.open) document.querySelectorAll('details.sub').forEach(o=>{ if(o!==d) o.open=false; });
}));
const btns=document.querySelectorAll('.filters button');
btns.forEach(b=>b.addEventListener('click',()=>{
  btns.forEach(x=>x.classList.remove('on')); b.classList.add('on');
  const f=b.dataset.f;
  document.querySelectorAll('.card').forEach(c=>{
    let show = f==='all'
      || ((f==='val'||f==='train') && c.dataset.split===f)
      || (f==='noerr' && c.dataset.noerr==='1')
      || (['hi','mid','lo'].includes(f) && c.dataset.band===f)
      || c.dataset.verdict===f;
    c.style.display = show ? '' : 'none';
  });
}));
"""

if STALE:
    STALE_NOTE = ('<br><b>Voti e testo:</b> su <b>%d</b> schede la narrativa &egrave; stata '
                  'rigenerata dopo il giudizio, quindi il voto accanto si riferisce al testo '
                  'precedente e non a quello mostrato; quelle schede lo dichiarano. La media '
                  'qui sopra li comprende.' % len(STALE))

PRIMA_DOPO = (
    '<b>Prima/dopo:</b> sulle stesse 66 immagini, aggiungere il contorno delle regioni '
    'corrette, i dettagli persi e &laquo;inglobata&raquo; al posto di &laquo;mancata&raquo; '
    'ha portato il voto da 5.7 a 6.7 (+1.0), con il guadagno maggiore sulla '
    '<b>correttezza</b> (+0.53/4): 32 immagini migliorate, 4 peggiorate, 30 uguali. '
    'Il pavimento del controllo &egrave; salito da 1.6 a 2.1 &mdash; le frasi nuove sono '
    'pi&ugrave; generiche e combaciano un po&#39; con qualunque annotazione che parli di '
    'bordi &mdash; ma lo scarto sopra di esso &egrave; comunque cresciuto, da 4.0 a 4.4. '
) if IS_REF else ""

INTRO = ('{n} immagini annotate a mano con maschera ground-truth: <b>{v}</b> in <b>val</b> '
         '(lo split su cui gira la pipeline) e <b>{t}</b> in <b>train</b>, processate per '
         'induzione delle slice gi&agrave; certificate. Su <b>{z}</b> immagini il Blocco 1 non '
         'trova errori tipati: l&igrave; il blocco dei pattern &egrave; vuoto e resta la sola '
         'narrativa, che parla delle regioni corrette (nota sul contorno, sotto). I verdetti '
         'ALTA/PARZIALE/BASSA sono quelli assegnati a mano e ci sono solo per le immagini di val.'
         '<br><b>Riga Blocco 1:</b> sotto ogni intestazione c&#39;&egrave; la tassonomia tipata '
         'di <i>tutti</i> gli errori dell&#39;immagine, non solo di quelli finiti in un pattern '
         'certificato &mdash; &egrave; l&#39;informazione che spiega molti disaccordi, perch&eacute; '
         'un errore <code>boundary</code> viene rilevato e tipizzato dal Blocco 1 anche quando '
         'nessuna slice certificata lo nomina.'
         '<br><b>Narrativa generata:</b> accanto all&#39;annotazione umana c&#39;&egrave; ora un '
         'testo generato che racconta <i>ogni</i> errore tipato nello stesso registro degli '
         'annotatori &mdash; tipo d&#39;errore, classe e posizione nella scena &mdash; ed &egrave; '
         'quello direttamente confrontabile con l&#39;annotazione. Il blocco sotto, con i pattern '
         'sistematici, resta ristretto agli errori che ricadono in una slice certificata.'
         '<br><b>Nota sul contorno:</b> l&#39;IoU d&#39;area con cui il Blocco 1 decide '
         '&laquo;corretto&raquo; &egrave; scala-dipendente &mdash; sugli oggetti pi&ugrave; grandi '
         'sta in media 0.60 sopra il Boundary IoU, contro 0.22 sui pi&ugrave; piccoli &mdash; '
         'quindi una regione pu&ograve; essere &laquo;corretta&raquo; a IoU 0.97 con un contorno '
         'praticamente sbagliato. Ora il Boundary IoU &egrave; riportato accanto all&#39;IoU sugli '
         'errori di bordo, e le regioni <i>corrette ma tracciate male</i> ricevono una frase '
         'dedicata: sono <b>{n_contour}</b> immagini (val + train36), di cui <b>{n_contour_noerr}</b> '
         'senza alcun errore tipato, che prima non producevano alcun testo &mdash; comprese '
         '{bassa_contour}. Nulla viene '
         'riclassificato come errore: la tassonomia e la popolazione di riferimento restano '
         'intatte.'
         '<br><b>Score del giudice:</b> quanto il testo generato combacia con l&#39;annotazione '
         'umana, che fa da riferimento. Giudice = <b>Gemma-4-12B in locale</b>, rubrica 0-10 '
         '<b>Completezza 4 + Correttezza 4 + Specificit&agrave; 2</b>, che fa 10. Il punteggio '
         '&egrave; la somma dei tre criteri, quindi ogni suo punto si pu&ograve; verificare '
         'guardando i criteri accanto. '
         'Il giudice dava anche un voto d&#39;insieme, che per&ograve; <i>non</i> coincide con la '
         'somma dei suoi stessi criteri (in media 1.0 punti sotto, uguale solo nel 30% dei casi): '
         '&egrave; un secondo giudizio, pi&ugrave; severo, ed &egrave; riportato solo nel report. '
         'Da tenere presente che la <b>specificit&agrave;</b> prende {spec_c:.2f}/2 anche sui testi di '
         'un&#39;altra immagine &mdash; la narrativa usa linguaggio preciso a prescindere &mdash; '
         'mentre completezza e correttezza scendono a {comp_c:.2f} e {corr_c:.2f}: sono quelle due a portare il '
         'segnale. <b>Come tararlo:</b> le stesse annotazioni, accoppiate alla descrizione di '
         'un&#39;<i>altra</i> immagine e mescolate a queste, prendono in media <b>{cn:.1f}/10</b> '
         '(narrativa) e <b>{cp:.1f}/10</b> (pattern): quello &egrave; il pavimento del rumore, e la '
         'media sulle coppie vere &egrave; <b>{gn:.1f}/10</b> e <b>{gp:.1f}/10</b>. '
         '{prima_dopo}'
         '<b>Nota:</b> il giudice penalizzava come &laquo;inventati&raquo; gli IoU e i conteggi dei '
         'buchi, assenti dall&#39;annotazione umana ma veri; il prompt ora dice che il dettaglio in '
         'pi&ugrave; conta contro solo se <i>contraddice</i> il riferimento.'
         '<br><b>Pattern candidati:</b> dove la descrizione cita un gruppo &laquo;vicino ma non certificato&raquo;, ora si legge che cosa contiene e su quale gate &egrave; stato respinto &mdash; distinguere &laquo;non ho visto nulla&raquo; da &laquo;ho visto qualcosa ma non posso dimostrarlo&raquo; &egrave; il punto.'
         '<br><b>Nota:</b> le immagini di train sono state usate per addestrare il segmentatore, '
         'quindi i suoi errori l&igrave; non sono rappresentativi; il confronto vale sul '
         '<i>come</i> si descrive, non sul <i>quanto</i> il modello sbaglia.'
         + CAUSE_NOTE + CLAIMS_NOTE + STALE_NOTE + RUN_NOTE
         ).format(n=stats["val"] + stats["train"], v=stats["val"], t=stats["train"],
                  z=stats["noerr"], prima_dopo=PRIMA_DOPO, **LEG)

# la media di pagina: il voto del giudice sulla narrativa, su quante immagini
# ne hanno effettivamente una (le altre non hanno errori tipati da raccontare)
_avg = sum(page_scores) / len(page_scores) if page_scores else float("nan")
AVG = ('<div class="avg"><div class="avgn">%.1f<small>/10</small></div>'
       '<div class="avgl">media del giudice<br><span>su %d immagini con narrativa'
       '</span></div></div>' % (_avg, len(page_scores))) if SHOW_AVG else ""

if cause_j:
    AVG += "".join(
        '<div class="avg avg-c"><div class="avgn">%.1f<small>/10</small></div>'
        '<div class="avgl">plausibilit&agrave; della causa<br>'
        '<span>errore raccontato %s &middot; %d immagini</span></div></div>'
        % (m, lab, n)
        for m, lab, n in ((CAUSE_MEAN_NAR, "dalla narrativa", CAUSE_N_NAR),
                          (CAUSE_MEAN_HUM, "dall&#39;annotazione", CAUSE_N_HUM)))
if CL and (CL["summary"].get("all_narrative") or CL["summary"].get("voc_narrative")):
    # every annotated image, wherever it comes from (VOC val, VOC train, synthetic)
    _s = CL["summary"].get("all_narrative") or CL["summary"]["voc_narrative"]
    if CLM and CLM["summary"].get("tutte"):
        _m = CLM["summary"]["tutte"]
        AVG += ('<div class="avg avg-f"><div class="avgn">%.1f<small>/10</small></div>'
                '<div class="avgl">voto per fatti medio<br><span>umano vs narrativa della pipeline '
                '&middot; %d immagini annotate &middot; verdetti di Claude (Gemma %.1f)</span></div></div>'
                % (_m["genuine"], _m["n"], _m["gemma"]))
    else:
        AVG += ('<div class="avg avg-f"><div class="avgn">%.1f<small>/10</small></div>'
                '<div class="avgl">voto per fatti medio<br><span>umano vs narrativa della pipeline '
                '&middot; %d immagini annotate</span></div></div>' % (_s["genuine"], _s["n"]))
if AVG:
    AVG = '<div class="avgs">%s</div>' % AVG

# ---------------------------------------------------------------------------
# sezione speciale: le 21 immagini a maschere sintetiche
#
# Vale piu' delle altre per una ragione sola: dieci delle sue annotazioni
# dichiarano la predizione PERFETTA. Le 80 di VOC descrivono tutte un difetto,
# quindi misurano quanto la pipeline vede e mai quanto tace quando non c'e'
# nulla da vedere. Qui la classe negativa esiste.
# ---------------------------------------------------------------------------
SYN = os.path.join(ROOT, r"datasets\cropped_synthetic_masks")
MUTO = "No segmentation error was found in this image."
syn_cards, syn_stats = [], None
if os.path.isdir(SYN):
    s_mp = {int(k): v.replace(".jpg", "") for k, v in
            json.load(io.open(os.path.join(SYN, "voc_patch__id_matches.json"),
                              encoding="utf-8")).items()}
    s_hum, s_order = {}, []
    for line in io.open(os.path.join(SYN, "answer_gts.jsonl"), encoding="utf-8"):
        if line.strip():
            o = json.loads(line)
            s_hum[s_mp[o["img_idx"]]] = (o["img_idx"], o["content"].strip())
            s_order.append(s_mp[o["img_idx"]])
    s_nar = json.load(io.open(os.path.join(SQ, r"synth\narratives_synth.json"),
                              encoding="utf-8"))
    s_judge, s_jall = {}, []
    _jp = os.path.join(SQ, "synth", SYNTH_JUDGE)
    if os.path.exists(_jp):
        s_jall = json.load(io.open(_jp, encoding="utf-8"))["judgements"]
        for r in s_jall:
            if r.get("pairing") == "genuine" and "score" in r:
                s_judge.setdefault(r["image"], {})[r.get("kind", "narrative")] = r
    # il ramo dei pattern (cluster + SliceLens indotti) sul set sintetico
    s_patt = {}
    _dp = os.path.join(SQ, r"synth\descriptions_synth.json")
    if os.path.exists(_dp):
        s_patt = {k: v.get("text", "")
                  for k, v in json.load(io.open(_dp, encoding="utf-8")).items()}
        # a synthetic image with an induced pattern HAS a cause: without this the
        # card would claim none was certified when the judge simply has not run
        with_cause.update(k for k, v in json.load(io.open(_dp, encoding="utf-8")).items()
                          if v.get("patterns"))

    def _self_iou(nm):
        """IoU fra predizione e ground truth della stessa immagine.

        Sulle 11 immagini corrotte del primo lotto vale 0.002-0.284, perche' la
        predizione e' la GT di un'ALTRA immagine VOC (accoppiamento per posizione
        in lista). Misurarlo qui rende l'avviso automatico: se la cartella viene
        rigenerata correttamente, l'avviso sparisce senza toccare il codice.
        """
        g = np.array(Image.open(os.path.join(SYN, "SegmentationClass",
                                             nm + ".png")).resize((256, 256), Image.NEAREST))
        p = np.array(Image.open(os.path.join(SYN, "SegmentationClass_prediction",
                                             nm + ".png")).resize((256, 256), Image.NEAREST))
        fg = lambda a: (a > 0) & (a != 255)
        u = (fg(g) | fg(p)).sum()
        return float((fg(g) & fg(p)).sum()) / u if u else 0.0

    ORPHAN = 0.35          # sotto questa soglia la predizione e' scollegata dalla GT
    orphans = [nm for nm in s_order if _self_iou(nm) < ORPHAN]

    n_good = n_good_silent = n_bad = n_bad_spoke = 0
    for nm in s_order:
        rec = s_judge.get(nm, {}).get("narrative", {})
        good = bool(rec.get("declared_good"))
        spoke = s_nar.get(nm, {}).get("text", "") != MUTO
        if good:
            n_good += 1
            n_good_silent += (not spoke)
        else:
            n_bad += 1
            n_bad_spoke += spoke

        o = rcc(Image.open(os.path.join(SYN, "JPEGImages", nm + ".jpg")).convert("RGB"),
                SIDE, Image.LANCZOS)
        g = rcc(Image.open(os.path.join(SYN, "SegmentationClass",
                                        nm + ".png")).convert("RGB"), SIDE, Image.NEAREST)
        pr = rcc(Image.open(os.path.join(SYN, "SegmentationClass_prediction",
                                         nm + ".png")).convert("RGB"), SIDE, Image.NEAREST)
        idx, htxt = s_hum[nm]
        sc = rec.get("score")
        band = "na" if sc is None else ("hi" if sc >= 7 else ("lo" if sc <= 2 else "mid"))
        jud = ""
        if sc is not None:
            crit = " ".join(
                '<span class="c"><b>%s</b>%g/%d</span>' % (lab, rec[k], hi)
                for k, hi, lab in (("completeness", 4, "completezza "),
                                   ("correctness", 4, "correttezza "),
                                   ("specificity", 2, "specificit&agrave; ")))
            jud = ('<div class="jud j-%s"><span class="sc">%.1f<small>/10</small></span>'
                   '<div class="jd"><div class="crit">%s<span class="eq">= %g</span></div>'
                   '<div class="why">%s</div></div></div>'
                   % (band, sc, crit, sc, escape(rec.get("rationale", ""))))
        tag = ('<span class="b v-ALTA">dichiarata corretta</span>' if good
               else '<span class="b v-BASSA">dichiarata errata</span>')
        if nm in orphans:
            tag += ('<span class="b orphan">predizione non appartenente '
                    'a questa immagine</span>')
        syn_cards.append('''
<article class="card" data-split="synth" data-verdict="" data-noerr="0" data-band="{band}">
  <header><span class="idx">{idx}</span><span class="nm">{nm}</span>
    <span class="b split-synth">sintetica</span>{tag}</header>
  <div class="panels">
    <figure><img src="data:image/jpeg;base64,{a}" alt="immagine"><figcaption>immagine</figcaption></figure>
    <figure><img src="data:image/jpeg;base64,{b}" alt="ground truth"><figcaption>ground truth</figcaption></figure>
    <figure><img src="data:image/jpeg;base64,{c}" alt="predizione"><figcaption>predizione sintetica</figcaption></figure>
  </div>
  <div class="texts">
    <section><h4>Annotazione umana</h4><p>{hum}</p></section>
    <section class="gen"><h4>Narrativa generata dalla pipeline</h4>{jud}{fnar}<p>{nar}</p></section>
  </div>
  <div class="patt"><h4>Pattern sistematici (slice certificate su val, indotte qui)</h4>{patt}</div>
</article>'''.format(
            band=band, idx=idx, nm=nm, tag=tag, a=b64(o), b=b64(g), c=b64(pr),
            hum=escape(htxt), jud=jud, fnar=facts_html("synth", nm, "narrative"),
            nar=escape(s_nar.get(nm, {}).get("text", "")),
            patt=(("<pre>%s</pre>" % escape(s_patt[nm]) + cause_html(nm)) if s_patt.get(nm)
                  else '<em class="none">nessun pattern indotto</em>')))

    _n = {n: s_judge[n]["narrative"] for n in s_order
          if n in s_judge and "narrative" in s_judge[n]}
    gs = [r["score"] for r in _n.values() if r.get("declared_good")]
    bs = [r["score"] for r in _n.values() if not r.get("declared_good")]
    syn_stats = dict(orphans=len(orphans), n=len(s_order), good=n_good,
                     good_silent=n_good_silent,
                     bad=n_bad, bad_spoke=n_bad_spoke,
                     mg=sum(gs) / len(gs) if gs else 0.0,
                     mb=sum(bs) / len(bs) if bs else 0.0)
    # the numbers the caveat paragraphs quote, from the judge file itself
    _good = {r["image"] for r in s_jall if r.get("declared_good")}
    _cn = [r for r in s_jall if r.get("kind", "narrative") == "narrative"
           and r.get("pairing") == "control" and r.get("score") is not None]
    _cg = [r for r in _cn if r["image"] in _good]
    _same = [r for r in _cg if r.get("control_from") in _good]
    _m = lambda rows: sum(r["score"] for r in rows) / len(rows) if rows else 0.0
    _pg = [r["score"] for r in s_jall if r.get("kind") == "pattern" and r.get("pairing") == "genuine"
           and r.get("score") is not None]
    syn_stats.update(
        same=len(_same), cg=len(_cg), floor_with=_m(_cg),
        floor_without=_m([r for r in _cg if r not in _same]),
        floor_bad=_m([r for r in _cn if r["image"] not in _good]),
        pat=_m([r for r in s_jall if r.get("kind") == "pattern" and r.get("pairing") == "genuine"
                and r.get("score") is not None]),
        pat_good=_m([r for r in s_jall if r.get("kind") == "pattern" and r.get("pairing") == "genuine"
                     and r.get("score") is not None and r["image"] in _good]),
        pat_bad=_m([r for r in s_jall if r.get("kind") == "pattern" and r.get("pairing") == "genuine"
                    and r.get("score") is not None and r["image"] not in _good]),
        nar=_m([r for r in s_jall if r.get("kind", "narrative") == "narrative"
                and r.get("pairing") == "genuine" and r.get("score") is not None]),
        n_cluster=LEG["n_cluster"], n_lang=LEG["n_lang"])
    print("sezione sintetica: %d schede" % len(syn_cards))

WARN = ""
if syn_cards and syn_stats["orphans"]:
    WARN = ('''<div class="warn"><b>Avviso: {orphans} di queste {n} immagini hanno una
    predizione che non appartiene loro.</b> La maschera &laquo;predetta&raquo; &egrave;
    la <i>ground truth di un&#39;altra immagine VOC</i>, ritagliata al centro e
    incollata: con IoU 1.000 in cinque casi su undici, cio&egrave; copie esatte. Le
    sorgenti sono id consecutivi in ordine crescente (2007_002597, 2007_002611,
    2007_002639, ...), la firma di un accoppiamento per <b>posizione in lista</b>
    invece che per contenuto.
    <br><br>Di conseguenza <b>i numeri qui sotto non misurano quello che sembrano</b>
    sulla met&agrave; &laquo;errata&raquo;: la pipeline descrive fedelmente ci&ograve;
    che c&#39;&egrave; nei file, ma quei file non rappresentano un errore di
    segmentazione realistico. Restano invece validi i risultati sulla met&agrave;
    dichiarata corretta, dove la predizione &egrave; la ground truth della propria
    immagine. Le schede coinvolte portano una targhetta arancione.
    <br><br>Questo riquadro &egrave; generato da un controllo sui dati
    (IoU predizione/GT della stessa immagine &lt; {thr}) e sparir&agrave; da solo
    quando la cartella sar&agrave; rigenerata.</div>''').format(
        orphans=syn_stats["orphans"], n=syn_stats["n"], thr="0.35")

SYNTH = ""
if syn_cards and not SHOW_SYNTH_INTRO:
    # only the cards: they carry "sintetica", "dichiarata corretta/errata" and the
    # orange orphan badge, so they still read on their own
    SYNTH = '<section class="synthblock">%s</section>' % "".join(syn_cards)
elif syn_cards:
    SYNTH = '''<section class="synthblock">
  <h2>Set sintetico &mdash; 21 immagini con maschere alterate</h2>
  {warn}
  <div class="synthintro">
    <p>Predizioni costruite a tavolino, con annotazioni che si dividono in due
    met&agrave; <b>dichiarate</b>: {bad} descrivono fallimenti catastrofici, {good}
    dichiarano la predizione <b>perfetta</b>. Sono queste ultime a rendere prezioso il
    set: le 80 annotazioni di VOC descrivono tutte un difetto, quindi misurano quanto la
    pipeline <i>vede</i> e mai quanto <i>tace</i> quando non c&#39;&egrave; nulla da
    vedere. Qui la classe negativa esiste.</p>
    <div class="synthnums">
      <div class="sn"><b>{good_silent}/{good}</b><span>dichiarate corrette,<br>la pipeline tace</span></div>
      <div class="sn"><b>{bad_spoke}/{bad}</b><span>dichiarate errate,<br>la pipeline parla</span></div>
      <div class="sn"><b>{mg:.1f}/10</b><span>voto medio<br>sulle corrette</span></div>
      <div class="sn"><b>{mb:.1f}/10</b><span>voto medio<br>sulle errate</span></div>
    </div>
    <p class="cav"><b>Una cautela sul controllo.</b> Su questo set l&#39;accoppiamento di
    controllo &egrave; meno severo del solito su una met&agrave;: se a un&#39;immagine
    dichiarata perfetta si presta il testo di <i>un&#39;altra</i> immagine perfetta, quel
    testo dice comunque il vero, perch&eacute; &laquo;nessun errore&raquo; &egrave; la
    risposta giusta per entrambe. Succede in {same} casi su {cg} e porta il pavimento del rumore
    da <b>{floor_without:.2f}</b> a {floor_with:.2f}; sulle immagini errate, dove il problema non si pone, resta
    <b>{floor_bad:.2f}</b>. Per la classe negativa una coppia disallineata non &egrave;
    automaticamente una coppia sbagliata.</p>
    <p class="cav">Il ramo dei pattern &egrave; stato <b>indotto</b> anche qui, con lo
    stesso metodo di <code>train36</code>: le {n_cluster} slice cluster certificate su
    <code>val</code> applicate per centroide pi&ugrave; vicino, le {n_lang} ipotesi SliceLens
    validate ancorate con Qwen. Nessuna ri-certificazione: 21 immagini non certificano
    nulla. Nel giudizio il ramo pattern prende {pat:.1f}/10 ({pat_good:.1f} sulle corrette, {pat_bad:.1f} sulle
    errate) contro {nar:.1f} della narrativa.</p>
  </div>
  {cards}
</section>'''.format(cards="".join(syn_cards), warn=WARN, **syn_stats)

html = """<!doctype html>
<html lang="it"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Confronto annotazioni umane vs pipeline</title>
<style>%s</style></head><body>
<header class="top">
  <div class="hrow">
    <h1>Annotazioni umane vs descrizioni della pipeline</h1>
    %s
  </div>
  <div class="filters">
    <details class="sub"><summary>Come leggere questa pagina</summary>
    <div class="subtext">%s</div></details>
    <details class="sub prompt"><summary>Il prompt dato a Gemma</summary>
    <div class="subtext">%s</div></details>
    <button data-f="all" class="on">tutte</button>
    <button data-f="val">val</button>
    <button data-f="train">train</button>
    <button data-f="ALTA">ALTA</button>
    <button data-f="PARZIALE">PARZIALE</button>
    <button data-f="BASSA">BASSA</button>
    <button data-f="noerr">senza errori</button>
    <button data-f="synth">sintetiche</button>
    <button data-f="hi">giudice &ge;7/10</button>
    <button data-f="mid">giudice 3-6</button>
    <button data-f="lo">giudice &le;2</button>
  </div>
</header>
<main id="list">%s%s</main>
<script>%s</script></body></html>""" % (CSS, AVG, INTRO, PROMPT, SYNTH, "".join(cards), JS)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w", encoding="utf-8").write(html)
print("scritto", OUT)
print("schede: %d | val: %d | train: %d | senza errori: %d"
      % (len(cards), stats["val"], stats["train"], stats["noerr"]))
print("dimensione: %.1f MB" % (os.path.getsize(OUT) / 1e6))
