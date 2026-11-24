# -*- coding: utf-8 -*-
"""Panoramica in italiano della pipeline, per il relatore.

Documento 'a grandi linee ma chiaro': obiettivo, idea dei due rami, figura,
passo-per-passo, metriche, paper di riferimento, integrazione, risultati.
Il dettaglio tecnico completo resta in pipeline_implementation.docx (EN).
"""
import os

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

DOCS = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(DOCS, "pipeline_figure_it.png")

doc = Document()
st = doc.styles["Normal"]
st.font.name = "Calibri"
st.font.size = Pt(11)


def h(text, level=1):
    doc.add_heading(text, level=level)


def p(text, italic=False):
    par = doc.add_paragraph(text)
    if italic:
        for r in par.runs:
            r.font.italic = True
    return par


def bullet(text):
    doc.add_paragraph(text, style="List Bullet")


def table(headers, rows, widths=None):
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, htxt in enumerate(headers):
        cell = t.rows[0].cells[j]
        cell.text = htxt
        for r in cell.paragraphs[0].runs:
            r.font.bold = True
    for i, row in enumerate(rows, 1):
        for j, val in enumerate(row):
            t.rows[i].cells[j].text = str(val)
    if widths:
        for j, w in enumerate(widths):
            for row in t.rows:
                row.cells[j].width = Inches(w)
    for row in t.rows:
        for cell in row.cells:
            for par in cell.paragraphs:
                for r in par.runs:
                    r.font.size = Pt(9.5)
    doc.add_paragraph()


# ============================ TITOLO ============================
title = doc.add_heading(
    "Scoperta e descrizione automatica di error-slice per la segmentazione", 0)
p("Panoramica della pipeline implementata — Daniele Spini", italic=True)

# ============================ 1. OBIETTIVO ============================
h("1. L'obiettivo, in una frase")
p("Dato un modello di segmentazione e un dataset di validazione, il codice individua i "
  "gruppi SISTEMATICI di errori (le “error slice”) e produce, per ogni immagine in cui il "
  "modello sbaglia, una descrizione testuale del perché: la SITUAZIONE che induce "
  "l'errore e il TIPO di errore commesso. Non “il modello ha sbagliato qui”, ma “il "
  "modello tende a non rilevare gli oggetti tagliati dal bordo dell'immagine — e in "
  "questa immagine è successo a questi due oggetti”.")
p("Esempio reale di output per una singola immagine:")
ex = doc.add_paragraph()
run = ex.add_run(
    'Image 2010_003531: 14 wrong objects, 11 in systematic patterns and 3 isolated.\n'
    'Systematic patterns:\n'
    '  - objects mostly of class ‘chair’ not detected (missed)  [cluster]\n'
    '  - "an object cut off at the image border" (SliceLens) -> not detected  [language]')
run.font.name = "Consolas"
run.font.size = Pt(9)

# ============================ 2. L'IDEA ============================
h("2. L'idea centrale: due vie di scoperta complementari")
p("Il cuore del lavoro è che gli errori sistematici vengono cercati per DUE strade "
  "indipendenti, che rispondono a due domande diverse:")
bullet("RAMO CLUSTER (visivo, bottom-up): raggruppa gli errori che si assomigliano — "
       "stesso aspetto visivo, stessa firma d'errore. Risponde a “COME fallisce il "
       "modello?” (quali oggetti, con quale tipo di errore).")
bullet("RAMO LINGUISTICO / SliceLens (semantico, top-down): parte da ipotesi in "
       "linguaggio naturale (“un oggetto occluso da un altro”, “un oggetto piccolo”) e le "
       "verifica sulle immagini con un modello visione-linguaggio. Risponde a “IN QUALE "
       "SITUAZIONE fallisce?”.")
p("I due rami sono DELIBERATAMENTE indipendenti: nessuno dei due usa i risultati "
  "dell'altro per formarsi. Si incontrano solo alla fine, nella validazione. Questo ha due "
  "conseguenze importanti: (1) quando concordano su un pattern è una conferma incrociata "
  "genuina, non un artefatto (non-circolarità); (2) empiricamente tagliano lo spazio "
  "degli errori lungo assi diversi — visivo vs situazionale — quindi nessuno dei due è "
  "ridondante: insieme coprono più di ciascuno da solo.")
p("Entrambi i rami poggiano sullo stesso SUBSTRATO: le istanze d'errore tipate estratte "
  "dal Blocco 1. Ogni istanza ha un identificativo univoco, quindi i risultati dei due "
  "rami si fondono con un join esatto, senza euristiche di riallineamento.")

# ============================ 3. FIGURA ============================
h("3. La pipeline")
doc.add_picture(FIG, width=Inches(6.5))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
cap = p("I colori indicano il paper di provenienza di ogni blocco; in arancione i "
        "contributi di questo lavoro.", italic=True)
cap.alignment = WD_ALIGN_PARAGRAPH.CENTER

# ============================ 4. PASSO PER PASSO ============================
h("4. Che cosa succede, passo per passo")

h("Blocco 1 — Dare un nome a ogni errore (Beyond IoU)", 2)
p("Il punto di partenza è semplice: confrontare la predizione del modello con la verità "
  "(GT), oggetto per oggetto. Le maschere vengono spezzate in regioni connesse — ogni "
  "regione è un oggetto — e ogni oggetto sbagliato riceve un'etichetta che dice CHE TIPO "
  "di errore è:")
table(
    ["Tipo", "In parole povere"],
    [["FN (miss)", "l'oggetto c'era, il modello non l'ha visto"],
     ["FP (allucinazione)", "il modello ha visto un oggetto che non c'è"],
     ["boundary", "oggetto trovato, ma i bordi sono imprecisi"],
     ["extent", "oggetto trovato, ma ne manca un pezzo (o ce n'è in più)"],
     ["confusion", "oggetto trovato, ma con la classe sbagliata (cane → gatto)"],
     ["split", "un oggetto spezzato in più pezzi separati"],
     ["merge", "più oggetti fusi in un'unica maschera"]],
    widths=[1.6, 5.1])
p("Questo blocco è deterministico (solo geometria delle maschere, nessuna rete neurale) "
  "e produce l'unità di lavoro di TUTTO il resto della pipeline: l'elenco delle istanze "
  "d'errore, ognuna con tipo, gravità (IoU), classe e un identificativo univoco. Gli "
  "oggetti segmentati bene (IoU ≥ 0.85) vengono tenuti da parte come termine di "
  "paragone. Su VOC: 6406 errori e 522 oggetti corretti.")

h("Ramo cluster (Blocchi 2–4) — trovare gli errori che si assomigliano", 2)
p("L'intuizione: un modello non sbaglia a caso. Se mettiamo insieme gli errori che si "
  "somigliano, i gruppi che emergono sono i suoi punti deboli ricorrenti. Il ramo lavora "
  "in tre passi:")
bullet("1. L'IMPRONTA VISIVA. Ogni oggetto sbagliato viene trasformato in un vettore di "
       "numeri (embedding) da DINOv2, una rete pre-addestrata e congelata che cattura "
       "l'aspetto: due oggetti che si somigliano (colore, forma, contesto) hanno vettori "
       "vicini. Il vettore si calcola guardando SOLO i pixel dell'oggetto (pooling sulla "
       "maschera), non tutta l'immagine; per gli oggetti mancati (FN), che non hanno "
       "maschera predetta, si usa la maschera vera.")
bullet("2. IMPRONTA + FIRMA DELL'ERRORE. All'aspetto si affianca il tipo e la gravità "
       "dell'errore. Così due oggetti finiscono nello stesso gruppo solo se si "
       "assomigliano E sbagliano nello stesso modo — è la differenza tra “gruppo di "
       "biciclette” e “gruppo di biciclette che il modello non vede”.")
bullet("3. RAGGRUPPAMENTO IN DUE LIVELLI. Prima si separa per tipo di errore (tutti gli "
       "FN da una parte, tutti i boundary dall'altra, …): ogni gruppo parlerà di UN solo "
       "tipo di problema. Poi, dentro ogni tipo, un algoritmo di clustering (Gaussian "
       "Mixture) raggruppa per aspetto. Quanti gruppi formare non lo decide una scelta "
       "manuale: lo sceglie un criterio statistico (BIC) che premia i gruppi che spiegano "
       "bene i dati senza frammentarli inutilmente.")
p("Il risultato sono le slice candidate del ramo visivo: ad esempio “oggetti piccoli e "
  "scuri che il modello non rileva” — dove “piccoli e scuri” emerge dai dati, non da una "
  "regola scritta a mano. Questo ramo risponde a: COME fallisce il modello.")

h("Ramo linguistico — SliceLens (in parallelo)", 2)
p("L'intuizione opposta: invece di far emergere i gruppi dai dati, si parte da SOSPETTI "
  "formulati in linguaggio naturale e si verifica quali sono fondati. Tre passi:")
bullet("1. LA LISTA DEI SOSPETTI. Un modello di linguaggio (Gemma, locale) genera ipotesi "
       "di situazioni difficili: “un oggetto piccolo”, “un oggetto occluso da un altro”, "
       "“un oggetto tagliato dal bordo dell'immagine”, più ipotesi specifiche per classe.")
bullet("2. IL RISCONTRO SULLE IMMAGINI (grounding). Per ogni coppia (sospetto, oggetto), "
       "un modello visione-linguaggio (Qwen) guarda l'immagine con l'oggetto evidenziato "
       "e risponde: quanto questo oggetto corrisponde alla situazione descritta? Ne esce "
       "un punteggio da 0 a 1 (la slice-confidence).")
bullet("3. LA PROVA (slope-trend, dal paper SliceLens). Un sospetto è promosso a pattern "
       "sistematico solo se vale la relazione: PIÙ un oggetto somiglia alla situazione, "
       "PIÙ il modello ci sbaglia. Se il tasso d'errore non cresce con la somiglianza, "
       "il sospetto era infondato e viene scartato.")
p("I sospetti promossi vengono poi ri-verificati su TUTTE le istanze delle loro classi "
  "(non su un campione), così ogni immagine che contiene davvero la situazione riceve la "
  "sua descrizione. Questo ramo risponde a: IN QUALE SITUAZIONE fallisce il modello.")

h("Slice-Quality Vector e validazione (Blocchi 5–6) — la pagella", 2)
p("Ogni slice candidata, di entrambi i rami, riceve una pagella a 6 voci (la tabella "
  "nella sezione 5): quanto è grave, quanto è coesa, quanto è affidabile. Poi ogni ramo "
  "ha il suo criterio di promozione, perché le domande sono diverse: le slice visive "
  "devono essere pure — e “pura” non vuol dire superare una percentuale fissata a "
  "tavolino, ma battere il caso: la coerenza dei suoi attributi deve essere più alta di "
  "quella di gruppi casuali di errori dello stesso tipo (p ≤ 0.05) — abbastanza grandi e "
  "statisticamente significative rispetto agli altri errori; le slice linguistiche "
  "devono aver superato la prova errore-vs-somiglianza. Un accorgimento trasversale: la "
  "gravità si confronta sempre TRA ERRORI DELLO STESSO TIPO — un errore di bordo ha per "
  "natura IoU alto e, con una soglia unica, non potrebbe mai competere con un oggetto "
  "mancato del tutto; normalizzare per tipo rende il confronto equo.")

h("Refine e residual peeling (Blocco 7) — sbucciare la cipolla", 2)
p("Le slice gravi ma confuse (un miscuglio di cose diverse) non vengono buttate: si "
  "prova a spezzarle in sotto-gruppi più puliti e a ri-valutarle. E quando una slice "
  "viene promossa, le sue istanze si tolgono dal tavolo e il clustering riparte su ciò "
  "che resta — come sbucciare una cipolla: prima emergono i pattern dominanti, poi, "
  "strato dopo strato (fino a 5 round), quelli più fini che i dominanti nascondevano.")

h("Spiegazione per immagine (Blocco 8) — il deliverable", 2)
p("Infine si torna alle immagini. Qwen etichetta un campione di ogni slice promossa con "
  "attributi (small / dark / occluded / …), e per ogni immagine con errori si scrive che "
  "cosa le è successo, aggregando i pattern a cui appartengono i SUOI oggetti: dal ramo "
  "cluster frasi come “small objects of class ‘bicycle’ not detected”, dal ramo language "
  "“‘a cow cut off at the image border’ → not detected” (l'ipotesi generica viene "
  "specializzata sulla classe dominante di quell'immagine). Gli errori isolati (fuori "
  "da ogni slice certificata) non restano muti: il testo indica il pattern candidato più "
  "vicino. Output finali: descrizioni per immagine (JSON + testo), gallery HTML "
  "navigabile, indici slice→immagini e descrizione→immagini, e per ogni slice un "
  "consiglio di riparazione (che dati aggiungere o aumentare).")

# ============================ 5. METRICHE ============================
h("5. Le sei metriche del Slice-Quality Vector")
p("Coprono tre proprietà: quanto è GRAVE la slice (severità), quanto è COESA (una cosa "
  "sola, non un miscuglio), quanto è AFFIDABILE (non un artefatto del caso).")
table(
    ["Asse", "Che cosa misura", "Domanda a cui risponde", "Fonte"],
    [["ω² (omega-squared)",
      "effect size della disparità di performance: quanta varianza spiega la partizione "
      "slice-vs-resto",
      "Quanto è grave?", "SEG4SEG"],
     ["Manifold compactness",
      "frazione dei k-vicini (grafo kNN su DINOv2) che restano dentro la slice",
      "È una regione coesa dello spazio visivo?", "MCSD"],
     ["Misprediction coherence",
      "1 − entropia normalizzata degli output sbagliati",
      "Sbagliano tutti nello stesso modo?", "CB-SLICE"],
     ["Error-type purity",
      "frazione del tipo d'errore dominante",
      "È un solo tipo di errore?", "nuovo (da Beyond IoU)"],
     ["Slice purity",
      "accordo sull'attributo semantico dominante (tag Qwen), contando solo i tag "
      "informativi e testato CONTRO IL CASO: la slice deve essere più coerente di "
      "gruppi casuali di errori dello stesso tipo (test di permutazione)",
      "È una situazione identificabile?", "SEG4SEG (Def.1), adattata"],
     ["Significatività",
      "bootstrap a blocchi (per immagine) del divario di severità vs gli ALTRI errori; "
      "per i tipi a severità costante (FN/FP), test di permutazione della compattezza; "
      "correzione FDR di Benjamini-Hochberg",
      "Potrebbe essere caso?", "nuovo"]],
    widths=[1.4, 2.6, 1.6, 1.1])
p("Due note metodologiche che il codice tratta esplicitamente: (a) la manifold "
  "compactness è quasi tautologica per le slice del ramo cluster (formate proprio per "
  "vicinanza), ma è un test GENUINO per le slice language, nate dalla semantica — per "
  "questo i due rami si validano a vicenda; (b) contro il rischio di “double dipping” "
  "(slice scoperte e testate sugli stessi dati) c'è una validazione split-half: le slice "
  "scoperte su metà delle immagini vengono ri-certificate sull'altra metà — nella "
  "configurazione finale replica il 94.1% (16/17).")

# ============================ 6. PAPER ============================
h("6. I paper di riferimento e che cosa si prende da ciascuno")
table(
    ["Paper", "Che cosa propone", "Che cosa usiamo"],
    [["SliceLens (Apple, 2025)",
      "error-slice discovery groundata per task multi-istanza: ipotesi LLM + verifica "
      "VLM (Eq. 1) + slope-trend",
      "l'intero ramo linguistico (generazione, grounding, verifica di sistematicità)"],
     ["Beyond IoU",
      "decomposizione dell'errore di segmentazione in boundary / extent / segment",
      "il Blocco 1; esteso a un matching a 7 casi per istanze (split/merge/confusion)"],
     ["MCSD",
      "manifold compactness come criterio di qualità delle slice + residual peeling",
      "l'asse di compattezza e il ciclo di peeling"],
     ["CB-SLICE",
      "coerenza della mis-predizione + explanation card",
      "l'asse di coerenza dell'output e lo schema di spiegazione"],
     ["SEG4SEG",
      "clustering GMM pesato dalla performance + criterio di successo Def.1 + ω²",
      "il clustering del ramo visivo (reimplementato), Def.1 e ω²"]],
    widths=[1.3, 2.6, 2.8])
p("Contributi propri di questo lavoro (in arancione nella figura): il substrato condiviso "
  "di istanze tipate con join esatto tra i rami; la tassonomia a 7 casi con soglia di "
  "frammento; il Slice-Quality Vector a 6 assi; il clustering stratificato con selezione "
  "BIC; i gate di validazione consapevoli del tipo d'errore; la verifica statistica "
  "(bootstrap, permutazione, FDR, split-half); le descrizioni testuali per immagine; la "
  "corroborazione incrociata tra i rami; la gallery e gli indici consultabili.")

# ============================ 7. INTEGRAZIONE ============================
h("7. L'integrazione dei due rami: le tre idee chiave")
bullet("UNA SOLA ESTRAZIONE, DUE VIE DI SCOPERTA. Le istanze tipate del Blocco 1 sono "
       "l'unità condivisa: il cluster le raggruppa per aspetto, SliceLens le interroga "
       "per situazione, e i risultati si fondono per identificativo, senza ambiguità.")
bullet("INDIPENDENZA (impostazione open-world). La generazione delle ipotesi NON è "
       "condizionata dai cluster trovati (scelta deliberata): questo preserva la "
       "capacità di scoprire situazioni che lo spazio visivo non separa, e rende la "
       "concordanza tra i rami una validazione incrociata non circolare. Empiricamente i "
       "due rami risultano complementari: quasi mai la stessa slice — il visivo e il "
       "situazionale sono assi diversi dello stesso spazio di errori.")
bullet("A OGNI RAMO IL SUO TEST. Le slice visive si certificano con purezza + "
       "significatività contro i pari; le slice linguistiche con la verifica "
       "errore-vs-confidence del paper SliceLens. Usare il test dell'uno per l'altro "
       "produce risultati vacui o distorti — è stata una delle lezioni principali "
       "dell'implementazione.")

# ============================ 8. RISULTATI ============================
h("8. Risultati sul banco di prova (VOC2012 val, LR-ASPP MobileNetV3)")
table(
    ["Quantità", "Valore"],
    [["Immagini / errori tipati / riferimenti corretti", "1449 / 6406 / 522"],
     ["Slice cluster certificate", "14 (FN 13, FP 1) — ogni slice batte dimostrabilmente "
       "il raggruppamento casuale sugli attributi"],
     ["Ipotesi linguistiche validate (20 classi, grounding completo: ~24.000 "
       "valutazioni VLM)", "35, tutte verificate dallo slope-trend"],
     ["Immagini con descrizione situazionale", "826 su 1232 immagini con errori (67%)"],
     ["Copertura sistematica (unione dei due rami, istanze d'errore coperte)", "36.7%"],
     ["Errore residuo non spiegato (frazione di severità fuori dalle slice)", "0.755"],
     ["Replicazione split-half (ramo cluster)", "94.1% (16/17)"]],
    widths=[4.3, 2.4])
p("Dove guardare: la gallery HTML (viste per-slice e per-immagine, con originale / GT / "
  "predizione), gli indici slices_index e descriptions_index (ogni descrizione con "
  "l'elenco completo delle sue immagini), e il report tecnico esteso in inglese "
  "(pipeline_implementation.docx) con ogni blocco, formula e decisione di progetto.",
  italic=True)

# ============================ 9. SECONDO MODELLO ============================
h("9. Il metodo trasferisce? Verifica su un segmentatore molto più forte")
p("Ogni risultato sopra è vero di UN modello. Per rispondere se le situazioni "
  "scoperte siano proprietà del compito o solo di quel modello, l'intera pipeline "
  "è stata rieseguita, senza toccare soglie o parametri, su DeepLabV3-ResNet101 "
  "(pesi pre-addestrati, nessun training; mIoU 0.82 contro 0.48 del modello di "
  "riferimento).")
table(
    ["Quantità", "MobileNetV3 (riferimento)", "DeepLabV3-R101 (più forte)"],
    [["Errori tipati / riferimenti corretti", "6406 / 522", "4818 / 883"],
     ["Slice cluster certificate", "14 (FN 13, FP 1)", "11 (FN 9, FP 1, boundary 1)"],
     ["Ipotesi linguistiche ri-certificate", "35 / 35", "31 / 35"],
     ["Immagini con un pattern certificato", "826 / 1232 (67%)", "698 / 1091 (64%)"],
     ["Copertura sistematica (unione dei due rami)", "36.7%", "34.7%"]],
    widths=[3.2, 2.0, 2.4])
p("Due letture. Primo: la composizione degli errori cambia natura, non solo "
  "quantità — le allucinazioni quasi si dimezzano, ma gli errori di bordo "
  "aumentano in numero assoluto e per la prima volta si certifica una slice di "
  "tipo boundary (mai successo sul modello debole). Secondo: il metodo trasferisce "
  "senza ritocchi — 11 slice contro 14, con purezza mediana più alta (0.62 "
  "contro 0.48) — e le 4 ipotesi linguistiche che non si ri-certificano "
  "(occlusione su horse, parti sottili su car, somiglianza di categoria su cat, "
  "piccolezza su tvmonitor) sono situazioni plausibilmente più rare quando il "
  "modello sbaglia meno.", italic=True)



# ===========================================================================
# 10. Dal tipo d'errore alla narrativa umana, e un giudice a rubrica
#     (settembre 2026). Numeri da docs/_revision_numbers.load_narrative_stage.
# ===========================================================================
import sys as _sys
_sys.path.insert(0, DOCS)
from _revision_numbers import load_narrative_stage as _load_stage
try:
    S1 = _load_stage()
except Exception as _ex:
    S1 = None
    print("  (numeri della fase narrativa non disponibili, sezione 10 saltata:", _ex, ")")

if S1:
    _nar, _pat, _hol, _sy, _sc = (S1["narrative"], S1["pattern"], S1["holistic"],
                                  S1["synth"], S1["synth_control"])

    h("10. Dal tipo d'errore alla narrativa umana, e un giudice a rubrica (settembre 2026)")
    p("Mettere il testo per immagine della pipeline accanto a quello che scrive una persona "
      "ha mostrato che non erano lo stesso genere di testo. La descrizione del Blocco 8 parla "
      "solo degli errori caduti in una slice certificata; gli annotatori descrivono ogni "
      "errore che vedono, in un registro fisso — che tipo di errore, su quale oggetto, dove "
      "nella scena. Tutto quello che segue nasce dal prendere quel registro come obiettivo: "
      "un secondo testo generato che racconta OGNI errore tipato, i fatti di maschera che gli "
      "servono e che il Blocco 1 non registrava, e un giudice che misura l'accordo con una "
      "rubrica. I numeri di questa sezione sono quelli del run di riferimento "
      "(outputs/slice_quality); la sezione 11 rivede poi il Blocco 1 stesso.")

    h("La narrativa: una frase per ogni gruppo (tipo d'errore, classe)", 2)
    p("describe_errors.narrate traduce in inglese gli errori tipati di un'immagine con delle "
      "regole, senza alcun modello di linguaggio: il testo è una resa del Blocco 1 e dei fatti "
      "di maschera, quindi ogni frase risale a un numero. Le istanze si raggruppano per "
      "(sottotipo, classe), i gruppi si ordinano come fanno gli annotatori (mancati, "
      "allucinati, confusi, merge, split, estensione, contorno), se ne scrivono al massimo "
      "sei e il resto si conta. Ogni frase porta una parola di taglia (tiny / small / large, "
      "dalla quota di fotogramma della regione di ground truth — non della predizione, o un "
      "oggetto sotto-segmentato leggerebbe «tiny»), una posizione su una griglia 3×3 più i "
      "bordi («on the bottom-left», «on the right edge», «scattered across the scene») e i "
      "numeri che la graduano: IoU d'area e, accanto, boundary IoU.")
    p("Il registro: «The two small ground truth BOAT regions on the right edge have been "
      "missed entirely.» — «Three tiny CHAIR patches have been hallucinated, scattered across "
      "the scene.» — «The ground truth DOG region in the center of the scene has been "
      "under-segmented, with whole parts of the region missing (IoU 0.47, boundary IoU "
      "0.31).»", italic=True)

    h("I fatti di maschera che il Blocco 1 non registra (mask_facts.py)", 2)
    p("Il Blocco 1 tipizza un errore e lo gradua con [E_bnd, E_ext, E_seg]; non dice com'è "
      "FATTO l'errore, ed è quello che la gente scrive. mask_facts misura il resto, in modo "
      "deterministico, dalle maschere:")
    bullet("Boundary IoU (Cheng et al., 2021): l'IoU delle due maschere ristretto a una banda "
           "di larghezza 2d intorno ai contorni, d = 1% della diagonale (≈14.7 px a 520). "
           "Riportato come attributo continuo accanto all'IoU d'area, mai usato per "
           "riclassificare. Il motivo è la scala: l'IoU d'area dipende dalla scala e su un "
           "oggetto grande resta alta anche con il contorno sbagliato — sul val lo scarto "
           "IoU − boundary IoU cresce da +0.22 sugli oggetti piccoli a +0.60 sui grandi "
           "(2011_002685: IoU 0.968, boundary IoU 0.042).")
    bullet("Note di contorno sulle regioni CORRETTE. Una regione a IoU ≥ 0.85 non produce "
           "errori e quindi, prima, nessun testo; tutte e sei le immagini giudicate BASSA a "
           "mano («la pipeline non ha visto nulla e doveva») contengono esattamente una "
           "regione così. La nota («The large ground truth TRAIN region is correctly located, "
           "but its outline is loosely followed (boundary IoU 0.26)») scatta sotto una soglia "
           "ancorata alla distribuzione misurata del boundary IoU sulle 522 regioni di "
           "riferimento: sotto la mediana (0.39) se l'immagine non ha alcun errore tipato, "
           "sotto il primo quartile (0.30) se ne ha, per non diluirli. Le soglie sono state "
           "calibrate sulle 79 annotate, e la calibrazione ha deciso la forma: l'etichetta "
           "«l'annotazione parla del bordo» non discrimina (67% di prevalenza, AUC 0.512: "
           "vince la maggioranza), il verdetto BASSA sì (AUC 0.802), e le istanze BASSA "
           "stanno dentro il grosso della distribuzione, non in una coda — da qui soglie di "
           f"segnalazione per quantile, non una classe d'errore nuova. {S1['contour']['all']} "
           f"immagini ricevono una nota ({S1['contour']['val']} del val, le altre in train36), "
           f"{S1['contour']['noerr']} delle quali non hanno alcun errore tipato e prima non "
           "producevano testo.")
    bullet("Dettagli sottili persi: un frammento mancato è un DETTAGLIO se un'apertura "
           "morfologica di raggio 2 lo cancella (sotto 15 px è rumore di bordo) — è ciò che "
           "separa un'antenna, una zampa, la punta di un'ala da una toppa mancante, che il "
           "termine di estensione già copre. Si riporta da 8 frammenti in su.")
    bullet("Regioni di ground truth sotto soglia: le componenti sotto 200 px l'adapter le "
           "scarta prima del Blocco 1, quindi non sono mai state istanze né mancate; si "
           "contano a 15 px e fino al 26 settembre 2026 si dichiaravano anche nel testo. Quella "
           "frase, come la gemella sulle toppe predette sotto soglia, parlava dello strumento e "
           "non dell'immagine, ed è stata tolta dalla narrativa (capitolo 11): la misura "
           "resta e viaggia nei metadati.")
    bullet("Buchi interni alla maschera predetta (≥ 20 px), il SEGNO di un errore di "
           "estensione (sotto- o sovra-, dai pixel mancati e in eccesso, che il vettore di "
           "Beyond IoU non porta) e — per ogni FN — quanta parte ne copre davvero una "
           "predizione della sua classe, che trasforma «mancata del tutto» in «inglobata in "
           "una predizione BOAT vicina: coperta al 66%, ma senza un contorno proprio». È il "
           "fatto che ha portato alla sezione 11.")
    p("Anche il testo dei pattern sistematici ha ricevuto gli stessi fatti come riga di "
      "dettaglio sotto ogni pattern, così i due testi di una scheda non si contraddicono.")

    h("Il giudice: una rubrica, una somma e un controllo negativo", 2)
    p("Per le narrative è stato costruito un giudice a rubrica: Gemma-4-12B-it, in locale, "
      "decodifica greedy, che riceve due testi e nient'altro — non vede l'immagine, quindi "
      "misura l'ACCORDO con l'annotazione umana, che fa da riferimento; dove l'annotatore "
      "sbaglia, la pipeline paga. La rubrica è completezza 0–4, correttezza 0–4, specificità "
      "0–2, e il voto riportato ovunque è la loro SOMMA, così ogni punto si verifica sui "
      "criteri accanto. Il giudice dà anche un voto d'insieme, che non coincide con la somma "
      f"dei suoi stessi criteri (scarto medio {_hol['mean_gap']:+.2f} su {_hol['n']} giudizi, "
      f"sotto in {_hol['below']} casi): un secondo giudizio, più severo, tenuto nel report e "
      "non usato come voto. Il prompt dice esplicitamente che un dettaglio in più assente "
      "dal riferimento — IoU, buchi, posizioni — non è un'invenzione e conta contro solo se "
      "CONTRADDICE: una versione precedente penalizzava numeri veri come allucinazioni.")
    p("Il controllo negativo: ogni annotazione viene accoppiata anche, con una permutazione "
      "senza punti fissi a seme fisso, al testo di un'ALTRA immagine; le coppie di controllo "
      "sono mescolate alle vere e indistinguibili. Quello che il giudice dà su di esse è il "
      "pavimento del rumore, e solo lo scarto sopra vale come evidenza.")
    table(["Testo", "Immagini", "Coppie vere", "Controlli", "Scarto"],
          [["narrativa", str(_nar["n"]), f"{_nar['genuine']:.2f}", f"{_nar['control']:.2f}",
            f"{_nar['genuine'] - _nar['control']:+.2f}"],
           ["pattern sistematici", str(_pat["n"]), f"{_pat['genuine']:.2f}",
            f"{_pat['control']:.2f}", f"{_pat['genuine'] - _pat['control']:+.2f}"]],
          widths=[1.8, 0.9, 1.2, 1.2, 0.9])
    p(f"Sui controlli la specificità paga ancora {_nar['crit_c']['specificity']:.2f}/2 — la "
      "narrativa usa un linguaggio preciso a prescindere — mentre completezza e correttezza "
      f"scendono a {_nar['crit_c']['completeness']:.2f} e {_nar['crit_c']['correctness']:.2f}: "
      f"sono quelle due a portare il segnale. La narrativa fa {S1['split']['val']:.2f} sulle "
      f"43 immagini di val e {S1['split']['train']:.2f} sulle 36 di train; contro i verdetti "
      f"a mano fa {S1['by_verdict'].get('ALTA', (0, 0))[0]:.2f} sulle ALTA, "
      f"{S1['by_verdict'].get('PARZIALE', (0, 0))[0]:.2f} sulle PARZIALE e "
      f"{S1['by_verdict'].get('BASSA', (0, 0))[0]:.2f} sulle BASSA — le BASSA sono le più "
      "alte perché le note di contorno sono state scritte proprio per loro. Misurato "
      "prima/dopo sulle stesse 66 immagini mentre si aggiungevano i fatti di maschera, il "
      "voto è passato da 5.7 a 6.7 (+1.0), con il guadagno maggiore sulla correttezza "
      "(+0.53/4): 32 immagini migliorate, 4 peggiorate, 30 uguali; il pavimento del "
      "controllo è salito da 1.6 a 2.1, ma lo scarto è cresciuto da 4.0 a 4.4.")
    p("Una lezione sullo strumento, costata un run: la prima ripetizione con la rubrica "
      "cambiata ha dato risultati identici al centesimo, perché la sostituzione del prompt "
      "era fallita in silenzio e la cache aveva restituito i voti vecchi. Ora il prompt si "
      "stampa e si controlla prima di lanciare, ogni rubrica scrive su un nome proprio e "
      "(sezione 11) la cache è indicizzata anche per testo.")

    h("Un set sintetico con la classe negativa", 2)
    p("Le 80 annotazioni di VOC descrivono tutte un difetto: misurano quanto la pipeline "
      "VEDE, mai quanto tace quando non c'è nulla da vedere. Un set di 21 immagini con "
      "maschere alterate a mano colma quel vuoto: 11 annotazioni descrivono fallimenti "
      "catastrofici, 10 dichiarano la predizione perfetta. Stessa catena (Blocco 1, fatti di "
      "maschera, narrativa; pattern per induzione delle slice certificate e delle ipotesi "
      "validate, senza ri-certificazione: 21 immagini non certificano nulla) e stesso "
      f"giudice. Sulla metà dichiarata corretta entrambi i testi fanno "
      f"{_sy['narrative']['good']:.2f}: la pipeline dice «No segmentation error was found in "
      f"this image» e il giudice concorda. Sull'intero set la narrativa fa "
      f"{_sy['narrative']['all']:.2f} (metà errata {_sy['narrative']['bad']:.2f}) e i pattern "
      f"{_sy['pattern']['all']:.2f} ({_sy['pattern']['bad']:.2f}).")
    p("Due cautele fanno parte del risultato. Il controllo è meno severo sulla classe "
      "negativa: se a un'immagine perfetta si presta il testo di un'altra immagine perfetta, "
      f"quel testo dice comunque il vero — succede in {_sc['same']} controlli su "
      f"{_sc['good_n']} e alza il pavimento da {_sc['floor_without']:.2f} a "
      f"{_sc['floor_with']:.2f}; sulla metà errata resta {_sc['floor_bad']:.2f}. E un "
      "controllo sui dati ha trovato che le prime 11 predizioni del set sono le ground truth "
      "di ALTRE immagini VOC incollate (id consecutivi, IoU 1.000 in cinque casi): un "
      "accoppiamento per posizione in lista, non per contenuto. La pagina di confronto "
      "genera l'avviso da quel controllo e lo toglierà da sola quando la cartella sarà "
      "rigenerata; su quelle 11 immagini la pipeline descrive fedelmente un artefatto, non "
      "un errore di segmentazione realistico.")

# ===========================================================================
# 11. Revisione del Blocco 1 dopo il confronto con le annotazioni umane
# I numeri vengono dai file dei due run (docs/_revision_numbers.py), non
# sono ricopiati a mano.
# ===========================================================================
import sys as _sys
_sys.path.insert(0, DOCS)
from _revision_numbers import load as _load_rev
try:
    N = _load_rev()
except Exception as _ex:
    N = None
    print("  (numeri della revisione non disponibili, sezione 11 saltata:", _ex, ")")

if N:
    J, C = N["judge"], N["criteria"]
    tb, ta = N["tax_before"], N["tax_after"]

    h("11. Revisione del Blocco 1 dopo il confronto con le annotazioni umane (settembre 2026)")
    p("Le sezioni precedenti descrivono la pipeline com'era quando è stata valutata. Il "
      "confronto delle sue narrative per immagine con le 79 annotazioni scritte a mano ha "
      "fatto emergere un difetto del Blocco 1 che nessuna metrica aggregata mostrava. La "
      "correzione cambia la popolazione degli errori, quindi è stata eseguita come "
      "configurazione separata e riproducibile (claim_cov = 0.5, cartella "
      "outputs/slice_quality_cov05b): il run di riferimento resta intatto e i due si "
      "confrontano istanza per istanza.")

    h("Che cosa non andava", 2)
    p("La frase più frequente nelle narrative generate era «la regione X è stata mancata "
      "del tutto». Misurata sulle maschere delle 80 immagini annotate, era falsa più spesso "
      "che vera: dei 223 FN, il 52% era coperto per almeno metà da una predizione della "
      "propria classe, un altro 8% da una predizione di un'altra classe, e solo il 27% era "
      "davvero sfondo. L'oggetto non era stato mancato: era stato inglobato in un blob "
      "vicino e aveva perso il suo contorno — che è esattamente quello che scrivono gli "
      "annotatori («le due barche sono fuse», «la persona è etichettata come moto»).")
    p("La causa è il test di claim del matcher a due stadi: split e merge scattano su IoU "
      "≥ 0.25 fra predizione e regione GT, e l'IoU è cieca al contenimento asimmetrico. Una "
      "GT piccola inglobata al 70% in un blob che copre anche una GT grande ha IoU ≈ 0.1 con "
      "quel blob, non lo rivendica e finisce in FN. Beyond IoU stesso, il paper che il blocco "
      "segue, chiama «mancato» un segmento solo se nessun suo pixel è un vero positivo: la "
      "regola a livello di istanza era più severa di quella a livello di pixel che voleva "
      "riprodurre.")

    h("La regola nuova", 2)
    p("Per le regioni che la regola IoU lascia orfane si aggiunge un claim per "
      "contenimento: una GT rivendica anche la predizione che ne contiene almeno metà; una "
      "predizione della stessa classe contenuta per almeno metà in una GT la rivendica. A "
      "0.5 il contenitore è unico (due maschere disgiunte non possono tenere più di metà "
      "della stessa regione), quindi il criterio è ben posto quanto τ_match = 0.5. Ne "
      "seguono tre cose:")
    bullet("una predizione il cui unico claim è una GT che contiene diventa la coppia 1↔1 di "
           "quella GT (boundary / extent / confusion) invece di un FP più un FN;")
    bullet("una GT il cui miglior claimer le sta quasi tutto dentro viene accoppiata a quello, "
           "e ogni altro claimer è un frammento (split);")
    bullet("una predizione con due o più claim è un merge, come prima — ma ogni GT che ingloba "
           "viene TENUTA come istanza propria (sottotipo MERGE, lato GT, embedding sulla "
           "maschera GT come un FN). Un merge produce quindi 1 + M istanze: un doppio "
           "conteggio dichiarato, che sostituisce quello non dichiarato di prima, quando ogni "
           "maschera sovra-estesa produceva un FP e un FN.")
    p("Perché tenere gli oggetti inglobati come istanze: il primo tentativo non lo faceva e "
      "il ramo cluster ha perso il suo substrato. Tredici delle quattordici slice cluster "
      "certificate del run di riferimento erano slice di FN; ripiegare 1 800 istanze FN "
      "dentro i loro blob ne ha lasciate quattro, perché l'embedding del blob è dominato "
      "dall'oggetto grande che per lo più è, non da quello piccolo che ha inghiottito. Una "
      "slice come «le persone piccole accanto a quelle grandi vengono assorbite» si trova "
      "solo dal lato degli oggetti inglobati. Tenerli ha riportato le slice certificate a "
      f"{N['slices_after']['cluster']}, di cui {N['slices_after']['cluster_types'].get('merge', 0)} "
      "di tipo MERGE fatte in maggioranza di oggetti inglobati.")
    p("Trovato per strada anche un bug: un merge su una predizione già accoppiata per IoU "
      "non toglieva le GT inglobate dai FN, che venivano contate due volte. Con claim_cov "
      "disattivo il codice riproduce il run di riferimento istanza per istanza, salvo questo.")

    h("Effetto sulla popolazione degli errori (VOC2012 val)", 2)
    table(["Tipo", "Regola IoU (riferimento)", "Regola per contenimento (revisione)"],
          [["FN", str(tb["fn"]), f"{ta['fn']}  (−{100 - 100 * ta['fn'] // tb['fn']}%)"],
           ["FP", str(tb["fp"]), str(ta["fp"])],
           ["merge (blob)", str(tb["merge"]), str(N["merge_blobs"])],
           ["merge (oggetti inglobati, lato GT)", "—", str(N["merge_gt_side"])],
           ["split", str(tb["split"]), str(ta["split"])],
           ["extent / boundary", f"{tb['extent']} / {tb['boundary']}",
            f"{ta['extent']} / {ta['boundary']}"],
           ["confusion", str(tb["confusion"]), str(ta["confusion"])],
           ["correct (popolazione di riferimento)", str(N["ref_before"]), str(N["ref_after"])]],
          widths=[2.6, 1.9, 2.4])
    p(f"{N['pairs']} coppie sono nate per contenimento, ciascuna un FP e un FN prima. "
      f"{N['merge_mixed']} dei {N['merge_blobs']} blob mescolano classi — un tavolo "
      "inglobato in un blob «sedia» è un merge e una confusione insieme, e la narrativa "
      "ora lo dice. Nota di metodo: "
      f"{N['ref_moved']} dei {N['ref_before']} riferimenti CORRECT "
      f"({100 * N['ref_moved'] / N['ref_before']:.0f}%) diventano MERGE perché il blob, pur a "
      "IoU ≥ 0.85 sul proprio oggetto, ingloba anche una GT piccola: la popolazione di "
      f"riferimento di ω² e delle note di contorno scende a {N['ref_after']} regioni.")

    h("I merge misurati contro l'unione delle regioni fuse", 2)
    p("Una maschera fusa non ha una singola ground truth con cui confrontarsi. Il contorno "
      "di un merge veniva misurato contro la GT più simile, e la narrativa non ne diceva "
      "nulla se il blob non aveva un match IoU proprio (257 blob su 559). Il caso pulito è "
      "2007_000121: due monitor affiancati, 5 px di distanza, una sola maschera predetta. "
      "Contro ciascun monitor la maschera ha IoU 0.44 — sotto τ_match, quindi merge con "
      "entrambe le regole — e la frase si fermava a «fonde due regioni in una sola "
      "maschera». L'annotatore scriveva di piedistalli mancati e bordi meno definiti. Contro "
      "l'UNIONE dei due monitor la maschera ha IoU 0.80 e boundary IoU 0.08: la lettura "
      "dell'annotatore. Ora il Blocco 1 registra quali GT un merge ha fuso, i fatti di "
      "maschera si misurano contro la loro unione, e la frase continua: «…; taken together, "
      "the merged regions are segmented with heavily imprecise boundaries (IoU 0.80, "
      "boundary IoU 0.08)». Corretto anche l'aggettivo del contorno, che dipendeva dalla "
      "sola IoU d'area (scala-dipendente: «leggermente più grossolano» a boundary IoU 0.08): "
      "scende di un grado sotto boundary IoU 0.30, il quartile peggiore dei contorni di "
      "riferimento, e di due sotto 0.15.")

    h("Che cosa dice il giudice, appaiato sulle stesse immagini", 2)
    _r = lambda d: f"{d['a']:.2f} → {d['b']:.2f}"
    _u = lambda d: f"{d['up']} / {d['dn']} / {d['eq']}"
    table(["Testo", "Immagini", "Riferimento → revisione", "migliorate / peggiorate / uguali"],
          [["narrativa", f"tutte ({J['narrative']['all']['n']})", _r(J['narrative']['all']), _u(J['narrative']['all'])],
           ["narrativa", f"val ({J['narrative']['val']['n']})", _r(J['narrative']['val']), _u(J['narrative']['val'])],
           ["narrativa", f"train ({J['narrative']['train']['n']})", _r(J['narrative']['train']), _u(J['narrative']['train'])],
           ["pattern", f"tutte ({J['pattern']['all']['n']})", _r(J['pattern']['all']), _u(J['pattern']['all'])],
           ["pavimento del controllo, narrativa / pattern", "tutte",
            f"{J['narrative']['control_a']:.2f} / {J['pattern']['control_a']:.2f} → "
            f"{J['narrative']['control_b']:.2f} / {J['pattern']['control_b']:.2f}", "—"]],
          widths=[2.0, 1.1, 1.9, 1.9])
    p(f"Sulla narrativa il guadagno sta nei due criteri che portano il segnale: completezza "
      f"{C['completeness'][0]:.2f} → {C['completeness'][1]:.2f} e correttezza "
      f"{C['correctness'][0]:.2f} → {C['correctness'][1]:.2f} su 4; la specificità non si "
      "muove. Le immagini che guadagnano di più sono proprio i casi del contenimento "
      "(2007_000129: da 4 a 10, «four BICYCLE regions now labelled PERSON»). Il ramo dei "
      f"pattern è piatto: slice cluster {N['slices_before']['cluster']} → "
      f"{N['slices_after']['cluster']}, linguistiche {N['slices_before']['language']} → "
      f"{N['slices_after']['language']}; immagini con almeno un pattern "
      f"{N['cov_before']['with_pattern']} → {N['cov_after']['with_pattern']} su "
      f"{N['cov_after']['images']}. Quel che resta della perdita ha una causa precisa: le "
      f"{N['pairs']} coppie per contenimento hanno l'istanza dal lato della predizione, con "
      "l'embedding sul blob, dove il vecchio FN aveva la sua sulla maschera GT.")
    p(f"Set sintetico (21 immagini, metà dichiarate perfette): narrativa "
      f"{N['synth_before']['narrative']['all']:.2f} → {N['synth_after']['narrative']['all']:.2f}, "
      f"pattern {N['synth_before']['pattern']['all']:.2f} → {N['synth_after']['pattern']['all']:.2f}; "
      f"sulla metà dichiarata corretta entrambi i rami restano a "
      f"{N['synth_after']['narrative']['good']:.2f}. Il calo sta tutto nella metà dichiarata "
      "errata, le cui predizioni sono ground truth di altre immagini incollate (difetto del "
      "dataset, documentato nella pagina di confronto): lì la regola per contenimento dice "
      "«TRAIN etichettato SHEEP» dove l'annotatore ha scritto «mancato» — pixel per pixel ha "
      "ragione la regola, ma il riferimento è l'annotazione.")

    h("Due strumenti sistemati lungo la strada", 2)
    bullet("I buchi interni a una maschera predetta erano calcolati da uno script a parte solo "
           "per il val; rigenerando train36 da zero la frase «la maschera ha due buchi» "
           "spariva in silenzio, fino a 4 punti del giudice su un'immagine. Ora sono un fatto "
           "di maschera in mask_facts, misurato da ogni runner.")
    bullet("La cache del giudice era indicizzata solo per (immagine, tipo, accoppiamento): un "
           "testo rigenerato avrebbe tenuto il voto vecchio. Ora riusa un giudizio solo se il "
           "testo candidato è identico, e dopo una modifica ri-giudica solo i testi cambiati "
           "(14 minuti invece di due ore per 304 coppie).")
    p("Riproducibilità: retype_pickle.py rifà il Blocco 1 su CPU e riattacca la nuova "
      "tipizzazione agli embedding DINOv2 e ai tag Qwen già calcolati, per id di istanza (le "
      "componenti connesse non cambiano). L'intera revisione non ha richiesto alcun "
      "re-embedding; la GPU è servita solo al giudice e alle induzioni su train36 e "
      "sintetiche.")


# ---- Istanze GT dagli oggetti VOC (numeri da _revision_numbers.load_inst) ----
from _revision_numbers import load_inst as _load_inst
try:
    NI = _load_inst()
except Exception as _ex:
    NI = None
    print("  (numeri del run a oggetti non disponibili, sottosezione saltata:", _ex, ")")

if N and NI:
    _a, _j, _t = NI["amb"], NI["judge"], NI["tax"]
    h("Istanze di ground truth dagli oggetti VOC: un oggetto occluso è un oggetto solo", 2)
    p("Leggendo le narrative è emerso che sbagliano intorno all'occlusione: un oggetto "
      "coperto in parte da un altro esce come merge, come mancato, o come due oggetti. La causa "
      "è la definizione di istanza — una componente connessa della maschera di classe, perché "
      "VOC è segmentazione semantica. L'occlusione la rompe in entrambi i versi: un oggetto "
      "tagliato in due da ciò che gli sta davanti diventa due istanze (e la predizione che le "
      "copre entrambe un merge), due oggetti della stessa classe che si toccano diventano uno. "
      "VOC ha la risposta: SegmentationObject, un id per oggetto, sulle stesse immagini.")
    p(f"Misurato sul val con il substrato della pipeline: {_a['components']} componenti GT "
      f"stanno per {_a['objects']} oggetti; {_a['cut']} oggetti sono tagliati in ≥ 2 componenti "
      f"({_a['parts']} parti) in {_a['img_cut']} immagini ({100 * _a['img_cut'] / _a['images']:.0f}% "
      f"del val) — occlusione, ma anche la struttura sottile dell'oggetto stesso; "
      f"{_a['touching']} componenti contengono ≥ 2 oggetti che si toccano. Confrontati con gli "
      f"id degli oggetti, {_a['false_merge']} dei {_a['false_merge'] + _a['real_merge'] + _a['unknown_merge']} "
      f"merge del run a contenimento erano un solo oggetto occluso, e {_a['partial_miss']} delle "
      f"{_a['partial_miss'] + _a['whole_miss']} istanze «mancate» erano parti di un oggetto di cui "
      "un'altra parte era stata accoppiata.")
    p("La modifica sta tutta nell'adapter, ed è il punto: il Blocco 1 lavora su maschere e non "
      "gli importa che siano connesse. Con SQ_GT_INSTANCES=object un'istanza GT è un oggetto "
      "(classe per maggioranza dei pixel, numero di parti registrato); le predizioni restano "
      "componenti, perché il modello è semantico; le immagini senza maschera di oggetto (le "
      "sintetiche) ricadono sulle componenti. mask_facts individua poi l'OCCLUSORE di un oggetto "
      "tagliato — l'oggetto di un'altra classe che tocca almeno due delle sue parti; se nessuno "
      "lo fa, l'oggetto è spezzato dalla propria struttura e la frase tace — e la narrativa "
      "acquista un inciso: «The ground truth DININGTABLE region in the center of the scene, "
      "partly hidden behind the CHAIR, has been segmented with heavily imprecise boundaries». La "
      "dipendenza dal dataset è così esplicita: qualunque dataset con id di istanza (COCO, "
      "Cityscapes, ADE20K) ottiene il comportamento esatto dal suo adapter; uno semantico puro "
      "tiene l'approssimazione a componenti, dichiarata.")
    p(f"Popolazione: FN {ta['fn']} → {_t['fn']}, merge {N['tax_after']['merge']} → {_t['merge']} "
      f"(blob più oggetti inglobati), split {ta['split']} → {_t['split']} — un oggetto tagliato "
      "accoppiato da una parte ha ora gli altri frammenti predetti come split e non come FP —, "
      f"riferimenti {N['ref_after']} → {NI['ref']}. {NI['cut_inst']} istanze sono oggetti "
      f"tagliati; l'occlusore è stato individuato per {NI['occluded']} ("
      + ", ".join(f"{c.upper()} {k}" for c, k in NI['occluders']) +
      f"); {NI['hidden']} narrative su 1 483 portano l'inciso «partly hidden» "
      f"({NI['hidden79']} fra le 79 annotate). Solo le 664 istanze con maschera nuova sono "
      "passate da DINOv2 e Qwen; tutto il resto è stato riusato dalla cache.")
    _r = lambda d: f"{d['a']:.2f} → {d['b']:.2f}"
    _u = lambda d: f"{d['up']} / {d['dn']} / {d['eq']}"
    table(["Testo", "Immagini", "Riferimento → oggetti", "migliorate / peggiorate / uguali", "Contenimento → oggetti"],
          [["narrativa", f"tutte ({_j['narrative']['vs_ref']['n']})", _r(_j['narrative']['vs_ref']), _u(_j['narrative']['vs_ref']), _r(_j['narrative']['vs_cov'])],
           ["narrativa", f"val ({_j['narrative']['val']['n']})", _r(_j['narrative']['val']), _u(_j['narrative']['val']), "—"],
           ["narrativa", f"train ({_j['narrative']['train']['n']})", _r(_j['narrative']['train']), _u(_j['narrative']['train']), "—"],
           ["pattern", f"tutte ({_j['pattern']['vs_ref']['n']})", _r(_j['pattern']['vs_ref']), _u(_j['pattern']['vs_ref']), _r(_j['pattern']['vs_cov'])]],
          widths=[1.5, 1.1, 1.6, 1.7, 1.6])
    p(f"È la correttezza a pagare gli oggetti: {C['correctness'][0]:.2f} → {C['correctness'][1]:.2f} "
      f"→ {NI['criteria']['correctness']:.2f} su 4 nei tre run, con completezza e specificità "
      f"ferme. Le immagini di train guadagnano di più ({_j['narrative']['train']['a']:.2f} → "
      f"{_j['narrative']['train']['b']:.2f}): è lì che le frasi «il cavallo dietro la staccionata» "
      "degli annotatori venivano contraddette.")
    p(f"Il costo è di nuovo sul ramo cluster, per la stessa ragione della sezione precedente: "
      f"slice cluster certificate {N['slices_after']['cluster']} → {NI['slices']['cluster']} "
      f"(linguistiche {NI['slices']['language']}), immagini con pattern "
      f"{N['cov_after']['with_pattern']} → {NI['cov']['with_pattern']}, copertura degli errori "
      f"{N['cov_after']['share']:.2f} → {NI['cov']['share']:.2f}. Fondere le parti di un oggetto "
      f"in una sola istanza dimezza la popolazione lato GT (2 207 → {NI['gt_side']}) e fa "
      "pooling dell'embedding su una maschera più grande ed eterogenea — proprio le istanze "
      "piccole e omogenee che il GMM certificava. Il giudice non se ne accorge (pattern "
      f"{_j['pattern']['vs_ref']['a']:.2f} → {_j['pattern']['vs_ref']['b']:.2f}), perché quel voto lo "
      "reggono le righe linguistiche; il lato scoperta sì. La tipizzazione a oggetti è quella "
      "vera; il modo onesto di dirlo è che il ramo cluster, così com'è configurato, ha bisogno "
      "di molte istanze piccole — un limite che le maschere di oggetto mettono a nudo, non "
      "creano.")


# ---- Quello che gli annotatori nominano per primo (numeri da _revision_numbers.load_narr) ----
from _revision_numbers import load_narr as _load_narr
try:
    NN = _load_narr()
except Exception as _ex:
    NN = None
    print("  (numeri della revisione narrativa non disponibili, sottosezione saltata:", _ex, ")")

if N and NN:
    _jn, _cn, _k = NN["judge"]["narrative"], NN["criteria"], NN["counts"]
    h("Quello che gli annotatori nominano per primo: direzione, lato, e le regioni che vanno bene", 2)
    p("Leggendo le 79 schede del run a oggetti accanto alle note scritte a mano, ordinate per "
      f"voto del giudice, i voti bassi ({NN['low_before']} narrative a 6 o meno) si spiegano con "
      "cinque cause. In ordine di peso: (1) uno sconfinamento nello sfondo raccontato come "
      "«contorno grossolano» o «sotto-segmentato» — Beyond IoU chiama errore di bordo ogni pixel "
      "sbagliato entro 2d dal vero positivo, quindi uno sconfinamento di 20 px lungo un intero "
      "lato è «boundary» per la tassonomia e «over-extended on the right» per una persona (11 "
      "immagini); (2) una regione corretta senza alcuna frase, o la frase dei dettagli senza "
      "soggetto (6); (3) toppe della stessa classe attaccate all'oggetto dette «allucinate», "
      "dove l'annotatore vede la maschera dell'oggetto che sborda (4); (4) una toppa di classe "
      "sbagliata sotto i 200 px, scartata dall'adapter e quindi mai nominata (4); (5) ordine e "
      "posizione: sei puntini prima dell'oggetto di cui parla la nota, «scattered across the "
      "scene» per toppe tutte nella metà destra, e — sulle sintetiche — un blob su mezza "
      "immagine dopo tre piccoli mancati (3). Il resto è rumore dell'annotazione o del giudice.")
    p("La radice comune delle prime tre: la narrativa diceva quanto era sbagliato un contorno, "
      "gli annotatori dicono in che direzione e dove. Cinque modifiche, tutte in mask_facts e "
      "describe_errors, nessuna nel Blocco 1:")
    bullet("Direzione e lato su ogni coppia, merge e regione corretta, dai pixel mancati e in "
           "eccesso, come quota dell'area GT e per lati del suo riquadro: «segmented with coarser "
           "boundaries, spilling over the object on the top and right (8% of its area in "
           "excess)», «under-segmented, 82% of its area missing on the bottom». Muta sotto il 3% "
           "dell'area, che è la normale frastagliatura che il boundary IoU già gradua.")
    bullet("Una frase per ogni regione corretta non già coperta da una nota di contorno, graduata "
           "dal boundary IoU e con la stessa clausola: «segmented closely, spilling over the "
           "object on the top (5% of its area in excess) (IoU 0.93, boundary IoU 0.50)». «No "
           "segmentation error was found» resta solo per IoU ≥ 0.99, così le dieci sintetiche "
           "dichiarate perfette lo conservano. La frase dei dettagli nomina il suo oggetto.")
    bullet("Una toppa della stessa classe entro 8 px da una regione GT di quella classe è la "
           "maschera dell'oggetto che sborda, non un'allucinazione.")
    bullet("Le toppe predette sotto i 200 px di una classe che la ground truth non ha lì vengono "
           "contate in una riga di chiusura, come già le regioni GT sotto soglia. Due trappole, "
           "entrambe incontrate e corrette: la toppa va confrontata con la ground truth a soglia "
           "BASSA, altrimenti una toppa minuscola sopra una GT altrettanto minuscola della sua "
           "classe si legge come errore (le dieci sintetiche perfette erano scese a 6.6/10 "
           "proprio per questo), e i frammenti sulla banda «don't care» di VOC non sono né "
           "giusti né sbagliati.")
    bullet("Un verdetto in apertura quando il fallimento è globale — «The prediction is largely "
           "wrong: 76% of the annotated area has been labelled DOG, a class that is not in the "
           "image» — oltre metà dell'area annotata; ogni gruppo che copre il 15% del fotogramma "
           "viene detto per primo qualunque sia il suo tipo; «on the right half of the scene» "
           "quando tutte le toppe stanno in una metà; le toppe minuscole raccolte in una riga.")
    p(f"Sul val le narrative portano ora {_k['spill']} clausole di sconfinamento e {_k['missing']} "
      f"di parti mancanti su {_k['images']} immagini, {_k['correct']} frasi sulle regioni "
      f"corrette, {_k['spill_patch']} toppe che sbordano, {_k['verdict']} verdetti in apertura e "
      f"{_k['half']} posizioni «half of the scene»; nessuna immagine resta con «No segmentation "
      "error was found».")
    _r = lambda d: f"{d['a']:.2f} → {d['b']:.2f}"
    _u = lambda d: f"{d['up']} / {d['dn']} / {d['eq']}"
    table(["Testo", "Immagini", "Oggetti → narrativa rivista", "migliorate / peggiorate / uguali", "Riferimento → rivista"],
          [["narrativa", f"tutte ({_jn['vs_obj']['n']})", _r(_jn['vs_obj']), _u(_jn['vs_obj']), _r(_jn['vs_ref'])],
           ["narrativa", f"val ({_jn['val']['n']})", _r(_jn['val']), _u(_jn['val']), "—"],
           ["narrativa", f"train ({_jn['train']['n']})", _r(_jn['train']), _u(_jn['train']), "—"],
           ["pavimento del controllo", "tutte", f"{_jn['control_a']:.2f} → {_jn['control']:.2f}", "—", "—"]],
          widths=[1.5, 1.1, 1.8, 1.7, 1.6])
    p(f"Completezza {_cn['completeness'][0]:.2f} → {_cn['completeness'][1]:.2f}, correttezza "
      f"{_cn['correctness'][0]:.2f} → {_cn['correctness'][1]:.2f}, specificità "
      f"{_cn['specificity'][0]:.2f} → {_cn['specificity'][1]:.2f} su 4 / 4 / 2; "
      f"{NN['ten_before']} → {NN['ten_after']} narrative su 79 prendono 10, "
      f"{NN['low_before']} → {NN['low_after']} restano a 6 o meno, e il pavimento del controllo "
      "non si muove: il guadagno è accordo, non verbosità. Contro il run di riferimento la "
      f"narrativa passa da {_jn['vs_ref']['a']:.2f} a {_jn['vs_ref']['b']:.2f}, "
      f"{_jn['vs_ref']['up']} immagini migliorate e {_jn['vs_ref']['dn']} peggiorate. Sul set "
      f"sintetico la metà dichiarata errata sale da {NN['synth_before']['narrative']['bad']:.2f} a "
      f"{NN['synth_after']['narrative']['bad']:.2f} — il verdetto in apertura è la prima cosa che "
      f"quelle annotazioni dicono — e la metà perfetta resta a "
      f"{NN['synth_after']['narrative']['good']:.2f}. Delle {NN['low_after']} schede ancora a 6 o "
      "meno, quasi tutte sono disaccordi che le maschere non possono dirimere: una nota che "
      "nomina una classe che la predizione non contiene, o il verso opposto sullo stesso "
      "oggetto (il BUS che l'annotatore dice leggermente sotto-segmentato, dove la maschera "
      "sborda del 5% in alto).")
    from _revision_numbers import stale_cards as _stale
    p("Tolte dopo, il 26 settembre 2026: le due frasi sotto soglia — quella sulle regioni "
      "di ground truth sotto i 200 px dell'adapter e quella sulle toppe predette — dicevano "
      "che cosa fa lo STRUMENTO («below the size the pipeline represents as an object»), "
      "e la descrizione di un'immagine non deve parlare dell'attrezzo che la legge. I fatti "
      "restano misurati in mask_facts e contati nel record della narrativa: semplicemente non "
      "si dicono più, e un'immagine il cui testo era solo quella frase ora legge «No "
      "segmentation error was found in this image». Tutti i voti di questa sezione sono "
      "stati misurati con quelle frasi presenti: su %d schede delle 79 il testo è poi "
      "cambiato, e sia la pagina di confronto sia gli esempi del capitolo 12 lo dichiarano."
      % _stale())


# ============================================================================
#  12. I tre run e la revisione della narrativa, uno accanto all'altro
#      (settembre 2026). Numeri da _revision_numbers.load_runs, testi citati
#      da load_examples: in questa sezione non c'e' nulla di digitato a mano.
# ============================================================================
from _revision_numbers import load_runs as _load_runs, load_examples as _load_examples

EX_IDS = ["2010_003383", "2007_000129", "2008_006784", "2008_003703"]
try:
    RR, EX = _load_runs(), _load_examples(EX_IDS)
except Exception as _ex:
    RR = EX = None
    print("  (numeri per run non disponibili, sezione 12 saltata:", _ex, ")")

if RR and EX:
    _COLS = ["ref", "cov05b", "inst", "inst_v3"]

    def _fig(img, caption):
        path = os.path.join(DOCS, "panels", img + ".png")
        if not os.path.exists(path):
            print("  (figura mancante, saltata:", img + ")")
            return
        doc.add_picture(path, width=Inches(6.3))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        c = doc.add_paragraph()
        c.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = c.add_run(caption)
        r.italic = True
        r.font.size = Pt(9)

    def _quote(label, rec, key):
        par = doc.add_paragraph()
        par.paragraph_format.left_indent = Inches(0.25)
        r = par.add_run("%s (%.0f/10)  " % (label, rec["runs"][key]["score"]))
        r.bold = True
        r.font.size = Pt(9)
        r = par.add_run(rec["runs"][key]["text"])
        r.italic = True
        r.font.size = Pt(9)

    def _example(img, title, comment):
        rec = EX[img]
        h(title, 2)
        _fig(img, "%s — immagine, ground truth, predizione (LR-ASPP MobileNetV3), "
                  "ritagliate come le ritaglia la pipeline." % img)
        par = doc.add_paragraph()
        par.paragraph_format.left_indent = Inches(0.25)
        r = par.add_run("annotazione umana  ")
        r.bold = True
        r.font.size = Pt(9)
        r = par.add_run(rec["human"])
        r.italic = True
        r.font.size = Pt(9)
        _quote("riferimento", rec, "ref")
        _quote("contenimento", rec, "cov05b")
        _quote("oggetti", rec, "inst")
        _quote("oggetti + narrativa v3", rec, "inst_v3")
        p(comment)

    h("12. I tre run e la revisione della narrativa, uno accanto all'altro "
      "(settembre 2026)", 1)
    p("Il capitolo 11 ha introdotto tre modifiche una alla volta, e ognuna è stata eseguita "
      "in una cartella propria invece di sovrascrivere la precedente: i numeri di riferimento "
      "del capitolo 8 restano riproducibili, e due run qualunque si confrontano immagine per "
      "immagine con lo stesso giudice, le stesse annotazioni e lo stesso disegno di controllo. "
      "Questa sezione è il riassunto che serve prima di scegliere quale run riportare nella "
      "tesi: che cos'è ciascuno, che cosa ha fatto alla popolazione degli errori, che cosa "
      "ha fatto al testo e — su quattro immagini — che cosa cambia davvero per chi "
      "legge. Un avvertimento sulle citazioni qui sotto: ogni testo è letto dal file "
      "delle narrative del run com'è adesso, mentre il voto accanto è quello che "
      "il giudice ha dato al testo che gli era stato mandato. Dove i due differiscono — "
      "le frasi sotto soglia tolte il 26 settembre 2026 — il voto precede la modifica, "
      "e la pagina di confronto segnala quelle schede.")
    table(["Run (cartella)", "Che cosa cambia", "Come si produce"],
          [["outputs/slice_quality",
            "Il riferimento. Il claim è un test di IoU; un'istanza GT è una componente "
            "connessa della maschera di classe.",
            "la catena originale"],
           ["outputs/slice_quality_cov05b",
            "Claim per contenimento (SQ_CLAIM_COV=0.5): una GT rivendica anche la predizione che "
            "ne contiene almeno metà, e una predizione contenuta per metà in una GT "
            "della sua classe la rivendica; le GT inglobate restano istanze lato GT (1 + M); un "
            "merge si misura contro l'unione delle regioni che ha fuso.",
            "retype_pickle.py, CPU, nessun re-embedding"],
           ["outputs/slice_quality_inst",
            "Claim per contenimento E istanza GT = OGGETTO di VOC (SQ_GT_INSTANCES=object): un "
            "oggetto tagliato da un occlusore è una sola istanza con maschera non connessa, "
            "due oggetti che si toccano sono due. Le predizioni restano componenti.",
            "retype_instances.py, GPU solo per le 664 istanze nuove"],
           ["stessa cartella, narrativa v3",
            "Nel Blocco 1 non cambia nulla: cambia il TESTO, che guadagna verso e lato, una frase "
            "su ogni regione corretta, gli sconfinamenti, le toppe sotto soglia e un verdetto in "
            "apertura.",
            "narrate_errors.py, poi di nuovo il giudice"]],
          widths=[1.5, 3.6, 1.5])
    _t = {k: RR[k]["tax"] for k in ("ref", "cov05b", "inst")}
    table(["Errori tipati sul val", "riferimento", "contenimento", "oggetti"],
          [[n] + [str(_t[k].get(n, 0)) for k in ("ref", "cov05b", "inst")]
           for n in ("fn", "fp", "boundary", "extent", "confusion", "split", "merge")]
          + [["istanze d'errore in totale"]
             + [str(RR[k]["n_err"]) for k in ("ref", "cov05b", "inst")],
             ["riferimenti CORRECT (ω², note di contorno)"]
             + [str(RR[k]["n_ref"]) for k in ("ref", "cov05b", "inst")]],
          widths=[2.4, 1.4, 1.4, 1.4])
    p("Le popolazioni non sono confrontabili istanza per istanza, ed è il motivo per cui i "
      "run stanno separati. Con la regola dell'IoU %d regioni risultano «mancate»; il "
      "claim per contenimento ne trasforma la maggior parte in merge che il testo può "
      "raccontare (%d istanze di merge, i blob più gli oggetti che hanno inglobato) e in "
      "coppie 1↔1 che prima erano un FP e un FN, da cui vengono anche %d split. Le istanze "
      "a oggetti dimezzano poi i mancati rimasti (%d → %d), perché un oggetto "
      "agganciato da una sua parte non è più mancato dalle altre, e tagliano i merge "
      "(%d → %d), perché un blob che copre due parti dello stesso oggetto non fonde "
      "nulla." % (_t["ref"]["fn"], _t["cov05b"]["merge"], _t["cov05b"]["split"],
                  _t["cov05b"]["fn"], _t["inst"]["fn"], _t["cov05b"]["merge"],
                  _t["inst"]["merge"]))
    table(["Risultato (stesse 79 immagini annotate)", "riferim.", "contenim.", "oggetti",
           "ogg. + v3"],
          [["narrativa, voto del giudice"] + ["%.2f" % RR[k]["narrative"]["score"] for k in _COLS],
           ["    completezza / 4"] + ["%.2f" % RR[k]["criteria"]["completeness"] for k in _COLS],
           ["    correttezza / 4"] + ["%.2f" % RR[k]["criteria"]["correctness"] for k in _COLS],
           ["    specificità / 2"] + ["%.2f" % RR[k]["criteria"]["specificity"] for k in _COLS],
           ["    a 10 / a 6 o meno"] + ["%d / %d" % (RR[k]["ten"], RR[k]["low"]) for k in _COLS],
           ["    pavimento del controllo"]
           + ["%.2f" % RR[k]["narrative"]["control"] for k in _COLS],
           ["testo dei pattern, voto"] + ["%.2f" % RR[k]["pattern"]["score"] for k in _COLS],
           ["slice cluster certificate"] + [str(RR[k]["slices"]["cluster"]) for k in _COLS],
           ["slice linguistiche"] + [str(RR[k]["slices"]["language"]) for k in _COLS],
           ["quota di errori coperta da un pattern"]
           + ["%.2f" % RR[k]["cov"]["share"] for k in _COLS],
           ["sintetiche, metà dichiarata errata"]
           + ["%.2f" % RR[k]["synth_scores"]["narrative"]["bad"] for k in _COLS],
           ["sintetiche, metà dichiarata perfetta"]
           + ["%.2f" % RR[k]["synth_scores"]["narrative"]["good"] for k in _COLS]],
          widths=[2.5, 1.0, 1.1, 0.9, 1.0])
    p("Basta leggere la prima riga per vedere il compromesso di tutta la revisione: la narrativa "
      "sale %.2f → %.2f → %.2f → %.2f mentre le slice cluster certificate scendono "
      "%d → %d → %d e la quota di errori coperta da un pattern scende %.2f → %.2f. "
      "Sono lo stesso fatto visto dai due lati: ogni modifica FONDE istanze che prima erano "
      "separate — una GT inglobata e il blob che la tiene, le parti di un oggetto occluso "
      "— e il GMM certificava slice proprio a partire da quelle istanze piccole, omogenee e "
      "numerose. Il lato descrizione diventa più vero, il lato scoperta ha meno gruppi da "
      "certificare. Il pavimento del controllo si muove appena nelle quattro colonne: il "
      "guadagno non è verbosità."
      % (RR["ref"]["narrative"]["score"], RR["cov05b"]["narrative"]["score"],
         RR["inst"]["narrative"]["score"], RR["inst_v3"]["narrative"]["score"],
         RR["ref"]["slices"]["cluster"], RR["cov05b"]["slices"]["cluster"],
         RR["inst"]["slices"]["cluster"], RR["ref"]["cov"]["share"], RR["inst"]["cov"]["share"]))
    _c, _c3 = RR["ref"]["text_counts"], RR["inst_v3"]["text_counts"]
    p("Lo stesso spostamento dentro le 1 483 narrative: «missed entirely» compare in %d "
      "con la regola dell'IoU e in %d con le istanze a oggetti; «absorbed into a neighbouring "
      "prediction», il fatto da cui è partita tutta la revisione, %d → %d (è "
      "diventato un merge con un nome); merge raccontati contro la loro unione %d → %d; "
      "«partly hidden behind» 0 → %d; clausole di sconfinamento %d → %d; "
      "verdetti in apertura 0 → %d; e «No segmentation error was found», che era "
      "tutto il testo su %d immagini, %d → %d."
      % (_c["missed"], _c3["missed"], _c["absorbed"], _c3["absorbed"], _c["union"], _c3["union"],
         _c3["hidden"], _c["spill"], _c3["spill"], _c3["verdict"], _c["silent"], _c["silent"],
         _c3["silent"]))

    _example("2010_003383", "Una mucca a pezzi: che cosa compra il claim per contenimento",
             "L'annotazione parla di una mucca con i bordi grossolani, soprattutto sulle zampe. "
             "Con la regola dell'IoU la pipeline riporta un merge di due regioni più sei "
             "regioni GT «inglobate»: ogni affermazione è vera, ma sta contando le "
             "parti sconnesse dello stesso animale. Il claim per contenimento le lega al blob e "
             "misura il blob contro la loro unione, così la frase sul contorno che "
             "l'annotatore ha scritto finalmente esiste. Le istanze a oggetti la dicono poi con "
             "l'aritmetica giusta: tre oggetti, non otto frammenti.")
    _example("2007_000129", "Biciclette dietro i ciclisti: che cosa comprano le istanze a oggetti",
             "«Sei regioni BICYCLE mancate del tutto» è quello che vede la regola "
             "delle componenti: i telai sono tagliati in pezzi dalle gambe dei ciclisti, e ogni "
             "pezzo è un'istanza a sé. Il claim per contenimento sistema l'aritmetica "
             "(sono inglobate dalle maschere PERSON, e il testo dice «four BICYCLE regions "
             "now labelled PERSON»), ma solo le maschere d'oggetto dicono quello che dice "
             "l'annotatore — una bicicletta, parzialmente nascosta dietro la persona, "
             "sotto-segmentata — e la revisione della narrativa aggiunge da che lato e di "
             "quanto.")
    _example("2008_006784", "Un divano letto come sfondo: che cosa compra il verdetto in apertura",
             "Tutti e quattro i testi descrivono lo stesso fallimento e tutti e quattro sono "
             "veri; solo l'ultimo dice per primo quello che l'annotatore dice per primo, cioè "
             "che l'oggetto è sostanzialmente sparito. Il verdetto si calcola dalle maschere "
             "(l'81% dell'area annotata predetta come sfondo, sopra la soglia VERDICT_MIN = 50%) "
             "e va in testa. Qui la tassonomia non cambia: il guadagno è l'ordine e una "
             "frase.")
    _example("2008_003703", "Un aereo segmentato bene: il silenzio che costava punti",
             "Nessun errore tipato, quindi per tre run il testo era «No segmentation error "
             "was found in this image», mentre l'annotatore scriveva che l'aereo è "
             "segmentato piuttosto bene con qualche bordo non perfettamente definito — una "
             "descrizione di una regione CORRETTA, per cui la pipeline non aveva una frase sopra "
             "la soglia delle note di contorno. È il caso che spiega tutta la revisione della "
             "narrativa: i voti bassi non erano affermazioni sbagliate, erano affermazioni "
             "mancanti.")
    p("Quale run riportare nella tesi è una scelta, e la versione onesta è questa. La "
      "tipizzazione a oggetti è quella vera — un oggetto occluso è un oggetto "
      "solo, e le narrative che il giudice, gli annotatori e il set sintetico preferiscono sono "
      "costruite su di essa — mentre il run di riferimento resta quello il cui ramo cluster "
      "certifica più pattern, e tutti i numeri del capitolo 8 appartengono a lui. I due rami "
      "sono misurati separatamente in tutto il documento proprio perché questa scelta non "
      "debba essere nascosta: il lato descrizione si può riportare dal run a oggetti, il "
      "lato scoperta dal run su cui è stato misurato, e la dipendenza del ramo cluster da "
      "tante istanze piccole si dichiara per il limite che è.")


# ============================================================================
#  13. La causa e' plausibile? (settembre 2026). Numeri da
#      _revision_numbers.load_cause: qui non c'e' nulla di digitato a mano.
# ============================================================================
from _revision_numbers import load_cause as _load_cause
try:
    CZ = _load_cause()
except Exception as _ex:
    CZ = None
    print("  (numeri sulla plausibilita' della causa non disponibili, sezione 13 saltata:",
          _ex, ")")

if CZ and "voc" in CZ:
    _v, _a, _sy = CZ["voc"], CZ["voc"]["agree"], CZ.get("synth")
    _cv = CZ["coverage"]

    h("13. La causa è plausibile? Misurare l'altra metà del deliverable "
      "(settembre 2026)", 1)
    p("Tutto quello che si è misurato finora dà un voto a ciò che la pipeline "
      "dice sia ANDATO STORTO. Ma i due rami non esistono per descrivere un errore: esistono "
      "per dare un nome alla SITUAZIONE che un gruppo di errori ha in comune — "
      "«an object cut off at the image border» dal ramo linguistico, «small "
      "objects of class ‘person’ with imprecise boundaries» dal ramo cluster "
      "— e quella affermazione è il deliverable. Questa sezione la mette alla prova: "
      "dato quello che va storto in un'immagine, quanto è plausibile la situazione che il "
      "sistema incolpa?")
    p("La domanda va allo stesso Gemma locale, con una rubrica di due criteri, e viene posta "
      "DUE volte per immagine contro due racconti indipendenti degli stessi errori — "
      "l'annotazione scritta a mano e la narrativa generata dalla pipeline — senza che il "
      "giudice sappia quale dei due sta leggendo. L'accordo fra i due voti è il risultato "
      "che conta: dove coincidono, il testo generato può sostituire quello umano quando si "
      "giudica un modo di fallire, ed è esattamente ciò che serve sulle 1 483 "
      "immagini che nessuno ha annotato.")
    bullet("Coerenza (0–5): le classi, i tipi d'errore e le circostanze nominate nella "
           "causa concordano con la descrizione dell'errore? Ogni contraddizione costa un punto.")
    bullet("Forza esplicativa (0–5): se la situazione nominata fosse vera per questa "
           "immagine, spiegherebbe gli errori descritti — quello principale, o solo una "
           "parte marginale? Una causa così generica da andar bene per qualunque immagine "
           "perde qui.")
    bullet("La plausibilità è la somma dei due, 0–10. Il testo della causa porta "
           "la situazione, la classe e il tipo d'errore che rivendica, e NON i numeri di "
           "maschera (IoU medio, boundary IoU, frammenti persi): quelli misurano gli stessi "
           "errori che si stanno descrivendo, e un giudice a cui li si desse combacerebbe "
           "misure invece di pesare una spiegazione.")
    bullet("Calibrazione del caso, come ovunque qui dentro: ogni coppia ha il suo controllo, in "
           "cui la stessa descrizione d'errore riceve la causa di un'ALTRA immagine, con seme, "
           "derangement e mescolata alle vere. Solo lo scarto sopra quel pavimento è "
           "evidenza.")
    p("La domanda si può porre solo dove una causa esiste. Delle %d immagini annotate, %d "
      "hanno almeno un pattern certificato (%d in val, %d in train) e sono quelle misurate qui; "
      "su altre %d il Blocco 1 tipizza errori che non ricadono in nessuna slice certificata, "
      "quindi non si incolpa nulla e non c'è nulla da giudicare; le restanti %d non hanno "
      "alcun errore tipato. La pagina di confronto dichiara in quale dei tre casi si trova ogni "
      "scheda, invece di lasciare lo stesso silenzio per tutte. Il set sintetico aggiunge %d "
      "immagini, con il ramo indotto come in train36."
      % (_cv["annotated"], _v["human"]["n"], _v["n_val"], _v["n_train"],
         _cv["no_pattern"], _cv["no_error"], (_sy or {}).get("human", {}).get("n", 0)))
    table(["Plausibilità della causa (0–10)", "errore raccontato dall'annotazione",
           "errore raccontato dalla narrativa"],
          [["coppie genuine", "%.2f" % _v["human"]["score"], "%.2f" % _v["pipeline"]["score"]],
           ["pavimento del controllo (causa di un'altra immagine)",
            "%.2f" % _v["human"]["control"], "%.2f" % _v["pipeline"]["control"]],
           ["scarto sopra il pavimento",
            "%.2f" % (_v["human"]["score"] - _v["human"]["control"]),
            "%.2f" % (_v["pipeline"]["score"] - _v["pipeline"]["control"])],
           ["coerenza / 5", "%.2f" % _v["human"]["cons"], "%.2f" % _v["pipeline"]["cons"]],
           ["forza esplicativa / 5", "%.2f" % _v["human"]["expl"], "%.2f" % _v["pipeline"]["expl"]],
           ["parole per descrizione d'errore",
            "%.0f" % _v["human"]["words"], "%.0f" % _v["pipeline"]["words"]]],
          widths=[2.6, 2.0, 2.0])
    p("Cinque-sei punti sopra il pavimento del rumore: le situazioni che il sistema certifica "
      "non sono così generiche da andar bene per qualunque errore, ed era la prima cosa da "
      "escludere. Poi i due rami si separano, ed è il risultato di questa sezione.")
    _b = _v["by_branch"]
    table(["Causa proposta da", "con l'annotazione", "con la narrativa", "immagini"],
          [["ramo linguistico (un'ipotesi SliceLens)",
            "%.2f" % _b["human"]["language"][0], "%.2f" % _b["pipeline"]["language"][0],
            str(_b["human"]["language"][1])],
           ["ramo cluster (un gruppo visivo)",
            "%.2f" % _b["human"]["cluster"][0], "%.2f" % _b["pipeline"]["cluster"][0],
            str(_b["human"]["cluster"][1])],
           ["tutti e due sulla stessa immagine",
            "%.2f" % _b["human"]["both"][0], "%.2f" % _b["pipeline"]["both"][0],
            str(_b["human"]["both"][1])]],
          widths=[2.8, 1.7, 1.7, 1.0])
    p("Un'ipotesi SliceLens nomina una situazione che il lettore può immaginare e "
      "verificare — un oggetto tagliato dal bordo, uno occluso da un altro, più "
      "oggetti sovrapposti della stessa classe — e prende %.2f / %.2f. Una slice del ramo "
      "cluster nomina un gruppo e un tipo d'errore, e quando per quel gruppo nessun attributo "
      "è stato verificato la descrizione degenera in «objects of class "
      "‘person’ grouped by appearance only (no verified attribute) not detected», "
      "che non afferma quasi nulla: %.2f / %.2f, e le tre cause con il voto più basso di "
      "tutto l'insieme sono di questo tipo. Il ramo cluster è quello che TROVA il gruppo; "
      "il ramo linguistico è quello che sa DIRE perché. La complementarietà dei "
      "due rami (capitolo 2) era finora misurata dalla loro sovrapposizione, cioè da "
      "quante istanze d'errore ciascuno raggiunge da solo; qui la si misura dall'altro lato, "
      "sul testo che il lettore riceve davvero."
      % (_b["human"]["language"][0], _b["pipeline"]["language"][0],
         _b["human"]["cluster"][0], _b["pipeline"]["cluster"][0]))
    p("L'accordo fra i due racconti, immagine per immagine: differenza media %+.2f a favore "
      "della narrativa, differenza assoluta media %.2f punti, entro un punto su %d immagini "
      "delle %d ed entro due su %d, r di Pearson = %.2f. %d cause risultano plausibili "
      "(≥ 7) con entrambi i racconti e %d con nessuno dei due. La direzione è la "
      "stessa; la singola immagine non è intercambiabile."
      % (_a["diff"], _a["absdiff"], _a["within1"], _a["n"], _a["within2"], _a["r"],
         _a["both_high"], _a["both_low"]))
    p("Due cautele vanno accanto a quel +%.2f, perché non è un vantaggio pulito. La "
      "narrativa è lunga %.0f parole contro le %.0f dell'annotazione, quindi offre alla "
      "causa più cose con cui concordare — e più occasioni di contraddirla. "
      "Soprattutto, narrativa e causa nascono entrambe dalla tassonomia del Blocco 1 e ne "
      "condividono il vocabolario («merged», «missed», «imprecise "
      "boundaries»), quindi parte della coerenza più alta (%.2f contro %.2f) è "
      "allineamento lessicale e non evidenza migliore. La lettura onesta è l'accordo, non "
      "la classifica."
      % (_a["diff"], _v["pipeline"]["words"], _v["human"]["words"],
         _v["pipeline"]["cons"], _v["human"]["cons"]))
    _w = _v["worst"][0]
    p("I disaccordi sono istruttivi, non rumore. Il più grande, %s, prende %.1f con "
      "l'annotazione e %.1f con la narrativa: l'annotatore ha scritto solo della sedia mancata, "
      "che la causa spiega esattamente, mentre la narrativa racconta anche i monitor fusi, la "
      "persona sovra-segmentata e una pianta allucinata — così la stessa causa spiega "
      "una frazione di quello che viene detto, e la forza esplicativa crolla. La coppia di voti "
      "misura in effetti quanta parte di un'immagine una sola situazione certificata riesca a "
      "coprire, che è una proprietà della slice e non di nessuno dei due testi."
      % (_w[1], _w[2], _w[3]))
    if _sy:
        _s = _sy["agree"]
        p("Il set sintetico risponde alla domanda dove la risposta è nota. Delle sue 21 "
          "immagini solo %d ricadono in un pattern indotto, e NESSUNA di esse è fra le "
          "dieci la cui annotazione dichiara la predizione perfetta: dove non c'è nulla da "
          "spiegare non si incolpa nulla, ed è esattamente ciò per cui la classe "
          "negativa esiste. Sulle %d dichiarate errate le cause prendono %.2f con "
          "l'annotazione e %.2f con la narrativa, contro un pavimento di %.2f / %.2f: "
          "nettamente meno che su VOC, e per una ragione già documentata al capitolo 10, "
          "cioè che undici di quelle predizioni sono ground truth di altre immagini "
          "incollate, quindi nessuna situazione certificata su val può spiegarle "
          "onestamente. Anche l'accordo si allenta (r = %.2f, differenza assoluta media %.2f), "
          "come dev'essere su errori che sono artefatti."
          % (_sy["human"]["n"], _sy["human"]["n"], _sy["human"]["score"],
             _sy["pipeline"]["score"], _sy["human"]["control"], _sy["pipeline"]["control"],
             _s["r"], _s["absdiff"]))
    if CZ.get("all"):
        _al = CZ["all"]["pipeline"]
        _ab, _ae = CZ["all"]["by_branch"]["pipeline"], CZ["all"]["by_etype"]
        _bd, _np = CZ["all"]["bands"], CZ["all"]["by_npatterns"]
        h("La stessa domanda su tutta la copertura della pipeline", 2)
        p("Le %d immagini annotate sono il posto dove esiste un racconto umano, non il posto "
          "dove la pipeline lavora. Ripetendo la misura su OGNI immagine che ha una causa "
          "certificata — %d in tutto, %d di val e %d di train36 — giudicata con la "
          "sola narrativa, perché fuori dalle annotate un testo umano non c'è (ed "
          "è la ragione per cui la pipeline esiste): le cause prendono %.2f contro un "
          "pavimento di %.2f, uno scarto di %.2f, con coerenza %.2f/5 e forza esplicativa "
          "%.2f/5. Il controllo è stato estratto su un campione con seme di 150 immagini "
          "invece che su tutte e %d: il pavimento è una proprietà del giudice e del "
          "corpus, non della singola immagine. %d immagini su %d prendono 7 o più, %d "
          "stanno fra 5 e 6 e %d sotto 5."
          % (_v["human"]["n"], _al["n"], CZ["all"]["n_val"], CZ["all"]["n_train"],
             _al["score"], _al["control"], _al["score"] - _al["control"], _al["cons"],
             _al["expl"], _al["n"], _bd["high"], _al["n"], _bd["mid"], _bd["low"]))
        p("La misura più ampia corregge l'impressione che le 39 davano sui due rami. "
          "Lì le cause del ramo cluster prendevano %.2f contro %.2f del ramo linguistico, "
          "ma su sole %d immagini; su tutta la copertura il ramo cluster arriva a %.2f su %d "
          "immagini e quello linguistico a %.2f su %d, e le immagini in cui scattano ENTRAMBI "
          "i rami prendono il voto più alto di tutti (%.2f su %d). La direzione resta "
          "— una situazione nominata spiega un errore meglio di un gruppo visivo — "
          "ma lo scarto è di un punto, non di due e mezzo, e il dato sulle 39 va letto "
          "per il campione piccolo che è. Che i due rami insieme valgano più di "
          "ciascuno da solo è la stessa complementarietà del capitolo 2, misurata "
          "sulla spiegazione invece che sulla copertura."
          % (_b["human"]["cluster"][0], _b["human"]["language"][0], _b["human"]["cluster"][1],
             _ab["cluster"][0], _ab["cluster"][1], _ab["language"][0], _ab["language"][1],
             _ab["both"][0], _ab["both"][1]))
        table(["La causa rivendica un errore di tipo", "plausibilità", "immagini"],
              [[k, "%.2f" % m, str(n)] for k, (m, n) in
               sorted(_ae.items(), key=lambda kv: -kv[1][1])],
              widths=[3.0, 1.5, 1.2])
        p("Letta dall'alto, quella tabella è l'onestà della pipeline su se stessa: "
          "una causa che incolpa una detection spuria è la meno convincente (%.2f), "
          "perché «il modello si è inventato un oggetto qui» è la "
          "cosa più difficile da spiegare con una situazione, mentre le cause su regioni "
          "fuse, estensione sbagliata o classe sbagliata stanno sopra 8.7: sono fallimenti con "
          "una circostanza visibile dietro. Vale anche la pena dire che la plausibilità "
          "cresce con il numero di pattern certificati sull'immagine, da %.2f con uno a %.2f "
          "con quattro o più: più parte dell'immagine viene spiegata, ed è la "
          "forza esplicativa a guadagnarci."
          % (_ae["spurious"][0], _np[1][0], _np[4][0]))

    from _revision_numbers import load_gallery as _load_gal
    try:
        _GL = _load_gal()
    except Exception as _ex:
        _GL = None
        print("  (numeri della galleria non disponibili:", _ex, ")")
    if _GL:
        p("La galleria porta le stesse due cose, ciascuna al livello a cui appartiene. Una "
          "scheda-slice mostra la plausibilità MEDIA sulle immagini che la compongono, "
          "con il numero su cui è calcolata accanto: la media dice qualcosa della slice, "
          "il conteggio dice quanto fidarsene, e ce l'hanno %d schede su %d. Una "
          "scheda-immagine mostra il proprio voto e, da ora, i due testi che la pipeline "
          "scrive su quell'immagine, etichettati perché non si confondano: la NARRATIVA "
          "di ogni errore tipato e la descrizione dei PATTERN certificati a cui appartiene. "
          "Rispondono a domande diverse ed erano indistinguibili. Le schede portano tre "
          "immagini ciascuna e si fermano a %d; le %d narrative dell'intero run stanno come "
          "puro testo in fondo alla vista per immagine, ognuna con la sua plausibilità "
          "dove è stata misurata: costa meno di un megabyte e fa sì che il file "
          "risponda a «che cosa dice la pipeline di QUESTA immagine» per tutte, non "
          "solo per quelle con le figure."
          % (_GL["slice_cards_scored"], _GL["slice_cards"], _GL["image_cards"],
             _GL["narratives"]))
    p("Dove si legge: ogni scheda della pagina di confronto che ha una causa certificata porta "
      "i due voti sotto il blocco dei pattern, con entrambi i criteri e la motivazione del "
      "giudice, e le due medie stanno in testa alla pagina accanto a quella del giudice sulle "
      "descrizioni. Le schede con errori tipati ma senza slice certificata lo dichiarano. La "
      "misura è slice_quality/examples/judge_cause_plausibility.py (SYNTH=1 per il set "
      "sintetico), riprende dal proprio output e rifa' solo ciò che è cambiato.",
      italic=True)

# ---------------------------------------------------------------------------
# 14. Ampliare il vocabolario delle ipotesi (ottobre 2026). Numeri da
#     _revision_numbers.load_extra_hyp, figure da make_extra_hyp_figures.py.
# ---------------------------------------------------------------------------
from _revision_numbers import load_extra_hyp as _load_xh
try:
    XH = _load_xh()
except Exception as _ex:
    XH = None
    print("  (numeri delle ipotesi nuove non disponibili, sezione 14 saltata:", _ex, ")")


def _figxh(name, caption, width=5.8):
    path = os.path.join(DOCS, name)
    if not os.path.exists(path):
        print("  (figura mancante, saltata:", name + ")")
        return
    doc.add_picture(path, width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = c.add_run(caption)
    r.italic = True
    r.font.size = Pt(9)


if XH:
    _cn, _sp = XH["canon"], XH["specific"]
    _rt = lambda c, n: 100.0 * c / n
    _S = XH["sample"]["totals"]
    _F = XH["full"]["totals"] if "full" in XH else None
    _per = XH["sample"]["per_hypothesis"]
    _PF = XH["full"]["per_hypothesis"] if _F else _per
    _TY = {"fn": "mancati", "extent": "estensione", "confusion": "classe sbagliata",
           "boundary": "bordo", "fp": "allucinati", "merge": "merge", "split": "split"}
    _VD = {"holds": "regge", "mislabelled": "vera, frase sbagliata", "partial": "parziale",
           "vague": "vaga", "artefact": "artefatto"}
    h("14. Ampliare il vocabolario delle ipotesi del ramo linguistico (ottobre 2026)")
    p("Il ramo linguistico può certificare solo le situazioni su cui viene interrogato. Finora le "
      "domande venivano da due fonti: le 7 ipotesi canoniche, uguali per ogni classe («an object "
      "cut off at the image border», …), e fino a 15 ipotesi specifiche per classe scritte da "
      "Gemma con il prompt del paper SliceLens. La loro resa nella run inst è molto diversa.")
    table(["Ipotesi", "slice classe × ipotesi valutate", "certificate", "resa"],
          [["7 canoniche, generiche", _cn["evaluated"], _cn["certified"],
            "%.1f%%" % _rt(_cn["certified"], _cn["evaluated"])],
           ["specifiche per classe (Gemma)", _sp["evaluated"], _sp["certified"],
            "%.1f%%" % _rt(_sp["certified"], _sp["evaluated"])]],
          widths=[2.4, 1.9, 1.0, 0.8])
    _it = XH["inst_types"]
    p("Le due specifiche che certificano sono %s; le altre descrivono per lo più l'aspetto di un "
      "sottogruppo («matte black car», «square monitor») e non una situazione. Le %d situazioni "
      "certificate, poi, sono dominate da %s: nessuna da oggetti mancati, da estensione sbagliata "
      "o da classe sbagliata. La domanda è se ipotesi GENERICHE mirate a quei vuoti aggiungano "
      "situazioni che il vocabolario non aveva."
      % (" e ".join("«%s»" % x for x in XH["specific_certified"]), sum(_it.values()),
         ", ".join("%s %d" % (_TY[k], v) for k, v in sorted(_it.items(), key=lambda kv: -kv[1]))))
    h("Le otto ipotesi", 2)
    _mira = {
        "a picture, poster or screen showing an object": "allucinati: persone e animali in quadri, poster, schermi",
        "a reflection of an object in a mirror, window or water": "allucinati: barche riflesse, auto nelle vetrine",
        "a toy, statue or model of an object": "allucinati: cavalli di plastica, statue, modellini",
        "an object with holes or see-through parts": "bordo/estensione: sedie, bici, piante; la maschera riempie i vuoti",
        "an object behind a fence, net, bars or branches": "mancati/split: animali dietro recinti, auto dietro cancellate",
        "an object seen from an unusual viewpoint or in an unusual pose": "mancati: gatti sdraiati, aerei dall'alto",
        "a very large object seen up close that fills most of the image": "mancati/estensione: primi piani",
        "an object in an unusual setting for its kind": "mancati/classe: dipendenza dal contesto",
    }
    table(["Ipotesi", "a che cosa mira, dove ce la si aspetta"],
          [[q, _mira[q]] for q in XH["extra"]], widths=[3.0, 3.3])
    p("Regole di scelta, ognuna da una misura precedente: una situazione visibile per ipotesi e "
      "nessun nome di classe dentro (§6.2 del documento tecnico); niente di fotometrico («dark», "
      "«blurry» crollavano a 0.00–0.12 di precisione); niente che ripeta la geometria "
      "dell'errore (in un segmentatore semantico due oggetti uguali che si toccano diventano "
      "sempre un merge: un'ipotesi su oggetti a contatto certificherebbe per costruzione); "
      "nessun condizionamento sul ramo cluster, per restare open-world.")
    h("Protocollo", 2)
    p("Le otto sono state ancorate con lo stesso run_voc.py di luglio, sulla stessa cache del "
      "substrato e sullo stesso campione di 120 istanze per classe (identico a quello di luglio "
      "in tutte le classi tranne cow), con Qwen caricato una volta: %d valutazioni del VLM su due "
      "GPU. Poi la run inst è stata ricalcolata due volte, senza e con le nuove ipotesi: la prima "
      "riproduce esattamente la run inst (%d situazioni certificate), quindi le differenze "
      "vengono solo dalle ipotesi. Le slice certificate sul campione sono state infine "
      "riverificate su TUTTE le istanze della loro classe, come a luglio."
      % (XH["n_classes"] * 8 * 120, _S["certified_before"]))
    h("Risultati", 2)
    _figxh("fig_extra_yield_it.png", "Resa delle tre fonti di ipotesi: slice certificate su "
           "slice valutate.")
    rows = []
    for q in XH["extra"]:
        v = _per[q]
        row = [q, "%d / %d" % (len(v["systematic_classes"]), XH["n_classes"]),
               ", ".join(c["class"] for c in v["certified"]) or "—"]
        if _F:
            row.append(", ".join(XH["full_held"].get(q, [])) or "—")
        rows.append(row)
    table(["Ipotesi", "sistematica sul campione", "certificata sul campione"]
          + (["regge su tutte le istanze"] if _F else []), rows,
          widths=[2.5, 1.1, 1.6] + ([1.3] if _F else []))
    _figxh("fig_extra_per_hyp_it.png", "Le otto ipotesi, classe per classe.")
    p("Sul campione %d coppie classe × ipotesi su %d sono sistematiche e %d certificate, il "
      "%.1f%%: circa %.0f volte la resa delle specifiche, il %.0f%% di quella delle canoniche. "
      "Le situazioni certificate passano da %d a %d e le immagini con una descrizione "
      "situazionale da %d a %d (+%d): le slice nuove cadono quasi tutte su immagini che una "
      "situazione l'avevano già. Nessuna slice vecchia perde la certificazione."
      % (XH["sample_systematic"], XH["pairs"], XH["sample_certified"],
         _rt(XH["sample_certified"], XH["pairs"]),
         _rt(XH["sample_certified"], XH["pairs"]) / _rt(_sp["certified"], _sp["evaluated"]),
         100 * _rt(XH["sample_certified"], XH["pairs"]) / _rt(_cn["certified"], _cn["evaluated"]),
         _S["certified_before"], _S["certified_after"], _S["images_before"], _S["images_after"],
         _S["images_after"] - _S["images_before"]))
    if _F:
        _drop = {q: c for q, c in XH["full_dropped"].items() if c}
        _surv = [(q, c) for q in XH["extra"] for c in _PF[q]["certified"]]
        p("Riverificate su tutte le istanze, ne restano certificate %d su %d%s. Restano %s. Con "
          "loro le situazioni certificate vanno da %d a %d e le immagini descritte da %d a %d."
          % (XH["full_certified"], XH["sample_certified"],
             ("; il test non regge più per " + "; ".join(
                 "«%s» su %s" % (q, ", ".join(c)) for q, c in _drop.items())) if _drop else "",
             "; ".join("%s — «%s» (%d istanze su %d immagini, dominate da %s)"
                       % (c["class"], q, c["size"], c["n_images"], _TY[c["type"]]) for q, c in _surv),
             _F["certified_before"], _F["certified_after"], _F["images_before"], _F["images_after"]))
    _figxh("fig_extra_types_it.png", "Tipo d'errore dominante delle situazioni certificate, prima "
           "e dopo le ipotesi nuove.")
    _ta = (_F or _S)["types_after"]
    _hall_cert = [c for q in XH["extra"][:3] for c in _PF[q]["certified"]]
    _fp_new = [(q, c) for q in XH["extra"] for c in _PF[q]["certified"] if c["type"] == "fp"]
    p("Il vuoto a cui le otto miravano resta: %s. Delle tre ipotesi scritte per gli allucinati "
      "(quadri, riflessi, giocattoli) %s%s"
      % ("nessuna situazione certificata è dominata da mancati, estensione o classe sbagliata"
         if not any(_ta.get(k) for k in ("fn", "extent", "confusion")) else
         ", ".join("%s %d" % (_TY[k], _ta.get(k, 0)) for k in ("fn", "extent", "confusion")),
         ("ne certifica%s %d" % ("" if len(_hall_cert) == 1 else "no", len(_hall_cert)))
         if _hall_cert else "nessuna certifica",
         ("; le nuove situazioni dominate da allucinati, %s, vengono da ipotesi scritte per altro."
          % ", ".join("%s («%s»)" % (c["class"], q) for q, c in _fp_new))
         if _fp_new else ", e nessuna situazione nuova è dominata da allucinati."))
    h("Il controllo visivo", 2)
    p("Il test di andamento dice che le istanze che il VLM associa all'ipotesi sono più spesso "
      "errori, non che l'ipotesi le descriva. Ogni slice nuova è stata quindi guardata sulle sue "
      "istanze sbagliate con la confidenza più alta, nella fase sul campione.")
    _figxh("fig_extra_examples_it.png", "Istanze sbagliate a confidenza più alta delle slice "
           "nuove, con il giudizio del controllo visivo (verde regge, arancio in parte, rosso "
           "artefatto).", width=6.3)
    _vis = XH.get("visual", {}).get("slices", [])
    table(["Slice", "ipotesi", "che cosa mostrano le immagini", "giudizio"],
          [[s["class"], s["hypothesis"], s["it"], _VD[s["verdict"]]] for s in _vis],
          widths=[0.8, 1.9, 2.8, 0.9])
    _held = lambda s: s["class"] in XH["full_held"].get(s["hypothesis"], [])
    _hold = [s for s in _vis if s["verdict"] == "holds" and (not _F or _held(s))]
    _bus = next((s for s in _vis if s["verdict"] == "artefact"), None)
    _seen_dropped = [s for s in _vis if _F and s["verdict"] == "holds" and not _held(s)]
    _lucky = [s for s in _vis if _F and s["verdict"] != "holds" and _held(s)]
    if _bus:
        p("Il caso di %s è quello istruttivo: «%s» descrive qualunque fotografia, e il VLM "
          "risponde sì quando l'oggetto riempie l'inquadratura. Il test sul campione l'ha lasciata "
          "passare, il controllo visivo no%s."
          % (_bus["class"], _bus["hypothesis"],
             ", e poi l'ha scartata anche la riverifica" if _F and not _held(_bus) else ""))
    if _F:
        p("I due controlli rispondono a domande diverse, e si vede.%s%s Contando solo ciò che li "
          "supera entrambi, il vocabolario nuovo lascia %d situazion%s: %s."
          % ((" Lo sguardo conferma che la situazione c'è — %s — ma su tutte le istanze della "
              "classe non è più associata agli errori: sul campione l'associazione era in parte "
              "un caso." % "; ".join("%s «%s»" % (s["class"], s["hypothesis"]) for s in _seen_dropped))
             if _seen_dropped else "",
             (" Al contrario %s regg%s statisticamente, ma le immagini mostrano altro o qualcosa di "
              "vago (%s)." % (" e ".join(s["class"] for s in _lucky), "e" if len(_lucky) == 1 else "ono",
                              "; ".join("%s: %s" % (s["class"], _VD[s["verdict"]]) for s in _lucky)))
             if _lucky else "",
             len(_hold), "e" if len(_hold) == 1 else "i",
             ", ".join("%s — «%s»" % (s["class"], s["hypothesis"]) for s in _hold) or "nessuna"))
    h("Che cosa se ne conclude", 2)
    bullet("Il ramo linguistico non è limitato dal NUMERO di ipotesi: otto generiche in più, "
           "scelte per coprire vuoti noti, aggiungono %d immagini a %d."
           % ((_F or _S)["images_after"] - (_F or _S)["images_before"], (_F or _S)["images_before"]))
    bullet("Mancati, estensione e classe sbagliata restano senza una situazione certificata anche "
           "cercandola apposta: con ogni probabilità non sono spiegati da una situazione che un VLM "
           "sappia nominare e verificare su una regione. È la stessa linea dell'esperimento con i "
           "difetti iniettati: la scoperta funziona, la certificazione richiede una situazione "
           "nominabile.")
    bullet("La certificazione statistica ha bisogno di controlli indipendenti accanto, e i "
           "controlli non si sostituiscono l'uno all'altro%s. È la stessa ragione per cui esistono "
           "la precisione delle descrizioni e la plausibilità della causa."
           % ((": delle %d slice nuove solo %d li supera%s entrambi"
               % (len(_vis), len(_hold), "" if len(_hold) == 1 else "no")) if _F else ""))
    _MG = XH.get("merged", [])
    if _MG:
        bullet("Le %d situazioni che hanno retto alla riverifica sono entrate nella run inst "
               "(ottobre 2026) con la stessa regola delle 38 di luglio, senza un filtro manuale che "
               "le vecchie non hanno avuto: galleria e pagina di confronto le mostrano. Le frasi "
               "che non hanno retto non vengono riscritte: riscrivere un'ipotesi dopo averne visto "
               "il risultato e ritestarla sulle stesse immagini di val è il modo classico di "
               "fabbricare falsi positivi." % len(_MG))
    else:
        bullet("Le slice nuove non entrano nella run inst e le frasi non vengono riscritte: "
               "riscrivere un'ipotesi dopo averne visto il risultato e ritestarla sulle stesse "
               "immagini di val è il modo classico di fabbricare falsi positivi.")
    _MP = XH.get("merged_plaus")
    if _MG and _MP:
        h("Le situazioni aggiunte, giudicate come tutte le altre", 2)
        p("Entrate nella run inst, le situazioni nuove passano dallo stesso controllo indipendente "
          "di ogni altra causa: il giudice di plausibilità, contro la narrativa della stessa "
          "immagine. Su tutte le immagini giudicate la causa vale %.2f (%d immagini); quelle la "
          "cui causa nomina un'ipotesi nuova valgono così." % _MP["all"])
        table(["Ipotesi", "slice", "immagini giudicate", "plausibilità media (0–10)"],
              [[s["hypothesis"], s["slice_id"], _MP[s["hypothesis"]][1],
                ("%.2f" % _MP[s["hypothesis"]][0]) if _MP[s["hypothesis"]][0] is not None else "—"]
               for s in _MG], widths=[2.8, 1.4, 0.9, 1.2])
        p("Un voto molto sotto la media è il giudice che vede ciò che ha visto il controllo "
          "visivo; un voto pari o superiore vuol dire che la narrativa di quelle immagini è "
          "compatibile con la situazione anche dove lo sguardo sulle istanze era dubbio. La "
          "pagina mostra ognuna di queste immagini con il suo voto.", italic=True)

out = os.path.join(DOCS, "panoramica_pipeline_IT.docx")
doc.save(out)
print("saved", out)
