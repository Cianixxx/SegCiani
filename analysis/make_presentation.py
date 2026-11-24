"""Build the POLIMI-template deck in EN and IT. Speaker notes ('discorso') are IT
in both decks. Data-driven: one ITEMS list carries en/it text + shared IT note."""
import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.oxml.ns import qn
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from PIL import Image as PILImage

TPL = r"<project-root>\presentazione\TemplatePOLIMI.pptx"
DOCS = os.path.dirname(os.path.abspath(__file__))
FIG = {"en": os.path.join(DOCS, "pipeline_figure.png"),
       "it": os.path.join(DOCS, "pipeline_figure_it.png")}


def ph_by_idx(slide, idx):
    for ph in slide.placeholders:
        if ph.placeholder_format.idx == idx:
            return ph
    return None


def set_bullets(ph, items):
    tf = ph.text_frame
    tf.clear()
    for i, (text, lvl) in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = text
        p.level = lvl


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


# ----------------------------------------------------------------------------- data
COVER = {
    "en": ("Error-slice discovery & per-image failure descriptions "
           "for semantic segmentation", "Daniele Spini", "Implementation progress"),
    "it": ("Scoperta di error-slice e descrizioni di fallimento per immagine "
           "per la segmentazione semantica", "Daniele Spini", "Avanzamento implementazione"),
    "note": "Presento la pipeline che ho implementato: scoperta di error-slice e "
            "generazione di descrizioni testuali, per ogni immagine, delle situazioni "
            "sistematiche che fanno sbagliare il modello di segmentazione.",
}

GOAL = {
    "kind": "content",
    "en": ("Goal: describe why the model fails, per image",
           [("For each image where segmentation fails: a textual description of the "
             "systematic situations that induce the error — not just the error type.", 0),
            ("Two complementary branches over the same error instances:", 0),
            ("Cluster (class-agnostic): what / how it fails", 1),
            ("Language / SliceLens: in what situation it fails", 1),
            ("Combined per image: “situation (SliceLens) → error (cluster)”.", 0)]),
    "it": ("Obiettivo: descrivere perché il modello sbaglia, per immagine",
           [("Per ogni immagine in cui la segmentazione fallisce: una descrizione "
             "testuale delle situazioni sistematiche che inducono l’errore — non solo "
             "il tipo di errore.", 0),
            ("Due rami complementari sulle stesse istanze d’errore:", 0),
            ("Cluster (class-agnostic): cosa / come fallisce", 1),
            ("Linguistico / SliceLens: in quale situazione fallisce", 1),
            ("Combinati per immagine: “situazione (SliceLens) → errore (cluster)”.", 0)]),
    "note": "L'obiettivo non è descrivere l'errore in sé, ma le situazioni sistematiche "
            "che lo inducono, per ogni immagine. Uso due rami complementari sulle stesse "
            "istanze d'errore: il ramo cluster dice come fallisce, il ramo SliceLens in quale "
            "situazione. Poi li combino in una sola descrizione.",
}

RQ = {
    "kind": "content",
    "en": ("Five questions, in plain words",
           [("Can this work pixel by pixel, not just image by image?", 0),
            ("Can it work with no dataset that says WHY the model failed?", 0),
            ("Can a geometric error type and a plain-language explanation "
             "point to the same thing?", 0),
            ("Can it find NEW situations, not just pick from a fixed list?", 0),
            ("Can we tell if an explanation is right, with no answer key to "
             "check it against?", 0)]),
    "it": ("Cinque domande, in parole semplici",
           [("Funziona pixel per pixel, non solo immagine per immagine?", 0),
            ("Funziona senza un dataset che dice PERCHÉ il modello ha "
             "sbagliato?", 0),
            ("Un tipo di errore geometrico e una spiegazione a parole possono "
             "indicare la stessa cosa?", 0),
            ("Si possono scoprire situazioni NUOVE, non solo scegliere da una "
             "lista già scritta?", 0),
            ("Si può dire se una spiegazione è giusta, senza avere una "
             "risposta corretta con cui confrontarla?", 0)]),
    "note": "Prima di entrare nella pipeline, le cinque domande da cui parte questo "
            "lavoro — semplici, ma è tutto il resto della presentazione che risponde a "
            "queste cinque. Pixel per pixel: i metodi esistenti lavorano a livello di "
            "immagine intera; qui serve farlo per singolo oggetto. Senza un dataset di "
            "cause: nessuno ha mai scritto da nessuna parte perché un modello sbaglia un "
            "pixel, quindi il sistema deve scoprirlo da solo. Geometria e parole che "
            "coincidono: un errore descritto come 'bordo impreciso' e uno descritto come "
            "'l'oggetto è tagliato dal bordo' devono poter parlare dello stesso "
            "fenomeno — è il lavoro dei due rami che sto per mostrare. Situazioni "
            "nuove: non voglio un elenco di possibilità deciso a tavolino, voglio che il "
            "sistema possa proporre ipotesi che non avevo previsto. E l'ultima, la più "
            "difficile: come faccio a sapere se una spiegazione è corretta, se non ho "
            "una risposta giusta con cui confrontarla? La riprendo più avanti, prima "
            "delle verifiche.",
}

FIGURE = {
    "kind": "figure",
    "en": ("The two-branch pipeline",
           "Typed error → cluster branch (DINOv2 + GMM, top) + language branch "
           "(SliceLens, bottom) → Slice-Quality Vector → per-image explanation.   "
           "Colours = source method; orange = this work."),
    "it": ("La pipeline a due rami",
           "Errore tipato → ramo cluster (DINOv2 + GMM, sopra) + ramo linguistico "
           "(SliceLens, sotto) → Slice-Quality Vector → spiegazione per immagine.   "
           "Colori = metodo di origine; arancione = questo lavoro."),
    "note": "Questa è la pipeline completa. A sinistra input ed estrazione dell'errore "
            "tipato; poi si biforca in due rami — cluster sopra (DINOv2 + GMM) e "
            "SliceLens sotto (LLM + VLM); si ricongiungono nel Slice-Quality Vector che "
            "valida le slice; infine la spiegazione per-immagine. Nota che l'errore tipato "
            "(Beyond IoU) sta PRIMA della biforcazione: produce le istanze d'errore tipate "
            "che ENTRAMBI i rami condividono, ed è la metà '→ errore' di ogni descrizione. "
            "I colori indicano il paper di provenienza, l'arancione è il mio contributo.",
}

METHODS = {
    "kind": "content",
    "en": ("Five ideas this pipeline borrows, and why",
           [("SliceLens — failure patterns describable IN WORDS, not just by "
             "shape → the language branch", 0),
            ("Beyond IoU — error TYPES, not just how much is wrong → the "
             "typed taxonomy (Block 1)", 0),
            ("MCSD — is a group really tight, or just close by chance? → one "
             "of the six quality checks", 0),
            ("CB-SLICE — does the group make the SAME mistake every time? → "
             "another quality check", 0),
            ("SEG4SEG — cluster errors and decide automatically which groups "
             "are real → the cluster branch's backbone", 0),
            ("New here: typing errors per object, a 6-part quality score, one "
             "sentence per image", 0)]),
    "it": ("Cinque idee prese in prestito, e perché",
           [("SliceLens — pattern d'errore descrivibili A PAROLE, non solo "
             "per forma → il ramo linguistico", 0),
            ("Beyond IoU — TIPI di errore diversi, non solo quanto si "
             "sbaglia → la tassonomia tipata (Blocco 1)", 0),
            ("MCSD — un gruppo è davvero coeso, o solo vicino per caso? → "
             "uno dei sei assi di qualità", 0),
            ("CB-SLICE — il gruppo sbaglia sempre allo STESSO modo? → un "
             "altro asse di qualità", 0),
            ("SEG4SEG — raggruppare gli errori e decidere da soli quali "
             "gruppi sono veri → base del ramo cluster", 0),
            ("Nuovo qui: errori tipati per oggetto, un punteggio di qualità a "
             "6 parti, una frase per immagine", 0)]),
    "note": "Cinque idee prese in prestito, ognuna per un problema preciso. SliceLens "
            "risponde a: come trovo un pattern che si può descrivere a parole, non solo "
            "per come appare? Da lì viene il ramo linguistico. Beyond IoU: come "
            "distinguo TIPI di errore diversi, non solo quanto si sbaglia? Da lì viene "
            "la tassonomia del blocco uno. MCSD: come verifico che un gruppo sia "
            "davvero coeso e non vicino per puro caso? È uno dei sei assi di qualità. "
            "CB-SLICE: come verifico che un gruppo sbagli sempre nello stesso modo? Un "
            "altro asse. SEG4SEG: come raggruppo gli errori e decido automaticamente "
            "quali gruppi sono reali? È la base del ramo cluster e della "
            "certificazione. Quello che ho aggiunto io: l'errore tipato per singolo "
            "oggetto — non esisteva a livello di istanza — il punteggio di qualità a "
            "sei parti che li mette tutti insieme, e la frase finale per immagine.",
}

PRINCIPLES = {
    "kind": "content",
    "en": ("Three ideas that repeat everywhere in this design",
           [("Normalize before comparing — a boundary error and a missed "
             "object aren't on the same scale → Block 1, the cluster branch, "
             "the language gate", 0),
            ("Independence before judging — whatever FORMS a group cannot "
             "also certify it, or the test is circular → the two branches, "
             "every chance-calibrated check", 0),
            ("Facts stay outside claims — ground truth is asserted, "
             "everything else must be verified first → how every sentence is "
             "written", 0)]),
    "it": ("Tre idee che si ripetono in tutta la pipeline",
           [("Normalizzare prima di confrontare — un errore di bordo e un "
             "oggetto mancato non sono sulla stessa scala → Blocco 1, ramo "
             "cluster, gate del ramo linguistico", 0),
            ("Indipendenza prima del giudizio — chi FORMA un gruppo non può "
             "anche certificarlo, altrimenti il test è circolare → i due "
             "rami separati, ogni verifica calibrata al caso", 0),
            ("I fatti restano fuori dalle affermazioni — il ground truth si "
             "asserisce, il resto va verificato prima → come è scritta ogni "
             "frase", 0)]),
    "note": "Prima di entrare nei blocchi, tre idee che tornano più volte, così quando le "
            "rivedete le riconoscete invece di sembrare scelte scollegate. Primo: "
            "normalizzare prima di confrontare. Un errore di bordo ha per natura un IoU "
            "alto, un oggetto mancato ha IoU zero — metterli sulla stessa scala "
            "penalizzerebbe sempre il primo. La vedrete tre volte: nella tassonomia del "
            "blocco uno, nel clustering stratificato per tipo, e nel gate del ramo "
            "linguistico che normalizza la severità dentro ogni sottotipo. Secondo: "
            "indipendenza prima del giudizio. Se uso una misura per FORMARE un gruppo, "
            "non posso usare la stessa misura per dire che il gruppo è buono — è "
            "circolare, e ci sono cascato all'inizio. Per questo i due rami restano "
            "separati fino alla fine, e per questo ogni test è calibrato contro il "
            "caso invece che contro una soglia fissa. Terzo: i fatti restano fuori "
            "dalle affermazioni. La classe di un oggetto viene dal ground truth, è un "
            "fatto, non lo metto in discussione; tutto il resto — l'attributo, la "
            "situazione — è un'affermazione che va verificata prima di essere scritta. "
            "È il motivo per cui la classe sta fuori dalle virgolette nelle frasi che "
            "vi mostro tra poco.",
}

BLOCK1 = {
    "kind": "content",
    "en": ("Block 1 (Beyond IoU) — what exactly went wrong, object by object",
           [("An IoU number says how much is wrong; it never says WHAT is wrong", 0),
            ("VOC semantic seg → connected components per class = instances", 0),
            ("Two-threshold matching (τ=0.5 to claim, τ=0.25 for fragments) → 7 cases:", 0),
            ("boundary · extent · confusion · FP · FN · split · merge", 1),
            ("The second threshold is what makes split/merge possible at all: two "
             "fragments of one object can never both pass 0.5", 1),
            ("Per instance: IoU, error vector [E_bnd, E_ext, E_seg], class — masks only", 0)]),
    "it": ("Blocco 1 (Beyond IoU) — che cosa è andato storto, oggetto per oggetto",
           [("Un numero di IoU dice quanto si sbaglia; non dice mai CHE COSA si sbaglia", 0),
            ("VOC semantic seg → componenti connesse per classe = istanze", 0),
            ("Matching a due soglie (τ=0.5 per rivendicare, τ=0.25 per i frammenti) → 7 casi:", 0),
            ("boundary · extent · confusion · FP · FN · split · merge", 1),
            ("È la seconda soglia a rendere possibili split/merge: due frammenti dello "
             "stesso oggetto non possono superare entrambi 0.5", 1),
            ("Per istanza: IoU, vettore d’errore [E_bnd, E_ext, E_seg], classe — solo maschere", 0)]),
    "note": "Il primo blocco è deterministico, solo maschere, nessuna rete, ed è dove uso "
            "l'idea di Beyond IoU: l'IoU dice quanto sbagli, non che cosa sbagli. VOC è semantic "
            "segmentation, quindi spezzo ogni classe in componenti connesse, che diventano le "
            "istanze. Il matching usa due soglie: con IoU sopra zero virgola cinque una "
            "predizione rivendica l'oggetto, sopra zero virgola venticinque lo tocca soltanto. "
            "Questa seconda soglia non è un dettaglio: senza, split e merge sono geometricamente "
            "impossibili da rilevare, perché due frammenti dello stesso oggetto non possono "
            "superare entrambi il cinquanta per cento. Prima di accorgermene, quei casi "
            "risultavano zero — e non perché non ci fossero. Alla fine ogni istanza porta il suo "
            "IoU, il vettore d'errore tipato e la classe.",
}

CLUSTER = {
    "kind": "content",
    "en": ("Cluster branch (SEG4SEG) — “which failures LOOK alike?”",
           [("Every failed object becomes a vector describing its appearance "
             "(DINOv2); a false negative, with no predicted mask, uses the "
             "true one instead", 0),
            ("That vector is combined with the error type, the subtype, and "
             "how badly the instance fails — so grouping sees both "
             "appearance and severity", 0),
            ("Not one clustering for all errors, but a SEPARATE one per "
             "type — the number of groups is chosen automatically, "
             "statistically", 0),
            ("a boundary error has high IoU by nature: mixed in with the "
             "rest, it could never look as severe as a fully missed object", 1),
            ("Answers: WHAT failed? (error type + visual attribute + class)", 0)]),
    "it": ("Ramo cluster (SEG4SEG) — «quali fallimenti si ASSOMIGLIANO?»",
           [("Ogni oggetto fallito diventa un vettore che ne descrive "
             "l'aspetto (DINOv2); un falso negativo, senza maschera "
             "predetta, usa quella vera", 0),
            ("Quel vettore si combina con il tipo di errore, il sottotipo, "
             "e quanto l'istanza sbaglia — così il raggruppamento vede sia "
             "l'aspetto sia la gravità", 0),
            ("Non un solo raggruppamento per tutti gli errori, ma uno "
             "SEPARATO per ogni tipo — il numero di gruppi si sceglie da "
             "solo, statisticamente", 0),
            ("un errore di bordo ha per natura un IoU alto: mescolato con "
             "gli altri, non potrebbe mai risultare grave quanto un "
             "oggetto mancato del tutto", 1),
            ("Risponde a: CHE COSA è fallito? (tipo di errore + attributo "
             "visivo + classe)", 0)]),
    "note": "Il ramo cluster risponde alla domanda: quali fallimenti si assomigliano, "
            "guardandoli? DINOv2 congelato dà un embedding per istanza col mask-pooling; per i "
            "falsi negativi uso la maschera ground-truth, perché lì la predizione non c'è. "
            "Concateno embedding, errore tipato e one-hot del sottotipo, con un canale di "
            "performance pesato da gamma. Qui uso lo strumento di SEG4SEG: non faccio una GMM "
            "sola, ne faccio una per tipo d'errore, con il numero di componenti scelto dal BIC. Il "
            "motivo è che un errore di bordo ha per costruzione un IoU alto, quindi in un "
            "confronto unico non potrebbe mai risultare grave quanto un oggetto mancato del "
            "tutto: normalizzando dentro il tipo, il confronto torna equo.",
}

LANGUAGE = {
    "kind": "content",
    "en": ("Language branch (SliceLens) — “in WHICH SITUATION does it fail?”",
           [("An LLM (Gemma) proposes situational hypotheses per class", 0),
            ("e.g. “an object cut off at the image border”, “several overlapping objects”", 1),
            ("Qwen VLM grounding: P(yes | hypothesis, region) for every instance", 0),
            ("Same instances as the cluster branch (exact instance_id join, one "
             "shared substrate) — so the two views are comparable", 0),
            ("Verified with SliceLens’s own slope-trend, then re-grounded on the "
             "FULL class population (≈24k evaluations)", 0)]),
    "it": ("Ramo linguistico (SliceLens) — «in QUALE SITUAZIONE fallisce?»",
           [("Un LLM (Gemma) propone ipotesi situazionali per classe", 0),
            ("es. “an object cut off at the image border”, “several overlapping objects”", 1),
            ("Grounding VLM (Qwen): P(sì | ipotesi, regione) per ogni istanza", 0),
            ("Le stesse istanze del ramo cluster (join esatto per instance_id, un solo "
             "substrato condiviso) — così le due viste sono confrontabili", 0),
            ("Verificate con lo slope-trend di SliceLens, poi ri-groundate sull’INTERA "
             "popolazione della classe (≈24k valutazioni)", 0)]),
    "note": "Il ramo linguistico è SliceLens, e risponde all'altra domanda: in quale situazione "
            "il modello fallisce. Un LLM genera ipotesi situazionali per classe — un oggetto "
            "tagliato dal bordo, più oggetti sovrapposti — e Qwen fa il grounding: per ogni "
            "istanza calcola la probabilità che quella regione corrisponda all'ipotesi. Un punto "
            "su cui ho lavorato parecchio: le istanze sono ESATTAMENTE le stesse del ramo "
            "cluster, unite per identificatore e non per sovrapposizione di bounding box come "
            "facevo all'inizio. Serve perché le due viste siano confrontabili sullo stesso "
            "oggetto. Le ipotesi che passano la verifica vengono poi ri-groundate sull'intera "
            "popolazione della classe, non su un campione: circa ventiquattromila valutazioni. "
            "È una ri-verifica più severa, non più permissiva — qualche ipotesi che sembrava "
            "sistematica sul campione non lo è più sui dati completi.",
}

SQV_AXES_TABLE = [   # (axis, paper, question in EN, question in IT)
    ("ω² (severity)", "SEG4SEG",
     "How much worse does this group do than the REST of the whole dataset "
     "(errors + correct references) — not just the other errors?",
     "Quanto questo gruppo si comporta peggio del RESTO di tutto il dataset "
     "(errori + predizioni corrette) — non solo degli altri errori?"),
    ("manifold compactness", "MCSD",
     "Are these instances really close together in appearance space "
     "(DINOv2), or just close by chance?",
     "Le istanze sono davvero vicine tra loro nello spazio dell'aspetto "
     "visivo (DINOv2), o solo vicine per caso?"),
    ("misprediction coherence", "CB-SLICE",
     "Does the model fail the SAME way on these instances, or differently "
     "each time?",
     "Il modello sbaglia sempre nello STESSO modo su queste istanze, o in "
     "modi diversi ogni volta?"),
    ("error-type purity", "new",
     "Is this group dominated by ONE error type, or is it a mix?",
     "Il gruppo è dominato da UN tipo di errore, o è un mix?"),
    ("slice purity", "SEG4SEG, adapted",
     "Do the instances share a verified visual attribute more than a random "
     "same-type group would?",
     "Le istanze condividono un attributo visivo verificato più di "
     "quanto farebbe un gruppo casuale dello stesso tipo?"),
    ("significance", "new",
     "Could this have happened by pure chance, or does it beat a "
     "statistical test?",
     "Questo poteva succedere per puro caso, o supera un test statistico?"),
]

SQV_AXES = {
    "kind": "table1",
    "table": SQV_AXES_TABLE,
    "en": ("The six axes, one by one",
           ["axis", "paper", "question it answers"]),
    "it": ("I sei assi, uno per uno",
           ["asse", "paper", "domanda a cui risponde"]),
    "note": "Ogni slice candidata viene misurata su sei assi indipendenti, presi in "
            "prestito da quattro paper diversi più due che ho aggiunto io. ω² per la "
            "severità viene da SEG4SEG — quanto il gruppo si comporta peggio del RESTO "
            "di tutto il dataset, non solo degli altri errori. La compattezza sul "
            "manifold viene da MCSD — le istanze sono davvero vicine nello spazio "
            "dell'aspetto visivo, o vicine per caso? La coerenza della misprediction "
            "viene da CB-SLICE — il modello sbaglia sempre allo stesso modo? Purezza "
            "del tipo d'errore e significatività sono mie: la prima dice se il gruppo è "
            "dominato da un solo tipo di errore, la seconda se il risultato supera un "
            "test statistico o poteva succedere per caso. E la purezza di slice riprende "
            "l'idea di SEG4SEG ma cambiata: non una soglia fissa decisa a tavolino, un "
            "confronto col caso — la slice deve essere più coerente sugli attributi di "
            "gruppi casuali di errori dello stesso tipo. Le ultime due sono quelle "
            "'calibrate al caso' di cui ho parlato prima.",
}

SQV_GATE = {
    "kind": "content",
    "en": ("From six numbers to a verdict",
           [("The six axes are the measurement; each branch turns them into "
             "a KEEP/DISCARD verdict differently — “good” doesn't mean the "
             "same thing for a geometric group and a semantic one:", 0),
            ("Cluster slices → Def.1, SEG4SEG's success rule: severe (from "
             "ω² and significance) + big enough + pure (from the purity "
             "axes)", 0),
            ("“pure” means it BEATS CHANCE, not that it clears a fixed "
             "percentage decided in advance", 1),
            ("Language slices → adapted gate: the SAME typed severity from "
             "Block 1 (reused here) + high confidence + a positive "
             "slope-trend (does the error rate rise WITH the confidence?)", 0),
            ("the geometric axes don't apply here — these slices group by "
             "meaning, not by shape", 1)]),
    "it": ("Da sei numeri a un verdetto",
           [("I sei assi sono la misura; ogni ramo li trasforma in un "
             "verdetto KEEP/DISCARD in modo diverso — “buono” non "
             "vuol dire la stessa cosa per un gruppo geometrico e uno "
             "semantico:", 0),
            ("Slice cluster → Def.1, la regola di successo di SEG4SEG: "
             "grave (dagli assi ω² e significatività) + abbastanza grande "
             "+ pura (dagli assi di purezza)", 0),
            ("“pura” vuol dire che BATTE IL CASO, non che supera "
             "una percentuale fissa decisa a tavolino", 1),
            ("Slice linguistiche → gate adattato: la STESSA severità "
             "tipizzata del Blocco 1 (riusata qui) + confidence alta + "
             "uno slope-trend positivo (l'errore sale insieme alla "
             "confidenza?)", 0),
            ("gli assi geometrici non si applicano qui — queste slice "
             "raggruppano per significato, non per forma", 1)]),
    "note": "I sei assi sono la misura; questa slide è come diventano un verdetto — e "
            "vale la pena collegarla esplicitamente alla tabella di prima, perché ogni "
            "criterio qui viene da uno degli assi appena visti. Per i cluster uso il "
            "criterio Def.1 di SEG4SEG: severo viene dagli assi omega quadro e "
            "significatività, abbastanza grande è una soglia minima di istanze, puro "
            "viene dagli assi di purezza — con \"puro\" che vuol dire batte il caso, non "
            "una soglia fissa decisa a tavolino. Per le slice linguistiche gli assi "
            "geometrici non hanno senso, perché raggruppano per semantica, non per "
            "forma; quindi un gate adattato che riusa la STESSA severità tipizzata "
            "calcolata nel blocco uno — è il terzo posto dove torna il principio di "
            "normalizzare prima di confrontare, di cui ho parlato prima — più "
            "confidence alta e la verifica slope-trend di SliceLens: l'errore sale "
            "davvero insieme alla confidenza del VLM, o è piatto?",
}

GATE_TABLE_ROWS = [   # (criterion, cluster cell EN, language cell EN, cluster cell IT, language cell IT)
    ("severity",
     "z̄ ≥ 0 (mean severity z-score, normalised per error type)",
     "z̄ ≥ −0.25 (same z-score, softer threshold)",
     "z̄ ≥ 0 (z-score medio di severità, normalizzato per tipo d'errore)",
     "z̄ ≥ −0.25 (stesso z-score, soglia più morbida)"),
    ("size",
     "|S| ≥ N_min (minimum fraction of the dataset)",
     "|S| ≥ 3 instances",
     "|S| ≥ N_min (frazione minima del dataset)",
     "|S| ≥ 3 istanze"),
    ("purity / confidence",
     "purity permutation test, p ≤ α",
     "mean VLM confidence ≥ 0.6 over all members",
     "test di permutazione sulla purezza attributo, p ≤ α",
     "confidence media del VLM ≥ 0.6 su tutti i membri"),
    ("extra check",
     "coherence (same dominant type/output) + statistical significance",
     "slope-trend: does error rise WITH confidence?",
     "coerenza (stesso tipo/output dominante) + significatività statistica",
     "slope-trend: l'errore sale con la confidenza?"),
    ("verdict",
     "KEEP if all pass AND coherent; REFINE if severe (ω²) + significant "
     "but NOT coherent; else DISCARD",
     "certified only if ALL FOUR hold together — no REFINE path here",
     "KEEP se tutto passa ED è coerente; REFINE se grave (ω²) + "
     "significativo ma NON coerente; altrimenti DISCARD",
     "certificata solo se TUTTE E QUATTRO valgono insieme — nessun "
     "percorso REFINE qui"),
]

GATE_TABLE = {
    "kind": "gate_table",
    "rows": GATE_TABLE_ROWS,
    "en": ("How the verdict is reached",
           "The cluster branch uses FIVE independent checks: severity, "
           "size, purity, coherence, significance. The language branch "
           "uses FOUR, none geometric: severity (same measure, different "
           "threshold), size, VLM confidence, slope-trend.",
           ["criterion", "cluster → Def.1", "language → adapted gate"],
           "What REFINE does: a cluster slice that's severe (high ω²) and "
           "significant but NOT coherent isn't discarded right away — "
           "Block 7 splits it into purer sub-groups and re-tests, up to 5 "
           "rounds; dominant patterns surface first, finer ones only "
           "after. The language branch has no REFINE path: it either "
           "certifies, or it doesn't."),
    "it": ("Come si arriva al verdetto",
           "Il ramo cluster usa CINQUE controlli indipendenti: severità, "
           "dimensione, purezza, coerenza, significatività. Il "
           "linguistico ne usa QUATTRO, nessuno geometrico: severità "
           "(stessa misura, soglia diversa), dimensione, confidence del "
           "VLM, slope-trend.",
           ["criterio", "cluster → Def.1", "linguaggio → gate adattato"],
           "Che cosa fa REFINE: una slice cluster grave (ω² alto) e "
           "significativa ma NON coerente non viene scartata subito — il "
           "Blocco 7 la spacca in sottogruppi più puri e ripete il test, "
           "fino a 5 giri; i pattern dominanti emergono prima, quelli più "
           "fini solo dopo. Il ramo linguistico non ha un percorso "
           "REFINE: certifica, o no."),
    "note": "Questa slide mette a confronto i due gate riga per riga. Il ramo cluster "
            "usa cinque controlli indipendenti — severità, dimensione, purezza, "
            "coerenza, significatività. Il linguistico ne usa quattro, e nessuno è "
            "geometrico: la severità è calcolata esattamente come nel cluster, stesso "
            "z-score, ma con soglia più morbida — meno zero virgola venticinque invece "
            "di zero, perché lo z medio di una slice a dominanza di falsi negativi "
            "oscilla per puro rumore intorno allo zero, e un taglio esatto scarterebbe "
            "metà delle slice buone per sfortuna. L'ultimo punto, il più importante da "
            "spiegare bene: REFINE esiste SOLO per il ramo cluster. Se una slice è "
            "grave e significativa ma non coerente, il Blocco 7 non la scarta — la "
            "spacca in sottogruppi più puri e ripete il test fino a cinque volte, "
            "perché i pattern dominanti emergono per primi e quelli più fini restano "
            "nascosti finché non si toglie il residuo dei primi. Il ramo linguistico "
            "non ha questo passaggio intermedio: o l'ipotesi supera tutti e quattro i "
            "criteri insieme, o non viene certificata, punto.",
}

EXPL = {
    "kind": "content",
    "en": ("Per-image explanation — situation → error",
           [("Qwen attribute tagging on kept slices (small / dark / occluded / …)", 0),
            ("Per image, aggregate the systematic patterns:", 0),
            ("cluster: “small objects of class ‘bicycle’ not detected”", 1),
            ("language: “‘a cow cut off at the image border’ → not detected”", 1),
            ("Output: descriptions (JSON / txt), slice→images and description→images "
             "indices, interactive HTML gallery, repair advice", 0)]),
    "it": ("Spiegazione per immagine — situazione → errore",
           [("Tagging degli attributi con Qwen sulle slice tenute (small / dark / occluded / …)", 0),
            ("Per ogni immagine, aggrego i pattern sistematici:", 0),
            ("cluster: “small objects of class ‘bicycle’ not detected”", 1),
            ("linguistico: “‘a cow cut off at the image border’ → not detected”", 1),
            ("Output: descrizioni (JSON / txt), indici slice→immagini e "
             "descrizione→immagini, gallery HTML, consigli di riparazione", 0)]),
    "note": "La spiegazione: Qwen tagga un campione delle slice tenute con attributi — "
            "small, dark, occluso. Poi per ogni immagine aggrego i pattern sistematici: dal "
            "cluster viene il tipo d'errore, da SliceLens la situazione, combinati come "
            "'situazione freccia errore'. In output ho le descrizioni testuali per immagine e "
            "una gallery HTML navigabile.",
}

RESULTS = {
    "kind": "content",
    "en": ("Results — VOC2012 val (1449 images)",
           [("6406 typed errors + 522 correct references; FN 2651, FP 2628, "
             "boundary 799, extent 188, merge 98, confusion 38, split 4", 0),
            ("14 certified cluster slices (FN 13, FP 1) + 35 validated language "
             "hypotheses (all 20 classes, ≈24k VLM groundings)", 0),
            ("826 / 1232 error images get a description (67%) — instance coverage 36.7% "
             "(union of both branches)", 0),
            ("Example: “‘a cow cut off at the image border’ → not detected”", 0)]),
    "it": ("Risultati — VOC2012 val (1449 immagini)",
           [("6406 errori tipati + 522 riferimenti corretti; FN 2651, FP 2628, "
             "boundary 799, extent 188, merge 98, confusion 38, split 4", 0),
            ("14 slice cluster certificate (FN 13, FP 1) + 35 ipotesi linguistiche "
             "validate (tutte e 20 le classi, ≈24k grounding VLM)", 0),
            ("826 / 1232 immagini con errori ricevono una descrizione (67%) — copertura di istanze "
             "36.7% (unione dei due rami)", 0),
            ("Esempio: “cow — ‘an object cut off at the image border’ → non rilevata”", 0)]),
    "note": "Risultati su tutto il validation set di VOC, 1449 immagini: 6406 errori tipati e 522 "
            "riferimenti corretti, tassonomia dominata da falsi negativi e falsi positivi. "
            "Quattordici slice cluster superano la certificazione, e ognuna batte "
            "dimostrabilmente il caso. SliceLens, girato su tutte e venti le classi con circa "
            "ventiquattromila grounding, dà trentacinque ipotesi validate. In tutto 826 delle "
            "1232 immagini con errori ricevono una descrizione, il sessantasette per cento; la "
            "copertura a livello di istanze — l'unione di entrambi i rami — è al trentasei "
            "virgola sette per cento, perché certifico solo ciò che supera il test. Un esempio: "
            "una mucca, con la situazione 'un oggetto tagliato dal bordo dell'immagine', che "
            "non viene rilevata.",
}

# The full catalog behind the RESULTS headline count: all 14 cluster slices and all
# 35 language hypotheses, extracted straight from slices_index.json (2026-08-31) so
# the numbers can never drift from what the pipeline actually certified. 35 flat rows
# don't fit readably on one slide, so the language half is grouped by SITUATION
# (which class each one applies to) rather than listed as 35 separate lines -- this
# also surfaces something the flat list would hide: "cut off at the image border" is
# the most universal situation (11 of 20 classes), which is exactly why the
# model-repair loop (last slide) targets it.
CATALOG_CLUSTER = [  # (class [+ attribute], error type, size)
    ("chair", "FP", 156), ("sofa", "FN", 138), ("bicycle", "FN", 123),
    ("bottle (small)", "FN", 113), ("person", "FN", 104), ("motorbike", "FN", 103),
    ("car (small)", "FN", 91), ("cat (occluded)", "FN", 89),
    ("person (occluded)", "FN", 88), ("bird (occluded)", "FN", 87),
    ("horse (occluded)", "FN", 87), ("sheep (cluttered)", "FN", 86),
    ("bicycle", "FN", 71), ("aeroplane (small)", "FN", 62),
]
CATALOG_LANGUAGE = [  # (hypothesis, n classes, class list)
    ("an object cut off at the image border", 11,
     "boat, bottle, bus, car, cat, cow, diningtable, dog, sheep, train, tvmonitor"),
    ("an object occluded by another object", 5, "bird, bus, car, cat, horse"),
    ("several overlapping objects of the same type", 5, "bird, boat, bus, car, cat"),
    ("an object with thin or elongated parts", 4, "car, cat, dog, sheep"),
    ("a small object", 4, "cow, dog, motorbike, tvmonitor"),
    ("an object that looks similar to a different category", 3, "bus, cat, dog"),
    ("blurry <class> (class-specific wording)", 2, "bus, cat"),
    ("an object that blends into its background", 1, "person"),
]

CATALOG = {
    "kind": "catalog",
    "en": ("The full catalog: 14 cluster slices + 35 language hypotheses",
           "Cluster slices (14) — class, error type, size",
           "Language hypotheses (35) — grouped by situation"),
    "it": ("Il catalogo completo: 14 slice cluster + 35 ipotesi linguistiche",
           "Slice cluster (14) — classe, tipo di errore, dimensione",
           "Ipotesi linguistiche (35) — raggruppate per situazione"),
    "note": "Il catalogo completo dietro il numero di riepilogo. A sinistra le "
            "quattordici slice cluster, ordinate per dimensione: oggetti di classe "
            "sedia allucinati è la più grande, con centocinquantasei istanze. Notate "
            "bicicletta e persona che compaiono due volte: sono due cluster visivamente "
            "distinti per la stessa classe, non un doppione — DINOv2 ha trovato due "
            "gruppi diversi di biciclette mancate, per esempio. A destra le "
            "trentacinque ipotesi linguistiche, ma raggruppate per situazione invece "
            "che elencate una per una, perché il dato interessante è proprio questo: "
            "'tagliato dal bordo dell'immagine' copre undici classi su venti, la "
            "situazione più universale — non è un caso che sia proprio quella scelta "
            "per il repair loop nell'ultima slide.",
}

VALIDATION_THEORY = {
    "kind": "content",
    "en": ("How do you check an explanation with no answer key?",
           [("No dataset says WHY a model failed — nothing to check an "
             "explanation against directly", 0),
            ("The fix: compare against CHANCE, not a reference — would a "
             "random group of errors look just as coherent?", 0),
            ("Trap 1: don't use the SAME measurement to build a group and to "
             "prove it's real — that's circular", 1),
            ("Trap 2: a fixed number from someone else's experiment measures "
             "THEIR noise, not this one", 1),
            ("The rule used everywhere: compare against a random draw from "
             "the SAME noisy source, never an imported number", 0)]),
    "it": ("Come si controlla una spiegazione senza una risposta corretta?",
           [("Nessun dataset dice PERCHÉ un modello ha sbagliato — niente con "
             "cui confrontare direttamente una spiegazione", 0),
            ("La soluzione: confrontare col CASO, non con un riferimento — un "
             "gruppo di errori a caso sarebbe altrettanto coerente?", 0),
            ("Trappola 1: non uso la STESSA misura per formare un gruppo e "
             "per dire che è vero — sarebbe circolare", 1),
            ("Trappola 2: un numero fisso preso da un altro esperimento "
             "misura IL SUO rumore, non il mio", 1),
            ("Regola generale: confronto sempre con un'estrazione casuale "
             "dalla STESSA fonte rumorosa, mai un numero importato", 0)]),
    "note": "Questa è la domanda più difficile delle cinque: come controllo una "
            "spiegazione, se non ho una risposta corretta con cui confrontarla? Non "
            "esiste un dataset che dica perché un modello ha sbagliato, quindi non "
            "posso semplicemente confrontare la mia spiegazione con un riferimento. La "
            "soluzione non è rinunciare al controllo, è confrontare col CASO: un "
            "pattern è reale se batte quello che produrrebbe comunque un gruppo "
            "casuale dello stesso tipo. Due trappole facili da non notare. La prima: se "
            "uso una misura per FORMARE i gruppi — per esempio quanto sono vicini nello "
            "spazio dell'embedding — non posso usare la stessa misura per dire che sono "
            "davvero coerenti, sarebbe come darmi ragione da solo. La seconda: prendere "
            "in prestito un numero fisso da un altro esperimento misura il rumore del "
            "SUO strumento, non del mio. La regola che applico ovunque, e che vedrete "
            "ripetuta in ogni verifica dei prossimi minuti: il confronto col caso deve "
            "usare lo stesso strumento rumoroso della misura che sto testando, mai un "
            "numero preso da altrove.",
}

VALIDATION = {
    "kind": "content",
    "en": ("Does it hold up? — four independent checks",
           [("Split-half: slices found on half the images, re-certified on the "
             "other half → 94% (16/17)", 0),
            ("Synthetic corruption: injected failure mode is recovered — victims "
             "concentrate in 8–9 per cluster at both doses", 0),
            ("Human reference: 43 val images vs hand-written notes, re-scored by an LLM "
             "judge with 43 mismatched controls → class agreement 0.96 vs 0.06, "
             "error-type 0.24 vs 0.14, self-consistency 0.98", 0),
            ("Description precision (1497 VLM checks): situations 0.60 vs 0.22 "
             "on a same-class control; attribute phrases only +0.13 → the tag "
             "vocabulary is the weak link, not the slices", 0)]),
    "it": ("Regge? — quattro verifiche indipendenti",
           [("Split-half: slice scoperte su metà immagini, ri-certificate "
             "sull'altra metà → 94% (16/17)", 0),
            ("Corruzione sintetica: il fallimento iniettato viene ritrovato — "
             "le vittime si concentrano 8–9 per cluster a entrambe le dosi", 0),
            ("Riferimento umano: 43 immagini vs note scritte a mano, ri-valutate da un "
             "giudice LLM con 43 coppie mescolate → classe 0.96 contro 0.06, "
             "tipo d'errore 0.24 contro 0.14, auto-consistenza 0.98", 0),
            ("Precisione delle descrizioni (1497 verifiche VLM): situazioni 0.60 "
             "contro 0.22 su un controllo di stessa classe; le frasi con "
             "attributo solo +0.13 → il vocabolario dei tag è l'anello debole, "
             "non le slice", 0)]),
    "note": "Non basta che i numeri siano belli: le slice sono scoperte e testate sugli stessi "
            "dati, quindi il rischio di double dipping è reale. Ho quattro verifiche indipendenti. "
            "La split-half: scopro le slice su metà delle immagini e le ri-certifico sull'altra "
            "metà — replicano tutte, sette su sette. La corruzione sintetica: rompo di proposito "
            "il modello su una situazione nota e controllo che la pipeline la ritrovi — le vittime "
            "si concentrano in pochi cluster a entrambe le dosi provate. Il confronto con "
            "l'annotazione umana su 43 immagini, che ho fatto ri-valutare da un giudice LLM "
            "mettendogli dentro 43 coppie mescolate come controllo: sulle coppie false il giudice non concede mai nulla, quindi il confronto vale. E li' emerge il dato piu' istruttivo di tutti — sulla CLASSE dell'oggetto l'accordo con l'annotatore umano e' schiacciante, sul TIPO d'errore siamo quasi al livello del caso: diciamo giusto QUALE oggetto ha fallito, e lo diciamo in una tassonomia che non e' quella con cui lo racconta una persona. E infine la precisione: chiedo al VLM se la frase "
            "che consegno all'utente è davvero vera per quell'oggetto. Il confronto non è con un "
            "errore qualsiasi, ma con un errore della STESSA classe e dello stesso tipo preso "
            "fuori dalla slice: così il merito di aver indovinato la classe non conta, e resta "
            "solo la situazione. Le frasi situazionali passano sei volte su dieci contro due su "
            "dieci del controllo; quelle costruite sugli attributi guadagnano poco, e guardando "
            "frase per frase si vede perché: le etichette 'dark' e 'blurry' il modello non le "
            "riconferma quasi mai. È un difetto misurato del canale di tagging, non delle slice — "
            "gli stessi oggetti si raggruppano e replicano bene, li stiamo solo chiamando con una "
            "parola sbagliata.",
}

LIMITS = {
    "kind": "content",
    "en": ("Limitations & next steps",
           [("Instance coverage 36.7% (union of both branches): deliberately "
             "conservative — a slice must beat chance, not a threshold", 0),
            ("Attribute vocabulary is 6 words after removing dark; occlusion "
             "specifically fails to verify on small fragments — a per-region "
             "query may be the wrong instrument for it", 0),
            ("ω² stayed two orders of magnitude below a gating threshold even on "
             "the stronger model — a size-confounded axis, not a reference-"
             "scarcity artefact; the question is now closed, not deferred", 0),
            ("Next: external validation on data with known edits (pascal_ea, "
             "pending access); model-repair loop", 0)]),
    "it": ("Limiti & prossimi passi",
           [("Copertura di istanze 36.7% (unione dei due rami): volutamente "
             "conservativa — una slice deve battere il caso, non una soglia", 0),
            ("Il vocabolario di attributi è di 6 parole dopo aver tolto dark; "
             "l'occlusione in particolare non si verifica su frammenti piccoli — "
             "forse una domanda per-regione non è lo strumento giusto", 0),
            ("ω² resta due ordini di grandezza sotto una soglia di gate anche sul "
             "modello più forte — un asse condizionato dalla dimensione, non un "
             "artefatto di scarsità di riferimenti; la domanda è chiusa, non rimandata", 0),
            ("Prossimi: validazione esterna su dati con modifiche note (pascal_ea, "
             "accesso in attesa); loop di riparazione del modello", 0)]),
    "note": "I limiti, detti onestamente. La copertura a livello di istanze è al trentasette "
            "per cento, contando l'unione di entrambi i rami: è il prezzo di una certificazione "
            "che chiede a ogni slice di battere il caso, non una soglia. Il vocabolario di "
            "attributi è sceso a sei parole dopo aver tolto 'dark', che non superava la verifica "
            "in nessuna modalità; l'occlusione resta debole per un motivo strutturale, non di "
            "vocabolario — è una domanda su cosa NON si vede, e una regione ritagliata "
            "potrebbe essere lo strumento sbagliato per chiederlo. Sull'omega quadro: anche col "
            "modello più forte, con l'ottanta per cento in più di riferimenti corretti, resta "
            "due ordini di grandezza sotto la soglia — quindi non era mai stata scarsità di "
            "dati, è chiuso come risultato, non lasciato aperto. I prossimi passi: la "
            "validazione esterna su un dataset con modifiche note, appena arriva l'accesso; "
            "e chiudere il ciclo con la riparazione del modello.",
}

PANELS = os.path.join(DOCS, "panels")

EXAMPLE_COW = {
    "kind": "example",
    "image": os.path.join(PANELS, "2007_003088.png"),
    "en": ("What the system produces, on one image",
           ["Image 2007_003088: 2 wrong objects, 2 in systematic patterns.",
            "  - small objects of class 'bottle' not detected      [cluster_fn_8_r0]",
            '  - bottle - "an object cut off at the image border" -> not detected  [SliceLens]',
            "  - occluded objects of class 'person' not detected   [cluster_fn_16_r1]"],
           "The bottle is small AND clipped by the frame; the second person is behind "
           "the first. Both branches fire, and they say different things: WHAT failed "
           "(small bottle, occluded person) and IN WHICH SITUATION (cut off by the frame)."),
    "it": ("Che cosa produce il sistema, su una immagine",
           ["Image 2007_003088: 2 wrong objects, 2 in systematic patterns.",
            "  - small objects of class 'bottle' not detected      [cluster_fn_8_r0]",
            '  - bottle - "an object cut off at the image border" -> not detected  [SliceLens]',
            "  - occluded objects of class 'person' not detected   [cluster_fn_16_r1]"],
           "La bottiglia è piccola E tagliata dal bordo; la seconda persona è dietro la "
           "prima. Entrambi i rami rispondono, e dicono cose diverse: CHE COSA è fallito "
           "(bottiglia piccola, persona occlusa) e IN QUALE SITUAZIONE (tagliata dal bordo)."),
    "note": "Questa è la forma concreta del deliverable, su una immagine sola. La donna tiene "
            "una bottiglia, che nella verità è la macchia viola in basso: il modello non la "
            "vede affatto. E dietro di lei c'è una seconda persona, che il modello fonde con "
            "la prima. Il ramo cluster dice che cosa è fallito — oggetti piccoli di classe "
            "bottiglia non rilevati, oggetti occlusi di classe persona non rilevati. Il ramo "
            "SliceLens dice in quale situazione: un oggetto tagliato dal bordo dell'immagine. "
            "Le frasi non si ripetono, si completano — ed è esattamente il motivo per cui "
            "tengo i due rami separati. Notate anche come è scritta la frase di SliceLens: la "
            "classe sta fuori dalle virgolette, perché viene dal ground truth ed è un fatto; "
            "dentro le virgolette c'è solo ciò che la pipeline afferma e ha verificato.",
}

EXAMPLE_BUS = {
    "kind": "example",
    "image": os.path.join(PANELS, "2007_003621.png"),
    "en": ("A harder one — and what the output admits",
           ["Image 2007_003621: 6 wrong objects, 4 in systematic patterns and 2 isolated.",
            '  - bus - "cut off at the border" / "blurry" / "occluded" / "looks like '
            'another category" -> not detected   [5 SliceLens patterns]',
            "  - small objects of class 'bicycle' not detected     [cluster_fn_16_r1]",
            "  - objects of class 'bicycle' grouped by appearance only",
            "    (no verified attribute) not detected              [cluster_fn_2_r0]",
            "  ~ 2 isolated objects near candidate patterns (DISCARD, not certified)"],
           "The distant bus is missed and the bicycle's thin parts collapse into a blob. "
           "Two honesty devices are visible: a group we cannot NAME is reported as such "
           "rather than with a hollow class label, and errors outside any certified "
           "pattern stay isolated, with the candidate they came closest to."),
    "it": ("Un caso più difficile — e ciò che l'output ammette",
           ["Image 2007_003621: 6 wrong objects, 4 in systematic patterns and 2 isolated.",
            '  - bus - "cut off at the border" / "blurry" / "occluded" / "looks like '
            'another category" -> not detected   [5 pattern SliceLens]',
            "  - small objects of class 'bicycle' not detected     [cluster_fn_16_r1]",
            "  - objects of class 'bicycle' grouped by appearance only",
            "    (no verified attribute) not detected              [cluster_fn_2_r0]",
            "  ~ 2 isolated objects near candidate patterns (DISCARD, non certificati)"],
           "Il bus in lontananza è perso e le parti sottili della bici collassano in una "
           "macchia. Si vedono due accorgimenti di onestà: un gruppo che non sappiamo "
           "NOMINARE è dichiarato tale invece che con un'etichetta di classe vuota, e gli "
           "errori fuori da ogni pattern certificato restano isolati, col candidato più vicino."),
    "note": "Un caso più difficile, per mostrare che cosa succede quando la pipeline NON sa. "
            "Qui il bus in fondo viene perso e la bicicletta perde le parti sottili, che "
            "collassano in una macchia. Sei oggetti sbagliati, quattro rientrano in pattern "
            "certificati e due restano dichiarati come isolati, con indicato il pattern "
            "candidato a cui erano più vicini. Guardate però la penultima riga, perché è il "
            "risultato di una misura: quel gruppo di biciclette è certificato — le istanze "
            "ricorrono davvero insieme — ma non ha un attributo che il modello visione-"
            "linguaggio confermi, quindi la frase lo dichiara apertamente invece di dire "
            "'oggetti di classe bicicletta', che avevo misurato come una frase vuota: "
            "confermata sulle sue istanze meno spesso che su una bicicletta qualsiasi. "
            "Preferisco un output che ammette di non sapere, piuttosto che una descrizione "
            "che riempie il vuoto.",
}

PRECISION = {
    "kind": "picture",
    "image": {"en": os.path.join(DOCS, "precision_figure.png"),
              "it": os.path.join(DOCS, "precision_figure_it.png")},
    "en": ("Are the descriptions true? (the check that was missing)",
           "Each shipped phrase re-verified by the VLM on the instance's own region, "
           "against a control of the SAME class and error type outside the slice — so "
           "guessing the class earns nothing. SliceLens situations are confirmed on 90% of "
           "their instances against a 6% null; attribute phrases doubled their lift when "
           "the tagging channel was rebuilt; class-only phrases measure as empty."),
    "it": ("Le descrizioni sono vere? (la verifica che mancava)",
           "Ogni frase consegnata viene ri-verificata dal VLM sulla regione dell'istanza, "
           "contro un controllo della STESSA classe e dello stesso tipo d'errore fuori "
           "dalla slice — indovinare la classe non paga. Le situazioni SliceLens sono "
           "confermate sul 90% delle loro istanze contro un null del 6%; le frasi con "
           "attributo raddoppiano il lift col nuovo canale; quelle di sola classe "
           "risultano vuote."),
    "note": "Questa è la verifica che mancava, e la considero la più importante di oggi. "
            "Prendo ogni frase che il sistema consegna e chiedo al modello visione-linguaggio "
            "se è vera per quell'oggetto. Da sola la precisione non direbbe niente, perché un "
            "VLM dice di sì spesso; quindi la confronto con un controllo preso fuori dalla "
            "slice ma della stessa classe e dello stesso tipo d'errore. Le situazioni "
            "SliceLens passano sei volte su dieci contro due su dieci del controllo. Le frasi "
            "costruite sugli attributi guadagnano poco, e guardando frase per frase si scopre "
            "il colpevole: le etichette 'dark' e 'blurry' il modello non le riconferma quasi "
            "mai. È un difetto del canale di tagging, non della scoperta delle slice — e lo "
            "stesso difetto è emerso, in modo indipendente, nell'esperimento di corruzione. "
            "Un'ultima cosa che questa misura mi ha insegnato. Avevo reso le descrizioni "
            "specifiche per classe, per leggibilità: non 'un oggetto tagliato dal bordo' ma "
            "'una mucca tagliata dal bordo'. Ho rifatto la stessa misura sulle stesse istanze "
            "con la formulazione vecchia, e la precisione saliva da zero virgola sessanta a "
            "zero virgola settantotto. La classe viene dal ground truth, quindi non è mai "
            "sbagliata; ma infilata nella domanda sposta l'attenzione del modello da 'è "
            "tagliato dal bordo?' a 'è una mucca?'. Così ora la classe la scrivo fuori dalle "
            "virgolette: dentro le virgolette resta solo ciò che la pipeline afferma e ha "
            "verificato.",
}

CONCLUSIONS = {
    "kind": "content",
    "en": ("Conclusion — back to the five questions",
           [("Pixel by pixel, not just image by image? Yes — the typed "
             "taxonomy shows split/merge/boundary where a single IoU hides "
             "them", 0),
            ("No dataset of causes? Yes — certification always compares "
             "against chance, never a labelled cause", 0),
            ("Geometry and words pointing to the same thing? Yes — two "
             "independent branches, reconciled per image", 0),
            ("New situations, not a fixed list? Yes — the language branch "
             "proposes freely; 31/35 transfer untouched to a stronger model", 0),
            ("Judging with no answer key? Five chance-calibrated experiments "
             "— two of which found and FIXED real bugs, not just graded the "
             "pipeline", 0)]),
    "it": ("Conclusione — di nuovo le cinque domande",
           [("Pixel per pixel, non solo immagine per immagine? Sì — la "
             "tassonomia tipata mostra split/merge/bordo dove un solo IoU li "
             "nasconde", 0),
            ("Senza un dataset di cause? Sì — la certificazione confronta "
             "sempre col caso, mai con una causa etichettata", 0),
            ("Geometria e parole che indicano la stessa cosa? Sì — due rami "
             "indipendenti, riconciliati per immagine", 0),
            ("Situazioni nuove, non una lista fissa? Sì — il ramo linguistico "
             "propone liberamente; 31/35 trasferiscono senza ritocchi su un "
             "modello più forte", 0),
            ("Giudicare senza una risposta corretta? Cinque esperimenti "
             "calibrati al caso — due hanno trovato e RIPARATO bug reali, non "
             "solo votato la pipeline", 0)]),
    "note": "In conclusione, torno alle cinque domande da cui sono partito. Pixel per "
            "pixel: risolto — la tassonomia tipata a livello di istanza rende osservabili "
            "split, merge e bordo, che un singolo IoU rende geometricamente invisibili; "
            "la tassonomia finale ha 6406 errori tipati su 1449 immagini. Senza un "
            "dataset di cause: risolto per costruzione, perché ogni certificazione è un "
            "test contro il caso, mai contro un'etichetta che non esiste. Geometria e "
            "parole che coincidono: è il cuore della pipeline stessa, i due rami che "
            "restano indipendenti fino alla validazione e si ricongiungono solo nella "
            "frase finale. Situazioni nuove: il ramo linguistico non ha un vocabolario "
            "fisso, propone ipotesi ed è trasferito su un secondo modello molto più "
            "forte — trentuno delle trentacinque ipotesi si ri-certificano senza "
            "toccare nulla. E l'ultima, la più difficile: come giudico senza una "
            "risposta corretta? Cinque esperimenti indipendenti, tutti calibrati al "
            "caso, e due di loro non si sono limitati a dare un voto alla pipeline — "
            "hanno trovato difetti REALI e li hanno fatti correggere: un canale di "
            "tagging che assegnava attributi in modo inaffidabile, e una metrica di "
            "copertura che contava solo un ramo su due. È il risultato di cui vado più "
            "fiero: la valutazione non ha solo giudicato il sistema, lo ha riparato. I "
            "prossimi passi restano la validazione esterna su dati con modifiche note, "
            "appena arriva l'accesso, e chiudere il ciclo con un tentativo di riparare "
            "il modello stesso a partire dalla diagnosi.",
}

ITEMS = [GOAL, RQ, EXAMPLE_COW, FIGURE, METHODS, PRINCIPLES, BLOCK1, CLUSTER, LANGUAGE,
         SQV_AXES, SQV_GATE, GATE_TABLE, EXPL, RESULTS,
         CATALOG, EXAMPLE_BUS, VALIDATION_THEORY, VALIDATION, PRECISION, LIMITS,
         CONCLUSIONS]


def build_deck(lang, out_path):
    prs = Presentation(TPL)

    # drop existing slides (relationship + part) and set the master footer.
    xml_slides = prs.slides._sldIdLst
    for sld in list(xml_slides):
        prs.part.drop_rel(sld.get(qn("r:id")))
        xml_slides.remove(sld)
    for sh in prs.slide_masters[0].shapes:
        if sh.has_text_frame and "Nome struttura" in sh.text_frame.text:
            p = sh.text_frame.paragraphs[0]
            if p.runs:
                p.runs[0].text = "Daniele Spini · Politecnico di Milano"
                for r in p.runs[1:]:
                    r._r.getparent().remove(r._r)

    L_COVER, L_CONTENT = prs.slide_layouts[0], prs.slide_layouts[3]

    # cover
    title, name, sub = COVER[lang]
    s = prs.slides.add_slide(L_COVER)
    tph = ph_by_idx(s, 1)
    tph.text = title
    if lang == "it":                       # longer IT title: cap size so it fits the box
        for p in tph.text_frame.paragraphs:
            for r in p.runs:
                r.font.size = Pt(30)
    ph_by_idx(s, 10).text = name
    ph_by_idx(s, 11).text = sub
    notes(s, COVER["note"])

    def _blank_with_title(title):
        """Picture layout with its body placeholder removed (we position by hand)."""
        s = prs.slides.add_slide(prs.slide_layouts[1])
        ph_by_idx(s, 0).text = title
        cp = ph_by_idx(s, 1)
        if cp is not None:
            cp._element.getparent().remove(cp._element)
        return s

    def _fit_picture(s, img, top, max_h, max_w=Inches(12.4)):
        iw, ih = PILImage.open(img).size
        ar = iw / ih
        h, w = max_h, int(max_h * ar)
        if w > max_w:
            w, h = max_w, int(max_w / ar)
        left = int((prs.slide_width - w) / 2)
        s.shapes.add_picture(img, left, top, width=w, height=h)
        return top + h

    def _textbox(s, top, height, text, size, color=(0x33, 0x33, 0x33),
                 align=PP_ALIGN.LEFT, mono=False, left=Inches(0.6),
                 width=Inches(12.13)):
        box = s.shapes.add_textbox(left, top, width, height)
        tf = box.text_frame
        tf.word_wrap = True
        lines = text if isinstance(text, list) else [text]
        for i, ln in enumerate(lines):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = ln
            p.alignment = align
            for r in p.runs:
                r.font.size = Pt(size)
                r.font.color.rgb = RGBColor(*color)
                if mono:
                    r.font.name = "Consolas"
        return box

    def _table(s, left, top, width, header, rows, col_widths, font_size=9):
        tbl_shape = s.shapes.add_table(len(rows) + 1, len(header), left, top,
                                       width, Inches(0.05))
        tbl = tbl_shape.table
        for j, w in enumerate(col_widths):
            tbl.columns[j].width = w
        for j, htext in enumerate(header):
            cell = tbl.cell(0, j)
            cell.text = str(htext)
            cell.margin_top = cell.margin_bottom = Emu(18000)
            cell.margin_left = cell.margin_right = Emu(45000)
            for p in cell.text_frame.paragraphs:
                for r in p.runs:
                    r.font.bold = True
                    r.font.size = Pt(font_size + 0.5)
        for i, row in enumerate(rows, start=1):
            for j, val in enumerate(row):
                cell = tbl.cell(i, j)
                cell.text = str(val)
                cell.margin_top = cell.margin_bottom = Emu(14000)
                cell.margin_left = cell.margin_right = Emu(45000)
                for p in cell.text_frame.paragraphs:
                    for r in p.runs:
                        r.font.size = Pt(font_size)
        return tbl_shape

    for it in ITEMS:
        if it["kind"] == "content":
            title, bullets = it[lang]
            s = prs.slides.add_slide(L_CONTENT)
            ph_by_idx(s, 0).text = title
            set_bullets(ph_by_idx(s, 1), bullets)
            notes(s, it["note"])
        elif it["kind"] == "example":
            title, lines, takeaway = it[lang]
            s = _blank_with_title(title)
            bottom = _fit_picture(s, it["image"], Inches(1.15), Inches(3.5))
            _textbox(s, bottom + Inches(0.18), Inches(1.15), lines, 11.5,
                     color=(0x22, 0x33, 0x44), mono=True)
            _textbox(s, bottom + Inches(1.38), Inches(0.85), takeaway, 13,
                     color=(0x1F, 0x5F, 0x9E))
            notes(s, it["note"])
        elif it["kind"] == "picture":
            title, caption = it[lang]
            s = _blank_with_title(title)
            bottom = _fit_picture(s, it["image"][lang], Inches(1.3), Inches(4.4))
            _textbox(s, bottom + Inches(0.25), Inches(1.1), caption, 12.5,
                     color=(0x33, 0x33, 0x33))
            notes(s, it["note"])
        elif it["kind"] == "table1":
            title, header = it[lang]
            s = _blank_with_title(title)
            q_idx = 2 if lang == "en" else 3
            rows = [(r[0], r[1], r[q_idx]) for r in it["table"]]
            _table(s, Inches(0.5), Inches(1.25), Inches(12.3), header, rows,
                   [Inches(2.2), Inches(1.7), Inches(8.4)], font_size=11)
            notes(s, it["note"])
        elif it["kind"] == "gate_table":
            title, intro, header, footer = it[lang]
            s = _blank_with_title(title)
            _textbox(s, Inches(1.1), Inches(0.55), intro, 12.5,
                     color=(0x33, 0x33, 0x33))
            c1, c2 = (1, 2) if lang == "en" else (3, 4)
            rows = [(r[0], r[c1], r[c2]) for r in it["rows"]]
            _table(s, Inches(0.5), Inches(1.75), Inches(12.3), header, rows,
                   [Inches(1.9), Inches(5.0), Inches(5.4)], font_size=10)
            _textbox(s, Inches(4.65), Inches(1.3), footer, 12,
                     color=(0x1F, 0x5F, 0x9E))
            notes(s, it["note"])
        elif it["kind"] == "catalog":
            title, left_hd, right_hd = it[lang]
            s = _blank_with_title(title)
            top = Inches(1.15)
            _textbox(s, top, Inches(0.3), left_hd, 12.5, color=(0x1F, 0x5F, 0x9E),
                     left=Inches(0.5), width=Inches(5.9))
            _textbox(s, top, Inches(0.3), right_hd, 12.5, color=(0x1F, 0x5F, 0x9E),
                     left=Inches(6.7), width=Inches(6.1))
            tbl_top = top + Inches(0.42)
            _table(s, Inches(0.5), tbl_top, Inches(5.9),
                   ["class", "type", "n"], CATALOG_CLUSTER,
                   [Inches(3.6), Inches(0.9), Inches(1.4)], font_size=10)
            _table(s, Inches(6.7), tbl_top, Inches(6.1),
                   ["situation (SliceLens)", "n", "classes"], CATALOG_LANGUAGE,
                   [Inches(2.5), Inches(0.4), Inches(3.2)], font_size=9)
            notes(s, it["note"])
        else:  # figure
            title, caption = it[lang]
            s = prs.slides.add_slide(prs.slide_layouts[1])
            ph_by_idx(s, 0).text = title
            cp = ph_by_idx(s, 1)
            if cp is not None:
                cp._element.getparent().remove(cp._element)
            img = FIG[lang]
            iw, ih = PILImage.open(img).size
            ar = iw / ih
            h = Inches(5.0)
            w = int(h * ar)
            if w > Inches(12.7):
                w = Inches(12.7)
                h = int(w / ar)
            left = int((prs.slide_width - w) / 2)
            s.shapes.add_picture(img, left, Inches(1.2), width=w, height=h)
            cap = s.shapes.add_textbox(Inches(0.5), Inches(6.35), Inches(12.33), Inches(0.55))
            tf = cap.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.text = caption
            p.alignment = PP_ALIGN.CENTER
            for r in p.runs:
                r.font.size = Pt(11)
                r.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
            notes(s, it["note"])

    prs.save(out_path)
    print("saved", out_path, "-", len(prs.slides._sldIdLst), "slides")


if __name__ == "__main__":
    build_deck("en", os.path.join(DOCS, "pipeline_presentation.pptx"))
    build_deck("it", os.path.join(DOCS, "pipeline_presentation_IT.pptx"))
