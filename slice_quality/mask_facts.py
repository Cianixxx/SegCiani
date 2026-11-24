"""Fatti geometrici sulle maschere, che il Blocco 1 non registra.

Il Blocco 1 tipizza l'errore (fn / fp / boundary / extent / confusion / split /
merge) e ne da' la magnitudine con [E_bnd, E_ext, E_seg]. Non dice pero' come e'
FATTO l'errore, ed e' quello che gli annotatori scrivono. Qui si misura il resto,
tutto in modo deterministico dalle maschere:

    frame_frac      quota del fotogramma occupata -> "tiny / small / large"
    missed/excess   il SEGNO di un errore di estensione (sotto- o sovra-),
                    che il vettore di Beyond IoU non porta
    boundary_iou    quanto bene e' tracciato il contorno, indipendente dalla
                    scala al contrario dell'IoU d'area
    n_thin_lost     i DETTAGLI persi: ali, zampe, antenne
    n_gt_covered    quante regioni GT ha inglobato una predizione (merge)
    union_iou       per un MERGE, IoU / E_bnd / E_ext del blob contro l'UNIONE
                    delle GT che ha fuso: e' l'unico riferimento sensato per una
                    maschera a cavallo di due oggetti (contro la singola GT piu'
                    simile un blob su due monitor affiancati ha IoU 0.44 e nessun
                    contorno da raccontare; contro l'unione ha IoU 0.80 e un
                    contorno impreciso, che e' quello che scrive l'annotatore)
    excess_sides    su quali lati della GT si concentrano i pixel in eccesso
    missed_sides    e quelli mancanti -- "over-extended on the right" e' cio' che
                    scrivono gli annotatori; Beyond IoU chiama "boundary" ogni
                    pixel entro 2d dal vero positivo e cosi' nasconde la
                    direzione, che e' la cosa che loro nominano per prima
    gt_px           area della GT accoppiata, per dire "8% dell'area in eccesso"
    adjacent_gt     per un FP della stessa classe che tocca (entro ADJ_PX) una
                    regione GT: non e' un'allucinazione, e' l'oggetto che sborda
    n_holes         buchi interni alla maschera predetta (componenti di sfondo
                    racchiuse), con hole_px / hole_frac -- gli annotatori li
                    scrivono ("holes scattered within the region")
    covered_frac    per un FN, quanta parte e' in realta' coperta da una
                    predizione della sua classe -- vedi sotto
    occluded_by     per una GT-oggetto tagliata in piu' parti (SQ_GT_INSTANCES=
                    object, ``gt_parts`` > 1): la classe dell'oggetto di un'altra
                    classe che ne tocca almeno due parti -- "partly hidden
                    behind the CHAIR"; se nessuno lo fa, l'oggetto e' tagliato
                    dalla sua stessa struttura sottile e non si dice nulla
    relabelled_as   per l'istanza che parla di un oggetto GT (la coppia boundary /
                    extent / correct, o l'FN lato GT): l'ALTRA classe con cui e'
                    predetta la parte piu' grande dell'oggetto e la sua quota
                    dell'area -- "the DININGTABLE region is often misclassified as
                    CHAIR"; senza, la narrativa diceva solo "36% missing"
    over_gt         per un FP: la classe dell'oggetto GT su cui la toppa giace
                    (almeno meta' della toppa) -- "parts of the CHAIR labelled
                    SOFA" invece di "SOFA patches hallucinated"
    (v6) relabelled_as e' una LISTA di al piu' due [classe, quota]: "the SHEEP region
                    is totally misclassified with classes DOG and COW"; misurata
                    anche sugli oggetti fusi in un merge e sulle confusioni (la
                    classe della confusione esclusa: la frase la dice gia')
    boundary_where  la cella (griglia 3x3 dell'oggetto) dove si concentra il
                    contorno sbagliato, se ne tiene almeno BND_WHERE_MIN --
                    "coarse boundaries, especially towards the bottom-right"
    (v6) excess_sides / missed_sides nominano anche gli ANGOLI ("top-left"):
                    "incomplete in the top-left corner"
    (per immagine) small_patches  le toppe predette sotto la soglia dell'adapter ma
                    sopra SMALL_PATCH_PX, con classe, cosa c'e' sotto e bbox --
                    "a small PERSON region hallucinated on the right"
    (per immagine, negli ``extras`` restituiti)
    subthreshold        regioni GT sotto i 200 px dell'adapter (contate a 15 px)
    subthreshold_pred   toppe PREDETTE sotto i 200 px di una classe che la GT
                        a soglia bassa non ha li' (la banda void di VOC ignorata)
    verdict             quota dell'area annotata etichettata con una classe
                        sbagliata o come sfondo -> la frase d'apertura quando
                        il fallimento e' globale

Sul ``covered_frac``. Il Blocco 1 decide MERGE con un test di IoU fra la
predizione e ogni regione GT (``tau_frag`` = 0.25). Ma l'IoU e' cieca al
contenimento asimmetrico: una barca piccola inglobata al 68% in un blob che
copre anche la barca grande ha IoU 0.14 con quel blob, il merge non scatta, e
la barca piccola cade in FN -- "missed entirely", per una regione coperta per
due terzi. Su pascal_80 succede al 54% dei FN (41% coperti oltre l'80%); solo il
30% e' davvero mancato. La copertura, che non dipende dall'area del blob, lo
vede; l'IoU no. Qui la si misura e la narrativa la usa per dire "inglobata"
invece di "mancata". Questo fatto e' quello che ha portato a correggere il test
di claim nel Blocco 1 (``config.TypedErrorConfig.claim_cov``, env SQ_CLAIM_COV):
con il claim per contenimento la GT inglobata non e' piu' un FN ma un'istanza
MERGE lato GT (``swallowed_by``, ``covered_frac`` scritti dal Blocco 1), e il
blob viene misurato qui contro l'UNIONE delle GT che ha fuso (``gt_ids`` ->
``union_iou``). Senza ``claim_cov`` (run di riferimento) vale ancora il
comportamento sopra: ``covered_frac`` sul FN, tassonomia invariata.

Vive in un modulo e non dentro uno script perche' serve a piu' runner: il val +
train36 (narrate_errors.py) e il set sintetico (narrate_synth.py). Nessuna I/O
qui dentro: i campioni li costruisce il chiamante, che cosi' controlla dataset,
sottocartella delle predizioni e soglie.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from .config import SliceQualityConfig
from .typed_error import boundary_extent

_TE = SliceQualityConfig().typed_error
CROP = 520
# stessa banda con cui il Blocco 1 separa i pixel di bordo da quelli di estensione:
# d e' l'1% della diagonale e il test e' a 2*d, cosi' la misura del contorno sta
# sulla stessa scala spaziale della decomposizione E_bnd / E_ext che accompagna
BND_BAND = 2 * _TE.boundary_d_frac * math.sqrt(2) * CROP
TAU_FRAG = _TE.tau_match * _TE.frag_frac
MIN_PIXELS = 200          # la stessa soglia con cui build_voc_samples filtra

_DISK = ndimage.generate_binary_structure(2, 1)
HOLE_MIN_PX = 20        # sotto, e' rumore di bordo, non un buco
ADJ_PX = 8              # un FP a meno di tanto da una GT della sua classe le e' adiacente
SIDE_MIN = 0.30         # un lato si nomina se tiene almeno questa quota dei pixel sbagliati
THIN_R = 2                # raggio dell'apertura: cio' che sparisce e' sottile
THIN_NOISE = 15           # sotto quest'area e' rumore di bordo, non un dettaglio
# una parte dell'oggetto predetta con un'altra classe si dice da questa quota dell'area
# e da questo numero di pixel: le stesse soglie con cui la narrativa nomina un
# difetto di estensione (EXTENT_MIN) e con cui l'adapter tiene una regione
RELABEL_MIN = 0.03
RELABEL_PX = MIN_PIXELS
OVER_MIN = 0.5            # una toppa FP "sta su" un oggetto se questo ne tiene meta'
SECTOR_CUT = 1 / 3        # griglia 3x3 della bbox GT: oltre questo scarto normalizzato si e' su un lato
CORNER_MIN = 0.5          # un angolo si nomina se da solo tiene meta' dell'errore fuori dal centro
BND_WHERE_MIN = 0.4       # una cella si nomina per il contorno se ne tiene almeno questa quota
SMALL_PATCH_PX = 50       # una toppa sotto soglia si dice da qui: sotto e' un granello


def boundary_iou(pred, gt, d=BND_BAND):
    """IoU ristretta alla fascia di contorno di entrambe le maschere (Cheng 2021).

    L'IoU d'area e' scala-dipendente: un contorno sbagliato la muove di pochi
    centesimi su un oggetto grande, ed e' il motivo per cui immagini con bordi
    visibilmente rotti stanno sopra tau_iou_good senza produrre alcun errore.
    Togliendo l'area dal conto, un oggetto grande e uno piccolo sono giudicati
    con lo stesso metro.
    """
    pb = pred & (ndimage.distance_transform_edt(pred) <= d)
    gb = gt & (ndimage.distance_transform_edt(gt) <= d)
    u = (pb | gb).sum()
    return float((pb & gb).sum()) / u if u else 0.0


def thin_lost(pred, gt):
    """Le parti SOTTILI della ground truth che la predizione non copre.

    Un frammento mancato e' un DETTAGLIO se un'apertura morfologica lo cancella:
    e' cio' che distingue un'antenna, una zampa o un'ala da una macchia mancante,
    gia' coperta dal termine di estensione. Restituisce (frammenti, pixel).
    """
    missed = gt & ~pred
    if not missed.any():
        return 0, 0
    thin = missed & ~ndimage.binary_opening(missed, structure=_DISK,
                                            iterations=THIN_R)
    lab, k = ndimage.label(thin)
    n = px = 0
    for j in range(1, k + 1):
        a = int((lab == j).sum())
        if a >= THIN_NOISE:
            n += 1
            px += a
    return n, px


def label_map(items, shape):
    """Per-pixel class name ('' where nothing) from a list of {"mask", "cls"}."""
    lab = np.full(shape, "", dtype=object)
    for x in items:
        lab[np.asarray(x["mask"], dtype=bool)] = x["cls"]
    return lab


def relabelled_as(gm, gcls, pr_lab, exclude=()):
    """The other classes predicted on GT object ``gm``, with their share of the object's
    area: up to two [cls, frac], largest first, each above RELABEL_MIN and RELABEL_PX
    (None if none is). Background is not another class: an object left as background is
    an extent error, already told by its direction and sides. ``exclude``: classes the
    caller's sentence already names (the label of a confusion)."""
    under = pr_lab[gm]
    other = under[(under != "") & (under != gcls)]
    if not other.size:
        return None
    vals, cnt = np.unique(other.astype(str), return_counts=True)
    out = []
    for k in np.argsort(-cnt):
        frac = float(cnt[k]) / float(gm.sum())
        if str(vals[k]) in exclude or int(cnt[k]) < RELABEL_PX or frac < RELABEL_MIN:
            continue
        out.append([str(vals[k]), round(frac, 4)])
    return out[:2] or None


def over_gt(m, pcls, gt_lab):
    """The class of the GT object (another class) under predicted patch ``m`` when it
    holds at least OVER_MIN of the patch: [cls, frac], else None."""
    under = gt_lab[m]
    other = under[(under != "") & (under != pcls)]
    if not other.size:
        return None
    vals, cnt = np.unique(other.astype(str), return_counts=True)
    k = int(cnt.argmax())
    frac = float(cnt[k]) / float(m.sum())
    return [str(vals[k]), round(frac, 4)] if frac >= OVER_MIN else None


def subthreshold_gt(s_fine, min_px=MIN_PIXELS):
    """Componenti GT sotto min_pixels e senza corrispettivo nella predizione.

    Non sono errori "mancati": non sono MAI diventate istanze, perche'
    build_voc_samples le scarta prima che il Blocco 1 giri.

    ATTENZIONE: ``s_fine`` DEVE venire da una costruzione a soglia bassa. Con i
    campioni normali (min_pixels=200) queste componenti non ci sono, perche'
    l'adapter le ha gia' buttate, e il conteggio esce sempre zero -- uno zero che
    non insospettisce come farebbe un numero sbagliato.
    """
    n = px = 0
    by_cls = {}
    for p in s_fine["preds"]:
        m = np.asarray(p["mask"], dtype=bool)
        by_cls[p["cls"]] = by_cls.get(p["cls"], np.zeros_like(m)) | m
    for g in s_fine["gts"]:
        gm = np.asarray(g["mask"], dtype=bool)
        a = int(gm.sum())
        if a >= min_px or a < THIN_NOISE:
            continue
        pred_c = by_cls.get(g["cls"])
        cov = (gm & pred_c).sum() / float(a) if pred_c is not None else 0.0
        if cov < 0.5:
            n += 1
            px += a
    return n, px


def _sector_counts(err, gt):
    """Pixels of ``err`` per cell of a 3x3 grid on the bounding box of ``gt``:
    "top-left" ... "bottom-right" for the corners, "top" / "left" / ... for the sides,
    "center" for the middle cell. Returns (n, {cell: count})."""
    n = int(err.sum())
    if n == 0 or not gt.any():
        return n, {}
    ys, xs = np.where(gt)
    cy, cx = (ys.min() + ys.max()) / 2.0, (xs.min() + xs.max()) / 2.0
    hh, hw = max((ys.max() - ys.min()) / 2.0, 1.0), max((xs.max() - xs.min()) / 2.0, 1.0)
    ey, ex = np.where(err)
    dy, dx = (ey - cy) / hh, (ex - cx) / hw
    counts = {}
    for vy, hx in zip(np.where(dy < -SECTOR_CUT, 0, np.where(dy > SECTOR_CUT, 2, 1)),
                      np.where(dx < -SECTOR_CUT, 0, np.where(dx > SECTOR_CUT, 2, 1))):
        counts[(int(vy), int(hx))] = counts.get((int(vy), int(hx)), 0) + 1
    v_name, h_name = ("top", "", "bottom"), ("left", "", "right")
    out = {}
    for (vy, hx), c in counts.items():
        k = "-".join(x for x in (v_name[vy], h_name[hx]) if x) or "center"
        out[k] = c
    return n, out


def error_sides(err, gt):
    """Where on ``gt`` the pixels of ``err`` lie, most first.

    SIDES as until v5: each error pixel goes to the side of the GT's bounding box it
    points to from the box centre (top/bottom when the vertical offset, normalised by the
    box half-height, dominates; left/right otherwise), and sides holding at least
    SIDE_MIN of the pixels are returned, at most two. A CORNER ("top-left") replaces
    them when the corner cell of a 3x3 grid on the box holds CORNER_MIN of the error
    outside the middle cell: "incomplete in the top-left corner", which the sides alone
    read as "on the top and left".
    """
    n = int(err.sum())
    if n == 0 or not gt.any():
        return []
    _n, cells = _sector_counts(err, gt)
    n_eff = n - cells.get("center", 0)
    corners = {k: v for k, v in cells.items() if "-" in k}
    if n_eff > 0 and corners:
        k, v = max(corners.items(), key=lambda kv: kv[1])
        if v >= CORNER_MIN * n_eff:
            return [k]
    ys, xs = np.where(gt)
    cy, cx = (ys.min() + ys.max()) / 2.0, (xs.min() + xs.max()) / 2.0
    hh, hw = max((ys.max() - ys.min()) / 2.0, 1.0), max((xs.max() - xs.min()) / 2.0, 1.0)
    ey, ex = np.where(err)
    dy, dx = (ey - cy) / hh, (ex - cx) / hw
    vertical = np.abs(dy) >= np.abs(dx)
    counts = {"top": int((vertical & (dy < 0)).sum()), "bottom": int((vertical & (dy >= 0)).sum()),
              "left": int((~vertical & (dx < 0)).sum()), "right": int((~vertical & (dx >= 0)).sum())}
    return [k for k, v in sorted(counts.items(), key=lambda kv: -kv[1]) if v >= SIDE_MIN * n][:2]


def boundary_where(pred, gt, d=BND_BAND):
    """The cell of the 3x3 grid where the outline is wrong: the pixels on which
    prediction and ground truth disagree within ``d`` of the ground truth's edge.
    Named only when one cell holds BND_WHERE_MIN of them."""
    if not gt.any():
        return None
    edge = gt ^ ndimage.binary_erosion(gt)
    near = ndimage.distance_transform_edt(~edge) <= d
    n, counts = _sector_counts((pred ^ gt) & near, gt)
    if not n or not counts:
        return None
    k, v = max(counts.items(), key=lambda kv: kv[1])
    return k if k != "center" and v >= BND_WHERE_MIN * n else None


def subthreshold_pred(s_fine, s_full, min_px=MIN_PIXELS):
    """Predicted patches under min_pixels labelled with a class the ground truth
    does not have there: the "small area misclassified as DININGTABLE" a reader
    sees and the instrument silently drops. Returns [(pred_cls, gt_cls_under)].

    ``s_fine`` must come from a low-threshold build; ``s_full`` supplies the GT
    to look under (any threshold, the GT masks are the same).
    """
    # look under the patch with the LOW-threshold GT: a tiny predicted patch
    # that sits on an equally tiny GT region of its class is right, and only
    # s_fine still has that region (build_voc_samples dropped it at min_pixels)
    gts = [(np.asarray(g["mask"], dtype=bool), g["cls"]) for g in s_fine["gts"]]
    void = s_full.get("void", s_fine.get("void"))
    out = []
    for p in s_fine["preds"]:
        m = np.asarray(p["mask"], dtype=bool)
        if int(m.sum()) >= min_px:
            continue
        if void is not None:
            m = m & ~np.asarray(void, dtype=bool)   # slivers on the void band are not errors
        a = int(m.sum())
        if a < THIN_NOISE:
            continue
        under, best = "background", 0
        for gm, c in gts:
            v = int((gm & m).sum())
            if v > best:
                under, best = c, v
        if best < 0.5 * a:
            under = "background"
        if under != p["cls"]:
            out.append((p["cls"], under))
    return out


def small_patches(s_fine, s_full, min_px=MIN_PIXELS):
    """Predicted patches under the adapter's min_pixels but over SMALL_PATCH_PX, with a
    class the ground truth does not have there: class, what lies under it, its box and
    size. The same patches as ``subthreshold_pred``, with the place a reader needs to
    write "a small PERSON region hallucinated on the right"."""
    if not s_fine["preds"]:
        return []
    shape = np.asarray(s_fine["preds"][0]["mask"]).shape
    gt_lab = label_map(s_fine["gts"], shape)
    void = s_full.get("void", s_fine.get("void"))
    out = []
    for p in s_fine["preds"]:
        m = np.asarray(p["mask"], dtype=bool)
        if int(m.sum()) >= min_px:
            continue
        if void is not None:
            m = m & ~np.asarray(void, dtype=bool)
        a = int(m.sum())
        if a < SMALL_PATCH_PX:
            continue
        vals, cnt = np.unique(gt_lab[m].astype(str), return_counts=True)
        k = int(cnt.argmax())
        u = str(vals[k]) if vals[k] and cnt[k] >= 0.5 * a else "background"
        if u == p["cls"]:
            continue
        ys, xs = np.where(m)
        out.append({"cls": p["cls"], "under": u, "px": a,
                    "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]})
    return out


def image_verdict(s):
    """How wrong the prediction is as a whole: share of the annotated (GT
    foreground) area labelled with a wrong class, or left as background, and
    the wrong class that takes most of it. What annotators put first when the
    failure is global ("the model predicts SHEEP on most of the image")."""
    if not s["gts"]:
        return None
    H, W = np.asarray(s["gts"][0]["mask"]).shape
    gt_lab = np.zeros((H, W), dtype=object)
    fg = np.zeros((H, W), dtype=bool)
    for g in s["gts"]:
        gm = np.asarray(g["mask"], dtype=bool)
        gt_lab[gm] = g["cls"]
        fg |= gm
    pr_lab = np.zeros((H, W), dtype=object)
    covered = np.zeros((H, W), dtype=bool)
    for p in s["preds"]:
        pm = np.asarray(p["mask"], dtype=bool)
        pr_lab[pm] = p["cls"]
        covered |= pm
    n_fg = int(fg.sum())
    if n_fg == 0:
        return None
    wrong = fg & covered & (gt_lab != pr_lab)
    bg = fg & ~covered
    wrong_cls = {}
    for c in {p["cls"] for p in s["preds"]}:
        v = int((wrong & (pr_lab == c)).sum())
        if v:
            wrong_cls[c] = v
    top = max(wrong_cls.items(), key=lambda kv: kv[1]) if wrong_cls else (None, 0)
    return {"wrong_frac": round(int(wrong.sum()) / n_fg, 4), "bg_frac": round(int(bg.sum()) / n_fg, 4),
            "wrong_cls": top[0], "wrong_cls_frac": round(top[1] / n_fg, 4),
            "wrong_cls_in_gt": bool(top[0] and any(g["cls"] == top[0] for g in s["gts"]))}


def occluder_of(s, j, dilate=3):
    """Class of the object in front of GT ``j`` when ``j`` is cut into parts.

    The occluder is the other-class GT object that touches at least two parts
    of ``j`` (each part dilated by ``dilate`` px); if none does, the object is
    cut by its own thin structure (spokes, legs) or by the void band, and the
    narrative keeps quiet. Needs the ``obj_id`` the object adapter records.
    """
    g = s["gts"][j]
    m = np.asarray(g["mask"], dtype=bool)
    lab, n = ndimage.label(m)
    if n < 2:
        return None
    hits = {}
    for k in range(1, n + 1):
        ring = ndimage.binary_dilation(lab == k, iterations=dilate) & ~m
        for jj, o in enumerate(s["gts"]):
            if jj == j or o.get("obj_id") is None:
                continue
            if (ring & np.asarray(o["mask"], dtype=bool)).any():
                hits.setdefault(jj, set()).add(k)
    best = [jj for jj, parts in hits.items() if len(parts) >= 2]
    if not best:
        return None
    jj = max(best, key=lambda x: len(hits[x]))
    return s["gts"][jj]["cls"]


def gt_covered(s):
    """{indice pred -> quante regioni GT ha inglobato}, con la regola del Blocco 1.

    Riproduce ``pred_claims`` invece di indovinare: una regione GT conta come
    inglobata dalla predizione che la sovrappone di piu', e solo se quella
    sovrapposizione raggiunge tau_match * frag_frac.
    """
    preds, gts = s["preds"], s["gts"]
    if not preds or not gts:
        return {}
    pm = [np.asarray(p["mask"], dtype=bool) for p in preds]
    out = {}
    for g in gts:
        gm = np.asarray(g["mask"], dtype=bool)
        best_i, best_iou = -1, 0.0
        for i, m in enumerate(pm):
            inter = float((m & gm).sum())
            if not inter:
                continue
            v = inter / float((m | gm).sum())
            if v > best_iou:
                best_i, best_iou = i, v
        if best_i >= 0 and best_iou >= TAU_FRAG:
            out[best_i] = out.get(best_i, 0) + 1
    return out


def add_mask_facts(errors, samples, fine_samples=(), reference=()):
    """Misura tutti i fatti di cui sopra, in place su ``metadata``.

    Ritorna (istanze toccate, {image_id -> extras per immagine}).
    """
    by_img = {s["image_id"]: s for s in samples}
    extras = {}
    by_img_full = {s["image_id"]: s for s in samples}
    by_img_fine = {s["image_id"]: s for s in fine_samples}
    maps = {}

    def label_maps(s):
        """(prediction, ground truth) class maps of an image; the prediction from the
        low-threshold build when there is one, so that small bits count too."""
        if s["image_id"] not in maps:
            shape = np.asarray((s["gts"] or s["preds"])[0]["mask"]).shape
            src = by_img_fine.get(s["image_id"], s)
            maps[s["image_id"]] = (label_map(src["preds"], shape), label_map(s["gts"], shape))
        return maps[s["image_id"]]
    for s in fine_samples:                # per immagine, non per istanza
        ex = {}
        n, px = subthreshold_gt(s)
        if n:
            ex.update(subthreshold=n, subthreshold_px=px)
        if s["image_id"] in by_img_full:
            sp = subthreshold_pred(s, by_img_full[s["image_id"]])
            if sp:
                ex["subthreshold_pred"] = sp
            small = small_patches(s, by_img_full[s["image_id"]])
            if small:
                ex["small_patches"] = small
        if ex:
            extras[s["image_id"]] = ex
    for s in samples:
        v = image_verdict(s)
        if v:
            extras.setdefault(s["image_id"], {})["verdict"] = v

    # le GT che ogni blob ha fuso, ricostruite dalle istanze lato GT quando il
    # Blocco 1 non ha ancora scritto gt_ids (pickle ritipati prima di quel campo)
    swallowed_of = {}
    for inst in errors:
        if inst.seg_subtype.value == "merge" and inst.metadata.get("swallowed_by"):
            swallowed_of.setdefault(inst.metadata["swallowed_by"], []).append(inst.instance_id)

    covered, done = {}, 0
    for inst in list(errors) + list(reference):
        s = by_img.get(inst.image_id)
        if s is None:
            continue
        from_gt = bool(inst.metadata.get("from_gt_mask"))
        pool = s["gts"] if from_gt else s["preds"]
        idx = int(inst.instance_id.rsplit("#", 1)[1][1:])
        if idx >= len(pool):
            continue
        m = np.asarray(pool[idx]["mask"], dtype=bool)
        inst.metadata["frame_frac"] = round(float(m.sum()) / m.size, 5)
        done += 1

        # a GT object cut into parts (object adapter): name what hides it, on
        # the GT-side instance and on the prediction paired with it
        gj = idx if from_gt else (int(inst.metadata["gt_id"].rsplit("#g", 1)[1])
                                  if inst.metadata.get("gt_id") else None)
        if (gj is not None and gj < len(s["gts"]) and s["gts"][gj].get("n_parts", 1) > 1
                and "occluded_by" not in inst.metadata):
            occ = occluder_of(s, gj)
            inst.metadata["gt_parts"] = int(s["gts"][gj]["n_parts"])
            if occ:
                inst.metadata["occluded_by"] = occ

        # which OTHER class took part of the object -- the "table labelled as
        # chair" the annotators write -- on the instance that speaks for the
        # object; and, for a hallucinated patch, which object it lies on.
        # Recomputed every time (cheap), so a changed threshold never leaves a
        # stale value in a pickle measured before.
        inst.metadata.pop("relabelled_as", None)
        inst.metadata.pop("relabel_obj", None)
        inst.metadata.pop("over_gt", None)
        gm = gcls = None
        excl = ()
        sub = inst.seg_subtype.value
        if from_gt and sub in ("fn", "merge"):            # the object itself (missed, or swallowed)
            gm, gcls = m, inst.gt_class
        else:
            ref = inst.metadata.get("gt_id") or (inst.metadata.get("primary_gt") if sub == "merge" else None)
            if not from_gt and ref and sub in ("boundary", "extent", "correct", "confusion", "merge"):
                gj2 = int(ref.rsplit("#g", 1)[1])
                if gj2 < len(s["gts"]):
                    gm, gcls = np.asarray(s["gts"][gj2]["mask"], dtype=bool), s["gts"][gj2]["cls"]
                    if sub == "confusion":
                        excl = (inst.pred_class,)
        if gm is not None and gm.any():
            rl = relabelled_as(gm, gcls, label_maps(s)[0], exclude=excl)
            if rl:
                inst.metadata["relabelled_as"] = rl
                inst.metadata["relabel_obj"] = inst.instance_id if from_gt else ref
        if not from_gt and inst.seg_subtype.value == "fp" and m.any():
            og = over_gt(m, inst.pred_class, label_maps(s)[1])
            if og:
                inst.metadata["over_gt"] = og

        if not from_gt and "n_holes" not in inst.metadata:
            # gia' presenti sul val (add_geometry_metadata.py, stessa regola);
            # misurati qui per ogni altro runner, train36 e sintetico compresi
            gaps = ndimage.binary_fill_holes(m) & ~m
            px = int(gaps.sum())
            if px >= HOLE_MIN_PX:
                _lab, n_h = ndimage.label(gaps)
                inst.metadata.update(n_holes=int(n_h), hole_px=px,
                                     hole_frac=round(px / float(m.sum()), 4))

        if not from_gt and inst.seg_subtype.value == "fp":
            ring = ndimage.binary_dilation(m, iterations=ADJ_PX)
            for jj, g in enumerate(s["gts"]):
                if g["cls"] == inst.pred_class and (ring & np.asarray(g["mask"], dtype=bool)).any():
                    inst.metadata["adjacent_gt"] = f"{inst.image_id}#g{jj}"
                    break

        if from_gt and inst.seg_subtype.value == "fn":
            pc = np.zeros_like(m)
            for p in s["preds"]:
                if p["cls"] == inst.gt_class:
                    pc |= np.asarray(p["mask"], dtype=bool)
            inst.metadata["covered_frac"] = round(float((m & pc).sum()) / m.sum(), 4)

        if (inst.seg_subtype.value == "merge" and not from_gt
                and not inst.metadata.get("n_gt_covered")):
            # older pickles lack the count; Block 1 now records it (and, with
            # claim_cov, counts containment claims this IoU replica cannot see)
            if inst.image_id not in covered:
                covered[inst.image_id] = gt_covered(s)
            n_gt = covered[inst.image_id].get(idx)
            if n_gt and n_gt > 1:
                inst.metadata["n_gt_covered"] = int(n_gt)

        if from_gt or inst.seg_subtype.value not in ("boundary", "extent", "correct", "merge"):
            continue
        cls = inst.gt_class or inst.pred_class

        if inst.seg_subtype.value == "merge":
            # il blob contro l'unione delle GT che ha fuso (gt_ids dal Blocco 1;
            # per i pickle vecchi, le GT della sua classe che tocca)
            ids = inst.metadata.get("gt_ids")
            if not ids and inst.instance_id in swallowed_of:
                ids = list(swallowed_of[inst.instance_id])
                if inst.metadata.get("primary_gt"):
                    ids.append(inst.metadata["primary_gt"])
            if ids:
                picked = [s["gts"][int(i.rsplit("#g", 1)[1])] for i in ids
                          if int(i.rsplit("#g", 1)[1]) < len(s["gts"])]
            else:
                picked = [g for g in s["gts"]
                          if g["cls"] == cls and (np.asarray(g["mask"], dtype=bool) & m).any()]
            if not picked:
                continue
            gu = np.zeros_like(m)
            for g in picked:
                gu |= np.asarray(g["mask"], dtype=bool)
            e_bnd, e_ext, iou_u = boundary_extent(m, gu, BND_BAND / 2.0)
            inst.metadata.update(union_iou=round(float(iou_u), 4),
                                 union_e_bnd=round(float(e_bnd), 4),
                                 union_e_ext=round(float(e_ext), 4),
                                 boundary_iou=round(boundary_iou(m, gu), 4),
                                 boundary_where=boundary_where(m, gu),
                                 missed_px=int((gu & ~m).sum()),
                                 excess_px=int((m & ~gu).sum()),
                                 gt_px=int(gu.sum()),
                                 excess_sides=error_sides(m & ~gu, gu),
                                 missed_sides=error_sides(gu & ~m, gu))
            n_thin, px_thin = thin_lost(m, gu)
            if n_thin:
                inst.metadata["n_thin_lost"] = n_thin
                inst.metadata["thin_lost_px"] = px_thin
            continue

        # qualita' del contorno contro la componente GT che la predizione ha
        # davvero accoppiato -- calcolata anche sulle CORRECT, che e' il punto:
        # non producono errori, quindi questo e' l'unico numero che possa dire
        # al lettore che il loro contorno e' sbagliato
        best_g, best_v = None, 0.0
        for g in s["gts"]:
            if g["cls"] != cls:
                continue
            gm = np.asarray(g["mask"], dtype=bool)
            inter = (gm & m).sum()
            if not inter:
                continue
            v = inter / float((gm | m).sum())
            if v > best_v:
                best_g, best_v = gm, v
        if best_g is not None:
            inst.metadata["boundary_iou"] = round(boundary_iou(m, best_g), 4)
            n_thin, px_thin = thin_lost(m, best_g)
            if n_thin:
                inst.metadata["n_thin_lost"] = n_thin
                inst.metadata["thin_lost_px"] = px_thin

        # missed / excess and their sides, for the pairs AND the correct
        # regions: "very close, with some slight under-segmentation" is what
        # the annotators write about a correct region, and it needs the sign
        local = np.zeros_like(m)
        if inst.metadata.get("gt_id"):
            gj = int(inst.metadata["gt_id"].rsplit("#g", 1)[1])
            if gj < len(s["gts"]):
                local = np.asarray(s["gts"][gj]["mask"], dtype=bool)
        if not local.any():
            for g in s["gts"]:            # solo le parti GT che la pred tocca
                gm = np.asarray(g["mask"], dtype=bool)
                if g["cls"] == cls and (gm & m).any():
                    local |= gm
        if local.any():
            inst.metadata["missed_px"] = int((local & ~m).sum())
            inst.metadata["excess_px"] = int((m & ~local).sum())
            inst.metadata["gt_px"] = int(local.sum())
            inst.metadata["excess_sides"] = error_sides(m & ~local, local)
            inst.metadata["missed_sides"] = error_sides(local & ~m, local)
            inst.metadata["boundary_where"] = boundary_where(m, local)
            # holes of the mask that the object does not have: a prediction equal to a
            # ground truth with a hole in it (between an arm and the body) has no error
            gaps = ndimage.binary_fill_holes(m) & ~m & local
            lab_t, n_t = ndimage.label(gaps)
            sizes = ndimage.sum(gaps, lab_t, range(1, n_t + 1)) if n_t else []
            real = [int(x) for x in sizes if x >= HOLE_MIN_PX]
            inst.metadata["n_true_holes"] = len(real)
            inst.metadata["true_hole_frac"] = round(sum(real) / float(m.sum()), 4) if real else 0.0
    return done, extras
