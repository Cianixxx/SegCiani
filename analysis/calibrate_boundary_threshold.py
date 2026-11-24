# -*- coding: utf-8 -*-
"""Calibra la soglia di Boundary IoU sulle 79 immagini annotate a mano.

Il Blocco 1 decide CORRECT con l'IoU d'AREA (`tau_iou_good = 0.85`), che e'
scala-dipendente: su un oggetto grande un contorno sbagliato sposta l'IoU di
pochi centesimi. Tutte e 6 le immagini giudicate BASSA stanno sopra quella
soglia (IoU 0.848-0.968) con Boundary IoU fra 0.042 e 0.343: l'annotatore vede
il bordo rotto, la pipeline non emette nulla.

Regola proposta: un'istanza accoppiata resta CORRECT solo se e' corretta in area
E ha un contorno accettabile; sotto la soglia diventa un errore BOUNDARY.
Questo script sceglie la soglia invece di indovinarla.

SU QUALE ETICHETTA calibrare -- la scelta non e' ovvia e la prima e' una trappola:

  (a) "l'annotazione lamenta il contorno" (boundar*, coarse, irregular, ...).
      NON USABILE: lo fanno 53 annotazioni su 79, cioe' il 67%. La regola banale
      "di' sempre bordo" ottiene F1 0.80, piu' della migliore soglia (0.78), e il
      BoundaryIoU delle istanze area-corrette e' identico fra chi lamenta e chi no
      (0.344 contro 0.351, AUC 0.512 = nessun segnale). Su un segmentatore debole
      il contorno e' scadente quasi ovunque: l'etichetta non discrimina.

  (b) il verdetto BASSA scritto a mano, cioe' "qui la pipeline avrebbe dovuto dire
      qualcosa e non ha detto nulla". QUESTA discrimina: BoundaryIoU 0.230 sulle
      istanze area-corrette delle BASSA contro 0.372 sulle altre, AUC 0.802.

Si calibra su (b), e (a) resta nel report come controprova del perche' non serva.
`edge` e' escluso dal vocabolario: negli annotatori indica la posizione ("on the
right edge"), non un difetto.

Avvertenza sulla numerosita': le BASSA sono 6. La soglia che ne esce e' un valore
indicativo tarato su sei punti, non una soglia validata.

Il matching qui e' una re-implementazione di quello del Blocco 1 (componenti
connesse per classe, min 200 px, crop 520, greedy a IoU >= 0.5): lo script
verifica di riprodurre i conteggi reali di `typed_per_image.json` prima di
fidarsi dei suoi numeri.

CPU, locale. Run:  python docs/calibrate_boundary_threshold.py
"""
import json
import io
import os
import sys
import re

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = r"<project-root>\SlicelenceCode"
sys.path.insert(0, os.path.join(ROOT, "docs"))
from _revision_numbers import out_path as _out_path  # noqa: E402  (RUN_UFFICIALI, RUN_VECCHI or outputs/)
B = r"<datasets>\pascal_80"
CROP, MIN_PX = 520, 200
TAU_MATCH, TAU_GOOD = 0.5, 0.85
D = 0.01 * np.sqrt(2) * CROP          # 1% della diagonale, come boundary_d_frac
BAND = 2 * D                          # la soglia nel codice del Blocco 1 e' 2*d

# qualita' del contorno, non posizione: "edge" non entra
BND_RE = re.compile(
    r"boundar|coarse|irregular|erratic|impreci|inaccurate|blob|rough|flawed|"
    r"jagged|less defined|not well defined|details are lost|lost details", re.I)


def rcc(img, resample):
    w, h = img.size
    if w < h:
        nw, nh = CROP, int(round(h * CROP / w))
    else:
        nh, nw = CROP, int(round(w * CROP / h))
    img = img.resize((nw, nh), resample)
    l, t = (nw - CROP) // 2, (nh - CROP) // 2
    return img.crop((l, t, l + CROP, t + CROP))


def comps(arr):
    out = []
    for c in sorted(set(np.unique(arr)) - {0, 255}):
        lab, n = ndimage.label(arr == c)
        for k in range(1, n + 1):
            cc = lab == k
            if cc.sum() >= MIN_PX:
                out.append((int(c), cc))
    return out


def band(m, d):
    return m & (ndimage.distance_transform_edt(m) <= d)


def pairs_of(name):
    """Coppie accoppiate della stessa classe: (iou, boundary_iou, area)."""
    gp = os.path.join(B, "SegmentationClass", name + ".png")
    pp = os.path.join(B, "mobilenet_pred", "mask_pr_" + name + ".png")
    if not (os.path.exists(gp) and os.path.exists(pp)):
        return None
    gt = np.array(rcc(Image.open(gp), Image.NEAREST))
    pr = np.array(rcc(Image.open(pp).split()[0], Image.NEAREST))
    G, P, used, out = comps(gt), comps(pr), set(), []
    for gc, gm in G:
        best, best_iou = None, 0.0
        for i, (pc, pm) in enumerate(P):
            if i in used or not (gm & pm).any():
                continue
            v = (gm & pm).sum() / float((gm | pm).sum())
            if v > best_iou:
                best, best_iou = i, v
        if best is None or best_iou < TAU_MATCH:
            continue
        pc, pm = P[best]
        used.add(best)
        if pc != gc:                       # confusione, non un caso di bordo
            continue
        gb, pb = band(gm, BAND), band(pm, BAND)
        u = (gb | pb).sum()
        out.append((float(best_iou), float((gb & pb).sum()) / u if u else 0.0,
                    int(gm.sum())))
    return out


# ---- dati -----------------------------------------------------------------
mp = {int(k): v.replace(".jpg", "") for k, v in
      json.load(io.open(os.path.join(ROOT, r"_evaldata\dataset\voc_patch_matches.json"),
                        encoding="utf-8")).items()}
human = {}
for line in io.open(os.path.join(ROOT, r"_evaldata\dataset\answer_gts.jsonl"),
                    encoding="utf-8"):
    if line.strip():
        o = json.loads(line)
        human[mp[o["img_idx"]]] = o["content"].strip()

typed = json.load(io.open(_out_path("slice_quality", "typed_per_image.json"),
                          encoding="utf-8"))
verdict = {}
for line in io.open(_out_path("slice_quality", "gt_comparison.md"),
                    encoding="utf-8"):
    m = re.match(r"^## (\S+) - \*\*(ALTA|PARZIALE|BASSA)\*\*", line)
    if m:
        verdict[m.group(1)] = m.group(2)

imgs = {}
for name, txt in human.items():
    ps = pairs_of(name)
    if ps is None:
        continue                            # 2011_003554: nessuna GT di segmentazione
    tx = typed.get(name, {})
    imgs[name] = {
        "pairs": ps,
        "human_bnd": bool(BND_RE.search(txt)),
        "already": bool(tx.get("boundary", 0) or tx.get("extent", 0)),
        "verdict": verdict.get(name),
    }
print(f"immagini valutabili: {len(imgs)} | coppie accoppiate: "
      f"{sum(len(v['pairs']) for v in imgs.values())}")
print(f"annotazioni che lamentano il contorno: "
      f"{sum(v['human_bnd'] for v in imgs.values())}/{len(imgs)}")

# ---- il matching re-implementato riproduce quello vero? -------------------
mine = {n for n, v in imgs.items() if any(i < TAU_GOOD for i, _b, _a in v["pairs"])}
real = {n for n, v in imgs.items() if v["already"]}
print(f"\nvalidazione della re-implementazione: immagini con un errore "
      f"boundary/extent -- reali {len(real)}, ricostruite {len(mine)}, "
      f"in accordo {len(mine & real)} ({len(mine & real) / max(len(real | mine), 1):.0%})")

# ---- il segnale c'e'? test soglia-indipendente su entrambe le etichette ----
def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    return sum((a < b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))


ac = [(n, i, b) for n, v in imgs.items() for i, b, _a in v["pairs"] if i >= TAU_GOOD]
print(f"\nistanze area-corrette (oggi CORRECT, invisibili): {len(ac)}")

pos_a = [b for n, _i, b in ac if imgs[n]["human_bnd"]]
neg_a = [b for n, _i, b in ac if not imgs[n]["human_bnd"]]
hb = sum(v["human_bnd"] for v in imgs.values())
print(f"\n(a) etichetta 'l'annotazione lamenta il contorno': {hb}/{len(imgs)} = "
      f"{hb/len(imgs):.0%} -- caso base altissimo")
print(f"    BIoU {np.mean(pos_a):.3f} (lamenta) contro {np.mean(neg_a):.3f} (non lamenta) "
      f"| AUC {auc(pos_a, neg_a):.3f}")
print(f"    La regola banale 'di' sempre bordo' avrebbe F1 "
      f"{2*(hb/len(imgs))/(hb/len(imgs)+1):.2f}: questa etichetta NON e' utilizzabile.")

pos_b = [b for n, _i, b in ac if imgs[n]["verdict"] == "BASSA"]
neg_b = [b for n, _i, b in ac if imgs[n]["verdict"] != "BASSA"]
print(f"\n(b) etichetta 'verdetto BASSA' (la pipeline taceva e non doveva):")
print(f"    BIoU {np.mean(pos_b):.3f} (n={len(pos_b)}) contro {np.mean(neg_b):.3f} "
      f"(n={len(neg_b)}) | AUC {auc(pos_b, neg_b):.3f}")
print(f"    intervallo sulle BASSA: {min(pos_b):.3f} - {max(pos_b):.3f}")

# ---- sweep sulla sola etichetta utilizzabile ------------------------------
print("\n" + "=" * 74)
print("CALIBRAZIONE sul verdetto BASSA")
print("=" * 74)
bassa = [n for n, v in imgs.items() if v["verdict"] == "BASSA"]
n_corr = len(ac)
print(f"{'soglia':>7}{'BASSA prese':>13}{'nuove istanze':>15}{'CORRECT rimasti':>17}")
for tau in [round(x, 2) for x in np.arange(0.10, 0.55, 0.05)]:
    new_inst = [(n, b) for n, i, b in ac if b < tau]
    taken = len({n for n, _b in new_inst if imgs[n]["verdict"] == "BASSA"})
    print(f"{tau:7.2f}{taken:>9d}/{len(bassa)}{len(new_inst):15d}"
          f"{n_corr - len(new_inst):17d}")

tau_all = max(pos_b) + 0.005
print(f"\nSoglia minima che prende tutte le {len(bassa)} BASSA: {tau_all:.3f} "
      f"(la peggiore delle altre e' a {max(pos_b):.3f})")
n_at = sum(1 for _n, _i, b in ac if b < tau_all)
print(f"  costo: {n_at}/{n_corr} istanze area-corrette riclassificate, "
      f"{n_corr - n_at} restano come riferimento per omega^2")

print("\n-- perche' la pipeline taceva, che non e' un problema di tipizzazione --")
zero = [n for n in bassa if sum(typed.get(n, {}).values()) == 0]
print(f"  BASSA con ZERO errori tipati di qualunque tipo: {len(zero)}/{len(bassa)}")
print("  Un'istanza area-corretta non genera alcuna istanza d'errore: il bordo non")
print("  viene tipizzato male, semplicemente non esiste nulla da descrivere.")
