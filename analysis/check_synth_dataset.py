# -*- coding: utf-8 -*-
"""Controlla il dataset a maschere sintetiche prima di darlo in pasto alla pipeline.

Nasce da un errore vero: nella prima versione le predizioni delle 11 immagini
"sbagliate" erano le ground truth di ALTRE immagini VOC, ritagliate al centro e
incollate (IoU 1.000 con la GT di 2007_002597, 2007_002611, 2007_002639, ...).
Le sorgenti erano id consecutivi in ordine crescente: la firma di un accoppiamento
per POSIZIONE fra due liste invece che per contenuto. La pipeline non poteva
accorgersene -- descriveva fedelmente quello che c'era nei file -- e il difetto e'
emerso solo guardando le immagini.

I quattro controlli, dal piu' banale al piu' utile:

  1. dimensioni   immagine, GT e predizione devono coincidere;
  2. auto-IoU     quanto la predizione somiglia alla GT della PROPRIA immagine;
     e' bassa per costruzione sui casi corrotti, ma se e' ~0 su TUTTI e' sospetto;
  3. estraneita'  la predizione coincide con la GT di un'altra immagine? e' il
     controllo che avrebbe preso il bug: cerca fra le immagini VOC con la stessa
     identica composizione di classi e confronta il ritaglio quadrato centrale;
  4. annotazione  le classi nominate in maiuscolo nel testo umano compaiono
     davvero nella GT o nella predizione di quell'immagine.

Il (3) richiede VOC e quindi il server; senza, viene saltato con un avviso.

Run:  python docs/check_synth_dataset.py [cartella]
"""
import json
import io
import os
import re
import sys

import numpy as np
from PIL import Image

VOC = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
       "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
       "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
ALIAS = {"airplane": "aeroplane", "plane": "aeroplane", "monitor": "tvmonitor",
         "tv": "tvmonitor", "table": "diningtable", "bike": "bicycle",
         "motorcycle": "motorbike"}

D = sys.argv[1] if len(sys.argv) > 1 else \
    r"<project-root>\SlicelenceCode\datasets\cropped_synthetic_masks"
VOCD = os.environ.get("VOCD", "")          # SegmentationClass di VOC, se disponibile
R = 256


def sq(im):
    w, h = im.size
    s = min(w, h)
    return im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))


def fg(a):
    return (a > 0) & (a != 255)


def classes(a):
    return frozenset(int(c) for c in set(np.unique(a)) - {0, 255})


names = sorted(f[:-4] for f in os.listdir(os.path.join(D, "JPEGImages"))
               if f.endswith(".jpg"))
mp = {int(k): v.replace(".jpg", "") for k, v in
      json.load(io.open(os.path.join(D, "voc_patch__id_matches.json"),
                        encoding="utf-8")).items()}
human = {}
for line in io.open(os.path.join(D, "answer_gts.jsonl"), encoding="utf-8"):
    if line.strip():
        o = json.loads(line)
        human[mp[o["img_idx"]]] = o["content"].strip()

print(f"immagini: {len(names)} | annotazioni: {len(human)}\n")

# ---- 1. dimensioni --------------------------------------------------------
bad_size = []
for n in names:
    a = Image.open(os.path.join(D, "JPEGImages", n + ".jpg")).size
    b = Image.open(os.path.join(D, "SegmentationClass", n + ".png")).size
    c = Image.open(os.path.join(D, "SegmentationClass_prediction", n + ".png")).size
    if not (a == b == c):
        bad_size.append((n, a, b, c))
print(f"1. dimensioni coerenti: {len(names) - len(bad_size)}/{len(names)}")
for x in bad_size:
    print("   INCOERENTE:", x)

# ---- 2. auto-IoU ----------------------------------------------------------
print("\n2. quanto la predizione somiglia alla GT della propria immagine")
self_iou = {}
for n in names:
    g = np.array(Image.open(os.path.join(D, "SegmentationClass",
                                         n + ".png")).resize((R, R), Image.NEAREST))
    p = np.array(Image.open(os.path.join(D, "SegmentationClass_prediction",
                                         n + ".png")).resize((R, R), Image.NEAREST))
    u = (fg(g) | fg(p)).sum()
    self_iou[n] = float((fg(g) & fg(p)).sum()) / u if u else 0.0
lo = [n for n in names if self_iou[n] < 0.35]
print(f"   auto-IoU < 0.35 (predizione scollegata dalla propria GT): {len(lo)}/{len(names)}")
for n in lo:
    print(f"     {n}  {self_iou[n]:.3f}")

# ---- 3. la predizione e' la GT di un'ALTRA immagine? ----------------------
print("\n3. la predizione coincide con la ground truth di un'altra immagine?")
if not (VOCD and os.path.isdir(VOCD)):
    print("   SALTATO: serve VOCD=<.../VOC2012/SegmentationClass>.")
    print("   E' il controllo che avrebbe preso il bug: eseguilo prima di consegnare.")
else:
    pool = {}
    for f in sorted(os.listdir(VOCD)):
        a = np.array(Image.open(os.path.join(VOCD, f)))
        pool.setdefault(classes(a), []).append(f[:-4])
    stolen = []
    for n in names:
        p = Image.open(os.path.join(D, "SegmentationClass_prediction", n + ".png"))
        pa = np.array(p.resize((R, R), Image.NEAREST))
        best = (0.0, None)
        for cand in pool.get(classes(pa), []):
            if cand == n:
                continue
            ga = np.array(sq(Image.open(os.path.join(VOCD, cand + ".png")))
                          .resize((R, R), Image.NEAREST))
            u = (fg(pa) | fg(ga)).sum()
            v = float((fg(pa) & fg(ga)).sum()) / u if u else 0.0
            if v > best[0]:
                best = (v, cand)
        if best[0] >= 0.75:
            stolen.append((n, best[1], best[0]))
    print(f"   predizioni che sono la GT di un'altra immagine: {len(stolen)}/{len(names)}")
    for n, c, v in stolen:
        print(f"     {n}  <-  {c}  IoU {v:.3f}")
    if stolen:
        print("   -> accoppiamento per POSIZIONE invece che per contenuto: da correggere.")

# ---- 4. annotazione contro maschere ---------------------------------------
print("\n4. le classi nominate nell'annotazione compaiono nelle maschere?")
bad_ann = []
for n in names:
    g = np.array(Image.open(os.path.join(D, "SegmentationClass", n + ".png")))
    p = np.array(Image.open(os.path.join(D, "SegmentationClass_prediction", n + ".png")))
    present = {VOC[c] for c in classes(g) | classes(p)}
    named = {ALIAS.get(w.lower(), w.lower())
             for w in re.findall(r"\b[A-Z]{3,}\b", human.get(n, ""))}
    named = {w for w in named if w in VOC}
    missing = named - present
    if missing:
        bad_ann.append((n, sorted(missing), sorted(present)))
print(f"   annotazioni coerenti: {len(names) - len(bad_ann)}/{len(names)}")
for n, m, pres in bad_ann:
    print(f"     {n}  nomina {m} ma nelle maschere c'e' {pres}")

print("\n--- esito ---")
probs = len(bad_size) + len(bad_ann)
print("dimensioni e annotazioni: " + ("tutto a posto" if not probs else f"{probs} problemi"))
print("il controllo 3 e' quello decisivo: eseguilo con VOCD impostato.")
