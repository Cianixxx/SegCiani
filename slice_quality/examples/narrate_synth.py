"""Fa passare le 21 immagini a maschere sintetiche nel Blocco 1 + narrativa. CPU.

Perche' questo set vale piu' degli altri: le sue 21 annotazioni si dividono in
due meta' dichiarate. Undici descrivono fallimenti catastrofici ("the prediction
is totally wrong", "hallucinate DOG and SHEEP"), dieci descrivono predizioni
PERFETTE ("The prediction is perfect", "makes no mistakes"). Le seconde sono la
CLASSE NEGATIVA che mancava: con le sole 80 annotazioni di VOC si poteva misurare
quanto la pipeline vede, mai quanto tace quando non c'e' nulla da vedere. Qui si
misurano entrambe le cose sulla stessa scala.

Un dettaglio di formato che va sistemato prima, altrimenti i risultati sarebbero
silenziosamente sbagliati: le predizioni di questo set sono PNG in modalita' P
con l'indice di classe nella palette, mentre l'adapter legge
``Image.open(pred).convert("RGB")[:, :, 0]``, cioe' il canale ROSSO -- che per un
file a palette e' il colore, non l'indice (8 e 12 diventerebbero 64 e 224). Lo
script prepara quindi uno staging con le predizioni ricodificate come il resto
della pipeline le vuole, invece di toccare l'adapter che serve a tutti gli altri.

Run:  PYTHONPATH=/workspace python slice_quality/examples/narrate_synth.py
Env: SRC (cartella del dataset), STAGE, OUT.
"""
import json
import os
import pickle
import shutil
from collections import Counter

import numpy as np
from PIL import Image

from slice_quality.adapters.voc_seg_samples import build_voc_samples
from slice_quality.describe_errors import build_error_narratives, export_narratives
from slice_quality.mask_facts import THIN_NOISE, add_mask_facts
from slice_quality.typed_error import decompose_dataset

SRC = os.environ.get("SRC", "datasets/cropped_synthetic_masks")
STAGE = os.environ.get("STAGE", "outputs/slice_quality/synth/_stage")
OUT = os.environ.get("OUT", "outputs/slice_quality/synth")
CROPS = os.path.join(STAGE, "_crops")


def prepare_stage():
    """Ricodifica le predizioni a palette in RGB con l'indice di classe nel rosso."""
    for sub in ("JPEGImages", "SegmentationClass", "mobilenet_pred", "_crops"):
        os.makedirs(os.path.join(STAGE, sub), exist_ok=True)
    names = sorted(f[:-4] for f in os.listdir(os.path.join(SRC, "JPEGImages"))
                   if f.endswith(".jpg"))
    for nm in names:
        shutil.copyfile(os.path.join(SRC, "JPEGImages", nm + ".jpg"),
                        os.path.join(STAGE, "JPEGImages", nm + ".jpg"))
        shutil.copyfile(os.path.join(SRC, "SegmentationClass", nm + ".png"),
                        os.path.join(STAGE, "SegmentationClass", nm + ".png"))
        idx = np.array(Image.open(os.path.join(SRC, "SegmentationClass_prediction",
                                               nm + ".png")))     # indici, mode P
        rgb = np.dstack([idx, idx, idx]).astype(np.uint8)
        Image.fromarray(rgb, "RGB").save(
            os.path.join(STAGE, "mobilenet_pred", f"mask_pr_{nm}.png"))
    return names


names = prepare_stage()
print(f"immagini: {len(names)} | staging in {STAGE}", flush=True)

samples, image_paths = build_voc_samples(STAGE, CROPS, names,
                                         pred_subdir="mobilenet_pred")
fine, _f = build_voc_samples(STAGE, CROPS, names, pred_subdir="mobilenet_pred",
                             min_pixels=THIN_NOISE)
errors, reference = decompose_dataset(samples)
print(f"errori tipati: {len(errors)} | riferimenti corretti: {len(reference)}",
      flush=True)
print("per tipo:", dict(Counter(e.seg_subtype.value for e in errors).most_common()),
      flush=True)

done, extras = add_mask_facts(errors, samples, fine, reference)
print(f"fatti di maschera su {done} istanze | immagini con componenti sotto "
      f"soglia: {len(extras)}", flush=True)

nar = build_error_narratives(errors, names, reference, extras=extras)
os.makedirs(OUT, exist_ok=True)
export_narratives(nar, f"{OUT}/narratives_synth.json", f"{OUT}/narratives_synth.txt")
with open(f"{OUT}/instances_synth.pkl", "wb") as f:
    pickle.dump({"errors": errors, "reference": reference,
                 "image_paths": image_paths}, f)

# la meta' "perfetta" del set e' l'unica misura di precisione che abbiamo
mp = {int(k): v.replace(".jpg", "") for k, v in
      json.load(open(os.path.join(SRC, "voc_patch__id_matches.json"),
                     encoding="utf-8")).items()}
human = {}
for line in open(os.path.join(SRC, "answer_gts.jsonl"), encoding="utf-8"):
    if line.strip():
        o = json.loads(line)
        human[mp[o["img_idx"]]] = o["content"].strip()

MUTO = "No segmentation error was found in this image."
by_err = {n: sum(1 for e in errors if e.image_id == n) for n in names}
print("\n" + "=" * 74, flush=True)
for nm in names:
    t = nar[nm]["text"]
    print(f"\n## {nm}  ({by_err[nm]} errori tipati)", flush=True)
    print(f"   UMANO   : {human.get(nm, '')[:150]}", flush=True)
    print(f"   PIPELINE: {t[:150]}", flush=True)
print(f"\nsalvato in {OUT}/", flush=True)
print("NARRATE_SYNTH_DONE", flush=True)
