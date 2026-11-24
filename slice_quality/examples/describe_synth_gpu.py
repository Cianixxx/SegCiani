"""Ramo dei pattern (cluster + SliceLens) sulle 21 immagini sintetiche. GPU.

Stesso metodo di describe_train36_gpu.py: NON si rifa' la scoperta -- 21 immagini
non certificano nulla (N_min = 2% di 6928 istanze = 138) -- ma si APPLICANO le
14 slice cluster gia' certificate su val (centroide piu' vicino entro il raggio
al 90o percentile) e le 35 ipotesi linguistiche gia' validate (ancoraggio Qwen,
senza ri-verifica del trend). E' induzione di predicati certificati altrove.

Due differenze rispetto a train36:
  * i fatti di maschera (contorno, dettagli persi, copertura dei FN) vengono
    misurati PRIMA di scrivere i pattern, cosi' la riga "details" nasce completa
    invece di essere ricostruita a posteriori come fu per train36;
  * le predizioni sono nello staging preparato da narrate_synth.py, che le
    ricodifica nel formato che l'adapter legge (indice di classe nel rosso).

Avvertenza dichiarata anche nell'HTML: 11 delle 21 predizioni sono ground truth
di ALTRE immagini VOC incollate (accoppiamento per posizione in lista). Su quelle
l'induzione trova i pattern che trova, ma descrive un artefatto, non un errore di
segmentazione realistico.

Run (GPU 1):
  QWEN_4BIT=0 PYTHONPATH=/workspace python slice_quality/examples/describe_synth_gpu.py
Env: STAGE, OUT, QWEN_4BIT (default 0 -- bf16 su una RTX6000 da 24GB).
"""
import json
import os
import pickle
import re
import time
from collections import Counter, defaultdict

import numpy as np

from slicelens.data_types import Region
from slice_quality.adapters.voc_seg_samples import build_voc_samples
from slice_quality.config import SliceQualityConfig
from slice_quality.data_types import Slice
from slice_quality.describe import _detail_clause, describe_instances
from slice_quality.embeddings import attach_embeddings
from slice_quality.extractors import DINOv2Extractor
from slice_quality.features import build_features
from slice_quality.mask_facts import THIN_NOISE, add_mask_facts
from slice_quality.tagging import tag_instances_verify
from slice_quality.typed_error import decompose_dataset

# SQ_OUT: the run whose certified slices are induced here (a retyped run has
# its own directory); the staging of the synthetic masks is shared
SQ_OUT = os.environ.get("SQ_OUT", "outputs/slice_quality")
STAGE = os.environ.get("STAGE", "outputs/slice_quality/synth/_stage")
OUT = os.environ.get("OUT", f"{SQ_OUT}/synth")
PKL = f"{SQ_OUT}/result_tagged.pkl"
SLICES_INDEX = f"{SQ_OUT}/slices_index.json"
QWEN_4BIT = os.environ.get("QWEN_4BIT", "0") == "1"
LANG_CONF_MIN = 0.5
CROPS = os.path.join(STAGE, "_crops")

names = sorted(f[:-4] for f in os.listdir(os.path.join(STAGE, "JPEGImages"))
               if f.endswith(".jpg"))
print(f"immagini: {len(names)}", flush=True)

# ---- Blocco 1 + fatti di maschera ----------------------------------------
samples, image_paths = build_voc_samples(STAGE, CROPS, names, pred_subdir="mobilenet_pred")
fine, _f = build_voc_samples(STAGE, CROPS, names, pred_subdir="mobilenet_pred",
                             min_pixels=THIN_NOISE)
errors, reference = decompose_dataset(samples)
print(f"errori tipati: {len(errors)} | riferimenti: {len(reference)}", flush=True)
add_mask_facts(errors, samples, fine, reference)


def make_mask_lookup(samples):
    by_img = {s["image_id"]: s for s in samples}

    def lookup(inst):
        s = by_img.get(inst.image_id)
        if s is None:
            return None
        pool = s["gts"] if inst.metadata.get("from_gt_mask") else s["preds"]
        idx = int(inst.instance_id.rsplit("#", 1)[1][1:])
        return pool[idx]["mask"] if idx < len(pool) else None
    return lookup


# ---- Blocco 2 (GPU): embedding ------------------------------------------
print("DINOv2 ...", flush=True)
attach_embeddings(errors, DINOv2Extractor(image_paths), make_mask_lookup(samples))
print(f"embedding: {sum(1 for e in errors if e.embedding is not None)}/{len(errors)}",
      flush=True)

# ---- Qwen: attributi in modalita' verifica --------------------------------
from slicelens.backends.qwen_vlm import QwenVLM

t0 = time.time()
print(f"carico Qwen (4bit={QWEN_4BIT}) ...", flush=True)
vlm = QwenVLM(model="/models/qwen", max_new_tokens=10, load_in_4bit=QWEN_4BIT)
print(f"Qwen in {time.time() - t0:.0f}s", flush=True)
tag_instances_verify(errors, image_paths, vlm)
print("attributi:", dict(Counter(e.metadata.get("attribute") for e in errors).most_common()),
      flush=True)

# ---- ramo linguistico: applica le ipotesi gia' validate -------------------
idx = json.load(open(SLICES_INDEX, encoding="utf-8"))["slices"]
validated = [s for s in idx if s.get("source") != "cluster" and s.get("in_descriptions")
             and s.get("hypothesis")]
print(f"ipotesi validate da applicare: {len(validated)}", flush=True)
err_by_class = defaultdict(list)
for e in errors:
    cls = e.gt_class or e.pred_class
    if cls:
        err_by_class[cls].append(e)

lang_slices, n_eval = [], 0
for s in validated:
    cls = re.sub(r"(_extra|_dd)?_\d+$", "", s["slice_id"])   # "<cls>_<n>", "<cls>_extra_<n>", "<cls>_dd_<n>"
    members, membership = [], {}
    for inst in err_by_class.get(cls, []):
        if inst.bbox is None or inst.image_id not in image_paths:
            continue
        conf = float(vlm.slice_confidence(image_paths[inst.image_id],
                                          Region(bbox=inst.bbox), s["hypothesis"]))
        n_eval += 1
        if conf >= LANG_CONF_MIN:
            members.append(inst)
            membership[inst.instance_id] = conf
    if members:
        lang_slices.append((s, Slice(slice_id=s["slice_id"], members=members,
                                     source="language", hypothesis=s["hypothesis"],
                                     membership=membership)))
print(f"valutazioni linguistiche: {n_eval} | slice con >=1 membro: {len(lang_slices)}",
      flush=True)

# ---- induzione delle slice cluster certificate ----------------------------
data = pickle.load(open(PKL, "rb"))
kept = data["result"].kept
cfg = SliceQualityConfig()
old, seen = [], set()
for k in kept:
    for m in k.slice_.members:
        if m.instance_id not in seen and m.embedding is not None:
            seen.add(m.instance_id)
            old.append(m)
new = [e for e in errors if e.embedding is not None]
feats, kept_insts = build_features(old + new, cfg.feature)
fid = {i.instance_id: f for i, f in zip(kept_insts, feats)}
slice_of = {}
for k in kept:
    X = np.stack([fid[m.instance_id] for m in k.slice_.members if m.instance_id in fid])
    c = X.mean(axis=0)
    slice_of[k.slice_.slice_id] = (k, c, float(np.percentile(np.linalg.norm(X - c, axis=1), 90)),
                                   k.quality.dominant_error_type)
induced = defaultdict(list)
for inst in new:
    best, best_d = None, None
    for sid, (k, c, radius, subtype) in slice_of.items():
        if inst.seg_subtype != subtype:
            continue
        d = float(np.linalg.norm(fid[inst.instance_id] - c))
        if d <= radius and (best_d is None or d < best_d):
            best, best_d = sid, d
    if best is not None:
        induced[best].append(inst)
print(f"cluster: {sum(len(v) for v in induced.values())}/{len(new)} istanze in "
      f"{len(induced)} slice", flush=True)

# ---- descrizioni per immagine, entrambi i rami, con i dettagli -----------
by_img = defaultdict(lambda: defaultdict(list))
meta = {}
for sid, insts in induced.items():
    k = slice_of[sid][0]
    meta[sid] = ("cluster", k.slice_, k.quality)
    for i in insts:
        by_img[i.image_id][sid].append(i)


class _Q:
    def __init__(self, et):
        self.dominant_error_type = et


for s, sl in lang_slices:
    et = next((k.quality.dominant_error_type for k in kept
               if k.quality.dominant_error_type.value == s.get("dominant_error_type")), None)
    meta[sl.slice_id] = ("language", sl, _Q(et or sl.members[0].seg_subtype))
    for i in sl.members:
        by_img[i.image_id][sl.slice_id].append(i)

all_by_img = defaultdict(set)
for e in errors:
    all_by_img[e.image_id].add(e.instance_id)

out = {}
for name in names:
    n_total = len(all_by_img.get(name, ()))
    pats, sys_ids = [], set()
    for sid, members in by_img.get(name, {}).items():
        branch, sl, q = meta[sid]
        pats.append({"slice_id": sid, "branch": branch,
                     "error_type": q.dominant_error_type.value,
                     "description": describe_instances(members, sl, q),
                     "details": _detail_clause(members),
                     "n_instances": len(members)})
        sys_ids.update(m.instance_id for m in members)
    n_sys = len(sys_ids)
    n_iso = n_total - n_sys
    obj = "object" if n_total == 1 else "objects"
    if n_sys == 0:
        lines = [f"Image {name}: {n_total} wrong {obj}, no systematic pattern (isolated errors)."]
    else:
        lines = [f"Image {name}: {n_total} wrong {obj}, {n_sys} in systematic patterns"
                 + (f" and {n_iso} isolated." if n_iso else "."), "Systematic patterns:"]
        for p in sorted(pats, key=lambda x: -x["n_instances"]):
            n = p["n_instances"]
            lines.append(f"  - {p['description']} [{p['slice_id']}, {n} obj{'' if n == 1 else 's'}.]")
            if p.get("details"):
                lines.append(f"      · {p['details']}")
    out[name] = {"n_errors": n_total, "n_systematic": n_sys, "n_isolated": n_iso,
                 "patterns": pats, "text": "\n".join(lines)}

os.makedirs(OUT, exist_ok=True)
json.dump(out, open(f"{OUT}/descriptions_synth.json", "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
with open(f"{OUT}/descriptions_synth.txt", "w", encoding="utf-8") as f:
    for name in names:
        f.write(out[name]["text"] + "\n\n")
pickle.dump({"errors": errors, "reference": reference, "image_paths": image_paths},
            open(f"{OUT}/instances_synth.pkl", "wb"))
described = sum(1 for v in out.values() if v["n_systematic"])
print(f"\nimmagini con >=1 pattern: {described}/{len(names)} | righe per ramo: "
      f"{dict(Counter(p['branch'] for v in out.values() for p in v['patterns']))}", flush=True)
print("DESCRIBE_SYNTH_DONE", flush=True)
