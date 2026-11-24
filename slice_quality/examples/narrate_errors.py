"""Write a human-style narrative of the errors of every image. CPU only.

Runs ``describe_errors.narrate`` over both image sets that have hand-written
annotations to compare against, and over the whole val set besides:

    val      from result_tagged.pkl          (the pipeline's own run)
    train36  from train36/instances_train36.pkl

Before narrating, it measures the mask facts the narrative needs and that Block 1
does not store (see ``slice_quality.mask_facts`` for what and why): frame share,
missed/excess pixels, boundary IoU, thin lost details, sub-threshold GT regions,
and -- for every FN -- how much of it a same-class prediction actually covers,
which is what turns "missed entirely" into "absorbed" where that is the truth.

Run:  PYTHONPATH=/workspace python slice_quality/examples/narrate_errors.py
Env: PKL, TRAIN36, OUT, PRED_SUBDIR, VOC_DATA, CROPS.
"""
import os
import pickle
from collections import Counter

from slice_quality.adapters.voc_seg_samples import build_voc_samples
from slice_quality.describe_errors import build_error_narratives, export_narratives
from slice_quality.mask_facts import THIN_NOISE, add_mask_facts as _measure

# VOC_DATA / CROPS: the VOC tree and crop cache of the run (val-EA has its own)
DATA = os.environ.get("VOC_DATA", "/data/voc2012/VOCdevkit/VOC2012")
CROPS = os.environ.get("CROPS", "/data/voc2012/voc_crops")
PKL = os.environ.get("PKL", "outputs/slice_quality/result_tagged.pkl")
TRAIN36 = os.environ.get("TRAIN36", "outputs/slice_quality/train36")
OUT = os.environ.get("OUT", "outputs/slice_quality/error_narratives")
PRED_SUBDIR = os.environ.get("PRED_SUBDIR", "mobilenet_pred")


def add_mask_facts(errors, names, reference=()):
    """Build the samples for ``names`` and measure everything in place."""
    samples, _p = build_voc_samples(DATA, CROPS, names, pred_subdir=PRED_SUBDIR)
    # a second pass at a low threshold: the only way to see the components the
    # adapter drops at min_pixels, which are therefore absent from `samples`
    fine, _f = build_voc_samples(DATA, CROPS, names, pred_subdir=PRED_SUBDIR,
                                 min_pixels=THIN_NOISE)
    return _measure(errors, samples, fine, reference)


out_all = {}

# ---- val ------------------------------------------------------------------
d = pickle.load(open(PKL, "rb"))
val_err, val_ref = d["errors"], d["reference"]
# the image list must include images with NO typed error: those are exactly the
# ones that can only be described through the contour of a correct region
val_names = sorted({e.image_id for e in val_err} | {r.image_id for r in val_ref})
print(f"val: {len(val_err)} typed errors + {len(val_ref)} correct references "
      f"over {len(val_names)} images", flush=True)
_done, val_extra = add_mask_facts(val_err, val_names, val_ref)
print(f"  mask facts measured on {_done} instances | immagini con componenti GT "
      f"sotto soglia: {len(val_extra)}", flush=True)
val_nar = build_error_narratives(val_err, val_names, val_ref, extras=val_extra)
out_all.update(val_nar)

# ---- train36 --------------------------------------------------------------
t_pkl = f"{TRAIN36}/instances_train36.pkl"
if os.path.exists(t_pkl):
    c = pickle.load(open(t_pkl, "rb"))
    t_err, t_ref = c["errors"], c["reference"]
    t_names = sorted({e.image_id for e in t_err} | {r.image_id for r in t_ref})
    print(f"train36: {len(t_err)} typed errors + {len(t_ref)} correct references "
          f"over {len(t_names)} images", flush=True)
    _dt, t_extra = add_mask_facts(t_err, t_names, t_ref)
    print(f"  mask facts measured on {_dt} instances | immagini con componenti GT "
          f"sotto soglia: {len(t_extra)}", flush=True)
    t_nar = build_error_narratives(t_err, t_names, t_ref, extras=t_extra)
    out_all.update(t_nar)
    export_narratives(t_nar, f"{TRAIN36}/error_narratives_train36.json",
                      f"{TRAIN36}/error_narratives_train36.txt")
    with open(t_pkl, "wb") as f:
        pickle.dump(c, f)                     # keep the mask facts

export_narratives(out_all, OUT + ".json", OUT + ".txt")
with open(PKL, "wb") as f:
    pickle.dump(d, f)                         # keep the mask facts on the val side

nb = sum(1 for v in out_all.values() if v.get("n_contour_notes"))
only = sum(1 for v in out_all.values()
           if v.get("n_contour_notes") and not v["n_errors"])
print(f"\nimmagini con una nota sul contorno: {nb} "
      f"(di cui {only} che prima non producevano alcun testo)", flush=True)

sent = Counter(len(v["sentences"]) for v in out_all.values())
print(f"\nnarratives: {len(out_all)} images | sentences per image: "
      f"{dict(sorted(sent.items()))}", flush=True)
print("\n--- samples ---", flush=True)
for n in list(val_names[:3]) + (t_names[:2] if os.path.exists(t_pkl) else []):
    print(f"\n{n}\n{out_all[n]['text']}", flush=True)
print(f"\nsaved {OUT}.json / .txt", flush=True)
print("NARRATE_DONE", flush=True)
