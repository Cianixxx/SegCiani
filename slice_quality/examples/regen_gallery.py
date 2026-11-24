"""Regenerate the gallery + per-image descriptions from the pickled result.

CPU only -- no GPU, no DINOv2, no Qwen. Reuses result_tagged.pkl (result +
image_paths + tags) and the persistent crops, so you can tweak gallery.py /
describe.py and re-render instantly.

Run:  PYTHONPATH=/workspace python slice_quality/examples/regen_gallery.py
"""
import os
import pickle

from slice_quality.gallery import build_gallery
from slice_quality.describe import (build_image_descriptions,
                                    export_description_index, export_descriptions)
from slice_quality.report import export_slice_index, to_json

# SQ_OUT: output directory (a retyped run lives in its own, e.g.
# outputs/slice_quality_cov05, and must not overwrite the reference one)
SQ_OUT = os.environ.get("SQ_OUT", "outputs/slice_quality")
PKL = os.environ.get("PKL", f"{SQ_OUT}/result_tagged.pkl")
# VOC_DATA: the VOC tree whose images the panels show (val-EA has its own)
DATA = os.environ.get("VOC_DATA", "/data/voc2012/VOCdevkit/VOC2012")
# ORIG_DATA: the untouched VOC tree, for a run on edited images (val-EA): an image whose
# JPEG in DATA is a real file rather than a symlink was edited, and its card shows the
# original and the segmenter's prediction on it beside the edited one.
ORIG_DATA = os.environ.get("ORIG_DATA")
# IMAGE_NOTES: {"order": [...], "notes": {image: [label, text]}, optionally "badges":
# {image: [text, tooltip]} and "header": an HTML line under the title} (ea_gallery_notes.py)
IMAGE_NOTES = os.environ.get("IMAGE_NOTES")
TITLE = os.environ.get("GALLERY_TITLE", "VOC2012 val")
# GALLERY_OUT: the gallery file (a variant gallery gets its own name, never the run's default)
GALLERY_OUT = os.environ.get("GALLERY_OUT", f"{SQ_OUT}/gallery_full_tagged.html")

with open(PKL, "rb") as f:
    data = pickle.load(f)
res, image_paths = data["result"], data["image_paths"]

# rebuild descriptions (picks up any describe.py edits) + report
descs = build_image_descriptions(res)
from slice_quality.corroborate import cross_branch_corroboration
lang_ids = {p["slice_id"] for d in descs.values()
            for p in d["patterns"] if p["branch"] == "language"}
cor = cross_branch_corroboration(res, lang_ids)
print(f"corroborated cluster slices: {len(cor)}", flush=True)
export_descriptions(descs, f"{SQ_OUT}/image_descriptions.json",
                    f"{SQ_OUT}/image_descriptions.txt")
export_description_index(descs, f"{SQ_OUT}/descriptions_index.json",
                         f"{SQ_OUT}/descriptions_index.txt")
to_json(res, f"{SQ_OUT}/voc_full_val_tagged.json")
used_ids = {p["slice_id"] for d in descs.values() for p in d["patterns"]}
export_slice_index(res, f"{SQ_OUT}/slices_index.json",
                   f"{SQ_OUT}/slices_index.txt", used_ids=used_ids)
print(f"descriptions: {len(descs)} images", flush=True)

# plausibility of the CAUSE, when it has been measured for this run: one score
# per image, judged against the pipeline's own narrative (the only account that
# exists outside the 80 annotated images). The gallery averages it per slice and
# shows each image's own; without the files nothing is drawn.
import json as _json
cause_scores = {}
for _f in (os.environ.get("CAUSE_VAL", f"{SQ_OUT}/judge_cause_val.json"),
           os.environ.get("CAUSE_ANN", f"{SQ_OUT}/judge_cause.json")):
    if not os.path.exists(_f):
        continue
    for r in _json.load(open(_f, encoding="utf-8"))["judgements"]:
        if r.get("kind") == "pipeline" and r.get("pairing") == "genuine" and "score" in r:
            cause_scores.setdefault(r["image"], r["score"])
print(f"plausibilita' della causa: {len(cause_scores)} immagini con un voto", flush=True)

# the length-neutral facts score (judge_claims.py): only where a human annotation
# exists. Every annotated VOC image (val and train36) gets its badge -- the narrative
# list holds the train36 images too -- and the header mean is taken over EVERY
# annotated image, wherever it comes from (VOC val, VOC train36, synthetic)
facts_scores = {}
_fc = os.environ.get("CLAIMS", f"{SQ_OUT}/judge_claims.json")
if os.path.exists(_fc):
    _cj = _json.load(open(_fc, encoding="utf-8"))
    for r in _cj["judgements"]:
        if (r.get("rubric") == "facts" and r.get("pairing") == "genuine" and r.get("set") == "voc"
                and "score" in r):
            facts_scores.setdefault(r["image"], {})[r["kind"]] = r["score"]
            facts_scores[r["image"]]["n_facts"] = r.get("n_facts")
            facts_scores[r["image"]]["split"] = r.get("split")
facts_mean = None
if os.path.exists(_fc) and _cj.get("summary", {}).get("all_narrative"):
    _a, _sm = _cj["summary"]["all_narrative"], _cj["summary"]
    _vs = [f["narrative"] for f in facts_scores.values() if f.get("split") == "val"]
    _ts = [f["narrative"] for f in facts_scores.values() if f.get("split") == "train"]
    _det = []
    if _vs:
        _det.append(f"VOC val {sum(_vs) / len(_vs):.2f} su {len(_vs)}")
    if _ts:
        _det.append(f"VOC train {sum(_ts) / len(_ts):.2f} su {len(_ts)}")
    if _sm.get("synth_narrative"):
        _det.append(f"sintetiche {_sm['synth_narrative']['genuine']:.2f} su {_sm['synth_narrative']['n']}")
    facts_mean = (_a["genuine"], _a["n"], ", ".join(_det))
# the same facts with verdicts written by hand (judge_claims_manual.py), when present:
# they replace the automatic judge's scores, and the header says whose verdicts they are
_fm = os.environ.get("CLAIMS_MANUAL", f"{SQ_OUT}/judge_claims_manual.json")
if os.path.exists(_fm):
    _mj = _json.load(open(_fm, encoding="utf-8"))
    for r in _mj["rows"]:
        if r["pairing"] == "genuine" and r["set"] == "voc":
            facts_scores[r["image"]] = {"narrative": r["score"], "n_facts": r["n_facts"], "split": r.get("split")}
    _ms = _mj["summary"]
    _det = [f"VOC val {_ms['voc_val']['genuine']:.2f} su {_ms['voc_val']['n']}",
            f"VOC train {_ms['voc_train']['genuine']:.2f} su {_ms['voc_train']['n']}",
            f"sintetiche {_ms['synth']['genuine']:.2f} su {_ms['synth']['n']}",
            f"verdetti di Claude; giudice automatico Gemma {_ms['tutte']['gemma']:.2f}"
            + (f" (su {sum(1 for r in _mj['rows'] if r['pairing'] == 'genuine' and r.get('gemma_same_text') is False)}"
               f" immagini dato alla narrativa precedente)" if _mj.get("gemma_stale_items") else "")]
    facts_mean = (_ms["tutte"]["genuine"], _ms["tutte"]["n"], ", ".join(_det))
print(f"voto per fatti: badge su {len(facts_scores)} immagini VOC annotate; media "
      f"{facts_mean[0]:.2f} su {facts_mean[1]} annotate ({facts_mean[2]})" if facts_mean else
      f"voto per fatti: {len(facts_scores)} immagini annotate", flush=True)

# la narrativa per immagine (describe_errors.py): risponde a "che cosa e' andato
# storto qui", mentre la descrizione dei pattern dice "a quale pattern certificato
# appartiene". Nella galleria stanno una sopra l'altra, etichettate.
NARR = os.environ.get("NARR", f"{SQ_OUT}/error_narratives.json")
narratives = {}
if os.path.exists(NARR):
    narratives = {k: v["text"] for k, v in _json.load(open(NARR, encoding="utf-8")).items()}
print(f"narrative per immagine: {len(narratives)}", flush=True)

# colored GT + prediction panels
from PIL import Image
from slicelens.adapters.voc_seg import _resize_center_crop

val = [ln.strip() for ln in
       open(os.path.join(DATA, "ImageSets/Segmentation/val.txt")) if ln.strip()]
_PAL = Image.open(os.path.join(DATA, "SegmentationClass", val[0] + ".png")).getpalette()


def _pred(root, name):
    pr = Image.open(f"{root}/mobilenet_pred/mask_pr_{name}.png").split()[0]
    pr.putpalette(_PAL)
    return _resize_center_crop(pr.convert("RGB"), 520, Image.NEAREST)


def _panels(name):
    try:
        o = _resize_center_crop(Image.open(f"{DATA}/JPEGImages/{name}.jpg").convert("RGB"), 520, Image.LANCZOS)
        g = _resize_center_crop(Image.open(f"{DATA}/SegmentationClass/{name}.png").convert("RGB"), 520, Image.NEAREST)
        p = _pred(DATA, name)
        if ORIG_DATA and not os.path.islink(f"{DATA}/JPEGImages/{name}.jpg"):
            oo = _resize_center_crop(Image.open(f"{ORIG_DATA}/JPEGImages/{name}.jpg").convert("RGB"),
                                     520, Image.LANCZOS)
            return [("originale, non modificata", oo), ("predizione sull'originale", _pred(ORIG_DATA, name)),
                    ("modificata (Pascal-EA)", o), ("ground truth", g), ("predizione sulla modificata", p)]
        return [("original", o), ("ground truth", g), ("prediction", p)]
    except Exception:
        return None


# Panels for EXACTLY the images the gallery will show (first max_images of the
# sorted error images) -- val[:N] drifts out of sync because error-free val
# images are not shown, leaving the tail of the by-image view without GT/pred.
MAX_IMAGES = int(os.environ.get("MAX_IMAGES", "80"))
image_order, image_notes, image_badges, header_note = None, None, None, None
if IMAGE_NOTES:
    _gn = _json.load(open(IMAGE_NOTES, encoding="utf-8"))
    image_order, image_notes = _gn["order"], {k: tuple(v) for k, v in _gn["notes"].items()}
    image_badges = {k: tuple(v) for k, v in (_gn.get("badges") or {}).items()}
    header_note = _gn.get("header")
    _have = {m.image_id for s, _q in (getattr(res, "scored", None) or
                                      [(k.slice_, k.quality) for k in res.kept]) for m in s.members}
    _first = [n for n in image_order if n in _have or n in narratives]
    shown = (_first + [n for n in sorted(_have) if n not in set(_first)])[:MAX_IMAGES]
else:
    shown = sorted(descs)[:MAX_IMAGES]
image_panels = {n: pn for n in shown if (pn := _panels(n))}
build_gallery(res, image_paths, GALLERY_OUT,
              image_panels=image_panels, image_descriptions=descs, max_images=MAX_IMAGES,
              cause_scores=cause_scores, image_narratives=narratives,
              n_images_total=len(val), facts_scores=facts_scores, facts_mean=facts_mean,
              title=TITLE, image_order=image_order, image_notes=image_notes,
              image_badges=image_badges, header_note=header_note)
print("gallery regenerated from pickle (CPU, no GPU)", flush=True)
