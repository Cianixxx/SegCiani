"""One self-contained HTML page with the cluster slices the relaxed gates certify and the thesis gates do not.
CPU only, reads only.

For each variant directory (rescored by outputs/chain_relaxed_cluster.sh) the certified cluster slices whose
member set is not one of the official run's are shown: the description the pipeline gives them, size, error
type, the values of every gate (severity z, significance, attribute purity and its permutation p, coherence),
the IoU of their class on their images against the class's other images (slice_image_iou.json), and up to
N_THUMBS members, each a crop around the object with the ground truth of its class outlined in green, the
prediction of that class filled in red and the object's box in orange. The split-half replication of each
variant is reported for the variant as a whole (the halves discover their own groups, not these ids).
Run (CPU, server):  python -u slice_quality/examples/relaxed_gallery.py
Env: OFFICIAL, VARIANTS ("A=dir B=dir"), ROOT (VOC), OUT (outputs/relaxed_clusters.html), N_THUMBS (10)
"""
import base64
import html
import io
import json
import os
import pickle
import random
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw

from slicelens.adapters.voc_seg import _resize_center_crop

OFFICIAL = os.environ.get("OFFICIAL", "outputs/slice_quality_inst")
VARIANTS = [v.split("=", 1) for v in os.environ.get(
    "VARIANTS", "A=outputs/slice_quality_inst_relaxA B=outputs/slice_quality_inst_relaxB").split()]
ROOT = os.environ.get("ROOT", "/data/voc2012/VOCdevkit/VOC2012")
OUT = os.environ.get("OUT", "outputs/relaxed_clusters.html")
N_THUMBS = int(os.environ.get("N_THUMBS", "10"))
CLS = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
       "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
       "train", "tvmonitor"]
TYPE_IT = {"fn": "mancati", "fp": "inventati", "boundary": "bordo impreciso", "merge": "fusi",
           "split": "frammentati", "extent": "estensione sbagliata", "confusion": "classe sbagliata"}
J = lambda p: json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None
rng = random.Random(0)
_masks = {}


def masks(img):
    if img not in _masks:
        gt = np.array(_resize_center_crop(Image.open(f"{ROOT}/SegmentationClass/{img}.png"), 520, Image.NEAREST))
        pr = np.array(Image.open(f"{ROOT}/mobilenet_pred/mask_pr_{img}.png").convert("RGB"))[:, :, 0]
        _masks[img] = (gt, pr)
    return _masks[img]


def thumb(inst, path, size=190):
    """Crop around the object with context; GT of its class outlined green, prediction of it filled red."""
    try:
        im = Image.open(path).convert("RGB")
        gt, pr = masks(inst.image_id)
        c = CLS.index(inst.gt_class or inst.pred_class)
        x0, y0, x1, y1 = (float(v) for v in inst.bbox)
        w, h = x1 - x0, y1 - y0
        m = max(w, h) * 0.35 + 12
        X0, Y0 = max(0, int(x0 - m)), max(0, int(y0 - m))
        X1, Y1 = min(im.width, int(x1 + m)), min(im.height, int(y1 + m))
        a = np.array(im).astype(float)
        red = pr == c
        a[red] = a[red] * 0.55 + np.array([220, 40, 30]) * 0.45
        g = gt == c
        edge = g & ~(np.roll(g, 1, 0) & np.roll(g, -1, 0) & np.roll(g, 1, 1) & np.roll(g, -1, 1))
        a[edge] = [40, 220, 70]
        out = Image.fromarray(a.clip(0, 255).astype(np.uint8))
        ImageDraw.Draw(out).rectangle([x0, y0, x1, y1], outline=(255, 150, 0), width=2)
        out = out.crop((X0, Y0, X1, Y1))
        out.thumbnail((size, size))
        buf = io.BytesIO()
        out.save(buf, "JPEG", quality=78)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def certified(d):
    idx = J(f"{d}/slices_index.json")["slices"]
    return {s["slice_id"]: s for s in idx if s.get("in_descriptions") and s["slice_id"].startswith("cluster")}


def members(s):
    return frozenset(m for ids in s["images"].values() for m in ids)


off_sets = {members(s) for s in certified(OFFICIAL).values()}
sections = []
for vname, vd in VARIANTS:
    cert = certified(vd)
    res = pickle.load(open(f"{vd}/result_tagged.pkl", "rb"))
    err = {e.instance_id: e for e in res["errors"]}
    paths = res["image_paths"]
    qual = {s.slice_id: q for s, q in res["result"].scored if s.slice_id in cert}
    iou = {r["slice_id"]: r for r in (J(f"{vd}/slice_image_iou.json") or {}).get("rows", [])}
    phr = {}
    for img, v in (J(f"{vd}/image_descriptions.json") or {}).items():
        for p in v.get("patterns", []):
            if p.get("branch") == "cluster":
                phr.setdefault(p["slice_id"], Counter())[p["description"]] += 1
    sh = J(f"{vd}/split_half.json")
    seen, new = set(), []
    for sid, s in sorted(cert.items(), key=lambda kv: -kv[1]["size"]):
        ms = members(s)
        if ms in seen or ms in off_sets:
            continue
        seen.add(ms)
        new.append(sid)
    cards = []
    for sid in new:
        s, q = cert[sid], qual.get(sid)
        e = q.extras if q else {}
        ids = sorted(members(s))
        rng.shuffle(ids)
        thumbs = []
        for mid in ids[:N_THUMBS]:
            inst = err.get(mid)
            b = thumb(inst, paths.get(inst.image_id, "")) if inst is not None else None
            if b:
                thumbs.append(f'<figure><img src="data:image/jpeg;base64,{b}"><figcaption>{html.escape(inst.image_id)}'
                              f' · IoU {inst.iou:.2f}</figcaption></figure>')
        r = iou.get(sid, {})
        ciou = (f'<b>{100 * r["cls_iou_in"]:.0f}</b> sulle sue immagini, {100 * r["cls_iou_out"]:.0f} sulle altre'
                if r.get("cls_iou_in") is not None and r.get("cls_iou_out") is not None else "–")
        desc = phr.get(sid, Counter()).most_common(1)
        desc = desc[0][0] if desc else sid
        no_attr = "appearance only" in desc
        pp = e.get("purity_p")
        gates = (f'gravità z {e.get("mean_sev_z", float("nan")):+.2f} · p {e.get("p_value", float("nan")):.3f} '
                 f'({html.escape(str(e.get("sig_test", "")))}) · attributo: {e.get("n_attr", 0)} membri taggati, '
                 f'p {("%.3f" % pp) if pp is not None else "–"} · coerenza {q.misprediction_coherence:.2f}'
                 if q else "")
        cards.append(f'''<section class="card{' noattr' if no_attr else ''}">
<h3>{html.escape(desc)}</h3>
<p class="meta"><span class="id">{sid}</span> · <b>{s["size"]}</b> errori su {s["n_images"]} immagini ·
{TYPE_IT.get(s["dominant_error_type"], s["dominant_error_type"])} · IoU della classe: {ciou}</p>
<p class="gates">{gates}</p>
<div class="thumbs">{"".join(thumbs)}</div></section>''')
    summ = (f'{len(cert)} gruppi cluster certificati ({len(new)} nuovi rispetto al run ufficiale)'
            + (f' · sulle due metà dei dati si ritrovano {sh["n_replicated"]} su {sh["n_discovered"]}' if sh else ""))
    sections.append(f'<h2>Variante {html.escape(vname)}</h2><p class="sum">{summ}</p>' + "".join(cards))

rules = {"A": "attributo comune con p ≤ 0,10 (era 0,05) e coerenza ≥ 0,5 (era 0,6)",
         "B": "nessun attributo comune richiesto; il resto come nella tesi"}
page = f'''<!doctype html><html lang="it"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gruppi cluster nuovi</title>
<style>
:root {{ --ink:#1E293B; --muted:#5B6675; --line:#D8DDE6; --card:#F7F8FA; --amber:#C2410C; --green:#1E7B4A; }}
body {{ margin:0; padding:24px 16px 48px; font-family:Calibri, "Segoe UI", Arial, sans-serif; color:var(--ink);
       background:#fff; max-width:1240px; margin-inline:auto; }}
h1 {{ font-size:28px; margin:0 0 6px; }} h2 {{ font-size:22px; margin:34px 0 4px; border-bottom:2px solid var(--line);
       padding-bottom:4px; }} h3 {{ font-size:17px; margin:0 0 6px; }}
.lead, .sum {{ color:var(--muted); font-size:15px; line-height:1.45; }}
.legend span {{ display:inline-block; margin-right:16px; font-size:14px; }}
.sw {{ display:inline-block; width:14px; height:10px; vertical-align:middle; margin-right:5px; border-radius:2px; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 16px; margin:14px 0; }}
.card.noattr h3::after {{ content:"senza attributo verificato"; font-size:12px; font-weight:600; color:var(--amber);
       background:#FBECE2; border-radius:10px; padding:2px 8px; margin-left:10px; vertical-align:middle; }}
.meta {{ font-size:14.5px; margin:2px 0; }} .gates {{ font-size:13px; color:var(--muted); margin:2px 0 10px; }}
.id {{ font-family:Consolas, monospace; font-size:12.5px; color:var(--muted); }}
.thumbs {{ display:flex; flex-wrap:wrap; gap:8px; }}
figure {{ margin:0; }} figure img {{ display:block; height:150px; width:auto; max-width:100%; border-radius:4px; }}
figcaption {{ font-size:11.5px; color:var(--muted); margin-top:2px; }}
@media (max-width:600px) {{ figure img {{ height:110px; }} }}
</style></head><body>
<h1>Gruppi cluster nuovi con vincoli allentati</h1>
<p class="lead">Run di VOC val, copie del run ufficiale con vincoli di certificazione allentati. Variante A: {rules["A"]}.
Variante B: {rules["B"]}. Sono mostrati solo i gruppi che il run ufficiale non certifica (nella B anche quelli già nella A).
Per ogni gruppo: la descrizione che dà la pipeline, i valori dei vincoli, l'IoU della sua classe sulle sue immagini e sulle
altre, e alcuni membri presi a caso.</p>
<p class="legend"><span><i class="sw" style="background:#28DC46"></i>maschera vera della classe (contorno)</span>
<span><i class="sw" style="background:#DC281E"></i>predizione del modello per la classe</span>
<span><i class="sw" style="background:#FF9600"></i>riquadro dell'oggetto sbagliato</span></p>
{"".join(sections)}
</body></html>'''
open(OUT, "w", encoding="utf-8").write(page)
print(f"saved {OUT} ({len(page) / 1e6:.1f} MB)", flush=True)
print("RELAXED_GALLERY_DONE", flush=True)
