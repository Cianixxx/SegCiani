"""Visual evaluation -- a self-contained HTML gallery of slices + member images.

Two views (tabs in the header):
* **by slice**  -- one card per slice (both branches, all verdicts) with the
  Slice-Quality Vector, a thumbnail grid of the grouped images, and a JSON link.
* **by image**  -- one card per original image: the original / ground-truth /
  prediction panels, plus, for each of its instances, ALL the slices it belongs to.

Output is ONE ``.html`` file (images embedded as base64) -> scp it and open in any
browser, no server needed. Filter buttons toggle branch / verdict.

``build_gallery(result, image_paths, out_html, image_panels=...)``
* ``image_paths``  : image_id -> path of the full image (for crops + fallback panel).
* ``image_panels`` : optional image_id -> list[(caption, PIL.Image)] for the
  "by image" view (e.g. [("original", img), ("ground truth", gt), ("prediction", pred)]).
  Build them in the run script (dataset-specific); the gallery just renders them.
"""

from __future__ import annotations

import base64
import html as _html
import io
import json
import os
import re
from collections import defaultdict

AXES = [
    ("omega^2", "omega_squared"),
    ("manifold", "manifold_compactness"),
    ("mispred", "misprediction_coherence"),
    ("err-type", "error_type_purity"),
    ("purity", "slice_purity"),
]

_CSS = """
*{box-sizing:border-box} body{font-family:system-ui,Segoe UI,sans-serif;margin:0;background:#0f1115;color:#e6e6e6}
header{position:sticky;top:0;background:#171a21;padding:12px 18px;border-bottom:1px solid #2a2f3a;z-index:5}
h1{font-size:16px;font-weight:600;margin:0 0 8px} .muted{color:#9aa3b2;font-weight:400}
.filters button{background:#222734;color:#cfd6e4;border:1px solid #313847;border-radius:6px;padding:4px 10px;margin:2px;cursor:pointer;font-size:12px}
.filters button.on{background:#3b82f6;border-color:#3b82f6;color:#fff}
[hidden]{display:none!important}
.wrap{padding:16px;display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:14px}
.iwrap{padding:16px}
.card{background:#171a21;border:1px solid #2a2f3a;border-radius:10px;padding:12px}
.hd{display:flex;align-items:center;gap:8px;flex-wrap:wrap;font-size:12px;margin-bottom:8px}
.badge{font-weight:700;padding:2px 8px;border-radius:5px;font-size:11px}
.inarr{background:#151a22;border:1px solid #232a36;border-left:3px solid #3b82f6;
border-radius:6px;padding:8px 10px;margin:0 0 8px;font-size:12.5px;line-height:1.55;color:#d6dce7}
.idesc .lbl,.inarr .lbl{display:block;font-size:10px;letter-spacing:.06em;
text-transform:uppercase;color:#6b7687;margin-bottom:3px}
.nall{margin:14px 0 6px}
.nall summary{cursor:pointer;color:#cfd6e4;background:#222734;border:1px solid #313847;
border-radius:6px;padding:6px 12px;font-size:12px;display:inline-block}
.nbox{margin-top:10px;display:flex;flex-direction:column;gap:8px}
.nrow{background:#151a22;border:1px solid #232a36;border-radius:6px;padding:8px 10px}
.nid{font:600 12px ui-monospace,Consolas,monospace;color:#9aa3b2;margin-bottom:3px;
display:flex;align-items:center;gap:8px}
.ntx{font-size:12.5px;line-height:1.55;color:#d6dce7}
.plaus{margin-left:auto;font-size:11px;color:#8fb8f0;background:#1b2330;
border:1px solid #2a3547;border-radius:10px;padding:1px 7px;white-space:nowrap}
.plaus i{font-style:normal;color:#6b7687}
.icard h3 .plaus{margin-left:8px;font-weight:400}
.plaus.facts{margin-left:6px;color:#9fe0b0;border-color:#2f5a40;background:#17261d}
.nid .plaus + .plaus.facts{margin-left:6px}
.plaus.extra{margin-left:6px;color:#f5c97a;border-color:#5a4a2a;background:#261f14}
.hnote{font-size:12.5px;color:#cbd5e1;margin:0 0 8px;line-height:1.5}
.hnote b{color:#f5c97a}
.KEEP{background:#16331f;color:#4ade80} .REFINE{background:#33270f;color:#fbbf24} .DISCARD{background:#331616;color:#f87171}
.sid{font-weight:600;color:#e6e6e6;font-size:13px;margin-right:2px}
.sname{font-weight:700;color:#fff;font-size:14px;margin-right:6px}
.sname+.sid{font-weight:400;color:#8a8a8a;font-size:11px}
.cname{font-weight:600}
.type{font-weight:600} .src{color:#9aa3b2} .n,.sig{color:#9aa3b2}
.card.flash{outline:2px solid #3b82f6;outline-offset:2px}
.hyp{font-style:italic;color:#cbd5e1;font-size:12px;margin:4px 0 8px}
.diag{font-size:12px;color:#a7b0c0;margin:6px 0}
.ax{display:flex;align-items:center;gap:6px;font-size:11px;margin:2px 0}
.ax span{width:62px;color:#9aa3b2} .ax b{width:34px;text-align:right;color:#e6e6e6}
.bar{flex:1;height:7px;background:#252b36;border-radius:4px;overflow:hidden}
.bar i{display:block;height:100%;background:#3b82f6}
.grid{display:flex;flex-wrap:wrap;gap:3px;margin-top:9px}
.grid img{width:62px;height:62px;object-fit:cover;border-radius:4px;border:1px solid #2a2f3a}
.more{align-self:center;color:#9aa3b2;font-size:12px;padding:0 6px}
.js{margin-top:9px;font-size:11px} .js summary{cursor:pointer;color:#60a5fa;user-select:none}
.js pre{background:#0c0e12;border:1px solid #2a2f3a;border-radius:6px;padding:8px;margin:6px 0;max-height:240px;overflow:auto;white-space:pre-wrap;color:#cbd5e1}
.js a{color:#60a5fa;text-decoration:none}
.icard{background:#171a21;border:1px solid #2a2f3a;border-radius:10px;padding:12px;margin-bottom:14px}
.icard h3{margin:0 0 8px;font-size:13px;color:#cbd5e1}
.idesc{background:#0c0e12;border:1px solid #2a2f3a;border-left:3px solid #3b82f6;border-radius:6px;padding:8px 10px;margin-bottom:10px;font-size:12px;color:#cbd5e1;white-space:pre-wrap;line-height:1.5}
.panels{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:10px}
.panels figure{margin:0;text-align:center}
.panels img{height:200px;border-radius:6px;border:1px solid #2a2f3a;display:block}
.panels figcaption{font-size:11px;color:#9aa3b2;margin-top:4px}
.ilist{display:flex;flex-wrap:wrap;gap:10px}
.irow{display:flex;gap:8px;align-items:center;background:#0f1218;border:1px solid #232936;border-radius:8px;padding:6px}
.irow img{width:54px;height:54px;object-fit:cover;border-radius:4px}
.it{font-size:11px;color:#cbd5e1;margin-bottom:3px}
.sbs{display:flex;flex-wrap:wrap;gap:3px;max-width:260px}
.sb{font-size:10px;padding:1px 5px;border-radius:4px;border:1px solid #313847;cursor:pointer;text-decoration:none}
.sb.KEEP{color:#4ade80;border-color:#16331f} .sb.REFINE{color:#fbbf24;border-color:#33270f} .sb.DISCARD{color:#f87171;border-color:#331616}
"""

_JS = """
function flt(){
 const b=document.querySelector('.branch.on').dataset.v;
 const v=document.querySelector('.verdict.on').dataset.v;
 document.querySelectorAll('#byslice .card').forEach(c=>{
  const ok=(b=='all'||c.dataset.branch==b)&&(v=='all'||c.dataset.verdict==v);
  c.style.display=ok?'':'none';});
}
function bind(sel,fn){document.querySelectorAll(sel).forEach(b=>b.onclick=()=>{
 b.parentElement.querySelectorAll('button').forEach(x=>x.classList.remove('on'));
 b.classList.add('on');fn(b);});}
bind('.branch',flt);bind('.verdict',flt);
bind('.view',b=>{document.getElementById('byslice').hidden=b.dataset.v!='byslice';
 document.getElementById('byimage').hidden=b.dataset.v!='byimage';});
document.querySelectorAll('a[data-goto]').forEach(a=>a.onclick=e=>{
 e.preventDefault();
 document.querySelectorAll('.view').forEach(x=>x.classList.remove('on'));
 document.querySelector('.view[data-v=byslice]').classList.add('on');
 document.getElementById('byslice').hidden=false;document.getElementById('byimage').hidden=true;
 ['.branch','.verdict'].forEach(sel=>{document.querySelectorAll(sel).forEach(x=>x.classList.remove('on'));
  document.querySelector(sel+'[data-v=all]').classList.add('on');});
 flt();
 const c=document.getElementById(a.dataset.goto);
 if(c){c.scrollIntoView({behavior:'smooth',block:'center'});
  c.classList.add('flash');setTimeout(()=>c.classList.remove('flash'),1600);}
});
"""


def _img_b64(src, size, quality=72):
    """base64 JPEG of a PIL image OR an image path, resized to fit `size`."""
    try:
        from PIL import Image
        im = src if hasattr(src, "convert") else Image.open(src)
        im = im.convert("RGB").copy()
        im.thumbnail((size, size))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=quality)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


def _thumb(image_paths, inst, size):
    try:
        from PIL import Image
        path = image_paths.get(inst.image_id)
        if not path or inst.bbox is None or not os.path.exists(path):
            return None
        img = Image.open(path).convert("RGB")
        x0, y0, x1, y1 = (int(round(v)) for v in inst.bbox)
        x0, y0 = max(0, x0), max(0, y0)
        crop = img.crop((x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)))
        crop.thumbnail((size, size))
        buf = io.BytesIO()
        crop.save(buf, "JPEG", quality=70)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def _bar(label, val):
    pct = int(round(max(0.0, min(1.0, float(val))) * 100))
    return (f'<div class="ax"><span>{label}</span>'
            f'<div class="bar"><i style="width:{pct}%"></i></div>'
            f'<b>{float(val):.2f}</b></div>')


def _record(s, q, kept):
    rec = {
        "slice_id": q.slice_id, "branch": s.source, "verdict": q.verdict,
        "hypothesis": s.hypothesis, "size": q.size,
        "dominant_error_type": q.dominant_error_type.value,
        "passes_def1": bool(q.passes_def1), "significant": bool(q.significant),
        "axes": {a: round(float(getattr(q, a)), 4) for _, a in AXES},
        "members": [{
            "id": m.instance_id, "image": m.image_id,
            "bbox": [round(v, 1) for v in m.bbox] if m.bbox else None,
            "type": m.seg_subtype.value, "iou": round(m.iou, 3),
            "attribute": m.metadata.get("attribute"),
        } for m in s.members],
    }
    k = kept.get(q.slice_id)
    if k is not None:
        rec["explanation"] = k.explanation.sentence
        rec["advice"] = k.advice.text
    cor = (getattr(s, "metadata", None) or {}).get("corroborated_by")
    if cor:
        rec["corroborated_by"] = cor
    return rec


def _json_block(rec):
    pretty = json.dumps(rec, ensure_ascii=False, indent=2)
    b64 = base64.b64encode(pretty.encode("utf-8")).decode()
    return (f'<details class="js"><summary>JSON</summary>'
            f'<pre>{_html.escape(pretty)}</pre>'
            f'<a download="{rec["slice_id"]}.json" '
            f'href="data:application/json;base64,{b64}">download .json</a></details>')


def build_gallery(result, image_paths, out_html, image_panels=None,
                  image_descriptions=None, thumb=124, max_per_slice=24,
                  max_images=None, title="slice_quality results", cause_scores=None,
                  image_narratives=None, n_images_total=None, facts_scores=None,
                  facts_mean=None, image_order=None, image_notes=None, image_badges=None,
                  header_note=None, slice_names=None):
    """Write a one-file HTML gallery (by-slice + by-image views). Returns the path.

    ``image_narratives`` : optional image_id -> the human-style narrative of EVERY
    typed error of that image (describe_errors.py). It answers a different
    question from the pattern description already shown -- what went wrong here,
    rather than which certified pattern it belongs to -- so the two are labelled
    and kept apart. Every image that has one also appears in a text-only listing
    at the end of the by-image view, which is how all 1 483 fit in a file meant
    to be shared: the cards carry pictures and stop at ``max_images``.

    ``cause_scores`` : optional image_id -> plausibility of the cause proposed for
    that image (judge_cause_plausibility.py, the narrative-told score). A slice
    card shows the MEAN over its member images -- the only level at which the
    number says something about the slice itself -- with the count it is averaged
    over; an image card shows its own. Absent, nothing is drawn.

    ``facts_scores`` : optional image_id -> {"narrative": score, "n_facts": n} -- the
    length-neutral facts score (judge_claims.py): the human annotation against the
    pipeline's narrative, so it exists only where an annotation does (the pattern text
    is not scored). Shown beside the cause on the image cards and in the narrative
    listing.

    ``facts_mean`` : optional (mean, n, detail) for the header -- the facts score
    averaged over EVERY annotated image, wherever it comes from (VOC val, VOC train,
    synthetic), most of which are not in this gallery; ``detail`` is a short
    per-source breakdown. Without it the header averages ``facts_scores``.

    ``image_order`` : optional list of image ids shown FIRST in the by-image view, in
    that order, before the remaining images (sorted); an image listed here gets its
    card even when none of its instances is in a slice. ``max_images`` caps the total.

    ``image_notes`` : optional image_id -> (label, text), a labelled line on the image
    card above the narrative (e.g. what was changed in the image and what it did).

    ``image_badges`` : optional image_id -> (text, tooltip), one more score badge beside
    the cause and facts badges, on the image card and in the narrative listing.

    ``header_note`` : optional HTML line under the title (the caller builds and escapes
    it), e.g. the mean of the scores the badges carry.
    """
    cause_scores = cause_scores or {}
    facts_scores = facts_scores or {}

    def _fbadge(iid):
        f = facts_scores.get(iid)
        if not f or f.get("narrative") is None:
            return ''
        return (f'<span class="plaus facts" title="voto per fatti: i {f.get("n_facts", "?")} fatti '
                f'dell&#39;annotazione umana verificati uno per uno nella narrativa della pipeline; '
                f'indipendente dalla lunghezza">fatti {f["narrative"]:.1f}/10</span>')
    image_badges = image_badges or {}

    def _xbadge(iid):
        b = image_badges.get(iid)
        if not b:
            return ''
        return (f'<span class="plaus extra" title="{_html.escape(b[1])}">'
                f'{_html.escape(b[0])}</span>')
    image_narratives = image_narratives or {}
    pairs = list(getattr(result, "scored", None) or
                 [(ks.slice_, ks.quality) for ks in result.kept])
    seen, items = set(), []
    for s, q in pairs:
        if q.slice_id in seen:
            continue
        seen.add(q.slice_id)
        items.append((s, q))
    # short names for the cluster slices (naming.py), shown instead of the id, which stays
    # beside it in small print and in the tooltips; the texts themselves are not changed
    if slice_names is None:
        from .naming import cluster_names
        slice_names = cluster_names(items)
    _cid = re.compile(r"\bcluster_[a-z]+_\d+_r\d+\b")

    def _named(text_html):
        return _cid.sub(lambda m: (f'<span class="cname" title="{m.group(0)}">'
                                   f'{_html.escape(slice_names[m.group(0)])}</span>')
                        if m.group(0) in slice_names else m.group(0), text_html)
    vorder = {"KEEP": 0, "REFINE": 1, "DISCARD": 2}
    items.sort(key=lambda sq: (sq[0].source, vorder.get(sq[1].verdict, 3),
                               -sq[1].omega_squared))
    kept_map = {k.slice_.slice_id: k for k in result.kept}

    # ---- by-slice cards --------------------------------------------------
    cards = []
    for s, q in items:
        thumbs = []
        for inst in s.members[:max_per_slice]:
            b = _thumb(image_paths, inst, thumb)
            if b:
                tip = f"{inst.instance_id} | {inst.seg_subtype.value}"
                thumbs.append(f'<img src="data:image/jpeg;base64,{b}" '
                              f'title="{_html.escape(tip)}">')
        more = len(s.members) - len(thumbs)
        more_html = f'<span class="more">+{more}</span>' if more > 0 else ''
        axes = "".join(_bar(l, getattr(q, a)) for l, a in AXES)
        hyp = (f'<div class="hyp">&ldquo;{_html.escape(s.hypothesis)}&rdquo;</div>'
               if s.hypothesis else '')
        aliases = (getattr(s, "metadata", None) or {}).get("aliases")
        if aliases:  # paraphrase hypotheses merged into this slice by dedup (#7)
            ali = "; ".join(f'&ldquo;{_html.escape(a)}&rdquo;' for a in aliases)
            hyp += f'<div class="hyp" style="opacity:.65">also: {ali}</div>'
        _cs = [cause_scores[m.image_id] for m in s.members if m.image_id in cause_scores]
        plaus = (f'<span class="plaus" title="plausibilit&agrave; media della causa sulle '
                 f'immagini di questa slice, giudicata contro la narrativa generata">'
                 f'causa {sum(_cs) / len(_cs):.1f}/10 <i>n={len(_cs)}</i></span>') if _cs else ''
        kept = kept_map.get(s.slice_id)
        diag = kept.explanation.sentence if kept else ''
        diag_html = f'<div class="diag">{_html.escape(diag)}</div>' if diag else ''
        # cross-branch corroboration: verified hypotheses this cluster contains
        cor = (getattr(s, "metadata", None) or {}).get("corroborated_by") or []
        for c in cor[:2]:
            diag_html += (f'<div class="diag">&#8776; &ldquo;{_html.escape(c["hypothesis"])}&rdquo; '
                          f'(SliceLens, C={c["containment"]:.2f}, '
                          f'n&#8745;={c["n_overlap"]})</div>')
        # Repair advice (block 9) stays out of the card on purpose: it's in the
        # per-slice JSON export (kept.advice.text) for whoever wants it, but the
        # card itself shows only the diagnosis sentence, as before.
        json_html = _json_block(_record(s, q, kept_map))
        cards.append(
            f'<div class="card" id="s_{q.slice_id}" '
            f'data-branch="{s.source}" data-verdict="{q.verdict}">'
            f'<div class="hd">'
            + (f'<span class="sname">{_html.escape(slice_names[q.slice_id])}</span>'
               if q.slice_id in slice_names else '')
            + f'<span class="sid">{_html.escape(q.slice_id)}</span>'
            f'<span class="badge {q.verdict}">{q.verdict}</span>'
            f'<span class="type">{q.dominant_error_type.value}</span>'
            f'<span class="src">{s.source}</span><span class="n">n={q.size}</span>'
            f'<span class="sig">{"sig+" if q.significant else "sig-"}</span>'
            f'{plaus}</div>'
            f'{hyp}<div class="axes">{axes}</div>{diag_html}'
            f'<div class="grid">{"".join(thumbs)}{more_html}</div>{json_html}</div>'
        )

    # ---- by-image cards: original/GT/pred + each instance -> its slices ---
    inst_slices = defaultdict(list)
    inst_obj, img_insts = {}, defaultdict(set)
    for s, q in items:
        for m in s.members:
            inst_slices[m.instance_id].append((s.slice_id, s.source, q.verdict))
            inst_obj[m.instance_id] = m
            img_insts[m.image_id].add(m.instance_id)

    panels_map = image_panels or {}
    descs_map = image_descriptions or {}
    icards = []
    image_ids = sorted(img_insts)
    if image_order:
        first = [i for i in dict.fromkeys(image_order)
                 if i in img_insts or i in panels_map or i in image_narratives]
        _f = set(first)
        image_ids = first + [i for i in image_ids if i not in _f]
    notes_map = image_notes or {}
    if max_images is not None:
        image_ids = image_ids[:max_images]
    for image_id in image_ids:
        panels = panels_map.get(image_id)
        if panels:
            panels_html = "".join(
                f'<figure><img src="data:image/jpeg;base64,{_img_b64(im, 320)}">'
                f'<figcaption>{_html.escape(cap)}</figcaption></figure>'
                for cap, im in panels)
        else:
            p = image_paths.get(image_id)
            b = _img_b64(p, 320) if p else ''
            panels_html = (f'<figure><img src="data:image/jpeg;base64,{b}">'
                           f'<figcaption>original</figcaption></figure>') if b else ''
        rows = []
        for iid in sorted(img_insts.get(image_id, ())):
            m = inst_obj[iid]
            tb = _thumb(image_paths, m, 96)
            tb_html = f'<img src="data:image/jpeg;base64,{tb}">' if tb else ''
            badges = "".join(
                f'<a class="sb {v}" data-goto="s_{_html.escape(sid)}" title="{_html.escape(sid)}" '
                f'href="#s_{_html.escape(sid)}">{_html.escape(slice_names.get(sid, sid))} &middot; {src}</a>'
                for sid, src, v in inst_slices[iid])
            rows.append(
                f'<div class="irow">{tb_html}<div>'
                f'<div class="it">{m.seg_subtype.value} &middot; iou={m.iou:.2f}</div>'
                f'<div class="sbs">{badges}</div></div></div>')
        desc = descs_map.get(image_id, {}).get("text", "")
        desc_html = (f'<div class="idesc"><span class="lbl">pattern sistematici</span>'
                     f'{_named(_html.escape(desc))}</div>') if desc else ''
        nar = (image_narratives.get(image_id) or "").strip()
        nar_html = (f'<div class="inarr"><span class="lbl">narrativa: ogni errore '
                    f'tipato</span>{_html.escape(nar)}</div>') if nar else ''
        desc_html = nar_html + desc_html
        if image_id in notes_map:
            _nl, _nt = notes_map[image_id]
            desc_html = (f'<div class="idesc"><span class="lbl">{_html.escape(_nl)}</span>'
                         f'{_html.escape(_nt)}</div>') + desc_html
        _ics = cause_scores.get(image_id)
        ip = (f'<span class="plaus" title="plausibilit&agrave; della causa proposta '
              f'per questa immagine">causa {_ics:.1f}/10</span>') if _ics is not None else ''
        icards.append(
            f'<div class="icard"><h3>{_html.escape(image_id)} '
            f'<span class="muted">{len(img_insts.get(image_id, ()))} instances in slices</span>{ip}'
            f'{_fbadge(image_id)}{_xbadge(image_id)}</h3>'
            f'{desc_html}<div class="panels">{panels_html}</div>'
            f'<div class="ilist">{"".join(rows)}</div></div>')

    # ---- every narrative, as text ----------------------------------------
    # The cards above carry three pictures each and are capped; these are the
    # same texts for the whole run, which cost nothing to ship and make the file
    # answer "what does the pipeline say about THIS image" for every image.
    nlist = ""
    if image_narratives:
        rows_n = []
        for iid in sorted(image_narratives):
            t = (image_narratives[iid] or "").strip()
            if not t:
                continue
            sc = cause_scores.get(iid)
            sch = (f'<span class="plaus">causa {sc:.1f}/10</span>') if sc is not None else ''
            rows_n.append(f'<div class="nrow"><div class="nid">{_html.escape(iid)}{sch}{_fbadge(iid)}{_xbadge(iid)}</div>'
                          f'<div class="ntx">{_html.escape(t)}</div></div>')
        nlist = (f'<details class="nall"><summary>narrative generate, tutte le '
                 f'{len(rows_n)} immagini (testo, senza figure)</summary>'
                 f'<div class="nbox">{"".join(rows_n)}</div></details>')

    # ---- assemble --------------------------------------------------------
    # Header: how many images the run covers, how many receive a description of a
    # certified pattern, how many slices were scored (both branches, every verdict)
    # and the mean plausibility of the cause over the images shown here that have
    # one. n_images_total comes from the caller (the split's image list); without
    # it the images that carry an instance stand in.
    n_total = n_images_total if n_images_total is not None else len(img_insts)
    n_desc = sum(1 for d in descs_map.values() if d.get("patterns"))
    _shown_cs = [v for k, v in cause_scores.items() if not descs_map or k in descs_map]
    plaus_h = (f' &middot; plausibilit&agrave; media della causa '
               f'{sum(_shown_cs) / len(_shown_cs):.2f}/10 '
               f'<i>(n={len(_shown_cs)})</i>') if _shown_cs else ''
    if facts_mean:
        _fm, _fn, _fd = facts_mean
        facts_h = (f' &middot; voto per fatti medio, umano vs pipeline, {_fm:.2f}/10 '
                   f'<i>(n={_fn} immagini annotate{": " + _fd if _fd else ""})</i>')
    else:
        _fs = [f["narrative"] for f in facts_scores.values() if f.get("narrative") is not None]
        facts_h = (f' &middot; voto per fatti medio, umano vs pipeline, {sum(_fs) / len(_fs):.2f}/10 '
                   f'<i>(n={len(_fs)} annotate)</i>') if _fs else ''
    head = (f'<h1>{_html.escape(title)} '
            f'<span class="muted">&mdash; {n_total} immagini &middot; {n_desc} descritte '
            f'(almeno un pattern certificato) &middot; {len(items)} slice{plaus_h}{facts_h}'
            f'</span></h1>') + (f'<div class="hnote">{header_note}</div>' if header_note else '')
    view = ('<span class="filters">view: '
            '<button class="view on" data-v="byslice">by slice</button>'
            '<button class="view" data-v="byimage">by image</button></span>')
    bbtn = ('<span class="filters">branch: '
            '<button class="branch on" data-v="all">all</button>'
            '<button class="branch" data-v="cluster">cluster</button>'
            '<button class="branch" data-v="language">language</button></span>')
    vbtn = ('<span class="filters">verdict: '
            '<button class="verdict on" data-v="all">all</button>'
            '<button class="verdict" data-v="KEEP">keep</button>'
            '<button class="verdict" data-v="REFINE">refine</button>'
            '<button class="verdict" data-v="DISCARD">discard</button></span>')

    doc = (f'<!doctype html><meta charset="utf-8"><title>{_html.escape(title)}</title>'
           f'<style>{_CSS}</style><header>{head}'
           f'<div>{view} &nbsp; {bbtn} &nbsp; {vbtn}</div></header>'
           f'<div class="wrap" id="byslice">{"".join(cards)}</div>'
           f'<div class="iwrap" id="byimage" hidden>{"".join(icards)}{nlist}</div>'
           f'<script>{_JS}</script>')

    parent = os.path.dirname(os.path.abspath(out_html))
    os.makedirs(parent, exist_ok=True)
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(doc)
    return out_html
