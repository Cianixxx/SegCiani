# -*- coding: utf-8 -*-
"""Read the pictures out of the HTML galleries of the runs (they embed every picture as base64), so that the figures of
the thesis use exactly what the galleries show.
  slices(path)  -> {slice_id: {"name", "verdict", "branch", "type", "hyp", "thumbs": [(title, bytes)]}}
  images(path)  -> {image_id: {"narr", "desc", "panels": [(caption, bytes)]}}
"""
import base64
import html
import re

_IMG = re.compile(r'<img src="data:image/(?:jpeg|png);base64,([A-Za-z0-9+/=]+)"(?: title="([^"]*)")?')


def _read(path):
    return open(path, encoding="utf-8").read()


def slices(path):
    h = _read(path)
    out = {}
    for m in re.finditer(r'<div class="card" id="s_([^"]+)" data-branch="([a-z]+)" data-verdict="([A-Z]+)">', h):
        sid, branch, verdict = m.groups()
        end = h.find('<details class="js">', m.end())
        body = h[m.end():end]
        name = re.search(r'<span class="sname">(.*?)</span>', body)
        typ = re.search(r'<span class="type">(.*?)</span>', body)
        hyp = re.search(r'<div class="hyp">(.*?)</div>', body)
        out[sid] = {"name": html.unescape(name.group(1)) if name else None, "branch": branch, "verdict": verdict,
                    "type": typ.group(1) if typ else None, "hyp": html.unescape(hyp.group(1)) if hyp else None,
                    "thumbs": [(html.unescape(t or ""), base64.b64decode(b)) for b, t in _IMG.findall(body)]}
    return out


def images(path):
    h = _read(path)
    out = {}
    for m in re.finditer(r'<div class="icard"><h3>([0-9_]+)', h):
        iid = m.group(1)
        end = h.find('<div class="icard">', m.end())
        body = h[m.end():end if end > 0 else len(h)]
        narr = re.search(r'<div class="inarr"><span class="lbl">[^<]*</span>(.*?)</div>', body, re.S)
        desc = re.search(r'<div class="idesc"><span class="lbl">[^<]*</span>(.*?)</div>', body, re.S)
        panels = []
        for f in re.finditer(r'<figure><img src="data:image/(?:jpeg|png);base64,([A-Za-z0-9+/=]+)"[^>]*>'
                             r'(?:<figcaption>(.*?)</figcaption>)?', body):
            panels.append((html.unescape(re.sub(r"<[^>]+>", "", f.group(2) or "")), base64.b64decode(f.group(1))))
        out[iid] = {"narr": html.unescape(re.sub(r"<[^>]+>", "", narr.group(1))) if narr else "",
                    "desc": html.unescape(re.sub(r"<[^>]+>", "", desc.group(1))) if desc else "", "panels": panels}
    return out
