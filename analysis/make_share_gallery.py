# -*- coding: utf-8 -*-
"""Shrink the full gallery into a shareable one, without changing what it says.

The full gallery inlines every crop and panel as a base64 data URI, which is what
makes it self-contained and also what makes it 50 MB -- too large for chat apps.
Two steps, in order of how much they actually save:

  1. DEDUPLICATE. The same crop appears in several views (per-slice grid, per-image
     panel), so the file carries roughly three times more image bytes than it has
     distinct pictures. Base64 in HTML cannot be shared between tags, so each unique
     picture is emitted ONCE in a JS array and assigned at load. This is where the
     saving is -- the crops were already thumbnail-sized, their NUMBER was the problem.
  2. RE-ENCODE. Each unique picture is downscaled and re-saved as JPEG.

Text, patterns, slice ids and structure are untouched, so the shared copy says
exactly what the real one says.

Run: python docs/make_share_gallery.py [max_side] [quality]
"""
import base64
import io as _io
import os
import re
import sys

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# SQ_OUT: the run whose gallery is shrunk (a retyped run lives in its own directory)
SQ = os.path.join(ROOT, os.environ.get("SQ_OUT", os.path.join("outputs", "slice_quality")))
# GALLERY_SRC / GALLERY_DST: file names inside SQ_OUT (a gallery with its own name, e.g. val-EA)
SRC = os.path.join(SQ, os.environ.get("GALLERY_SRC", "gallery_full_tagged.html"))
DST = os.path.join(SQ, os.environ.get("GALLERY_DST", "gallery_share.html"))
MAX_SIDE = int(sys.argv[1]) if len(sys.argv) > 1 else 160
QUALITY = int(sys.argv[2]) if len(sys.argv) > 2 else 55

PAT = re.compile(r"data:image/(png|jpe?g);base64,([A-Za-z0-9+/=]+)")


def main():
    html = _io.open(SRC, encoding="utf-8", errors="ignore").read()
    seen = {}
    stats = {"n": 0, "failed": 0}

    def repl(m):
        raw = m.group(2)
        stats["n"] += 1
        if raw in seen:
            return seen[raw]
        try:
            im = Image.open(_io.BytesIO(base64.b64decode(raw))).convert("RGB")
            im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
            buf = _io.BytesIO()
            im.save(buf, "JPEG", quality=QUALITY, optimize=True)
            out = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            stats["failed"] += 1
            out = m.group(0)
        seen[raw] = out
        return out

    small = PAT.sub(repl, html)

    # ---- emit every distinct picture once, reference it by index ----
    uniq, index = [], {}

    def dedupe(m):
        uri = m.group(0)
        if uri not in index:
            index[uri] = len(uniq)
            uniq.append(uri)
        return "IMG#%d" % index[uri]

    small = re.sub(r"data:image/[a-z]+;base64,[A-Za-z0-9+/=]+", dedupe, small)
    small = re.sub(r'src="IMG#(\d+)"',
                   lambda m: 'src="" data-img="' + m.group(1) + '"', small)
    leftover = small.count("IMG#")
    loader = ("<script>const IMGS=[" + ",".join('"%s"' % u for u in uniq) + "];"
              "document.querySelectorAll('[data-img]').forEach(function(e){"
              "e.src=IMGS[+e.dataset.img];});</script>")
    # gallery.py emits a bare fragment: no <body> tags to anchor to, so append.
    if "</body>" in small:
        small = small.replace("</body>", loader + "</body>", 1)
    else:
        small += loader

    banner = ("<p style='font:12px system-ui,sans-serif;color:#9aa3b2;margin:6px 18px'>"
              "Shared copy: images downscaled to {}px / JPEG q{} and de-duplicated. "
              "Text, patterns and slice ids are identical to the full gallery."
              "</p>").format(MAX_SIDE, QUALITY)
    anchor = small.find("</header>")
    if anchor != -1:
        small = small[:anchor + 9] + banner + small[anchor + 9:]
    elif "<body>" in small:
        small = small.replace("<body>", "<body>" + banner, 1)
    _io.open(DST, "w", encoding="utf-8", newline="\n").write(small)

    print("image tags: {n} ({failed} left as-is) | distinct pictures: {u}".format(
        u=len(uniq), **stats))
    if leftover:
        print("WARNING: {} placeholders outside src= left unresolved".format(leftover))
    print("source {:.1f} MB -> share {:.1f} MB".format(
        os.path.getsize(SRC) / 1e6, os.path.getsize(DST) / 1e6))
    # the zip is what actually travels through chat apps
    import zipfile
    zp = DST[:-5] + ".zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.write(DST, os.path.basename(DST))
    print("zip {:.1f} MB -> {}".format(os.path.getsize(zp) / 1e6, zp))


if __name__ == "__main__":
    main()
