"""Summary of SliceLens's data-driven hypotheses on val-EA (outputs/chain_dd_ea.sh), for the thesis document.

Does the generator, run on the images where the cause is planted (the Pascal-EA material edits), ever name
the material? Two counts, both from the generation records /data/outputs_dd_ea/dd_<class>.json:
  - the condensed phrases (6 per class) that contain a material or replica word;
  - the Qwen captions behind them, split by whether the captioned instance lies in an EDITED image or an
    untouched one (an instance of an edited image is not always the edited subject: the split is coarse).
Standard library only (host python3 -I); writes outputs/slice_quality_ea_dd/dd_ea_summary.json.
Run: python3 -I slice_quality/examples/dd_ea_summary.py $HOME/storage/pascal_vlm
"""
import json
import re
import sys
from pathlib import Path

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "/data")
OUT = Path("outputs/slice_quality_ea_dd/dd_ea_summary.json")
WORDS = re.compile(r"\b(toy|toys|statue|model|stuffed|figurine|sculpture|fake|artificial|paper|cardboard|origami|"
                   r"metal|metallic|plastic|wood|wooden|replica|miniature|craft)\b", re.I)

edited = {l.strip() for l in open(DATA / "voc2012_ea" / "edited_stems.txt") if l.strip()}
per_class = {}
for f in sorted((DATA / "outputs_dd_ea").glob("dd_*.json")):
    r = json.load(open(f, encoding="utf-8"))
    hyps = [h["hypothesis"] for h in r.get("condensed", [])]
    caps = {"edited": [0, 0], "untouched": [0, 0]}
    words = {"edited": set(), "untouched": set()}
    for a in r.get("attributes", []):
        for iid, cap in zip(r["sample"], a.get("captions", [])):
            k = "edited" if iid.split("#")[0] in edited else "untouched"
            caps[k][0] += 1
            m = WORDS.search(cap or "")
            if m:
                caps[k][1] += 1
                words[k].add(m.group(0).lower())
    per_class[r["label"]] = {
        "seeds_from": r.get("seeds_from"), "attributes": [a["attribute"] for a in r.get("attributes", [])],
        "hypotheses": hyps, "material_phrases": [h for h in hyps if WORDS.search(h)],
        "n_sampled": len(r["sample"]),
        "n_sampled_edited": sum(1 for i in r["sample"] if i.split("#")[0] in edited),
        "captions": caps, "caption_words": {k: sorted(v) for k, v in words.items()}}
OUT.write_text(json.dumps({"classes": sorted(per_class), "per_class": per_class}, indent=1), encoding="utf-8")
tot = {k: [sum(v["captions"][k][i] for v in per_class.values()) for i in (0, 1)] for k in ("edited", "untouched")}
print("classes", len(per_class), "| phrases with a material word",
      sum(len(v["material_phrases"]) for v in per_class.values()), "| captions", tot, "| written", OUT)
