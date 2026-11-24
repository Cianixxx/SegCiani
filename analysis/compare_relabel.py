"""Before / after the "which class took the place of which" narrative (CPU, local).

Reads the inst narratives (before) and the ones regenerated on the copy
outputs/slice_quality_inst_relabel (after), for val, train36 and the synthetic set,
and prints: how many narratives changed, how many new sentences of each kind, and
for every category-A missed fact (claims_manual/missed_facts_categories.json) the
new sentences of its image.
Run: python docs/compare_relabel.py  ->  outputs/slice_quality_inst_relabel/compare.txt
"""
import json
import os
import sys
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "docs"))
from _revision_numbers import out_path as _out_path  # noqa: E402  (RUN_UFFICIALI, RUN_VECCHI or outputs/)
# OLD_RUN / NEW_RUN: the two narrative versions; CATS: the missed-fact categories listed
OLD = _out_path(os.environ.get("OLD_RUN", "slice_quality_inst"))
NEW = _out_path(os.environ.get("NEW_RUN", "slice_quality_inst_relabel"))
CATS = os.environ.get("CATS", "A").split(",")
MISSED = os.environ.get("MISSED", _out_path("slice_quality_inst", "claims_manual",
                                               "missed_facts_categories.json"))
FILES = ["error_narratives.json", "train36/error_narratives_train36.json", "synth/narratives_synth.json"]


def load(base):
    out = {}
    for f in FILES:
        p = os.path.join(base, f)
        if os.path.exists(p):
            for k, v in json.load(open(p, encoding="utf-8")).items():
                out[k] = v["sentences"] if isinstance(v, dict) else v
    return out


old, new = load(OLD), load(NEW)
lines = []
say = lambda s="": (lines.append(s), print(s))
changed = [k for k in new if old.get(k) != new[k]]
added = [(k, x) for k in new for x in new[k] if x not in (old.get(k) or [])]
kinds = {"patch on another object": r"^Parts? of the ground truth \w+ ha(ve|s) been labelled",
         "part of an object, its own sentence": r"^Parts? of the (\w+ )?\w+ (regions )?\(",
         "tiny bits on another object": r"^Besides, (bits|a bit) of the ground truth",
         "v6: a corner named": r" corner",
         "v6: the smaller defect too": r"and a smaller (spill|gap)",
         "v6: where the outline is ragged": r"especially towards the",
         "v6: two classes on one object": r"instead, and parts? ",
         "v6: small patches": r"small \w+ patch(es)? ha(ve|s) been hallucinated|\(\w+ small patch",
         "v6: holes of a correct region": r"^The \w+ mask has \w+ holes?"}
say(f"narratives: {len(new)} | changed: {len(changed)}")
for name, rx in kinds.items():
    say(f"  new sentences, {name}: {sum(1 for _k, x in added if re.search(rx, x))}")
gone = [(k, x) for k in new for x in (old.get(k) or []) if x not in new[k]]
say(f"  sentences removed (patches now told on their object): {len(gone)}")
cat = json.load(open(MISSED, encoding="utf-8"))
say()
say(f"CATEGORY {'/'.join(CATS)} facts and what the new narrative says about them")
for f in [f for f in cat["facts"] if f["category"] in CATS]:
    k = f["image"]
    say(f"\n{k} [{f['category']}, {f['set']}, before: {f['verdict']}] human: {f['fact']}")
    nw = [x for x in new.get(k, []) if x not in (old.get(k) or [])]
    for x in nw or ["(no new sentence)"]:
        say(f"   + {x}")
open(os.path.join(NEW, "compare.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
