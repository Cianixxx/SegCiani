"""Drop the judgements whose REFERENCE text is no longer the one on file.

A judge's resume cache is keyed on the texts it was given. The candidate side is
checked by the judges themselves; the reference side -- the hand-written
annotation -- can also change, and on 2026-09-29 six of them did, when the class
names the masks contradicted were corrected. Older records carry no fingerprint
of the reference (``gt_hash`` / ``err_hash`` were added with this change), so for
them nothing in the file can tell whether the annotation moved.

This script settles it from the data instead of a hardcoded list: it diffs the
current annotation files against the backup they replaced, takes the images whose
text actually changed, and removes from each judge file the records that used
that text as the reference. The next run of the judge re-asks exactly those and
reuses everything else.

Which records use the annotation as reference:
    judge_descriptions / judge_synth   every record (both kinds, both pairings:
                                       a control pair is the SAME annotation
                                       against another image's text)
    judge_cause*                       only kind="human"; the "pipeline" records
                                       are judged against the narrative

Run (CPU, seconds):
    SQ_OUT=outputs/slice_quality_inst python slice_quality/examples/invalidate_judgements.py
Env: SQ_OUT, ANN/ANN_BAK and SYN_ANN/SYN_ANN_BAK (defaults below), DRY=1.
"""
import glob
import json
import os
import shutil

SQ_OUT = os.environ.get("SQ_OUT", "outputs/slice_quality_inst")
ANN = os.environ.get("ANN", "_evaldata/dataset/answer_gts.jsonl")
MAP = os.environ.get("MAP", "_evaldata/dataset/voc_patch_matches.json")
SYN_ANN = os.environ.get("SYN_ANN", "datasets/cropped_synthetic_masks/answer_gts.jsonl")
SYN_MAP = os.environ.get("SYN_MAP", "datasets/cropped_synthetic_masks/voc_patch__id_matches.json")
DRY = os.environ.get("DRY") == "1"


def latest_backup(path):
    b = sorted(glob.glob(path + ".bak_*"))
    return b[-1] if b else None


def read(path, mapping):
    mp = {int(k): v.replace(".jpg", "") for k, v in
          json.load(open(mapping, encoding="utf-8")).items()}
    out = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            o = json.loads(line)
            n = mp.get(o["img_idx"])
            if n:
                out[n] = " ".join(o["content"].split())
    return out


changed = set()
for ann, mapping in ((ANN, MAP), (SYN_ANN, SYN_MAP)):
    bak = latest_backup(ann)
    if not bak:
        print(f"nessun backup per {ann}: niente da confrontare")
        continue
    now, before = read(ann, mapping), read(bak, mapping)
    diff = {n for n in now if n in before and now[n] != before[n]}
    gone = set(before) - set(now)
    print(f"{os.path.basename(ann)}: {len(diff)} annotazioni cambiate, "
          f"{len(gone)} sparite (rispetto a {os.path.basename(bak)})")
    for n in sorted(diff):
        print("   cambiata:", n)
    for n in sorted(gone):
        print("   sparita: ", n)
    changed |= diff | gone

if not changed:
    raise SystemExit("nessuna annotazione cambiata: niente da invalidare")

FILES = [
    (f"{SQ_OUT}/judge_descriptions_scored.json", None),
    (f"{SQ_OUT}/judge_descriptions_v3.json", None),
    (f"{SQ_OUT}/synth/judge_synth.json", None),
    (f"{SQ_OUT}/judge_cause.json", {"human"}),
    (f"{SQ_OUT}/synth/judge_cause_synth.json", {"human"}),
]

for path, kinds in FILES:
    if not os.path.exists(path):
        continue
    data = json.load(open(path, encoding="utf-8"))
    rows = data["judgements"]
    keep = [r for r in rows
            if not (r["image"] in changed and (kinds is None or r.get("kind") in kinds))]
    n = len(rows) - len(keep)
    print(f"{os.path.basename(path):34s} {n:3d} giudizi da rifare, {len(keep)} conservati"
          + (" (DRY)" if DRY else ""))
    if n and not DRY:
        if not os.path.exists(path + ".bak_ann"):
            shutil.copy2(path, path + ".bak_ann")
        data["judgements"] = keep
        json.dump(data, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("INVALIDATE_DONE")
