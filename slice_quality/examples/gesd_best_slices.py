"""Which slice earns each expert slice's Precision@10 on GESD, for SliceLens (verdict only) and the pipeline: its
branch, class, phrase, size and the expert objects among its first 10. Also, per class, the errors, the expert slices
and the slices of each column. Reads what gesd_cmp.py MODE=report reads (its top half is run as is). CPU only.
  -> outputs/slice_quality_gesd/gesd_best_slices.{txt,json}
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "gesd_cmp.py"), encoding="utf-8").read()
exec(compile(src.split('out = {"n_expert"')[0], "gesd_cmp_top", "exec"))

res = {"expert": {}, "per_class": {}}
lines_ = []
for col in ("SliceLens, verdict only", "pipeline"):
    lines_.append(f"== {col}")
    for s in EXPERT:
        sl = COLS[col]
        best = max(sl, key=lambda o: p_at_k(o[3], s), default=None)
        p = p_at_k(best[3], s) if best else 0.0
        row = {"p10": p, "branch": best[0] if best else None, "cls": best[1] if best else None,
               "name": best[2] if best else None, "size": len(best[3]) if best else 0,
               "names_it": bool(best and best[0] == "language" and names(best[1], best[2], kw[s])),
               "n_slices_any": sum(p_at_k(o[3], s) > 0 for o in sl)}
        res["expert"].setdefault(s, {})[col] = row
        lines_.append(f"  {s[:52]:52s} {p:.1f}  {row['branch'] or '-':8s} {str(row['cls']):10s} "
                      f"size {row['size']:4d}  names {int(row['names_it'])}  slices>0 {row['n_slices_any']:2d}  "
                      f"{str(row['name'])[:70]}")
for s in EXPERT:
    res["expert"][s]["class"] = kw[s]["class"]
    res["expert"][s]["size"] = len(slice_units[s])
for c in CLASSES:
    ex = [s for s in EXPERT if kw[s]["class"] == c]
    res["per_class"][c] = {"expert_slices": ex, "errors": sum(len(slice_units[s]) for s in ex),
                           "SliceLens": sum(o[1] == c for o in COLS["SliceLens, verdict only"] if o[0] == "language"),
                           "pipeline_language": sum(o[1] == c for o in COLS["pipeline"] if o[0] == "language")}
    r = res["per_class"][c]
    lines_.append(f"{c:11s} errors {r['errors']:4d}  expert slices {len(ex)}  SliceLens slices {r['SliceLens']:2d}  "
                  f"pipeline phrase slices {r['pipeline_language']:2d}")
os.makedirs(L, exist_ok=True)
json.dump(res, open(f"{L}/gesd_best_slices.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
open(f"{L}/gesd_best_slices.txt", "w", encoding="utf-8").write("\n".join(lines_) + "\n")
print("\n".join(lines_))
print("GESD_BEST_DONE", flush=True)
