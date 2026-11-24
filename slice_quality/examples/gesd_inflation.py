"""Why SliceLens's numbers swell on GESD: for every phrase SliceLens certifies there (slope rule, thesis gates), how
many of the class's errors its group takes, how often Qwen says yes (P(yes) >= 0.5) on the wrong and on the correct
objects of the class, the AUC, and whether the rank test certifies it. CPU only; reads what gesd_cmp.py reads (its top
half is run as is).
  -> outputs/slice_quality_gesd/gesd_inflation.{json,txt}
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "gesd_cmp.py"), encoding="utf-8").read()
exec(compile(src.split('out = {"n_expert"')[0], "gesd_cmp_top", "exec"))

rows, txt = [], []
for _b, c, q, ids in COLS["SliceLens, thesis gates"]:
    if _b != "language" or (c, q) not in W:
        continue
    sc = W[(c, q)][0]["scored"]
    err = [x for x in sc if x["is_error"]]
    ok = [x for x in sc if not x["is_error"]]
    a = auc([x["conf"] for x in sc], [x["is_error"] for x in sc])
    rows.append({"cls": c, "q": q, "members": len(ids), "n_err": len(err), "n_ok": len(ok),
                 "share_of_class_errors": len(ids) / max(len(err), 1),
                 "yes_err": sum(x["conf"] >= 0.5 for x in err) / max(len(err), 1),
                 "yes_ok": sum(x["conf"] >= 0.5 for x in ok) / max(len(ok), 1),
                 "auc": a, "sent_by_rank_sample": (c, q) in SET_A,
                 "rank_certified": bool(V.get((c, q), (0, False))[1])})
rows.sort(key=lambda r: (r["rank_certified"], -r["share_of_class_errors"]))
txt.append("SliceLens's certified phrases on GESD (slope rule, thesis gates): the class's errors in its group, Qwen's yes "
           "on wrong / correct objects, AUC, rank test")
for r in rows:
    txt.append(f"  {r['cls']:10s} {r['q'][:60]:60s} group {r['members']:4d} = {100 * r['share_of_class_errors']:5.1f}% of "
               f"{r['n_err']:4d} errors | yes: wrong {100 * r['yes_err']:5.1f}% correct {100 * r['yes_ok']:5.1f}% "
               f"({r['n_ok']} correct) | AUC {r['auc'] if r['auc'] is None else round(r['auc'], 3)} | rank "
               f"{'yes' if r['rank_certified'] else ('no' if r['sent_by_rank_sample'] else 'not past the sample')}")
os.makedirs(L, exist_ok=True)
json.dump(rows, open(f"{L}/gesd_inflation.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
open(f"{L}/gesd_inflation.txt", "w", encoding="utf-8").write("\n".join(txt) + "\n")
print("\n".join(txt))
print("GESD_INFLATION_DONE", flush=True)
