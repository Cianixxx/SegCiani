"""Why language slices are (not) certified: which gate of describe.build_image_descriptions
each one fails, per run. CPU. Run: python slice_quality/examples/language_gate_stats.py <run dir> ..."""
import pickle
import sys
from collections import Counter

from slice_quality.describe import _severity_z, _subtype_severity_stats

for run in sys.argv[1:]:
    res = pickle.load(open(f"{run}/result_tagged.pkl", "rb"))["result"]
    stats = _subtype_severity_stats(res)
    kept = {k.slice_.slice_id for k in res.kept}
    seen, rows = set(), []
    for s, q in res.scored:
        if s.source != "language" or q.slice_id in seen:
            continue
        seen.add(q.slice_id)
        conf = (sum(s.membership.values()) / len(s.membership)) if s.membership else 0.0
        sev = _severity_z(s.members, stats)
        sysf = (getattr(s, "metadata", None) or {}).get("is_systematic")
        ver = q.significant if sysf is None else bool(sysf)
        fails = [n for n, ok in (("not systematic (slope-trend)", ver), ("Qwen confidence < 0.6", conf >= 0.6),
                                 ("milder than its error type", sev >= -0.25), ("size < 3", q.size >= 3)) if not ok]
        rows.append((q.slice_id, q.size, fails, q.slice_id in kept))
    cert = [r for r in rows if not r[2] or r[3]]
    print(f"== {run}: {len(rows)} language slices, certified {len(cert)}")
    print("   failing gates (a slice may fail several):", dict(Counter(f for r in rows if not r[3] for f in r[2]).most_common()))
    only = Counter(r[2][0] for r in rows if len(r[2]) == 1 and not r[3])
    print("   slices failing ONE gate only:", dict(only.most_common()))
    sys_ok = [r for r in rows if "not systematic (slope-trend)" not in r[2]]
    print(f"   systematic: {len(sys_ok)} of {len(rows)} ({100 * len(sys_ok) / len(rows):.0f}%), of which certified {len(cert)}")
