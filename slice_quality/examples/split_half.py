"""Split-half replication of the cluster branch, from the cached pickle. CPU.

Run:  PYTHONPATH=/workspace python slice_quality/examples/split_half.py
Env:  SQ_OUT, the run whose pickle is split (default outputs/slice_quality); split_half.json goes beside it.
"""
import json
import os
import pickle

from slice_quality.config import SliceQualityConfig
from slice_quality.validate_split import run_split_half

SQ_OUT = os.environ.get("SQ_OUT", "outputs/slice_quality")
PKL = f"{SQ_OUT}/result_tagged.pkl"

d = pickle.load(open(PKL, "rb"))
errors, reference = d["errors"], d["reference"]
print(f"loaded {len(errors)} errors, {len(reference)} reference", flush=True)

cfg = SliceQualityConfig()
cfg.def1.n_min_frac = 0.005
cfg.significance.n_boot = 200

report = run_split_half(errors, reference, cfg)
with open(f"{SQ_OUT}/split_half.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

for fold in report["folds"]:
    print(f"\n[{fold['fold']}] discovered {fold['discovered']}, "
          f"replication rate {fold['replication_rate']}")
    for s in fold["slices"]:
        print(f"   {s['slice_id']:28s} train={s['n_train']:4d} test={s['n_test']:4d} "
              f"{'REPLICATED' if s['replicated'] else '-'}")
print(f"\nOVERALL replication rate: {report['replication_rate']} "
      f"({report['n_replicated']}/{report['n_discovered']})")
print("SPLIT_HALF_DONE", flush=True)
