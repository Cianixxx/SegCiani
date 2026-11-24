"""Write <SQ_OUT>/cluster_names.json: the short name of every cluster slice of a run
(slice_quality/naming.py, the same names the gallery shows), for the pages built elsewhere
(docs/make_compare_html.py). CPU, reads the run's pickle only.

Run: SQ_OUT=outputs/slice_quality_inst python slice_quality/examples/write_cluster_names.py
"""
import json
import os
import pickle

from slice_quality.naming import cluster_names

SQ_OUT = os.environ.get("SQ_OUT", "outputs/slice_quality_inst")
res = pickle.load(open(f"{SQ_OUT}/result_tagged.pkl", "rb"))["result"]
names = cluster_names(res.scored)
json.dump(names, open(f"{SQ_OUT}/cluster_names.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(f"{len(names)} cluster names -> {SQ_OUT}/cluster_names.json")
