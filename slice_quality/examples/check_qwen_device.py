"""Does Qwen on this GPU give the groundings of the thesis runs? CPU-offloaded bf16 on an 11 GB card (GTX 1080 Ti)
against the bf16 RTX 6000 values already on disk: the same objects, the same query, the same call.

Run (GPU container, Qwen mounted): SQ_QWEN_DEVICE_MAP=auto python slice_quality/examples/check_qwen_device.py
Env: RUN (a voc_paper_<cls>_cc.json with groundings), QUERY, SUBSTRATE, N (24), QWEN_4BIT (0)
Prints the largest and mean absolute difference, how many objects change side of 0.5, and seconds per call.
"""
import json
import os
import pickle
import time

import numpy as np

from slicelens.backends import QwenVLM
from slice_quality.adapters.to_slicelens import typed_to_slicelens_instances

RUN = os.environ.get("RUN", "/data/outputs_dd_ground/voc_paper_bird_cc.json")
QUERY = os.environ.get("QUERY", "a bird in deep shadow with high contrast")
SUBSTRATE = os.environ.get("SUBSTRATE", "/data/voc2012/substrate_cc.pkl")
N = int(os.environ.get("N", "24"))

r = json.load(open(RUN))
old = {s["instance_id"]: s["conf"] for c in r["candidates"] if c["query"] == QUERY for s in c["scored"]}
d = pickle.load(open(SUBSTRATE, "rb"))
insts = [i for i in typed_to_slicelens_instances(d["errors"], d["reference"], d["image_paths"], only_label=r["label"])
         if i.instance_id in old]
# the most informative objects: those the RTX values put on both sides of 0.5, and near it
insts = sorted(insts, key=lambda i: abs(old[i.instance_id] - 0.5))[:N]
t0 = time.time()
vlm = QwenVLM(model="/models/qwen", grounding="bbox", load_in_4bit=os.environ.get("QWEN_4BIT", "0") == "1")
print(f"Qwen loaded in {time.time() - t0:.0f}s, device map: "
      f"{sorted({str(v) for v in getattr(vlm.model, 'hf_device_map', {'all': 'single'}).values()})}", flush=True)
a, b = [], []
t1 = time.time()
for i in insts:
    a.append(old[i.instance_id])
    b.append(float(vlm.slice_confidence(i.image_path, i.region, QUERY)))
dt = (time.time() - t1) / len(insts)
a, b = np.array(a), np.array(b)
diff = np.abs(a - b)
print(f"{len(insts)} objects | max |diff| {diff.max():.4f} | mean {diff.mean():.4f} | side of 0.5 changed "
      f"{int(((a >= 0.5) != (b >= 0.5)).sum())} | {dt:.1f} s per call", flush=True)
for x, y in zip(a, b):
    print(f"  {x:.4f} -> {y:.4f}")
print("CHECK_QWEN_DONE", flush=True)
