"""Caption (Qwen) delle 3 istanze di discrepanza."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from slicelens.adapters import load_voc_seg_instances
from slicelens.backends import QwenVLM

TARGETS = [("2008_000073", "cow"), ("2011_003151", "chair"), ("2010_003531", "diningtable")]
print("Loading Qwen2.5-VL ...", flush=True)
vlm = QwenVLM(model="/models/qwen", grounding="bbox", load_in_4bit=True)
insts = load_voc_seg_instances("/data/dataset", "/data/voc_crops")
bykey = {(i.metadata.get("name"), i.label): i for i in insts}
out = {}
attr = "the object and any reason its segmentation mask may be inaccurate (occlusion, size, contrast, similarity to another class)"
for name, lab in TARGETS:
    i = bykey.get((name, lab))
    if i is None:
        out[f"{name}_{lab}"] = "(istanza non trovata)"; print(name, lab, "NOT FOUND", flush=True); continue
    cap = vlm.caption_region(i.image_path, i.region, attr)
    out[f"{name}_{lab}"] = cap.strip()
    print(name, lab, "->", cap[:140].replace("\n", " "), flush=True)
Path("/data/outputs/discrepancy_captions.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
print("WROTE /data/outputs/discrepancy_captions.json")
