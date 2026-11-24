"""Giudice LLM (Gemma): classifica le 80 descrizioni manuali nei 7 pattern."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from slicelens.backends import HFLLM

PATTERNS = ["Boundary Imprecision", "Object Hallucination", "Under-segmentation",
            "Class Confusion", "Missing Instances", "Over-segmentation", "Detail Loss"]
DATA = "/data/dataset"
gts = {}
for line in open(DATA + "/answer_gts.jsonl", encoding="utf-8"):
    line = line.strip()
    if line:
        o = json.loads(line); gts[o["img_idx"]] = o["content"]
idx2name = {int(k): v.replace(".jpg", "") for k, v in
            json.load(open(DATA + "/voc_patch_matches.json", encoding="utf-8")).items()}

print("Loading Gemma (LLM, 4-bit) ...", flush=True)
llm = HFLLM(model="/models/gemma", load_in_4bit=True, max_new_tokens=64)
SYS = ("You label a semantic-segmentation error description with error categories. "
       "Allowed categories (use these exact names): " + "; ".join(PATTERNS) + ". "
       "A description can map to several. Output ONLY the categories that clearly apply, "
       "comma-separated, exactly as written above. No explanation.")
out = {}
for idx, text in sorted(gts.items()):
    name = idx2name.get(idx, f"idx{idx}")
    resp = llm.chat(SYS, f'Description: "{text.strip()}"')
    found = [p for p in PATTERNS if p.lower() in resp.lower()]
    out[name] = {"idx": idx, "patterns": found, "llm_raw": resp.strip()[:160]}
    print(name, found, flush=True)
Path("/data/outputs/descriptions_patterns_llm.json").write_text(
    json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
print("WROTE", len(out), "descriptions")
