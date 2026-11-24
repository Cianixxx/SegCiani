"""Diagnose the HFVLM (Gemma) slice-confidence calibration.

Compares two grounding strategies on the same instances with a probe
hypothesis ('an animal') that *should* cleanly separate animals from vehicles:

  * bbox  -> region given as text coordinates (needs a grounding-tuned VLM)
  * crop  -> region given as an actual image crop (works for any VLM)

If 'crop' yields graded P(yes) (animals high, vehicles low) while 'bbox' stays
flat ~0, crop-grounding is the fix; if even crop stays flat, Gemma's yes/no is
unusable for SliceLens and we switch to Qwen2.5-VL.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from slicelens.adapters import load_voc_seg_instances  # noqa: E402
from slicelens.backends import HFVLM  # noqa: E402

ANIMALS = {"cat", "dog", "cow", "sheep", "horse", "bird", "person"}
VEHICLES = {"aeroplane", "boat", "train", "car", "bus", "motorbike", "bicycle"}


def pick(insts, names, k):
    return [s for s in insts if s.label in names][:k]


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/models/gemma")
    ap.add_argument("--dataset-root", default="/data/dataset")
    ap.add_argument("--crops-dir", default="/data/voc_crops")
    ap.add_argument("--hyp", default="an animal")
    args = ap.parse_args()

    vlm = HFVLM(model=args.model, grounding="bbox", load_in_4bit=True)
    insts = load_voc_seg_instances(args.dataset_root, args.crops_dir)
    sample = pick(insts, ANIMALS, 5) + pick(insts, VEHICLES, 5)

    print(f"\nprobe hyp = {args.hyp!r}   (animals SHOULD score high, vehicles low)\n")
    print(f"  {'label':11s} {'kind':8s} {'P(yes) bbox':>12s} {'P(yes) crop':>12s}  gen(crop)")
    for s in sample:
        kind = "animal" if s.label in ANIMALS else "vehicle"
        vlm.grounding = "bbox"
        p_bbox = vlm.slice_confidence(s.image_path, s.region, args.hyp)
        vlm.grounding = "crop"
        p_crop = vlm.slice_confidence(s.image_path, s.region, args.hyp)
        # what does it actually generate under crop?
        from slicelens.backends.hf_vlm import _VERIFY_SYSTEM_PROMPT
        from slicelens.prompts import CROP_VERIFICATION_PROMPT
        img = vlm._crop_region(s.image_path, s.region)
        inp = vlm._build_inputs(img, _VERIFY_SYSTEM_PROMPT,
                                CROP_VERIFICATION_PROMPT.format(query=args.hyp))
        with torch.no_grad():
            g = vlm.model.generate(**inp, max_new_tokens=4, do_sample=False)
        gen = vlm._tok.batch_decode(g[:, inp["input_ids"].shape[1]:],
                                    skip_special_tokens=True)[0].strip()
        print(f"  {s.label:11s} {kind:8s} {p_bbox:12.3f} {p_crop:12.3f}  {gen[:12]!r}")


if __name__ == "__main__":
    main()
