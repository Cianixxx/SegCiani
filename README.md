# SegCiani — Why does a segmentation model fail?

Code of the MSc thesis *"Why Does a Segmentation Model Fail? Error Types, Error Slices and VLM Descriptions"*
(Daniele Spini, Politecnico di Milano, 2026; advisor Prof. Matteo Matteucci, co-advisor Nico Catalano).

A semantic segmenter is usually judged by one number, the mean IoU. SegCiani turns that number into a diagnosis:
for every image on which the model fails it says **what** went wrong, object by object, **how** (the type of error),
and **when** (the situation in which the failure tends to repeat).

| | |
|---|---|
| **Block 1: typed errors** | Matches predicted regions to ground-truth objects from the masks only and gives every wrong object one of seven types: missed, hallucinated, merged, split, wrong class, imprecise boundary, wrong extent. |
| **Cluster branch** | Groups the errors of each type by appearance (DINOv2) and error geometry with a Gaussian mixture, with no vocabulary. |
| **Language branch** | A language model writes situations per class; Qwen2.5-VL checks each one object by object. |
| **Rank test** | A situation is kept if the VLM is more confident on the wrong objects than on the correct ones (AUC > 0.5, image bootstrap, Benjamini–Hochberg), in place of the slope rule of SliceLens. |
| **Descriptions** | Every image with errors gets a text: geometry from the masks, the situation from the significant slice it belongs to. |

Main results (PASCAL VOC val, LR-ASPP MobileNetV3): 101 situations and 4 clusters are significant and hold 39.5% of the
4,967 wrong objects, against 20.7% for SliceLens on the same data, VLM and hypotheses (SBD: 45.7% vs 13.1%).

## Repository layout

| Folder | What is in it |
|---|---|
| `slice_quality/` | The SegCiani pipeline: typed errors (`typed_error.py`, `mask_facts.py`), clustering and slice tests, language branch, descriptions (`describe_errors.py`), gallery. `slice_quality/examples/` holds the entry points of every run and validation; its README lists them in the order they run. |
| `slicelens/` | Our reimplementation of SliceLens v1 (hypothesis generation, VLM grounding, slope rule), used as the comparison. The original code is not public. |
| `examples/`, `scripts/` | Hypothesis generation, data preparation, utilities. `scripts/server_chains/` are the shell chains used to run the experiments in Docker on the AIRLab GPU server (paths replaced by `$HOME`, `$STORAGE`, `$DATASETS`). |
| `docker/` | Dockerfile and helper scripts (`build_docker.sh`, `run_docker.sh`). |
| `analysis/` | Scripts that read the run outputs and produce tables, comparison pages and figures. |
| `thesis_figures/` | The figure builders of the thesis and `make_numbers.py`, which writes every number quoted in the text. |
| `tests/` | Unit tests. |
| `docs_setup/` | Setup notes: first run (`RUN.md`), server setup, AIRLab server recipe. |

## Requirements

- Python 3.10 (`pip install -e .` or `pip install -r requirements.txt`), PyTorch with CUDA for the GPU stages.
- Models (not included): DINOv2 ViT-S/14, Qwen2.5-VL-7B-Instruct (24 GB GPU in bf16, or 11 GB in 4-bit),
  Gemma for the hypotheses and the automatic judges (the scripts in `examples/` can also use Gemini via API).
- Datasets (not included): PASCAL VOC 2012, SBD, Pascal-EA (Yin et al., CVPR 2024), GESD (Zhang et al., ECCV 2026).
  Download them from their authors.

## Running

Start from `docs_setup/RUN.md` for a first run, then `slice_quality/examples/README.md` for the full pipeline in
order. The chains in `scripts/server_chains/` show the exact sequence and environment variables of each experiment
in the thesis. Paths are relative to the repository root; data are expected under `/data` inside the container.

## Related work

- SliceLens / GH-ESD: W. Zhang et al. (Apple), *Grounded Hypothesis-Driven Error Slice Discovery for Instance-Level
  Vision Tasks*, ECCV 2026, arXiv:2512.24592.
- Pascal-EA: Z. Yin et al., *Benchmarking Segmentation Models with Mask-Preserved Attribute Editing*, CVPR 2024.

## Citation

```bibtex
@mastersthesis{spini2026segciani,
  author = {Daniele Spini},
  title  = {Why Does a Segmentation Model Fail? Error Types, Error Slices and VLM Descriptions},
  school = {Politecnico di Milano},
  year   = {2026}
}
```
