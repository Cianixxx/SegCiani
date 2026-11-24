# slice_quality — example runs

Runnable entry points for the pipeline. **Run them from the repo root** (paths
are relative to it). On the AIRLab server they run inside `segciani:dev`
with the VOC data mounted at `/data` (see `../../AIRLAB.md`). Superseded
scripts (the original `server_voc_*.py` chain, an older `rescore.py`) live in
`_archive/` — kept for provenance, not meant to be run again.

## The current pipeline, in the order it actually runs

| Stage | Script | Needs |
|---|---|---|
| 1. Attribute tagging (verification mode) + re-score | `retag_current_slices.py` | GPU + Qwen |
| Rebuild descriptions/gallery from a cached pickle | `regen_gallery.py` | CPU |
| Add SliceLens language slices to a CPU-cached run | `rescore_with_language.py` | CPU |
| Re-type a cached pickle after a taxonomy change (`SQ_CLAIM_COV=0.5`, `OUT` to a new dir) | `retype_pickle.py` | CPU |
| Re-type with GT instances = VOC objects (`SQ_GT_INSTANCES=object`); embeds + tags only the fresh GT instances | `retype_instances.py` | GPU + Qwen |
| Split-half replication | `split_half.py` | CPU |

## Narratives and the rubric judge (§9–10 of `docs/pipeline_implementation.docx`)

Every run directory (`SQ_OUT`) gets the same chain; the three that exist are
`outputs/slice_quality` (reference, IoU claims), `outputs/slice_quality_cov05b`
(containment claims) and `outputs/slice_quality_inst` (containment + object
instances, narrative v3 — the current best). The chain scripts that ran them
(`outputs/chain_*.sh`) are on the server.

| Stage | Script | Needs | Env |
|---|---|---|---|
| 1. Mask facts + per-image narrative, val + train36 | `narrate_errors.py` | CPU | `PKL`, `TRAIN36`, `OUT` |
| 2. Pattern descriptions of the 36 train images (induction on certified slices) | `describe_train36_gpu.py` | GPU + Qwen | `SQ_OUT`, `QWEN_4BIT` |
| 3. Synthetic set: Block 1 + narrative / pattern descriptions | `narrate_synth.py`, `describe_synth_gpu.py` | CPU / GPU | `SQ_OUT`, `OUT`, `STAGE` |
| 4. Rubric judge (Gemma-4-12B) on the 79 annotated images, negative controls shuffled in; resumes only where the candidate text is identical | `judge_descriptions.py` | GPU + Gemma | `NARR`, `DESC`, `DESC_TR`, `OUT`, `JUDGE_4BIT` |
| 4b. Score = sum of the three criteria (writes `*_scored.json`) | `rescore_judge.py` | CPU | `SRC`, `OUT` |
| 5. Judge on the synthetic set | `judge_synth.py` | GPU + Gemma | `NARR`, `DESC`, `OUT` |
| 6. Per-image Block 1 counts for the comparison page | `typed_per_image.py` | CPU | `SQ_OUT` |
| 7. Comparison page (`docs/`), share gallery, instance-ambiguity numbers | `../../docs/make_compare_html.py`, `make_share_gallery.py`, `measure_instance_ambiguity.py` | CPU, local | `SQ_OUT`, `JUDGE_FILE`, `SYNTH_JUDGE`, `OUT_HTML` |

`enrich_train36_details.py` is the pre-`mask_facts` way of adding the detail
line to the train36 cards; `describe_train36_gpu.py` now measures the facts
itself, so it is no longer needed on a fresh run.

## Validation experiments (§6 of `docs/pipeline_implementation.docx`)

| Experiment | Script |
|---|---|
| Synthetic corruption (known-failure recovery) | `corruption_experiment.py` |
| Description precision vs matched chance | `eval_description_precision.py` |
| Human-annotation comparison, LLM judge w/ negative controls | `eval_gt_agreement.py` |
| Purity permutation-null export (for the report figure) | `export_purity_null.py` |
| Panel images for report figures | `export_example_panels.py` |
| True (both-branch union) instance coverage, from any pickle | `recompute_true_coverage.py` |

## Second-model transfer experiment (§6.6, three stages)

| Stage | Script | Needs |
|---|---|---|
| 1. Run the stronger segmenter, write its masks | `predict_second_model.py` | GPU |
| 1b. Typed-error taxonomy on both models' masks | `compare_models_block1.py` | CPU |
| 2. Cluster branch on the second model | `run_second_model_cluster.py` | GPU (embeddings + tagging) |
| 3. Language branch re-grounded on the second model's instances | `../../examples/reground_validated.py` | GPU |
| 3b. Combine stage 2 + stage 3 into one two-branch result | `run_second_model_language.py` | CPU |

## Model-repair loop (new — written, not yet run)

Closes the loop back to the thesis's own title: use the diagnosis to actually
try to fix the model, on the single best-evidenced situation across both
models — "an object cut off at the image border." Two arms from the SAME
pretrained checkpoint, fine-tuned on VOC `train` (1464 images, untouched by
any diagnosis so far); `val` (1449 images, the set the whole thesis's evidence
is built on) is read only afterward, for re-diagnosis.

| Step | Script | Needs |
|---|---|---|
| Fine-tune model1 (POLICY=control or POLICY=repair) | `repair_model1_train.py` | GPU |
| Predict a fine-tuned checkpoint's masks on val | `repair_model1_predict.py` | GPU |

Then re-diagnosis reuses the SAME chain as the second-model transfer
experiment above (Block 1 on the new masks → `run_second_model_cluster.py`-style
cluster branch → `reground_validated.py` re-grounding the existing 35
hypotheses), pointed at `mobilenet_control_pred` / `mobilenet_repair_pred`
instead of `deeplab_pred`.

`../../scripts/run_slice_quality.py` is the COCO entry point (`build_coco_samples`).

## Run pattern (server, GPU)

```bash
ssh ely && cd ~/storage/SLICELENS         # book the GPU on the Google Sheet first
docker run --rm --gpus device=N -u $(id -u):$(id -g) -e HOME=/tmp -e PYTHONPATH=/workspace \
  -v $HOME/storage/SLICELENS:/workspace \
  -v $HOME/storage/pascal_vlm:/data \
  -v $STORAGE/models/qwen2.5-vl-7b:/models/qwen \
  -v $STORAGE/models/gemma-4-12B-it:/models/gemma \
  -w /workspace segciani:dev \
  bash -c "pip install --user scipy scikit-learn && python slice_quality/examples/<script>.py"
```

The Gemma mount is only needed by the two judge scripts. On the 11 GB cards
(GPU 0 and 4) run Qwen 4-bit (`QWEN_4BIT=1`) and the judge 4-bit
(`JUDGE_4BIT=1`); when another user's memory would make Qwen offload to CPU,
pin it with `SQ_QWEN_DEVICE_MAP=cuda:0`.

`PYTHONPATH=/workspace` is required (the scripts live in a subfolder). Drop
`--gpus` and the qwen mount for the CPU-only scripts. Reports + the HTML
gallery go under `outputs/slice_quality/` (see that folder's own `_archive/`
for early demo/exploration JSONs that predate the current pipeline).

> gotcha: `DINOv2Extractor` and `predict_second_model.py` both disable cuDNN
> (the image's torch+driver throws `CUDNN_STATUS_NOT_INITIALIZED` under
> driver 520); re-enable with `SQ_DINOV2_CUDNN=1` / `USE_CUDNN=1`.
> gotcha: Docker containers here are `--rm` — pip installs (scipy,
> scikit-learn) do **not** persist across runs; always reinstall.
