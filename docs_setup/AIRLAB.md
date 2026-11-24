# Running SliceLens on AIRLab (PoliMi DEIB) servers

Tailored to the AIRLab onboarding ("Getting up to speed @ AIRLab"). It uses the
lab's **Docker** workflow and conventions. You run these steps; I cannot (they
need the VPN, your credentials and a GPU booking).

## Which server / GPU

Students get **`westworld`** and **`elysium`**. Qwen2.5-VL-7B needs ~16 GB in
bf16:

| Server | GPUs | SliceLens VLM mode |
| --- | --- | --- |
| **`elysium`** (`ely`, <server-ip>) | 4× RTX6000 **24 GB** | bf16 (default) — **preferred** |
| **`westworld`** (`ww`, <server-ip>) | 8× 1080Ti **11 GB** | **`--load-in-4bit`** (NF4) |

The LLM half (Gemini) runs via API, so it needs no GPU — only `GOOGLE_API_KEY`
and network (the VPN allows outbound).

## 0. One-time setup

1. **VPN** (if off the AIRLab LAN): GlobalProtect, gateway
   `gp-deib-saml.vpn.polimi.it` (see the deck's "Configure VPN" slide). Ask your
   co-advisor for server credentials + the Discord invite.
2. **SSH nicknames** — add to `~/.ssh/config` (and `/etc/hosts`):
   ```
   Host ely
     HostName <server-ip>
     User <your_username>
   Host ww
     HostName <server-ip>
     User <your_username>
   ```
   Then `ssh ely`. Add an SSH key (`ssh-copy-id ely`) to avoid passwords.
3. **GitHub on the server**: generate an SSH key on the server and add it to
   GitHub so you can `git clone`/`git pull` your repo.

## 1. Put the code on NAS-backed storage

Code goes under `~/storage` (a link to the `multiverse` NAS — visible from all
servers; keep your home otherwise empty):

```bash
ssh ely
cd ~/storage
git clone <your-fork-of-SLICELENS>.git SLICELENS    # -> ~/storage/SLICELENS
```

Datasets go under `~/dataset/private` (also NAS). Put COCO there, e.g.:

```
~/dataset/private/coco/val2017/                      # images
~/dataset/private/coco/annotations/instances_val2017.json
~/dataset/private/coco/preds/faster_rcnn.bbox.json   # your detector's results
```

## 2. Build the image

```bash
cd ~/storage/SLICELENS
bash docker/build_docker.sh          # -> <username>/slicelens:dev
```

(First build downloads the CUDA/PyTorch base + installs transformers, Qwen
deps, bitsandbytes, google-genai. Takes a while.)

## 3. Book + pick a free GPU

- Reserve your GPU/CPU in the **per-server Google Sheet** (links in the
  onboarding deck / Discord) — booking is manual, there is no scheduler.
- Find a free GPU index and free cores:
  ```bash
  nvidia-smi          # pick a GPU with ~0% util and free memory
  htop                # see free CPU cores
  ```

## 4. Run the container

`docker/run_docker.sh` bind-mounts your code at `~/exp`, your datasets at
`~/exp/datasets`, sets `HF_HOME` on the NAS (so the ~16 GB weights download
once and are shared), runs as your uid/gid, and passes `GOOGLE_API_KEY`:

```bash
export GOOGLE_API_KEY=...                 # for Gemini hypothesis generation
GPU_DEVICE=2 CPU_SET=0-7 bash docker/run_docker.sh
```

Then open a shell in it (or attach VS Code — see §7):

```bash
docker exec -it <username>_slicelens_<timestamp> bash
```

## 5. Smoke-test the VLM

Inside the container:

```bash
python scripts/check_qwen.py                 # elysium (24 GB)
python scripts/check_qwen.py  # add 4-bit on westworld via the runner flag below
```

Expect `[PASS] slice confidence discriminates the matching query.`

## 6. Run SliceLens on COCO

Inside the container (paths are the in-container mount points):

```bash
# On elysium (24 GB) — bf16:
python examples/run_coco.py \
    --gt        datasets/coco/annotations/instances_val2017.json \
    --results   datasets/coco/preds/faster_rcnn.bbox.json \
    --image-root datasets/coco/val2017 \
    --label bicycle \
    --out outputs/bicycle_slices.json

# On westworld (11 GB) — add 4-bit quantization:
python examples/run_coco.py ... --label bicycle --load-in-4bit --out outputs/bicycle_slices.json
```

The JSON report (each systematic slice's slope, error-rate curve, top-ranked
instances with image path + bbox + confidence) lands in `~/storage/SLICELENS/outputs`
on the host — ready for your thesis figures.

Tips: `--no-data-driven` for a fast first pass; `--score-threshold 0.3` to prune
low-score detections (the VLM is called once per instance × hypothesis, so this
dominates runtime).

**Instance segmentation** works the same way — generate mask predictions with
Mask R-CNN, then add `--task segmentation`:

```bash
python scripts/make_coco_predictions.py --model maskrcnn_resnet50_fpn \
    --images-dir datasets/coco/val2017 \
    --gt datasets/coco/annotations/instances_val2017.json \
    --out-results datasets/coco/preds/maskrcnn.seg.json \
    --only-label bicycle --limit 120 --score-threshold 0.3
python examples/run_coco.py --task segmentation --label bicycle \
    --gt datasets/coco/annotations/instances_bicycle.json \
    --results datasets/coco/preds/maskrcnn.seg.json \
    --image-root datasets/coco/val2017 --out outputs/bicycle_seg.json
```

## 6b. Run with an open VLM — Gemma 4 12B

`--vlm-backend qwen` (Qwen2.5-VL) is the paper's model. To run a *different* open
VLM — e.g. **Gemma 4 12B** (multimodal, grounding-capable) — use the generic
`--vlm-backend hf` backend: it reads the yes/no logits the same way, so the slice
confidence (Eq. 1) stays continuous.

Gemma 4 12B fits **elysium (24 GB) in 4-bit** (~8 GB). It is too big for
westworld 11 GB — there use a smaller variant (`google/gemma-4-e4b-it`).

```bash
# one-time: Gemma 4 needs a recent transformers; the HF repo is gated
python -m pip install -U transformers accelerate bitsandbytes pillow
huggingface-cli login          # accept the Gemma licence on HF

python examples/run_coco.py --task segmentation \
    --vlm-backend hf --vlm-model google/gemma-4-12B-it --load-in-4bit \
    --gt datasets/coco/annotations/instances_bicycle.json \
    --results datasets/coco/preds/maskrcnn.seg.json \
    --image-root datasets/coco/val2017 --label bicycle \
    --max-instances 100 --small-data --out outputs/bicycle_seg_gemma12b.json
```

Higher-quality / lighter 4-bit: the official QAT build
`google/gemma-4-12B-it-qat-w4a16-ct` (use *without* `--load-in-4bit`; needs
`pip install compressed-tensors`).

**Compare VLMs** (a nice thesis result): run the same command once with
`--vlm-backend qwen` and once with `--vlm-backend hf --vlm-model
google/gemma-4-12B-it`, then diff the two reports.

## 7. Develop with VS Code attached to the container

1. VS Code → Remote-SSH → connect to `ely`/`ww`.
2. Run `docker/run_docker.sh` (the `-d -it` container keeps running).
3. Remote Explorer → **Attach to Running Container** → pick
   `<username>_slicelens_<timestamp>`. Your project is at `~/exp`; `Ctrl+J` for a
   terminal. Notebooks: install the Jupyter extension *in the container* and pick
   the `base` Python env.

## 8. Long runs survive disconnects — use tmux

```bash
ssh ely
tmux new -s slicelens
GPU_DEVICE=2 bash docker/run_docker.sh
docker exec -it <container> bash -lc 'cd ~/exp && python examples/run_coco.py ... | tee outputs/run.log'
# detach: Ctrl+b then d     |  reattach later: tmux attach -t slicelens
```

## 8b. The slice_quality chains (thesis runs)

The thesis pipeline (`slice_quality/`) runs as detached, self-removing
containers driven by a chain script, one per run directory, so a run survives
the SSH session and leaves a log:

```bash
docker run -d --rm --name chain_x --gpus device=N -u $(id -u):$(id -g) -e HOME=/tmp   -v $HOME/storage/SLICELENS:/workspace -v $HOME/storage/pascal_vlm:/data   -v $STORAGE/models/qwen2.5-vl-7b:/models/qwen   -v $STORAGE/models/gemma-4-12B-it:/models/gemma   -w /workspace segciani:dev bash -lc "bash outputs/chain_x.sh > outputs/chain_x.log 2>&1"
```

The chains that exist (`outputs/chain_*.sh`) and what each run directory is are
listed in `slice_quality/examples/README.md`. GPU notes: 0 and 4 are 11 GB
(GTX 1080 Ti) — Qwen and the judge must run 4-bit there; 1/2/5/6 are 24 GB.
Never use a GPU that is not booked for you.

## 9. Clean up when done (free resources!)

```bash
docker stop <container>          # --rm auto-removes it
docker rmi <username>/slicelens:dev   # remove the image if you no longer need it
```
Release the GPU in the booking sheet.

---

### Notes / gotchas
- **CUDA version:** the image targets CUDA 11.8 (matches the driver in the deck).
  If `nvidia-smi` on your server shows newer CUDA, you may switch the base image
  in `docker/Dockerfile` to a cu121 build for speed.
- **HF cache on NAS:** `HF_HOME=~/exp/.hf-cache` keeps the big weights on
  `multiverse`, so they download once and are reused on every server/run.
- **Throughput:** this uses HuggingFace `transformers` (one forward pass per
  instance for the yes/no logits). For very large runs the paper uses vLLM —
  a vLLM backend can be added behind the same `VLMBackend` interface later.
