#!/usr/bin/env bash
# Run the SliceLens container on an AIRLab GPU server.
#
# !! Book the GPU/CPU first in the per-server Google Sheet, then pick the free
#    GPU index with `nvidia-smi`.
#
# Usage:
#   GPU_DEVICE=2 CPU_SET=0-7 docker/run_docker.sh
#   (override CODE_FOLDER / DATASET_FOLDER / TAG via env vars as needed)
set -euo pipefail

# ----------------------------- configure ------------------------------------
GPU_DEVICE="${GPU_DEVICE:-0}"          # GPU index from `nvidia-smi` (booked!)
CPU_SET="${CPU_SET:-0-7}"              # CPU cores, e.g. 0-7
TAG="${TAG:-dev}"
# Live code on the NAS-backed storage (edit on host / VS Code, shared by all servers):
CODE_FOLDER="${CODE_FOLDER:-/home/$(whoami)/storage/SLICELENS}"
# Datasets: COCO images + annotations + detection-results json:
DATASET_FOLDER="${DATASET_FOLDER:-/home/$(whoami)/dataset/private}"
# ----------------------------------------------------------------------------

IMAGE_NAME="$(whoami)/slicelens"
CONTAINER_NAME="$(whoami)_slicelens_$(date +%Y%m%d_%H%M%S)"
WORKDIR="/home/$(whoami)/exp"
UID_N="$(id -u)"
GID_N="$(id -g)"

if [ ! -d "${CODE_FOLDER}" ]; then
    echo "ERROR: CODE_FOLDER '${CODE_FOLDER}' not found. Clone the repo there or set CODE_FOLDER." >&2
    exit 1
fi

docker run -d -it --rm \
    --gpus "device=${GPU_DEVICE}" \
    --cpuset-cpus "${CPU_SET}" \
    --shm-size=8g \
    -w "${WORKDIR}" \
    --mount "type=bind,source=${CODE_FOLDER},target=${WORKDIR}" \
    --mount "type=bind,source=${DATASET_FOLDER},target=${WORKDIR}/datasets" \
    -e HOST_UID="${UID_N}" -e HOST_GID="${GID_N}" -u "${UID_N}:${GID_N}" \
    -e HOME="${WORKDIR}" \
    -e PYTHONPATH="${WORKDIR}" \
    -e HF_HOME="${WORKDIR}/.hf-cache" \
    -e GOOGLE_API_KEY="${GOOGLE_API_KEY:-}" \
    --name "${CONTAINER_NAME}" \
    "${IMAGE_NAME}:${TAG}"

cat <<EOF
Started container: ${CONTAINER_NAME}
  GPU device ${GPU_DEVICE} | CPUs ${CPU_SET}
  code     ${CODE_FOLDER}  -> ${WORKDIR}
  datasets ${DATASET_FOLDER} -> ${WORKDIR}/datasets
  HF cache ${WORKDIR}/.hf-cache  (persisted on the NAS)

Next:
  Open a shell:        docker exec -it ${CONTAINER_NAME} bash
  Attach VS Code:      Remote Explorer -> Dev Containers -> Attach to '${CONTAINER_NAME}'
  Smoke-test the VLM:  docker exec -it ${CONTAINER_NAME} python scripts/check_qwen.py
  Stop (auto-removed): docker stop ${CONTAINER_NAME}
EOF
