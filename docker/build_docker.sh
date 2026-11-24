#!/usr/bin/env bash
# Build the SliceLens docker image (AIRLab naming convention: <surname>/<name>).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"

IMAGE_NAME="${IMAGE_NAME:-segciani}"
TAG="${TAG:-dev}"

echo "Building ${IMAGE_NAME}:${TAG} (context: ${ROOT}) ..."
docker build --rm -t "${IMAGE_NAME}:${TAG}" -f "${HERE}/Dockerfile" "${ROOT}"
echo "Done -> ${IMAGE_NAME}:${TAG}"
echo "List your images: docker images | grep $(whoami)"
