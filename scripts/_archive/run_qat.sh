#!/usr/bin/env bash
# Run a check/command in the container after installing compressed-tensors
# (needed to load Google's official QAT / compressed-tensors models). The
# install lands in the ephemeral --rm container, so it adds no persistent disk.
set -e
# Install compressed-tensors with ALL its deps (loguru, etc.), but a constraints
# file pins torch/torchvision so pip cannot replace the image's cu118 build with
# a cu13x one (which the AIRLab driver cannot run).
printf 'torch==2.7.1+cu118\ntorchvision==0.22.1+cu118\n' > /tmp/constr.txt
pip install -q -c /tmp/constr.txt compressed-tensors
exec python "$@"
