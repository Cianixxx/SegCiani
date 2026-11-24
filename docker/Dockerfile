# SliceLens image for AIRLab GPU servers.
#
# Base = PyTorch + CUDA 11.8 to match the AIRLab driver (nvidia-smi shows
# "CUDA Version: 11.8" on the servers). If `nvidia-smi` on your booked server
# reports a newer CUDA, you may bump this to a cu121 base for speed.
#
# Build context must be the repo root:
#   docker build --rm -t $(whoami)/slicelens:dev -f docker/Dockerfile .
# (or just run docker/build_docker.sh)
FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1

# System packages: git (sync), tmux/htop (lab workflow), libGL/glib (image IO).
RUN apt-get update && apt-get install -y --no-install-recommends \
        git tmux htop ca-certificates libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install SliceLens + real backends (Gemini, Qwen2.5-VL, 4-bit quant).
# Copy only what pip needs so layer caching survives source edits; the *live*
# code is bind-mounted at run time and takes precedence via PYTHONPATH.
COPY pyproject.toml README.md /opt/slicelens/
COPY slicelens /opt/slicelens/slicelens
# Install SliceLens + all real-backend deps. Upgrade transformers to the latest
# so it knows recent architectures (e.g. Gemma 4) for the generic HFVLM backend.
# Then RE-PIN torch/torchvision to the CUDA 11.8 build (cu118 = latest is 2.7.1)
# matching the AIRLab driver (520 / CUDA 11.8); otherwise the transformers
# upgrade drags in a cu13x torch the driver cannot run.
RUN pip install "/opt/slicelens[gemini,qwen,quant,segmentation,detector]" \
    && pip install -U transformers accelerate timm \
    && pip install --force-reinstall --no-deps torch==2.7.1 torchvision==0.22.1 \
         --index-url https://download.pytorch.org/whl/cu118

# AIRLab runs containers as `-u <uid>:<gid>`. Make /etc/passwd writable so the
# entrypoint can register that uid (prevents getpwuid() errors in some libs).
RUN chmod 0666 /etc/passwd
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
