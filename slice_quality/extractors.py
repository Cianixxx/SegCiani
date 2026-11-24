"""BLOCK 2 backbone -- a frozen DINOv2 feature extractor (server-side).

Matches SEG4SEG's choice (DINOv2). Returns a dense (C, Hp, Wp) patch-feature map
per image, which BLOCK 2 (embeddings.mask_pool) pools over each instance mask.

Requires torch (+ torchvision transforms). First use downloads the model from
torch.hub; on an offline server, point ``DINOv2Extractor`` at a local checkpoint.

Where to modify
---------------
* ``model_name`` : "dinov2_vits14" (small) / "dinov2_vitb14" / ... .
* ``image_size`` : multiple of 14; larger -> finer feature grid, slower.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np


class DINOv2Extractor:
    """image_id -> (C, Hp, Wp) DINOv2 patch features (cached)."""

    def __init__(
        self,
        image_paths: dict[str, str],
        model_name: str = "dinov2_vits14",
        image_size: int = 518,          # 518 / 14 = 37 patches per side
        device: Optional[str] = None,
    ) -> None:
        import torch
        from torchvision import transforms

        # Some lab images ship a torch/cuDNN combo that throws
        # CUDNN_STATUS_NOT_INITIALIZED on conv2d with the installed driver.
        # DINOv2 is inference-only here, so disabling cuDNN (native conv) is a
        # safe, robust fix. Set SQ_DINOV2_CUDNN=1 to re-enable.
        if os.environ.get("SQ_DINOV2_CUDNN", "0") != "1":
            torch.backends.cudnn.enabled = False

        self.image_paths = image_paths
        self.image_size = image_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model.eval().to(self.device)
        self.patch = 14
        self.grid = image_size // self.patch
        self.tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        self._cache: dict[str, np.ndarray] = {}

    def feature_map(self, image_id: str) -> np.ndarray:
        if image_id in self._cache:
            return self._cache[image_id]
        import torch
        from PIL import Image

        img = Image.open(self.image_paths[image_id]).convert("RGB")
        x = self.tf(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model.forward_features(x)
            tokens = out["x_norm_patchtokens"][0]            # (grid*grid, C)
        C = tokens.shape[-1]
        fmap = tokens.reshape(self.grid, self.grid, C).permute(2, 0, 1)  # (C, Hp, Wp)
        arr = fmap.cpu().numpy().astype(np.float32)
        self._cache[image_id] = arr
        return arr
