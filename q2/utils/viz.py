from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torchvision.utils import make_grid, save_image


def save_image_grid(
    *,
    images: torch.Tensor,
    path: str | Path,
    nrow: int = 8,
    normalize: bool = True,
    value_range: Optional[tuple[float, float]] = None,
) -> str:
    """
    Save an image grid to disk.
    images: Tensor in shape (B,C,H,W), expected in [0,1] if normalize=False.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    grid = make_grid(images.detach().cpu(), nrow=nrow, normalize=normalize, value_range=value_range)
    save_image(grid, path)
    return str(path)


def denormalize_cifar10(x: torch.Tensor) -> torch.Tensor:
    # Inverse of q2.utils.data.cifar10_normalization()
    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010], device=x.device).view(1, 3, 1, 1)
    return x * std + mean


def clamp01(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, 0.0, 1.0)


def to_numpy_uint8(imgs01: torch.Tensor) -> np.ndarray:
    # (B,C,H,W) float in [0,1] -> uint8 (B,H,W,C)
    x = (imgs01.detach().cpu().clamp(0, 1) * 255.0).to(torch.uint8)
    return x.permute(0, 2, 3, 1).numpy()

