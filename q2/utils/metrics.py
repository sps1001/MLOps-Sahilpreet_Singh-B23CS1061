from __future__ import annotations

import torch


@torch.no_grad()
def accuracy_top1(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return float((preds == labels).float().mean().item())

