from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def accuracy_top1(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


def per_class_accuracy(preds: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    preds = preds.astype(np.int64, copy=False)
    labels = labels.astype(np.int64, copy=False)
    correct = np.zeros((num_classes,), dtype=np.int64)
    total = np.zeros((num_classes,), dtype=np.int64)

    for p, y in zip(preds, labels):
        total[y] += 1
        if p == y:
            correct[y] += 1

    acc = np.zeros((num_classes,), dtype=np.float32)
    nonzero = total > 0
    acc[nonzero] = correct[nonzero] / total[nonzero]
    return acc

