from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


@dataclass(frozen=True)
class CIFAR10Loaders:
    train: DataLoader
    test: DataLoader


def cifar10_normalization() -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    # Common CIFAR-10 normalization
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    return mean, std


def build_cifar10_loaders(
    *,
    root: str = "./data",
    batch_size: int = 128,
    num_workers: int = 2,
    seed: int = 42,
) -> CIFAR10Loaders:
    torch.manual_seed(seed)
    np.random.seed(seed)

    mean, std = cifar10_normalization()

    train_tfms = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_tfms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    train_ds = datasets.CIFAR10(root=root, train=True, download=True, transform=train_tfms)
    test_ds = datasets.CIFAR10(root=root, train=False, download=True, transform=test_tfms)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return CIFAR10Loaders(train=train_loader, test=test_loader)

