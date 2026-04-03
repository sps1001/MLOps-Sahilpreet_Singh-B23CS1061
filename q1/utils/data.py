from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class DataLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    class_names: List[str]


def _split_indices(n: int, val_split: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    val_idx = indices[:val_split]
    train_idx = indices[val_split:]
    return train_idx, val_idx


def build_cifar100_dataloaders(
    *,
    root: str = "./data",
    image_size: int = 224,
    batch_size: int = 128,
    num_workers: int = 4,
    val_split: int = 5000,
    seed: int = 42,
    pin_memory: bool | None = None,
) -> DataLoaders:
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    train_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    full_train = datasets.CIFAR100(root=root, train=True, download=True, transform=train_tfms)
    test = datasets.CIFAR100(root=root, train=False, download=True, transform=eval_tfms)

    if not (0 < val_split < len(full_train)):
        raise ValueError(f"val_split must be in (0, {len(full_train)}), got {val_split}")

    train_idx, val_idx = _split_indices(len(full_train), val_split=val_split, seed=seed)

    train_ds = Subset(full_train, train_idx.tolist())

    # Re-create dataset with eval transforms for validation
    full_train_eval = datasets.CIFAR100(root=root, train=True, download=False, transform=eval_tfms)
    val_ds = Subset(full_train_eval, val_idx.tolist())

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return DataLoaders(
        train=train_loader,
        val=val_loader,
        test=test_loader,
        class_names=list(full_train.classes),
    )

