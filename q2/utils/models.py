from __future__ import annotations

import torch
from torchvision.models import resnet18, resnet34


def build_resnet18_cifar10(*, num_classes: int = 10) -> torch.nn.Module:
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model


def build_resnet34_binary_detector() -> torch.nn.Module:
    model = resnet34(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    return model

