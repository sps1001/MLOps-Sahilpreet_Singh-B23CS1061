from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import ViTForImageClassification

from q1.utils.checkpoints import load_checkpoint
from q1.utils.data import build_cifar100_dataloaders
from q1.utils.metrics import per_class_accuracy
from q1.utils.wandb_utils import init_wandb


@dataclass(frozen=True)
class EvalConfig:
    ckpt: str
    mode: str
    model_name: str
    hf_token: Optional[str]
    image_size: int
    batch_size: int
    num_workers: int
    rank: Optional[int]
    alpha: Optional[int]
    dropout: float
    wandb: bool
    wandb_entity: str
    wandb_project: str
    wandb_run_name: Optional[str]
    limit_test_batches: Optional[int]
    device: str


def parse_args() -> EvalConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--mode", choices=["baseline", "lora"], required=True)
    p.add_argument("--model-name", default="WinKawaks/vit-small-patch16-224")
    p.add_argument("--hf-token", default=os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN"))
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--rank", type=int, default=None)
    p.add_argument("--alpha", type=int, default=None)
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument("--wandb", type=int, default=1)
    p.add_argument("--wandb-entity", default="ikamboj-919-iit-jodhpur")
    p.add_argument("--wandb-project", default="Assignment-5")
    p.add_argument("--wandb-run-name", default=None)

    p.add_argument("--limit-test-batches", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    return EvalConfig(
        ckpt=a.ckpt,
        mode=a.mode,
        model_name=a.model_name,
        hf_token=a.hf_token,
        image_size=a.image_size,
        batch_size=a.batch_size,
        num_workers=a.num_workers,
        rank=a.rank,
        alpha=a.alpha,
        dropout=a.dropout,
        wandb=bool(a.wandb),
        wandb_entity=a.wandb_entity,
        wandb_project=a.wandb_project,
        wandb_run_name=a.wandb_run_name,
        limit_test_batches=a.limit_test_batches,
        device=a.device,
    )


def configure_model(cfg: EvalConfig) -> torch.nn.Module:
    model = ViTForImageClassification.from_pretrained(
        cfg.model_name,
        num_labels=100,
        ignore_mismatched_sizes=True,
        token=cfg.hf_token,
    )

    if cfg.mode == "baseline":
        return model

    if cfg.rank is None or cfg.alpha is None:
        raise ValueError("LoRA mode requires --rank and --alpha to reconstruct adapters for eval.")

    lora_cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=cfg.rank,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        target_modules=["query", "key", "value"],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    return model


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader, device: torch.device, limit_batches: Optional[int]) -> Tuple[float, np.ndarray]:
    model.eval()
    all_preds = []
    all_labels = []
    for i, (images, labels) in enumerate(loader):
        if limit_batches is not None and i >= limit_batches:
            break
        images = images.to(device, non_blocking=True)
        outputs = model(pixel_values=images)
        preds = outputs.logits.argmax(dim=-1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())

    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    overall = float((preds == labels).mean())
    per_cls = per_class_accuracy(preds, labels, num_classes=100)
    return overall, per_cls


def main() -> None:
    cfg = parse_args()
    device = torch.device(cfg.device)

    dls = build_cifar100_dataloaders(
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        val_split=5000,
        seed=42,
    )

    model = configure_model(cfg).to(device)
    load_checkpoint(cfg.ckpt, model=model, optimizer=None, map_location=device)

    overall, per_cls = evaluate(model, dls.test, device, cfg.limit_test_batches)

    run_name = cfg.wandb_run_name or f"q1-eval-{cfg.mode}"
    run = init_wandb(
        enabled=cfg.wandb,
        entity=cfg.wandb_entity,
        project=cfg.wandb_project,
        name=run_name,
        config=asdict(cfg),
    )
    if run is not None:
        import wandb

        run.log({"test/acc": overall})
        table = wandb.Table(data=[[i, float(per_cls[i])] for i in range(100)], columns=["class_idx", "acc"])
        run.log({"test/per_class_acc": table})
        run.log({"test/per_class_acc_hist": wandb.Histogram(per_cls)})
        run.finish()

    print(f"Test accuracy: {overall:.4f}")


if __name__ == "__main__":
    main()

