from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from tqdm import tqdm
from transformers import ViTForImageClassification

from q1.utils.checkpoints import save_checkpoint
from q1.utils.data import DataLoaders, build_cifar100_dataloaders
from q1.utils.metrics import accuracy_top1
from q1.utils.wandb_utils import init_wandb


@dataclass(frozen=True)
class TrainConfig:
    mode: str
    model_name: str
    hf_token: Optional[str]
    image_size: int
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    seed: int
    val_split: int
    num_workers: int
    output_dir: str
    wandb: bool
    wandb_entity: str
    wandb_project: str
    wandb_run_name: Optional[str]
    rank: Optional[int]
    alpha: Optional[int]
    dropout: float
    limit_train_batches: Optional[int]
    limit_val_batches: Optional[int]
    log_grad_norm_every: int
    device: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def configure_model(cfg: TrainConfig) -> torch.nn.Module:
    model = ViTForImageClassification.from_pretrained(
        cfg.model_name,
        num_labels=100,
        ignore_mismatched_sizes=True,
        token=cfg.hf_token,
    )

    if cfg.mode == "baseline":
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith("classifier.")
        return model

    if cfg.mode == "lora":
        if cfg.rank is None or cfg.alpha is None:
            raise ValueError("LoRA mode requires --rank and --alpha")

        # Freeze everything; PEFT will mark LoRA params trainable
        for p in model.parameters():
            p.requires_grad = False

        # Ensure classifier is trainable
        for p in model.classifier.parameters():
            p.requires_grad = True

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

    raise ValueError(f"Unknown mode: {cfg.mode}")


def _limit_iter(loader, limit_batches: Optional[int]):
    if limit_batches is None:
        yield from loader
        return
    for i, batch in enumerate(loader):
        if i >= limit_batches:
            break
        yield batch


def compute_lora_grad_norm(model: torch.nn.Module) -> float:
    total_sq = 0.0
    for name, p in model.named_parameters():
        if "lora_" not in name:
            continue
        if p.grad is None:
            continue
        g = p.grad.detach()
        total_sq += float(torch.sum(g * g).item())
    return math.sqrt(total_sq)


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader, device: torch.device, limit_batches: Optional[int]) -> Tuple[float, float]:
    model.eval()
    losses = []
    accs = []
    for images, labels in _limit_iter(loader, limit_batches):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(pixel_values=images, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        losses.append(loss.item())
        accs.append(accuracy_top1(logits, labels))
    return float(np.mean(losses)) if losses else 0.0, float(np.mean(accs)) if accs else 0.0


def train_one_epoch(
    *,
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    cfg: TrainConfig,
    run,
) -> Tuple[float, float, Dict[str, float]]:
    model.train()
    losses = []
    accs = []

    extra_logs: Dict[str, float] = {}

    t0 = time.time()
    pbar = tqdm(
        _limit_iter(loader, cfg.limit_train_batches),
        desc=f"train epoch {epoch}",
        leave=False,
        mininterval=1.0,
    )
    for step, (images, labels) in enumerate(pbar, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(pixel_values=images, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if cfg.mode == "lora" and cfg.log_grad_norm_every > 0 and (step % cfg.log_grad_norm_every == 0):
            grad_norm = compute_lora_grad_norm(model)
            extra_logs["lora/grad_norm"] = grad_norm
            if run is not None:
                run.log({"lora/grad_norm": grad_norm})

        optimizer.step()

        losses.append(loss.item())
        acc = accuracy_top1(logits.detach(), labels)
        accs.append(acc)

        if step % 5 == 0 or step == 1:
            elapsed = max(1e-9, time.time() - t0)
            it_s = step / elapsed
            pbar.set_postfix(
                loss=f"{float(np.mean(losses)):.3f}",
                acc=f"{float(np.mean(accs)):.3f}",
                it_s=f"{it_s:.2f}",
            )

    return float(np.mean(losses)) if losses else 0.0, float(np.mean(accs)) if accs else 0.0, extra_logs


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["baseline", "lora"], required=True)
    # NOTE: `google/vit-small-patch16-224` is no longer available on HF in this environment.
    p.add_argument("--model-name", default="WinKawaks/vit-small-patch16-224")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-split", type=int, default=5000)
    # Default to 0 to avoid Docker /dev/shm DataLoader crashes.
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--output-dir", default="q1/outputs")

    p.add_argument("--wandb", type=int, default=1)
    p.add_argument("--wandb-entity", default="ikamboj-919-iit-jodhpur")
    p.add_argument("--wandb-project", default="Assignment-5")
    p.add_argument("--wandb-run-name", default=None)

    # If set, passed as HF token for model download. Otherwise rely on local HF cache mount.
    p.add_argument("--hf-token", default=os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN"))

    p.add_argument("--rank", type=int, default=None)
    p.add_argument("--alpha", type=int, default=None)
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument("--limit-train-batches", type=int, default=None)
    p.add_argument("--limit-val-batches", type=int, default=None)
    p.add_argument("--log-grad-norm-every", type=int, default=20)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    a = p.parse_args()
    return TrainConfig(
        mode=a.mode,
        model_name=a.model_name,
        hf_token=a.hf_token,
        image_size=a.image_size,
        epochs=a.epochs,
        batch_size=a.batch_size,
        lr=a.lr,
        weight_decay=a.weight_decay,
        seed=a.seed,
        val_split=a.val_split,
        num_workers=a.num_workers,
        output_dir=a.output_dir,
        wandb=bool(a.wandb),
        wandb_entity=a.wandb_entity,
        wandb_project=a.wandb_project,
        wandb_run_name=a.wandb_run_name,
        rank=a.rank,
        alpha=a.alpha,
        dropout=a.dropout,
        limit_train_batches=a.limit_train_batches,
        limit_val_batches=a.limit_val_batches,
        log_grad_norm_every=a.log_grad_norm_every,
        device=a.device,
    )


def main() -> None:
    cfg = parse_args()
    best_val_acc, best_path = train(cfg)
    print(f"Done. Best val acc: {best_val_acc:.4f}")
    print(f"Best checkpoint: {best_path}")


def train(cfg: TrainConfig) -> Tuple[float, str]:
    set_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device(cfg.device)

    dls: DataLoaders = build_cifar100_dataloaders(
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        val_split=cfg.val_split,
        seed=cfg.seed,
    )

    model = configure_model(cfg).to(device)
    trainable_params = count_trainable_params(model)

    exp_dir = Path(cfg.output_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    run_name = cfg.wandb_run_name
    if run_name is None:
        if cfg.mode == "baseline":
            run_name = "q1-baseline-head-only"
        else:
            run_name = f"q1-lora-r{cfg.rank}-a{cfg.alpha}-d{cfg.dropout}"

    run = init_wandb(
        enabled=cfg.wandb,
        entity=cfg.wandb_entity,
        project=cfg.wandb_project,
        name=run_name,
        config={
            **asdict(cfg),
            "trainable_params": trainable_params,
            "num_classes": 100,
        },
    )

    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val_acc = -1.0
    best_path = exp_dir / f"best_{run_name}.pt"

    # Save config next to outputs for reproducibility
    (exp_dir / f"config_{run_name}.json").write_text(json.dumps(asdict(cfg), indent=2))

    print(f"Device: {device}")
    print(f"Mode: {cfg.mode} | model: {cfg.model_name} | image_size: {cfg.image_size}")
    if cfg.mode == "lora":
        print(f"LoRA: rank={cfg.rank} alpha={cfg.alpha} dropout={cfg.dropout}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Output dir: {exp_dir}")
    print("Starting training.", flush=True)

    epoch_rows = []
    for epoch in range(1, cfg.epochs + 1):
        epoch_t0 = time.time()
        train_loss, train_acc, _ = train_one_epoch(
            model=model,
            loader=dls.train,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            cfg=cfg,
            run=run,
        )
        val_loss, val_acc = evaluate(model, dls.val, device, cfg.limit_val_batches)
        epoch_s = time.time() - epoch_t0

        epoch_rows.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "train_acc": float(train_acc),
                "val_acc": float(val_acc),
            }
        )

        if run is not None:
            run.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/acc": train_acc,
                    "val/loss": val_loss,
                    "val/acc": val_acc,
                    "trainable_params": trainable_params,
                    "time/epoch_s": epoch_s,
                }
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_metric=best_val_acc,
                extra={"run_name": run_name, "mode": cfg.mode, "rank": cfg.rank, "alpha": cfg.alpha, "dropout": cfg.dropout},
            )

        print(
            f"Epoch {epoch:02d}/{cfg.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
            f"{epoch_s/60.0:.2f} min",
            flush=True,
        )

    # Save epoch-wise metrics table locally (for report/README tables)
    metrics_csv = exp_dir / f"metrics_{run_name}.csv"
    with metrics_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "train_acc", "val_acc"])
        w.writeheader()
        w.writerows(epoch_rows)

    # Log epoch-wise metrics as a W&B Table (so "tables" are explicit, not only curves)
    if run is not None:
        import wandb

        table = wandb.Table(
            data=[[r["epoch"], r["train_loss"], r["val_loss"], r["train_acc"], r["val_acc"]] for r in epoch_rows],
            columns=["epoch", "train_loss", "val_loss", "train_acc", "val_acc"],
        )
        run.log({"epoch_metrics": table})

    if run is not None:
        run.summary["best_val_acc"] = best_val_acc
        run.summary["metrics_csv"] = str(metrics_csv)
        run.finish()

    return best_val_acc, str(best_path)


if __name__ == "__main__":
    main()

