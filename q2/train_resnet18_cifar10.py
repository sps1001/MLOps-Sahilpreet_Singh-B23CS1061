from __future__ import annotations

import argparse
import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from q2.utils.checkpoints import load_checkpoint, save_checkpoint
from q2.utils.data import build_cifar10_loaders
from q2.utils.metrics import accuracy_top1
from q2.utils.models import build_resnet18_cifar10
from q2.utils.wandb_utils import init_wandb


@dataclass(frozen=True)
class TrainCfg:
    epochs: int
    batch_size: int
    lr: float
    momentum: float
    weight_decay: float
    num_workers: int
    seed: int
    device: str
    wandb: bool
    wandb_entity: str
    wandb_project: str
    wandb_run_name: Optional[str]
    output_dir: str
    resume: Optional[str]


def parse_args() -> TrainCfg:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", default="q2/outputs")
    p.add_argument("--resume", default=None, help="Path to a checkpoint .pt to resume.")

    p.add_argument("--wandb", type=int, default=1)
    p.add_argument("--wandb-entity", default="ikamboj-919-iit-jodhpur")
    p.add_argument("--wandb-project", default="final-ops-ass-5")
    p.add_argument("--wandb-run-name", default=None)
    a = p.parse_args()
    return TrainCfg(
        epochs=a.epochs,
        batch_size=a.batch_size,
        lr=a.lr,
        momentum=a.momentum,
        weight_decay=a.weight_decay,
        num_workers=a.num_workers,
        seed=a.seed,
        device=a.device,
        wandb=bool(a.wandb),
        wandb_entity=a.wandb_entity,
        wandb_project=a.wandb_project,
        wandb_run_name=a.wandb_run_name,
        output_dir=a.output_dir,
        resume=a.resume,
    )


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    losses = []
    accs = []
    ce = torch.nn.CrossEntropyLoss()
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = ce(logits, y)
        losses.append(loss.item())
        accs.append(accuracy_top1(logits, y))
    return float(np.mean(losses)), float(np.mean(accs))


def main() -> None:
    cfg = parse_args()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device(cfg.device)
    dls = build_cifar10_loaders(batch_size=cfg.batch_size, num_workers=cfg.num_workers, seed=cfg.seed)

    model = build_resnet18_cifar10().to(device)
    opt = SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay, nesterov=True)
    sched = CosineAnnealingLR(opt, T_max=cfg.epochs)

    start_epoch = 0
    best_test_acc = -1.0

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.resume:
        payload = load_checkpoint(cfg.resume, model=model, optimizer=opt, map_location=device)
        start_epoch = int(payload.get("epoch", 0))
        best_test_acc = float(payload.get("best_metric", best_test_acc))

    run_name = cfg.wandb_run_name or "q2-resnet18-cifar10"
    run = init_wandb(
        enabled=cfg.wandb,
        entity=cfg.wandb_entity,
        project=cfg.wandb_project,
        name=run_name,
        config={**asdict(cfg), "model": "resnet18", "dataset": "cifar10"},
    )

    metrics_csv = out_dir / f"metrics_{run_name}.csv"
    ckpt_path = out_dir / f"best_{run_name}.pt"

    ce = torch.nn.CrossEntropyLoss()
    rows = []
    for epoch in range(start_epoch + 1, cfg.epochs + 1):
        t0 = time.time()
        model.train()
        train_losses = []
        train_accs = []
        pbar = tqdm(dls.train, desc=f"train {epoch}/{cfg.epochs}", leave=False, mininterval=1.0)
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = ce(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())
            train_accs.append(accuracy_top1(logits.detach(), y))
        sched.step()

        test_loss, test_acc = evaluate(model, dls.test, device)
        epoch_s = time.time() - t0

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "train_acc": float(np.mean(train_accs)),
            "test_loss": float(test_loss),
            "test_acc": float(test_acc),
            "lr": float(opt.param_groups[0]["lr"]),
            "time/epoch_s": float(epoch_s),
        }
        rows.append(row)

        if run is not None:
            run.log(
                {
                    "epoch": epoch,
                    "train/loss": row["train_loss"],
                    "train/acc": row["train_acc"],
                    "test/loss": row["test_loss"],
                    "test/acc": row["test_acc"],
                    "lr": row["lr"],
                    "time/epoch_s": row["time/epoch_s"],
                }
            )

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            save_checkpoint(ckpt_path, model=model, optimizer=opt, epoch=epoch, best_metric=best_test_acc, extra={"run_name": run_name})

        print(
            f"Epoch {epoch:02d} | train loss {row['train_loss']:.4f} acc {row['train_acc']:.4f} "
            f"| test loss {row['test_loss']:.4f} acc {row['test_acc']:.4f} | {epoch_s/60.0:.2f} min",
            flush=True,
        )

    with metrics_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["epoch"])
        w.writeheader()
        w.writerows(rows)

    if run is not None:
        run.summary["best_test_acc"] = best_test_acc
        run.summary["best_ckpt"] = str(ckpt_path)
        run.finish()

    print(f"Done. Best test acc: {best_test_acc:.4f}")
    print(f"Best checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()

