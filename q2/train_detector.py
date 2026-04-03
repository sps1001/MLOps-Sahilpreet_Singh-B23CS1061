from __future__ import annotations

import argparse
import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from q2.utils.checkpoints import save_checkpoint
from q2.utils.data import build_cifar10_loaders
from q2.utils.metrics import accuracy_top1
from q2.utils.models import build_resnet34_binary_detector
from q2.utils.viz import clamp01, denormalize_cifar10, save_image_grid, to_numpy_uint8
from q2.utils.wandb_utils import init_wandb


AttackName = Literal["pgd", "bim"]


@dataclass(frozen=True)
class Cfg:
    attack: AttackName
    base_ckpt: str
    eps: float
    steps: int
    step_size: float
    batch_size: int
    num_workers: int
    epochs: int
    lr: float
    seed: int
    device: str
    out_dir: str
    n_vis: int
    wandb: bool
    wandb_entity: str
    wandb_project: str
    wandb_run_name: Optional[str]


def parse_args() -> Cfg:
    p = argparse.ArgumentParser()
    p.add_argument("--attack", choices=["pgd", "bim"], required=True)
    p.add_argument("--base-ckpt", required=True, help="Trained ResNet18 checkpoint used to generate adversarial images.")
    p.add_argument("--eps", type=float, default=0.031)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--step-size", type=float, default=0.007)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", default="q2/outputs/detector")
    p.add_argument("--n-vis", type=int, default=10)

    p.add_argument("--wandb", type=int, default=1)
    p.add_argument("--wandb-entity", default="ikamboj-919-iit-jodhpur")
    p.add_argument("--wandb-project", default="final-ops-ass-5")
    p.add_argument("--wandb-run-name", default=None)
    a = p.parse_args()
    return Cfg(
        attack=a.attack,
        base_ckpt=a.base_ckpt,
        eps=a.eps,
        steps=a.steps,
        step_size=a.step_size,
        batch_size=a.batch_size,
        num_workers=a.num_workers,
        epochs=a.epochs,
        lr=a.lr,
        seed=a.seed,
        device=a.device,
        out_dir=a.out_dir,
        n_vis=a.n_vis,
        wandb=bool(a.wandb),
        wandb_entity=a.wandb_entity,
        wandb_project=a.wandb_project,
        wandb_run_name=a.wandb_run_name,
    )


def _make_art_attack(attack: AttackName, classifier, eps: float, steps: int, step_size: float):
    if attack == "pgd":
        from art.attacks.evasion import ProjectedGradientDescent

        return ProjectedGradientDescent(estimator=classifier, eps=eps, eps_step=step_size, max_iter=steps, targeted=False, batch_size=classifier.batch_size)
    if attack == "bim":
        from art.attacks.evasion import BasicIterativeMethod

        return BasicIterativeMethod(estimator=classifier, eps=eps, eps_step=step_size, max_iter=steps, targeted=False, batch_size=classifier.batch_size)
    raise ValueError(attack)


def _build_detector_dataset(
    *,
    base_model: torch.nn.Module,
    loader,
    device: torch.device,
    attack: AttackName,
    eps: float,
    steps: int,
    step_size: float,
    limit_batches: Optional[int] = None,
) -> Tuple[TensorDataset, TensorDataset]:
    """
    Returns (train_ds, test_ds) for binary detection:
    label 0 -> clean, label 1 -> adversarial.
    We generate adversarials on CIFAR-10 test split and then split it 80/20 for detector train/test.
    """
    from art.estimators.classification import PyTorchClassifier

    base_model.eval()

    # ART expects pixel space with clip_values; we'll feed pixel [0,1] and specify preprocessing=None,
    # but we actually run the base_model on normalized tensors. Easiest: use ART with preprocessing
    # that normalizes pixel space -> model space. We'll supply preprocessing=(mean,std) in NCHW.
    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
    std = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)

    classifier = PyTorchClassifier(
        model=base_model,
        loss=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.SGD(base_model.parameters(), lr=0.0),
        input_shape=(3, 32, 32),
        nb_classes=10,
        clip_values=(0.0, 1.0),
        preprocessing=(mean, std),
        device_type="gpu" if device.type == "cuda" else "cpu",
    )
    classifier.batch_size = loader.batch_size

    art_attack = _make_art_attack(attack, classifier, eps=eps, steps=steps, step_size=step_size)

    clean_list = []
    adv_list = []
    printed_shape = False
    for i, (x_norm, y) in enumerate(tqdm(loader, desc=f"generate {attack} adv", leave=False, mininterval=1.0)):
        if limit_batches is not None and i >= limit_batches:
            break
        x_norm = x_norm.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x01 = clamp01(denormalize_cifar10(x_norm))
        # ART expects NCHW because PyTorchClassifier has input_shape=(3,32,32).
        # Keep NCHW order here to avoid channel-order mismatch in ResNet.
        x_np = x01.detach().cpu().numpy().astype(np.float32)  # NCHW [0,1]
        x_adv_np = art_attack.generate(x=x_np, y=y.detach().cpu().numpy())
        # back to normalized tensor for detector dataset storage
        x_clean = x_norm.detach().cpu()
        x_adv = torch.tensor(x_adv_np, dtype=torch.float32)
        # FIX: If ART returned NHWC, convert to NCHW for the detector.
        if x_adv.ndim == 4 and x_adv.shape[-1] == 3:
            # NHWC -> NCHW
            x_adv = x_adv.permute(0, 3, 1, 2).contiguous()
        if not printed_shape:
            print(f"DEBUG x_adv.shape (should be NCHW): {tuple(x_adv.shape)}")
            printed_shape = True
        # normalize: (x-mean)/std
        mean_t = torch.tensor(mean).view(1, 3, 1, 1)
        std_t = torch.tensor(std).view(1, 3, 1, 1)
        x_adv = (x_adv - mean_t) / std_t
        clean_list.append(x_clean)
        adv_list.append(x_adv)

    x_clean = torch.cat(clean_list, dim=0)
    x_adv = torch.cat(adv_list, dim=0)

    # Binary labels
    y_clean = torch.zeros((x_clean.size(0),), dtype=torch.long)
    y_adv = torch.ones((x_adv.size(0),), dtype=torch.long)

    # CRITICAL sanity checks for detector dataset pipeline
    # (1) label correctness
    # (2) adversarial examples should differ from clean inputs
    print("DEBUG labels (first 10):")
    print("  y_clean[:10] =", y_clean[:10].tolist())
    print("  y_adv[:10]   =", y_adv[:10].tolist())
    print("DEBUG dataset shapes:", tuple(x_clean.shape), tuple(x_adv.shape))
    try:
        diff_mean = float((x_clean - x_adv).abs().mean().item())
    except Exception:
        diff_mean = None
    print("DEBUG mean |x_clean - x_adv|:", diff_mean)

    x_all = torch.cat([x_clean, x_adv], dim=0)
    y_all = torch.cat([y_clean, y_adv], dim=0)

    # Shuffle and split 80/20
    g = torch.Generator().manual_seed(42)
    idx = torch.randperm(x_all.size(0), generator=g)
    x_all = x_all[idx]
    y_all = y_all[idx]
    n_train = int(0.8 * x_all.size(0))
    train_ds = TensorDataset(x_all[:n_train], y_all[:n_train])
    test_ds = TensorDataset(x_all[n_train:], y_all[n_train:])
    return train_ds, test_ds


@torch.no_grad()
def _eval_detector(model: torch.nn.Module, loader, device: torch.device) -> float:
    model.eval()
    accs = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        accs.append(accuracy_top1(logits, y))
    return float(np.mean(accs))


def main() -> None:
    cfg = parse_args()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device)

    out_dir = Path(cfg.out_dir) / cfg.attack
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load base classifier (ResNet18) for adversarial generation
    from q2.utils.models import build_resnet18_cifar10
    from q2.utils.checkpoints import load_checkpoint

    base = build_resnet18_cifar10().to(device)
    load_checkpoint(cfg.base_ckpt, model=base, optimizer=None, map_location=device)

    dls = build_cifar10_loaders(batch_size=cfg.batch_size, num_workers=cfg.num_workers, seed=cfg.seed)
    train_ds, test_ds = _build_detector_dataset(
        base_model=base,
        loader=dls.test,
        device=device,
        attack=cfg.attack,
        eps=cfg.eps,
        steps=cfg.steps,
        step_size=cfg.step_size,
        limit_batches=None,
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    detector = build_resnet34_binary_detector().to(device)
    opt = torch.optim.AdamW(detector.parameters(), lr=cfg.lr, weight_decay=1e-4)
    ce = torch.nn.CrossEntropyLoss()

    run_name = cfg.wandb_run_name or f"q2-detector-{cfg.attack}-resnet34"
    run = init_wandb(
        enabled=cfg.wandb,
        entity=cfg.wandb_entity,
        project=cfg.wandb_project,
        name=run_name,
        config=asdict(cfg),
    )

    best_acc = -1.0
    ckpt_path = out_dir / f"best_{run_name}.pt"
    metrics_csv = out_dir / f"metrics_{run_name}.csv"

    rows = []
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        detector.train()
        train_losses = []
        train_accs = []
        for x, y in tqdm(train_loader, desc=f"detector train {epoch}/{cfg.epochs}", leave=False, mininterval=1.0):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = detector(x)
            loss = ce(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())
            train_accs.append(accuracy_top1(logits.detach(), y))

        test_acc = _eval_detector(detector, test_loader, device)
        epoch_s = time.time() - t0
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "train_acc": float(np.mean(train_accs)),
            "test_acc": float(test_acc),
            "time/epoch_s": float(epoch_s),
        }
        rows.append(row)

        if run is not None:
            run.log({"epoch": epoch, "detector/train_loss": row["train_loss"], "detector/train_acc": row["train_acc"], "detector/test_acc": row["test_acc"]})

        if test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint(ckpt_path, model=detector, optimizer=opt, epoch=epoch, best_metric=best_acc, extra={"attack": cfg.attack})

        print(f"Epoch {epoch:02d} | train acc {row['train_acc']:.4f} | test acc {row['test_acc']:.4f} | {epoch_s/60.0:.2f} min", flush=True)

    with metrics_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["epoch"])
        w.writeheader()
        w.writerows(rows)

    # log some samples (clean vs adv) from the generated dataset
    x0, y0 = test_ds[0]
    _ = (x0, y0)

    if run is not None:
        import wandb

        # reconstruct a small clean/adv visualization from CIFAR-10 test loader (first batch)
        x_norm, y = next(iter(dls.test))
        x_norm = x_norm[: cfg.n_vis].to(device)
        y = y[: cfg.n_vis].to(device)
        # reuse ART attack generation quickly on this batch
        from art.estimators.classification import PyTorchClassifier
        mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
        std = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)
        classifier = PyTorchClassifier(
            model=base,
            loss=torch.nn.CrossEntropyLoss(),
            optimizer=torch.optim.SGD(base.parameters(), lr=0.0),
            input_shape=(3, 32, 32),
            nb_classes=10,
            clip_values=(0.0, 1.0),
            preprocessing=(mean, std),
            device_type="gpu" if device.type == "cuda" else "cpu",
        )
        classifier.batch_size = cfg.n_vis
        attack = _make_art_attack(cfg.attack, classifier, eps=cfg.eps, steps=cfg.steps, step_size=cfg.step_size)
        x01 = clamp01(denormalize_cifar10(x_norm))
        # Keep NCHW for ART
        x_np = x01.detach().cpu().numpy().astype(np.float32)
        x_adv_np = attack.generate(x=x_np, y=y.detach().cpu().numpy())
        x_adv = torch.tensor(x_adv_np, dtype=torch.float32).to(device)
        # FIX: ensure NCHW for visualization
        if x_adv.ndim == 4 and x_adv.shape[-1] == 3:
            x_adv = x_adv.permute(0, 3, 1, 2).contiguous()
        orig_path = save_image_grid(images=x01, path=out_dir / "samples_clean.png", nrow=5, normalize=False)
        adv_path = save_image_grid(images=x_adv, path=out_dir / f"samples_{cfg.attack}_adv.png", nrow=5, normalize=False)

        run.log(
            {
                "detector/best_test_acc": best_acc,
                "viz/clean_samples": wandb.Image(orig_path, caption="Clean samples (pixel space)"),
                "viz/adv_samples": wandb.Image(adv_path, caption=f"{cfg.attack.upper()} adversarial samples (pixel space)"),
            }
        )
        run.finish()

    print(f"Done. Best detector test acc: {best_acc:.4f}")
    print(f"Best detector checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()

