from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from q2.utils.checkpoints import load_checkpoint
from q2.utils.data import build_cifar10_loaders, cifar10_normalization
from q2.utils.metrics import accuracy_top1
from q2.utils.models import build_resnet18_cifar10
from q2.utils.viz import clamp01, denormalize_cifar10, save_image_grid, to_numpy_uint8
from q2.utils.wandb_utils import init_wandb


@dataclass(frozen=True)
class Cfg:
    ckpt: str
    batch_size: int
    num_workers: int
    device: str
    eps_list: List[float]
    n_vis: int
    out_dir: str
    wandb: bool
    wandb_entity: str
    wandb_project: str
    wandb_run_name: Optional[str]


def parse_args() -> Cfg:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to trained ResNet18 checkpoint (.pt)")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--eps-list", default="0,0.002,0.004,0.008,0.016,0.031,0.062")
    p.add_argument("--n-vis", type=int, default=16, help="Number of images to visualize (orig vs adv).")
    p.add_argument("--out-dir", default="q2/outputs/fgsm")
    p.add_argument("--wandb", type=int, default=1)
    p.add_argument("--wandb-entity", default="ikamboj-919-iit-jodhpur")
    p.add_argument("--wandb-project", default="final-ops-ass-5")
    p.add_argument("--wandb-run-name", default=None)
    a = p.parse_args()
    eps_list = [float(x.strip()) for x in a.eps_list.split(",") if x.strip() != ""]
    return Cfg(
        ckpt=a.ckpt,
        batch_size=a.batch_size,
        num_workers=a.num_workers,
        device=a.device,
        eps_list=eps_list,
        n_vis=a.n_vis,
        out_dir=a.out_dir,
        wandb=bool(a.wandb),
        wandb_entity=a.wandb_entity,
        wandb_project=a.wandb_project,
        wandb_run_name=a.wandb_run_name,
    )


def fgsm_scratch(
    *,
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    model.eval()
    x_adv = x.detach().clone().requires_grad_(True)
    logits = model(x_adv)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    x_adv = x_adv + eps * torch.sign(x_adv.grad.detach())
    return x_adv.detach()


def _eval_clean(model: torch.nn.Module, loader, device: torch.device) -> float:
    model.eval()
    accs = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        accs.append(accuracy_top1(logits, y))
    return float(np.mean(accs))


def _eval_adv_scratch(model: torch.nn.Module, loader, device: torch.device, eps: float) -> float:
    model.eval()
    accs = []
    for x, y in tqdm(loader, desc=f"FGSM scratch eps={eps}", leave=False, mininterval=1.0):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x_adv = fgsm_scratch(model=model, x=x, y=y, eps=eps)
        logits = model(x_adv)
        accs.append(accuracy_top1(logits, y))
    return float(np.mean(accs))


def _eval_adv_art(model: torch.nn.Module, loader, device: torch.device, eps: float) -> float:
    # ART expects inputs in numpy with explicit clip_values; we use normalized tensors in [-?], so:
    # We generate adversarial in *pixel space [0,1]*, then normalize back to feed model.
    from art.attacks.evasion import FastGradientMethod
    from art.estimators.classification import PyTorchClassifier

    mean, std = cifar10_normalization()
    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1)

    def preprocess(x01: np.ndarray) -> np.ndarray:
        # x01: (N,C,H,W) float32 in [0,1] -> normalized NCHW
        # ART uses NCHW because `input_shape=(3,32,32)`; keep layout consistent.
        x = torch.from_numpy(x01).to(device)
        x = (x - mean_t) / std_t
        return x.detach().cpu().numpy().astype(np.float32)

    def loss_fn():
        return torch.nn.CrossEntropyLoss()

    classifier = PyTorchClassifier(
        model=model,
        loss=loss_fn(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
        input_shape=(3, 32, 32),
        nb_classes=10,
        clip_values=(0.0, 1.0),
        preprocessing=(np.array(mean, dtype=np.float32), np.array(std, dtype=np.float32)),
        device_type="gpu" if device.type == "cuda" else "cpu",
    )

    attack = FastGradientMethod(estimator=classifier, eps=eps, batch_size=loader.batch_size)

    accs = []
    for x_norm, y in tqdm(loader, desc=f"FGSM ART eps={eps}", leave=False, mininterval=1.0):
        # convert normalized -> pixel [0,1] for ART (reverse normalization)
        x_norm = x_norm.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x01 = clamp01(denormalize_cifar10(x_norm))
        # Keep NCHW order for ART. (to_numpy_uint8 would convert to NHWC and break channel order.)
        x_np = x01.detach().cpu().numpy().astype(np.float32)  # (N,C,H,W) in [0,1]
        x_adv_np = attack.generate(x=x_np, y=y.detach().cpu().numpy())
        x_adv_norm = torch.from_numpy(preprocess(x_adv_np)).to(device)
        logits = model(x_adv_norm)
        accs.append(accuracy_top1(logits, y))
    return float(np.mean(accs))


def main() -> None:
    cfg = parse_args()
    device = torch.device(cfg.device)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dls = build_cifar10_loaders(batch_size=cfg.batch_size, num_workers=cfg.num_workers, seed=42)
    model = build_resnet18_cifar10().to(device)
    load_checkpoint(cfg.ckpt, model=model, optimizer=None, map_location=device)

    run_name = cfg.wandb_run_name or "q2-fgsm-compare"
    run = init_wandb(
        enabled=cfg.wandb,
        entity=cfg.wandb_entity,
        project=cfg.wandb_project,
        name=run_name,
        config=asdict(cfg),
    )

    clean_acc = _eval_clean(model, dls.test, device)
    results = []
    for eps in cfg.eps_list:
        eps = float(eps)
        scratch_acc = _eval_adv_scratch(model, dls.test, device, eps=eps)
        art_acc = _eval_adv_art(model, dls.test, device, eps=eps)
        results.append({"eps": eps, "clean_acc": clean_acc, "fgsm_scratch_acc": scratch_acc, "fgsm_art_acc": art_acc})
        if run is not None:
            run.log({"eps": eps, "test/clean_acc": clean_acc, "test/fgsm_scratch_acc": scratch_acc, "test/fgsm_art_acc": art_acc})

    # Visuals on a single batch
    x_norm, y = next(iter(dls.test))
    x_norm = x_norm[: cfg.n_vis].to(device)
    y = y[: cfg.n_vis].to(device)
    eps_vis = max(cfg.eps_list) if cfg.eps_list else 0.031
    x_adv_s = fgsm_scratch(model=model, x=x_norm, y=y, eps=eps_vis)

    # save grids in pixel space
    x01 = clamp01(denormalize_cifar10(x_norm))
    xadv01 = clamp01(denormalize_cifar10(x_adv_s))
    orig_path = save_image_grid(images=x01, path=out_dir / "orig.png", nrow=int(math.sqrt(cfg.n_vis)) or 4, normalize=False)
    adv_path = save_image_grid(images=xadv01, path=out_dir / f"adv_scratch_eps{eps_vis}.png", nrow=int(math.sqrt(cfg.n_vis)) or 4, normalize=False)

    # ART-based FGSM visuals (same epsilon, same batch)
    from art.attacks.evasion import FastGradientMethod
    from art.estimators.classification import PyTorchClassifier

    mean, std = cifar10_normalization()
    mean_np = np.array(mean, dtype=np.float32)
    std_np = np.array(std, dtype=np.float32)

    classifier = PyTorchClassifier(
        model=model,
        loss=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
        input_shape=(3, 32, 32),
        nb_classes=10,
        clip_values=(0.0, 1.0),
        preprocessing=(mean_np, std_np),
        device_type="gpu" if device.type == "cuda" else "cpu",
    )
    art_attack = FastGradientMethod(estimator=classifier, eps=eps_vis, batch_size=x01.shape[0])
    x_np = x01.detach().cpu().numpy().astype(np.float32)  # (N,C,H,W) in [0,1]
    x_adv_art_np = art_attack.generate(x=x_np, y=y.detach().cpu().numpy())
    xadv01_art = clamp01(torch.from_numpy(x_adv_art_np).to(device))
    adv_art_path = save_image_grid(
        images=xadv01_art,
        path=out_dir / f"adv_art_eps{eps_vis}.png",
        nrow=int(math.sqrt(cfg.n_vis)) or 4,
        normalize=False,
    )

    if run is not None:
        import wandb

        run.log(
            {
                "viz/original": wandb.Image(orig_path, caption="Original (pixel space)"),
                "viz/adv_scratch": wandb.Image(adv_path, caption=f"FGSM scratch eps={eps_vis}"),
                "viz/adv_art": wandb.Image(adv_art_path, caption=f"FGSM ART eps={eps_vis}"),
                "tables/fgsm_eps_sweep": wandb.Table(
                    data=[[r["eps"], r["clean_acc"], r["fgsm_scratch_acc"], r["fgsm_art_acc"]] for r in results],
                    columns=["eps", "clean_acc", "fgsm_scratch_acc", "fgsm_art_acc"],
                ),
            }
        )
        run.finish()

    print(f"Clean acc: {clean_acc:.4f}")
    for r in results:
        print(f"eps={r['eps']:.4f} scratch={r['fgsm_scratch_acc']:.4f} art={r['fgsm_art_acc']:.4f}")
    print(f"Saved: {orig_path}")
    print(f"Saved: {adv_path}")


if __name__ == "__main__":
    main()

