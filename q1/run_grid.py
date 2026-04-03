from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class Args:
    epochs: int
    batch_size: int
    lr: float
    num_workers: int
    device: str
    wandb: int
    wandb_entity: str
    wandb_project: str
    model_name: str


def parse_args() -> Args:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--wandb", type=int, default=1)
    p.add_argument("--wandb-entity", default="ikamboj-919-iit-jodhpur")
    p.add_argument("--wandb-project", default="Assignment-5")
    p.add_argument("--model-name", default="WinKawaks/vit-small-patch16-224")
    a = p.parse_args()
    return Args(
        epochs=a.epochs,
        batch_size=a.batch_size,
        lr=a.lr,
        num_workers=a.num_workers,
        device=a.device,
        wandb=a.wandb,
        wandb_entity=a.wandb_entity,
        wandb_project=a.wandb_project,
        model_name=a.model_name,
    )


def run(cmd: List[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    a = parse_args()

    base = [
        sys.executable,
        "-m",
        "q1.train_vit_cifar100",
        "--epochs",
        str(a.epochs),
        "--batch-size",
        str(a.batch_size),
        "--lr",
        str(a.lr),
        "--num-workers",
        str(a.num_workers),
        "--device",
        a.device,
        "--wandb",
        str(a.wandb),
        "--wandb-entity",
        a.wandb_entity,
        "--wandb-project",
        a.wandb_project,
        "--model-name",
        a.model_name,
    ]

    # Baseline
    run(base + ["--mode", "baseline", "--wandb-run-name", "q1-baseline-head-only-full"])

    # LoRA grid
    ranks = [2, 4, 8]
    alphas = [2, 4, 8]
    for r in ranks:
        for alpha in alphas:
            run(
                base
                + [
                    "--mode",
                    "lora",
                    "--rank",
                    str(r),
                    "--alpha",
                    str(alpha),
                    "--dropout",
                    "0.1",
                    "--log-grad-norm-every",
                    "10",
                    "--wandb-run-name",
                    f"q1-lora-r{r}-a{alpha}-d0.1-full",
                ]
            )


if __name__ == "__main__":
    main()

