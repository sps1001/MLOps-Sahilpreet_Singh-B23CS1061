from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from typing import Optional

import optuna
import torch

from q1.train_vit_cifar100 import TrainConfig, train


@dataclass(frozen=True)
class SearchConfig:
    trials: int
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
    model_name: str
    hf_token: Optional[str]
    device: str
    limit_train_batches: Optional[int]
    limit_val_batches: Optional[int]


def parse_args() -> SearchConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=15)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-split", type=int, default=5000)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--output-dir", default="q1/outputs/optuna")

    p.add_argument("--wandb", type=int, default=0)
    p.add_argument("--wandb-entity", default="ikamboj-919-iit-jodhpur")
    p.add_argument("--wandb-project", default="Assignment-5")

    p.add_argument("--model-name", default="WinKawaks/vit-small-patch16-224")
    p.add_argument("--hf-token", default=None)

    p.add_argument("--limit-train-batches", type=int, default=50)
    p.add_argument("--limit-val-batches", type=int, default=10)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    return SearchConfig(
        trials=a.trials,
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
        model_name=a.model_name,
        hf_token=a.hf_token,
        device=a.device,
        limit_train_batches=a.limit_train_batches,
        limit_val_batches=a.limit_val_batches,
    )


def objective(trial: optuna.Trial, base: SearchConfig) -> float:
    rank = trial.suggest_categorical("rank", [2, 4, 8])
    alpha = trial.suggest_categorical("alpha", [2, 4, 8])

    cfg = TrainConfig(
        mode="lora",
        model_name=base.model_name,
        hf_token=base.hf_token,
        image_size=224,
        epochs=base.epochs,
        batch_size=base.batch_size,
        lr=base.lr,
        weight_decay=base.weight_decay,
        seed=base.seed,
        val_split=base.val_split,
        num_workers=base.num_workers,
        output_dir=f"{base.output_dir}/trial_{trial.number}_r{rank}_a{alpha}",
        wandb=False,
        wandb_entity=base.wandb_entity,
        wandb_project=base.wandb_project,
        wandb_run_name=None,
        rank=rank,
        alpha=alpha,
        dropout=0.1,
        limit_train_batches=base.limit_train_batches,
        limit_val_batches=base.limit_val_batches,
        log_grad_norm_every=0,
        device=base.device,
    )

    best_val_acc, _best_path = train(cfg)
    return best_val_acc


def main() -> None:
    cfg = parse_args()

    study = optuna.create_study(direction="maximize", study_name="q1_lora_rank_alpha")
    study.optimize(lambda t: objective(t, cfg), n_trials=cfg.trials)

    print("Best trial:")
    print(study.best_trial.number)
    print(study.best_trial.value)
    print(study.best_trial.params)

    if cfg.wandb:
        import wandb

        run = wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name="q1-optuna-lora-search",
            config={**asdict(cfg), "best_params": study.best_trial.params, "best_val_acc": study.best_value},
        )
        run.summary["best_val_acc"] = study.best_value
        run.summary.update({f"best/{k}": v for k, v in study.best_trial.params.items()})
        run.finish()


if __name__ == "__main__":
    main()

