# Assignment 5 — Q1 & Q2 Results

This repository contains:
- **Q1:** ViT-S (ImageNet-pretrained) + LoRA fine-tuning on **CIFAR-100**
- **Q2:** Adversarial attacks (FGSM/PGD/BIM) on **CIFAR-10** + adversarial detection using **IBM ART**

W&B project (for figures, class-wise histograms, and gradients):
https://wandb.ai/ikamboj-919-iit-jodhpur/final-ops-ass-5

### Docker build

From repo root:

```bash
docker build -f docker/Dockerfile -t ass5-q1 .
```

### Docker run

From repo root:

```bash
docker run --rm -it --gpus all -v "$PWD:/workspace/ops-ass-5" ass5-q1 bash
```

CPU-only: remove `--gpus all`.

If you hit DataLoader shared-memory errors, either keep `--num-workers 0` (default) or run Docker with a larger shm:

```bash
docker run --rm -it --shm-size=8g --gpus all -v "$PWD:/workspace/ops-ass-5" ass5-q1 bash
```

### Baseline (head-only, no LoRA)

Inside container (in `/workspace/ops-ass-5`):

```bash
python -m q1.train_vit_cifar100 \
  --mode baseline \
  --model-name WinKawaks/vit-small-patch16-224 \
  --num-workers 0 \
  --epochs 10 \
  --batch-size 128 \
  --wandb 1 \
  --wandb-entity ikamboj-919-iit-jodhpur \
  --wandb-project final-ops-ass-5
```

### LoRA run (Q/K/V)

```bash
python -m q1.train_vit_cifar100 \
  --mode lora \
  --model-name WinKawaks/vit-small-patch16-224 \
  --num-workers 0 \
  --rank 4 \
  --alpha 8 \
  --dropout 0.1 \
  --epochs 10 \
  --batch-size 128 \
  --wandb 1 \
  --wandb-entity ikamboj-919-iit-jodhpur \
  --wandb-project final-ops-ass-5
```

### Optuna search (LoRA rank/alpha only)

```bash
python -m q1.optuna_lora_search \
  --trials 15 \
  --epochs 3 \
  --limit-train-batches 50 \
  --limit-val-batches 10 \
  --device cpu \
  --wandb 1 \
  --wandb-entity ikamboj-919-iit-jodhpur \
  --wandb-project final-ops-ass-5
```

### Eval (overall + class-wise histogram)

```bash
python -m q1.eval_vit_cifar100 \
  --mode baseline \
  --model-name WinKawaks/vit-small-patch16-224 \
  --num-workers 0 \
  --ckpt q1/outputs/best_q1-baseline-head-only.pt \
  --wandb 1 \
  --wandb-entity ikamboj-919-iit-jodhpur \
  --wandb-project final-ops-ass-5
```

## Q1 — Table 1 (Baseline head-only training/validation, 10 epochs)

Epoch-wise training loss/accuracy and validation loss/accuracy for the baseline (**head-only, no LoRA**) run.

| Epoch | Training loss | Validation loss | Training accuracy | Validation accuracy |
|------:|--------------:|----------------:|------------------:|----------------------:|
| 1 | 2.2699 | 1.2275 | 0.5286 | 0.7146 |
| 2 | 1.0102 | 0.9497 | 0.7476 | 0.7527 |
| 3 | 0.8284 | 0.8504 | 0.7780 | 0.7648 |
| 4 | 0.7426 | 0.8067 | 0.7953 | 0.7727 |
| 5 | 0.6882 | 0.7745 | 0.8074 | 0.7807 |
| 6 | 0.6470 | 0.7567 | 0.8175 | 0.7836 |
| 7 | 0.6151 | 0.7461 | 0.8265 | 0.7824 |
| 8 | 0.5903 | 0.7298 | 0.8318 | 0.7881 |
| 9 | 0.5684 | 0.7269 | 0.8365 | 0.7889 |
| 10 | 0.5491 | 0.7232 | 0.8418 | 0.7910 |

## Testing Results (CIFAR-100 test set)

Your assignment requires the “Testing table” to include **overall test accuracy for all 9 LoRA settings**:
`rank in {2,4,8}` and `alpha in {2,4,8}` with `dropout=0.1`, plus the **baseline (no LoRA)**.

Use `python -m q1.eval_vit_cifar100` and the matching checkpoint in `q1/outputs/`.

| LoRA layers | Rank | Alpha | Dropout | Overall Test Accuracy | Trainable Parameters used |
|---|---:|---:|---:|---:|---:|
| Without LoRA (head-only) | - | - | - | 0.7901 | 38,500 |
| With LoRA (Q,K,V) | 2 | 2 | 0.1 | 0.5314 | 55,296 |
| With LoRA (Q,K,V) | 2 | 4 | 0.1 | 0.5901 | 55,296 |
| With LoRA (Q,K,V) | 2 | 8 | 0.1 | 0.6387 | 55,296 |
| With LoRA (Q,K,V) | 4 | 2 | 0.1 | 0.6104 | 110,592 |
| With LoRA (Q,K,V) | 4 | 4 | 0.1 | 0.6849 | 110,592 |
| With LoRA (Q,K,V) | 4 | 8 | 0.1 | 0.7408 | 110,592 |
| With LoRA (Q,K,V) | 8 | 2 | 0.1 | 0.6869 | 221,184 |
| With LoRA (Q,K,V) | 8 | 4 | 0.1 | 0.7530 | 221,184 |
| With LoRA (Q,K,V) | 8 | 8 | 0.1 | 0.8046 | 221,184 |

All 8 remaining LoRA evaluations are completed (see `test_updated.log`), so the Testing Results table above is final.

Optuna (LoRA-only) best hyperparameters: `rank=4`, `alpha=8`, `dropout=0.1` (overall test accuracy `0.7408`) with checkpoint `q1/outputs/best_q1-lora-r4-a8-d0.1-full.pt`.

### Q1 Figures (where to find them)

The following plots/tables are available in W&B for each Q1 run (baseline and LoRA grid runs):
- Train/val **loss** and **accuracy** curves (per epoch)
- **Class-wise** CIFAR-100 test accuracy histogram/table produced by `q1.eval_vit_cifar100`
- **LoRA gradient-norm / update** traces (enabled when using the gradient logging option in training)

## Q2 Results (CIFAR-10) — FGSM, PGD, BIM (IBM ART)

W&B project:
https://wandb.ai/ikamboj-919-iit-jodhpur/final-ops-ass-5

### Q2 ResNet18 training dynamics (logged per epoch)

The Q2 training log records **train loss/accuracy** and **test loss/accuracy** per epoch (no separate validation split in the logger). The table below summarizes epochs 1–10.

| Epoch | Train loss | Train acc. | Test loss | Test acc. |
|------:|-----------:|------------:|----------:|-----------:|
| 1 | 2.2954 | 0.2609 | 1.6602 | 0.3772 |
| 2 | 1.5774 | 0.4146 | 1.3920 | 0.4890 |
| 3 | 1.3653 | 0.5015 | 1.2159 | 0.5611 |
| 4 | 1.1877 | 0.5741 | 1.2381 | 0.5603 |
| 5 | 1.0499 | 0.6268 | 1.0248 | 0.6372 |
| 6 | 0.9690 | 0.6591 | 0.9519 | 0.6739 |
| 7 | 0.9029 | 0.6847 | 1.0633 | 0.6477 |
| 8 | 0.8528 | 0.7046 | 0.9086 | 0.6796 |
| 9 | 0.8173 | 0.7144 | 0.7935 | 0.7262 |
| 10 | 0.7790 | 0.7309 | 0.8277 | 0.7197 |

### Q2(i) ResNet18 + FGSM (scratch vs IBM ART) — Table 3

**Clean test accuracy:** `0.8583` (checkpoint: `q2/outputs/best_q2-resnet18-cifar10.pt`)

| ε | Clean acc. | FGSM scratch acc. | FGSM ART acc. | Δ clean → scratch | Δ clean → ART |
|--:|-----------:|------------------:|--------------:|------------------:|----------------:|
| 0.000 | 0.8601 | 0.8601 | 0.8601 | 0.0000 | 0.0000 |
| 0.002 | 0.8601 | 0.8376 | 0.7453 | 0.0225 | 0.1148 |
| 0.004 | 0.8601 | 0.8134 | 0.6237 | 0.0467 | 0.2364 |
| 0.008 | 0.8601 | 0.7702 | 0.4222 | 0.0899 | 0.4379 |
| 0.016 | 0.8601 | 0.6727 | 0.1849 | 0.1874 | 0.6752 |
| 0.031 | 0.8601 | 0.4997 | 0.0480 | 0.3604 | 0.8121 |
| 0.062 | 0.8601 | 0.2667 | 0.0157 | 0.5934 | 0.8444 |

Qualitative comparison (FGSM ε=0.062):
- Original: `q2/outputs/fgsm/orig.png`
- FGSM scratch: `q2/outputs/fgsm/adv_scratch_eps0.062.png`
- FGSM via IBM ART: `q2/outputs/fgsm/adv_art_eps0.062.png`

W&B run with the sweep and image panels:
https://wandb.ai/ikamboj-919-iit-jodhpur/final-ops-ass-5/runs/gmgefz8f

### Q2(ii) Adversarial Detectors (ResNet34) — Tables 4–5

Requirement: detector accuracy ≥ 70% for each attack type.

| Detector training data | Best test accuracy | Checkpoint |
|------------------------|---------------------:|------------|
| Clean + PGD (ART) | 0.9756 | `q2/outputs/detector/pgd/best_q2-detector-pgd-resnet34.pt` |
| Clean + BIM (ART) | 0.9600 | `q2/outputs/detector/bim/best_q2-detector-bim-resnet34.pt` |

Table 4 — PGD detector: train/test accuracy by epoch

| Epoch | Train loss | Train acc. | Test acc. |
|------:|-----------:|-----------:|----------:|
| 1 | 0.7485 | 0.5372 | 0.5510 |
| 2 | 0.3391 | 0.8419 | 0.7208 |
| 3 | 0.1327 | 0.9487 | 0.9443 |
| 4 | 0.0861 | 0.9665 | 0.8954 |
| 5 | 0.0783 | 0.9699 | 0.9433 |
| 6 | 0.0578 | 0.9794 | 0.9623 |
| 7 | 0.0410 | 0.9851 | 0.9488 |
| 8 | 0.0303 | 0.9887 | 0.9756 |
| 9 | 0.0368 | 0.9868 | 0.8045 |
| 10 | 0.0417 | 0.9857 | 0.9463 |

Table 5 — BIM detector: train/test accuracy by epoch

| Epoch | Train loss | Train acc. | Test acc. |
|------:|-----------:|-----------:|----------:|
| 1 | 0.7262 | 0.5642 | 0.6687 |
| 2 | 0.3290 | 0.8541 | 0.7671 |
| 3 | 0.1401 | 0.9456 | 0.8808 |
| 4 | 0.0973 | 0.9634 | 0.9468 |
| 5 | 0.0701 | 0.9746 | 0.9327 |
| 6 | 0.0715 | 0.9731 | 0.9574 |
| 7 | 0.0464 | 0.9827 | 0.9346 |
| 8 | 0.0480 | 0.9831 | 0.7567 |
| 9 | 0.0390 | 0.9859 | 0.8696 |
| 10 | 0.0403 | 0.9851 | 0.9600 |

Detector qualitative sample panels (also logged to W&B):
- PGD: `q2/outputs/detector/pgd/samples_clean.png`, `q2/outputs/detector/pgd/samples_pgd_adv.png`
- BIM: `q2/outputs/detector/bim/samples_clean.png`, `q2/outputs/detector/bim/samples_bim_adv.png`

W&B runs:
- PGD detector: https://wandb.ai/ikamboj-919-iit-jodhpur/final-ops-ass-5/runs/1lfee2h4
- BIM detector: https://wandb.ai/ikamboj-919-iit-jodhpur/final-ops-ass-5/runs/p7l4uhui

## How to reproduce Q2 (Docker + exact commands)

Run Q2 inside the required Docker container from repo root (`/workspace/ops-ass-5` inside the container).

### 1) Docker

Build:
```bash
docker build -f q2/Dockerfile -t ass5-q2 .
```

Run:
```bash
docker run --rm -it --gpus all -v "$PWD:/workspace/ops-ass-5" ass5-q2 bash
```

CPU-only: remove `--gpus all`.

Optional: login to W&B inside container:
```bash
wandb login
```

All Q2 scripts accept:
- `--wandb 1`
- `--wandb-entity ikamboj-919-iit-jodhpur`
- `--wandb-project final-ops-ass-5`

### 2) Q2(i) Train ResNet18 (clean)

```bash
python -m q2.train_resnet18_cifar10 \
  --epochs 30 \
  --batch-size 128 \
  --device cuda \
  --wandb 1 \
  --wandb-entity ikamboj-919-iit-jodhpur \
  --wandb-project final-ops-ass-5
```

This produces the checkpoint `q2/outputs/best_q2-resnet18-cifar10.pt` (or your `--wandb-run-name` variant).

### 3) Q2(i) FGSM: scratch vs IBM ART

```bash
python -m q2.fgsm_compare \
  --ckpt q2/outputs/best_q2-resnet18-cifar10.pt \
  --device cuda \
  --eps-list 0,0.002,0.004,0.008,0.016,0.031,0.062 \
  --wandb 1 \
  --wandb-entity ikamboj-919-iit-jodhpur \
  --wandb-project final-ops-ass-5
```

This saves `q2/outputs/fgsm/orig.png` and adversarial examples under `q2/outputs/fgsm/` (and logs the epsilon-sweep table to W&B).

### 4) Q2(ii) Adversarial detectors (PGD and BIM via ART)

PGD detector:
```bash
python -m q2.train_detector \
  --attack pgd \
  --base-ckpt q2/outputs/best_q2-resnet18-cifar10.pt \
  --eps 0.031 \
  --steps 10 \
  --step-size 0.007 \
  --epochs 10 \
  --device cuda \
  --wandb 1 \
  --wandb-entity ikamboj-919-iit-jodhpur \
  --wandb-project final-ops-ass-5
```

BIM detector:
```bash
python -m q2.train_detector \
  --attack bim \
  --base-ckpt q2/outputs/best_q2-resnet18-cifar10.pt \
  --eps 0.031 \
  --steps 10 \
  --step-size 0.007 \
  --epochs 10 \
  --device cuda \
  --wandb 1 \
  --wandb-entity ikamboj-919-iit-jodhpur \
  --wandb-project final-ops-ass-5
```

