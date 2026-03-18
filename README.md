# Assignment 4 — Optimizing Transformer Translation with Ray Tune & Optuna

**Roll Number:** B23CS1061
**Task:** English → Hindi Neural Machine Translation using a from-scratch PyTorch Transformer, optimized via Ray Tune + Optuna hyperparameter search.

---

## Repository Contents

| File | Description |
|------|-------------|
| `b23cs1061_ass_4_tuned_en_to_hi.ipynb` | Main notebook — baseline + Ray Tune + Optuna + final comparison |
| `b23cs1061_ass_4_best_model.pth` | Saved weights of the best-performing model from the tuning sweep |
| `b23cs1061_ass_4_report.pdf` | 1–2 page report with baseline metrics, hyperparameter ranges, best config, and final results |
| `English-Hindi.tsv` | Dataset — 13,186 English-Hindi sentence pairs |

---

## Part 1 — Baseline Results

The model was trained as-is for **100 epochs** with hardcoded hyperparameters, without any changes to architecture or training logic.

| Metric | Value |
|--------|-------|
| Total Training Time | 02h 13m 50s (8030.2s) |
| Epochs | 100 |
| Final Training Loss | 0.0988 |
| BLEU Score (NLTK) | **69.39%** |

**Sample translations (baseline model):**

| English | Hindi (Predicted) |
|---------|-------------------|
| I love you. | मुझे तुमसे प्यार है। |
| What is your name? | आपका नाम क्या है? |
| How are you? | आप कैसे हो? |
| The weather is nice today. | आज मौसम अच्छा सा है। |

---

## Part 2 — Hyperparameter Search (Ray Tune + Optuna)

### Search Space (5 hyperparameters tuned)

| Hyperparameter | Search Range | API Used |
|----------------|-------------|----------|
| Learning Rate (`lr`) | `1e-5` to `1e-3` (log scale) | `tune.loguniform()` |
| Batch Size (`batch_size`) | `[16, 32, 64]` | `tune.choice()` |
| Attention Heads (`num_heads`) | `[4, 8]` | `tune.choice()` |
| Feedforward Dim (`d_ff`) | `[1024, 2048]` | `tune.choice()` |
| Dropout Rate (`dropout`) | `0.1` to `0.4` | `tune.uniform()` |

Architecture constants (fixed): `d_model=512`, `num_layers=6`

### Tuner Configuration

- **Search Algorithm:** `OptunaSearch` (metric=`loss`, mode=`min`)
- **Scheduler:** `ASHAScheduler` (max_t=25, grace_period=5, reduction_factor=2) — early terminates underperforming trials
- **Trials:** 10 total, 1 concurrent (GPU memory constrained)
- **Epochs per trial:** capped at 25

### Best Configuration Found

```python
{
    'lr':         0.00011477645445159093,   # ~1.15e-4
    'batch_size': 32,
    'num_heads':  4,
    'd_ff':       2048,
    'dropout':    0.11420284465820767,
    'd_model':    512,
    'num_layers': 6,
    'num_epochs': 25
}
```

---

## Part 3 — Final Results (Best Config Retrained for 50 Epochs)

The best configuration from the sweep was retrained for **50 epochs** (half the baseline), yielding:

| Metric | Baseline | Best Config (Ray Tune) |
|--------|----------|------------------------|
| Training Time | 02h 13m 50s | **63.0 min** |
| Epochs | 100 | **50** |
| Final Training Loss | 0.0988 | 0.1564 |
| BLEU Score (NLTK) | 69.39% | **73.70%** |

---

## Key Finding — Why This Beats the Baseline

> **73.70% BLEU in 50 epochs (63 min) vs 69.39% BLEU in 100 epochs (2h 13m)**

1. **Exceeded baseline BLEU** — 73.70% > 69.39%, achieving better translation quality
2. **2.1× faster training** — 63 minutes vs 2 hours 13 minutes
3. **50% fewer epochs** — 50 epochs instead of 100
4. **Smarter convergence** — Optuna identified `lr≈1.15e-4` with `batch=32` and low dropout (`0.114`) as a significantly better starting point than the hardcoded defaults; the model converges faster without sacrificing quality
5. **ASHA early stopping** — automatically killed 5 underperforming trials within their first 5 epochs, saving additional compute

### Assignment Efficiency Goal — Met ✅

The rubric states: *"Matched Baseline OR exceeded BLEU (0.50) score using ≤ X epochs (X < 100)"*

- Best config achieves **73.70% BLEU** (exceeds baseline 69.39%) in **50 epochs** (< 100) ✅

---

## Dataset

- **Source:** English-Hindi parallel corpus
- **Total pairs:** 13,186
- **English vocab size:** 4,117 tokens
- **Hindi vocab size:** 4,044 tokens
- **Max sequence length:** 50 tokens

---

## Model Architecture

A from-scratch PyTorch Transformer with:
- Positional Encoding
- Multi-head Self-Attention + Cross-Attention
- Feed-Forward sublayers
- Label-smoothed cross-entropy loss (`nn.CrossEntropyLoss` with `<pad>` ignored)
- Adam optimizer

---

## How to Run

1. Open `b23cs1061_ass_4_tuned_en_to_hi.ipynb` in Google Colab
2. Mount Google Drive and place `English-Hindi.tsv` in `My Drive/`
3. Run all cells sequentially:
   - **Cells 1–61:** Data loading, vocab building, baseline training (100 epochs)
   - **Cells 62–77:** Ray Tune + Optuna sweep (10 trials × 25 epochs)
   - **Cells 78–83:** Best config retrain (50 epochs) + BLEU evaluation + comparison
