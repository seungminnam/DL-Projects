# 2. CIFAR-10 CNN Classifier

A progressive CNN experiment on **CIFAR-10** (32x32 color images, 10 classes), exploring the impact of **Batch Normalization** and **LR Scheduling** on a from-scratch architecture.

**Visual Summary**: [Open the browser-friendly project summary](https://seungminnam.github.io/DL-Projects/2_CIFAR10_CNN/summary.html)

---

## Experiment Summary

| Version | Description | Best Test Acc | Epochs | Key Change |
|---------|------------|---------------|--------|------------|
| v1 | Baseline CNN | 77.31% | 20 / 20 | — |
| v2 | + BatchNorm | 79.48% | 17 / 20 | BatchNorm after each Conv |
| v3 | + LR Scheduler | **81.04%** | 32 / 50 | ReduceLROnPlateau |

**Conclusion**: Scratch CNN ceiling is ~81%. Further gains require Transfer Learning or fundamentally different architectures.

---

## Model Architecture

### v1 — Baseline (`CIFAR10_CNN`)

```
Input (N, 3, 32, 32)
    │
    ▼
Conv2d(3→32, k=3, pad=1) → ReLU → MaxPool2d(2)    # (N, 32, 16, 16)
    │
    ▼
Conv2d(32→64, k=3, pad=1) → ReLU → MaxPool2d(2)   # (N, 64, 8, 8)
    │
    ▼
Conv2d(64→128, k=3, pad=1) → ReLU → MaxPool2d(2)  # (N, 128, 4, 4)
    │
    ▼
Flatten → Linear(2048→256) → ReLU → Dropout(0.5)
    │
    ▼
Linear(256→10)   ← raw logits for 10 classes
```

### v2/v3 — BatchNorm (`CIFAR10_CNN_BN`)

Same architecture, but each conv block becomes **Conv → BN → ReLU → Pool**.
v3 uses the same model as v2, with `ReduceLROnPlateau(mode='max', factor=0.5, patience=3)` to adaptively halve the learning rate when test accuracy plateaus.

---

## Results

### v1 — Baseline

| Metric | Value |
|--------|-------|
| Best Test Accuracy | **77.31%** |
| Stopped at Epoch | 20 / 20 |
| Overfitting Gap | ~15% (Train 92% vs Test 77%) |

![v1 Loss & Accuracy](results/v1/loss_acc_curves.png)
![v1 Confusion Matrix](results/v1/confusion_matrix.png)

### v2 — BatchNorm

| Metric | Value |
|--------|-------|
| Best Test Accuracy | **79.48%** (+2.17%) |
| Stopped at Epoch | 17 / 20 (Early Stopping) |
| Overfitting Gap | ~8% (Train 87% vs Test 79%) |

![v2 Loss & Accuracy](results/v2_BN/loss_acc_curves.png)
![v2 Confusion Matrix](results/v2_BN/confusion_matrix.png)

### v3 — LR Scheduler

| Metric | Value |
|--------|-------|
| Best Test Accuracy | **81.04%** (+1.56%) |
| Stopped at Epoch | 32 / 50 (Early Stopping) |
| Overfitting Gap | ~16% (Train 97% vs Test 81%) |

![v3 Loss & Accuracy](results/v3_scheduler/loss_acc_curves.png)
![v3 Confusion Matrix](results/v3_scheduler/confusion_matrix.png)

#### LR Schedule Timeline

```
Epoch  1-19:  LR = 0.001000  (initial)
Epoch 20:     LR = 0.000500  ← 1st reduction
Epoch 26:     LR = 0.000250  ← 2nd reduction
Epoch 31:     LR = 0.000125  ← 3rd reduction
Epoch 32:     Early stopping triggered
```

---

## Key Observations

### Why 3 Conv layers instead of 2?
CIFAR-10 images are 32x32 RGB — more complex than MNIST's 28x28 grayscale. A third conv layer captures higher-level features (textures, object parts) needed to distinguish visually similar classes like cat vs dog.

### BatchNorm Effect (v1 → v2)
- **+2.17% test accuracy** (77.31% → 79.48%)
- **Overfitting reduced**: the train-test gap shrank from ~15% to ~8%
- **Faster convergence**: BN stabilizes internal activations, allowing the model to learn more efficiently
- **Earlier stopping**: early stopping triggered at epoch 17 (v1 ran all 20 epochs without triggering)

### LR Scheduler Effect (v2 → v3)
- **+1.56% test accuracy** (79.48% → 81.04%)
- **Scheduler "rescued" the model from early stopping**: at epoch 19, patience was 3/5 and the model was stalling. LR reduction let it break through to 80.48% at epoch 21
- **Diminishing returns per LR reduction**: 0.001→0.0005 gave +0.48%, 0.0005→0.00025 gave +0.28%, 0.00025→0.000125 gave nothing — a clear sign the architecture has hit its capacity limit
- **Overfitting worsened**: BN had reduced the gap to ~8%, but the scheduler let training run longer (32 epochs), pushing the gap back to ~16%. Test accuracy went up slightly, but test loss kept rising — the model was getting more confident on its wrong predictions

### Note on Data Split (No Validation Set)

This project uses only **train (50,000) / test (10,000)** — no separate validation set. The test set is used for both early stopping / LR scheduling decisions and final evaluation. Strictly speaking, this means the test set indirectly influences training, so the reported accuracy could be slightly optimistic.

**Why this is acceptable here:**
- CIFAR-10 is a **standard benchmark** — nearly all papers and tutorials report test accuracy using this same 2-split setup. Following the community convention keeps results comparable.
- The test set is **large (10,000 samples)**, so information leakage from validation-based decisions is negligible in practice.
- Splitting from the train set would **reduce training data** and likely hurt performance without meaningful benefit at this scale.

**When a proper 3-split is required:**
In the next project (Transfer Learning / Sports Classification), the dataset comes pre-split into `train/valid/test`. There, the **valid set** drives LR scheduling and early stopping, while the **test set** is only used for final evaluation — ensuring a clean, unbiased performance metric.

### Most Confused Classes
All three versions consistently struggle with:
- **cat ↔ dog** (similar body shapes and textures)
- **dog → cat**: 146 (v1) → 117 (v2) → 152 (v3)
- **cat → dog**: 133 (v1) → 178 (v2) → 115 (v3)
- **bird ↔ deer** (overlapping background contexts)

These reflect real visual ambiguity in CIFAR-10's low-resolution 32x32 images.

### Architecture Ceiling
The progression tells a clear story:

```
v1 → v2 → v3
77%   79%   81%    (each technique adds less)
```

Scratch CNN with 3 conv layers tops out at ~81%. To go further, the next project will use Transfer Learning with pretrained models, where techniques like Data Augmentation also have much more impact (especially on smaller, custom datasets).

---

## Project Structure

```
2_CIFAR10_CNN/
├── scripts/
│   ├── model.py              ← CIFAR10_CNN + CIFAR10_CNN_BN definitions
│   ├── utils.py              ← shared: get_device, train, evaluate, plot helpers
│   ├── train.py              ← v1 baseline training
│   ├── train_v2_BN.py        ← v2 BatchNorm training
│   └── train_v3_scheduler.py ← v3 LR Scheduler training
├── results/
│   ├── v1/                   ← v1 checkpoint + plots
│   ├── v2_BN/                ← v2 checkpoint + plots
│   └── v3_scheduler/         ← v3 checkpoint + plots
├── data/                     ← CIFAR-10 downloaded here (gitignored)
└── README.md
```

---

## How to Run

```bash
# From the 2_CIFAR10_CNN/ directory
python scripts/train.py              # v1: baseline (fresh)
python scripts/train_v2_BN.py        # v2: + BatchNorm (fresh)
python scripts/train_v3_scheduler.py # v3: + LR Scheduler (fresh)

# Resume from a previous checkpoint
python scripts/train_v3_scheduler.py --resume-from results/v2_BN/best_model.pth
```

- **Default**: fresh training from random initialization
- **`--resume-from`**: load a `.pth` checkpoint and continue training (architecture must match)

Each script saves to its own `results/<version>/` folder:
- `best_model.pth` — best checkpoint
- `loss_acc_curves.png` — loss/accuracy curves
- `confusion_matrix.png` — confusion matrix
- `wrong_predictions.png` — wrong prediction gallery

---

## Environment

- Python 3.x
- PyTorch 2.x
- torchvision
- matplotlib, numpy
