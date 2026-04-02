# 3. Sports Classification — Transfer Learning

Classify **100 sports categories** using **ResNet18 pretrained on ImageNet**, demonstrating how Transfer Learning breaks through the scratch CNN ceiling.

**Visual Summary**: [Open the browser-friendly project summary](https://seungminnam.github.io/DL-Projects/3_Transfer_Learning/summary.html)

---

## Why Transfer Learning?

### The Problem: Scratch CNN Ceiling

In the previous project, our 3-layer scratch CNN topped out at **81% on CIFAR-10**. Two fundamental bottlenecks:

1. **Shallow network = limited representational capacity**
   - 3 Conv layers can learn edges, textures, and simple patterns — but not high-level features like "dog face" vs "cat face"
   - Deeper networks could learn these, but...

2. **Not enough data to train a deep network from scratch**
   - CIFAR-10 has only 5,000 images per class. A deep model (e.g., ResNet18 with 11M parameters) would massively overfit on this amount
   - The Sports dataset is even worse: ~135 images per class on average

**The dilemma**: we need a deeper model for better features, but we don't have enough data to train one.

### The Solution: Transfer Learning

Instead of training from scratch, we **reuse a model already trained on ImageNet (14M images, 1,000 classes)**. This model has already learned a rich hierarchy of visual features:

```
Early layers:  edges, colors, textures        (universal — works for any image)
Middle layers: eyes, wheels, fur, patterns     (reusable parts)
Later layers:  "dog face", "car front"         (high-level concepts)
```

These features transfer to new tasks. For sports classification, we need the same low/mid-level features (people, balls, fields, equipment) — no need to learn them again from just ~135 images per class.

### How It Works

```
ResNet18 (pretrained on ImageNet)
├── Conv layers (FROZEN) ← keep the learned feature extraction
└── FC layer (1000 classes) ← REPLACE with:
         ↓
    FC layer (100 sports classes) ← train only this
```

- **Feature Extraction (Step 2)**: Freeze all conv layers (`requires_grad = False`), train only the new FC head. The frozen layers still produce outputs in the forward pass, but no gradients are computed during backprop — the optimizer never touches them. This prevents catastrophic forgetting (our ~135 images/class overwriting 14M-image knowledge) and reduces trainable params from 11M to just 51,300 (512 × 100 weights + 100 biases).
- **Fine-Tuning (Step 3)**: Unfreeze some/all layers, train end-to-end with a small LR

> **Why not fine-tune from the start?**
> With limited data, updating all 11M weights at once can destroy pretrained knowledge (catastrophic forgetting).
> Standard practice: stabilize the classifier first via feature extraction, then fine-tune.

---

## Dataset

**Source**: [Kaggle — Sports Classification](https://www.kaggle.com/datasets/gpiosenka/sports-classification) (100 classes)

| Split | Images | Per Class | Purpose |
|-------|--------|-----------|---------|
| Train | 13,493 | avg 135 (min 59, max 191) | Model weight updates |
| Valid | 500 | 5 per class | Early stopping, LR scheduling |
| Test  | 500 | 5 per class | Final evaluation only (used once) |

### Data Split: Train / Valid / Test

Unlike MNIST and CIFAR-10 (which only provide train/test), this dataset comes with a **proper 3-way split**. This means:

- **Valid set** drives all training decisions (early stopping, LR scheduler) — the model never learns from it, but we use its feedback to tune *when to stop* and *how fast to learn*
- **Test set** is only touched **once**, after training is completely finished — giving an unbiased estimate of real-world performance

In the previous projects, we used the test set for both validation and final evaluation (standard practice for MNIST/CIFAR-10 benchmarks, but technically a mild form of data leakage). This project follows the proper protocol.

### Preprocessing

- **Train**: `Resize(224)` → `RandomHorizontalFlip` → `RandomCrop(224, padding=8)` → ImageNet normalization
- **Val/Test**: `Resize(224)` → `CenterCrop(224)` → ImageNet normalization

Data augmentation is **essential** here — with only ~135 images per class, the model needs to see varied versions of each image to generalize.

---

## Experiment Summary

| Version | Description | Test Acc (Colab) | Test Acc (Local) | Key Change |
|---------|------------|------------------|------------------|------------|
| v1 | Feature Extraction | 91.40% | 93.40% | Frozen backbone, FC only |
| v2 | Fine-Tuning | **98.20%** | **96.60%** | All layers unfrozen, LR 0.0001 |

---

## Lessons Learned

### Device Matters
Training on CPU vs GPU makes a dramatic difference. Running feature extraction on Colab CPU took 6+ hours for 16 epochs; switching to T4 GPU brought it down to ~10-15 minutes. Always verify the model is on the correct device before training:
```python
print(next(model.parameters()).device)  # should match your target device
```
Common pitfall: model and input tensors must be on the **same device**, otherwise PyTorch throws `RuntimeError: Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be the same`.

### Optimizer Efficiency
Pass only trainable parameters to the optimizer — not the entire model:
```python
# Wasteful: optimizer tracks 11M params (frozen ones have no grad, but still use memory)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Efficient: optimizer only tracks 51,300 FC params
optimizer = optim.Adam(model.resnet.fc.parameters(), lr=0.001)
```

### Hyperparameter Choices
- **CrossEntropyLoss**: Standard for multi-class classification (2+ classes). Combines softmax + negative log loss.
- **EPOCHS = 30**: Only 51,300 params to train — converges fast. Generous ceiling; early stopping handles the rest.
- **PATIENCE = 5**: Gives the model enough room to recover from temporary plateaus without wasting time.

---

## Results

### Series Progression

| | Colab (T4 GPU) | Local (M4 MPS) |
|---|---|---|
| MNIST scratch CNN | 99.35% (10 classes) | — |
| CIFAR-10 scratch CNN | 81.04% (10 classes, ceiling) | — |
| Sports frozen TL (v1) | 91.40% (100 classes) | 93.40% |
| Sports fine-tuned TL (v2) | **98.20%** (100 classes) | **96.60%** |

From 81% ceiling on 10 classes → 96–98% on 100 classes. That's the power of transfer learning.

### Step 2 — Feature Extraction (v1)

| Metric | Colab (T4 GPU) | Local (M4 MPS) |
|--------|----------------|----------------|
| Best Val Accuracy | 89.60% (epoch 11) | 90.20% (epoch 12) |
| Final Test Accuracy | **91.40%** | **93.40%** |
| Stopped at Epoch | 16 / 30 | 17 / 30 |
| Trainable Params | 51,300 / 11M (0.5%) | 51,300 / 11M (0.5%) |

> Results vary slightly across devices due to different floating-point implementations (CUDA vs MPS) and random initialization. The overall pattern and insights are consistent.

#### Training Timeline

```
Epoch  1-6:   Rapid improvement (47% → 88%) — FC layer learning fast
Epoch  7-11:  Slowing gains, val acc oscillating around 88-89%
Epoch 12-16:  Plateau — train acc still rising (91% → 93%) but val acc stalls
Epoch 16:     Early stopping triggered (patience 5/5)
```

#### Observations

- **Fast convergence**: With only 51,300 trainable params, the FC layer learned quickly — most gains happened in the first 6 epochs
- **Overfitting signal**: Train acc reached 93% while val acc plateaued at ~89% — a ~4% gap. The frozen features are powerful but the FC layer started memorizing training data
- **Test > Val accuracy (91.40% vs 89.60%)**: Not a bug — both sets have only 500 images (5 per class). One wrong prediction per class = 20% swing for that class, so this difference is within expected noise
- **Validation set in action**: This is the first project using a proper 3-split. Val accuracy drove early stopping at epoch 16, and the test set was only touched once at the end — giving an unbiased final metric

### Step 3 — Fine-Tuning (v2)

| Metric | Colab (T4 GPU) | Local (M4 MPS) |
|--------|----------------|----------------|
| Best Val Accuracy | 96.00% (epoch 7) | 96.00% (epoch 3) |
| Final Test Accuracy | **98.20%** | **96.60%** |
| Stopped at Epoch | 12 / 30 | 8 / 30 |
| Trainable Params | 11M (all layers unfrozen) | 11M (all layers unfrozen) |
| Learning Rate | 0.0001 (10x smaller than v1) | 0.0001 (10x smaller than v1) |

#### Why a smaller LR?

The pretrained weights are already 95% correct for our task. A large LR would overwrite this knowledge (catastrophic forgetting). A 10x smaller LR (0.0001 vs 0.001) makes gentle updates — nudging the features toward sports-specific patterns without destroying what ImageNet taught.

#### Training Timeline

```
Epoch  1:     Starts at 92% train (continuing from v1) → 94% val immediately
Epoch  1-7:   Rapid climb, val acc 94% → 96% — conv layers adapting to sports
Epoch  8-12:  Train hits 99.93% but val drops — overfitting, early stopping triggers
```

#### Observations

- **Massive jump from v1**: +5–6% test accuracy across both environments. Unfreezing let the conv layers learn sports-specific features that frozen ImageNet features couldn't capture (e.g., ice rink textures for figure skating, grass + net combos for specific sports)
- **Fast overfitting**: Train acc reached 99.93% by epoch 8 while val acc peaked at 96% — a ~4% gap. With only ~135 images per class, the model memorizes quickly once all 11M params are free
- **Early stopping saved us**: Val acc actually *dropped* from 96% to 93% in epochs 10-11. Without early stopping, we'd have an overfit model. The best checkpoint (epoch 7) was correctly preserved
- **v1 → v2 transition worked**: Starting from a trained FC layer (v1) gave fine-tuning a stable foundation. The stable classifier produced meaningful gradients from epoch 1, allowing the conv layers to adapt sensibly instead of thrashing

#### v1 vs v2 Comparison

| | Feature Extraction (v1) | Fine-Tuning (v2) |
|---|---|---|
| Trainable params | 51,300 (0.5%) | 11M (100%) |
| Learning rate | 0.001 | 0.0001 |
| Best Val Acc | 89–90% | 96% |
| Final Test Acc | 91–93% | 96–98% |
| Key strength | Fast, safe, no risk of forgetting | Adapts features to domain |

> Ranges reflect results across Colab (T4 GPU) and Local (M4 MPS). Different floating-point implementations and random seeds cause ~2% variation, but the pattern is consistent: fine-tuning gives a significant boost over frozen features.

### Confusion Matrix Analysis (v2)

**Colab**: 9 wrong out of 500 (98.20%) | **Local**: 17 wrong out of 500 (96.60%)

Top confused pairs (consistent across both runs):
| True | Predicted | Reason |
|------|-----------|--------|
| sidecar racing | motorcycle racing | Both are motorcycles on a track |
| hydroplane racing | sailboat racing | Both are boats on water |
| snow boarding | giant slalom | Both are snow sports on slopes |
| cheerleading | football / basketball | Cheerleaders appear at these sports |
| steer wrestling | bull riding | Both involve rodeo + cattle |
| shot put | javelin | Both are throwing events in athletics |
| speed skating | rollerblade racing | Both involve skating at speed |

All confusions are **context-based** (similar environments/equipment), not random — the model understands visual structure but struggles when two sports share the same scene.

**Hypothesis check from Step 1:**
Before training, we predicted `figure skating men/women/pairs`, `horse racing / harness racing`, and `football / field hockey` would be the hardest pairs. The model actually got all of these right in both runs. The real confusions came from **context overlap** (e.g., cheerleaders at sports events, rodeo variants) rather than the visual similarity we expected.

---

## Project Structure

```
3_Transfer_Learning/
├── scripts/
│   ├── model.py              ← SportsClassifier (ResNet18 wrapper with modified FC)
│   ├── utils.py              ← get_device, dataloaders (3-split), train/eval, plot helpers
│   ├── train.py              ← v1 Feature extraction training
│   └── train_finetune.py     ← v2 Fine-tuning training
├── notebooks/                ← Colab development notebooks
├── results/
│   ├── v1_feature_extraction/ ← v1 checkpoint + plots
│   └── v2_finetune/           ← v2 checkpoint + plots
├── data/                     ← Sports dataset (gitignored)
│   ├── train/
│   ├── valid/
│   └── test/
└── README.md
```

---

## How to Run

```bash
# From the 3_Transfer_Learning/ directory

# v1: Feature Extraction (frozen backbone, FC only)
python scripts/train.py

# v2: Fine-Tuning (loads v1 best model, unfreezes all layers)
python scripts/train_finetune.py

# Resume from a specific checkpoint
python scripts/train.py --resume-from results/v1_feature_extraction/best_model.pth
python scripts/train_finetune.py --resume-from results/v2_finetune/best_model.pth
```

- **v1 must run first** — v2 loads the best model from v1 as its starting point
- Each script saves to its own `results/<version>/` folder:
  - `best_model.pth` — best checkpoint
  - `loss_acc_curves.png` — loss/accuracy curves
  - `confusion_matrix.png` — confusion matrix
  - `wrong_predictions.png` — wrong prediction gallery
- Dataset must be downloaded from [Kaggle](https://www.kaggle.com/datasets/gpiosenka/sports-classification) and unzipped into `data/`

---

## Environment

- Python 3.x
- PyTorch 2.x
- torchvision (pretrained ResNet18)
- matplotlib, numpy
- Tested on: Google Colab (T4 GPU), Mac M4 Air (MPS)
