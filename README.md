# DL-Projects

A progressive series of deep learning projects built with **PyTorch**, going from scratch CNN implementations to Transfer Learning and beyond (RNN, GAN planned).
Each project is self-contained with its own notebook, scripts, results, and README documenting design decisions.

---

## Project Series

| # | Project | Dataset | Best Test Acc | Key Technique | Status |
|---|---|---|---|---|---|
| 1 | [MNIST CNN](./1_MNIST_CNN/) | MNIST (10 digits) | 99.35% | Scratch CNN, Early Stopping | ✅ Done |
| 2 | [CIFAR-10 CNN](./2_CIFAR10_CNN/) | CIFAR-10 (10 classes) | 81.04% | BatchNorm, LR Scheduler | ✅ Done |
| 3 | [Sports Classification](./3_Transfer_Learning/) | Sports (100 classes) | 96–98% | ResNet18 Transfer Learning | ✅ Done |

---

## Learning Progression

```
1_MNIST_CNN         → CNN basics, training loop, Early Stopping          → 99.35%
2_CIFAR10_CNN       → BatchNorm, LR Scheduler, scratch CNN ceiling       → 81.04%
3_Transfer_Learning → ResNet18 freeze → fine-tune, 100-class real data   → 96–98%
```

The narrative: scratch CNNs hit a ceiling at ~81% on CIFAR-10. Transfer Learning with a pretrained ResNet18 broke through to 96–98% on a harder 100-class problem — training only 0.5% of the model's parameters at first, then fine-tuning end-to-end.

Each project answers:
- What limitation did the previous project hit?
- What new technique addresses it?
- How much did it improve, and why?

---

## Skills Covered

- Custom CNN architecture design
- PyTorch `nn.Module`, `DataLoader`, `transforms`
- Training loop with Early Stopping & best model checkpointing
- Loss/Accuracy curve visualization
- Confusion Matrix & wrong-prediction analysis
- Transfer Learning with pretrained ResNet18 (feature extraction + fine-tuning)
- Proper train/valid/test 3-split evaluation
- Real dataset pipeline (Kaggle, ImageFolder)

---

## Background

Built after reading:
- *Deep Learning from Scratch* (DLFS) Ch. 1–7
- *Hands-On Machine Learning* (HOML) up to the Deep Learning chapters

**Environment**: Google Colab (T4 GPU) for development/experimentation, Mac M4 Air (MPS) for local training and script validation.
