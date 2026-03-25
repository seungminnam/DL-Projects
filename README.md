# DL-Projects

A progressive series of deep learning projects built with **PyTorch**, going from scratch CNN implementations to Transfer Learning and beyond (RNN, GAN planned).
Each project is self-contained with its own notebook, scripts, results, and README documenting design decisions.

---

## Project Series

| # | Project | Dataset | Best Result | Key Technique | Status |
|---|---|---|---|---|---|
| 1 | [MNIST CNN](./1_MNIST_CNN/) | MNIST (10 digits) | 99.35% acc | Scratch CNN, Early Stopping | ✅ Done |
| 2 | [CIFAR-10 CNN](./2_CIFAR10_CNN/) | CIFAR-10 (10 classes) | 81.04% acc | BatchNorm, LR Scheduler | ✅ Done |
| 3 | [Sports Classification](./3_Transfer_Learning/) | Sports (100 classes) | 96–98% acc | ResNet18 Transfer Learning | ✅ Done |
| 4 | [RNN Shakespeare](./4_RNN_Shakespeare/) | Tiny Shakespeare | 6.33 PPL | Vanilla RNN, Gradient Clipping | ✅ Done |

---

## Learning Progression

```
1_MNIST_CNN         → CNN basics, training loop, Early Stopping          → 99.35%
2_CIFAR10_CNN       → BatchNorm, LR Scheduler, scratch CNN ceiling       → 81.04%
3_Transfer_Learning → ResNet18 freeze → fine-tune, 100-class real data   → 96–98%
4_RNN_Shakespeare   → Vanilla RNN, hidden states, BPTT, text generation  → 6.33 PPL
```

The narrative: scratch CNNs hit a ceiling at ~81% on CIFAR-10. Transfer Learning with a pretrained ResNet18 broke through to 96–98% on a harder 100-class problem. Then we moved beyond images to **sequential data** — a vanilla RNN learned to generate Shakespeare character-by-character, but its short-term memory limitations motivate the next step: LSTM.

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
- Custom `torch.utils.data.Dataset` for text data
- Recurrent Neural Networks (RNN) for sequential processing
- Gradient clipping for BPTT stability
- Temperature-controlled text generation
- Perplexity as evaluation metric for language models

---

## Background

Built after reading:
- *Deep Learning from Scratch* (DLFS) Ch. 1–7
- *Hands-On Machine Learning* (HOML) up to the Deep Learning chapters

**Environment**: Google Colab (T4 GPU) for development/experimentation, Mac M4 Air (MPS) for local training and script validation.
