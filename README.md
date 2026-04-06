# DL-Projects

A progressive series of deep learning projects built with **PyTorch**, going from scratch CNN implementations to Transfer Learning, RNNs, and beyond (GAN planned).
Each project is self-contained with its own notebook, scripts, results, and README documenting design decisions.

---

## Project Series

| # | Project | Dataset | Best Result | Key Technique | Status |
|---|---|---|---|---|---|
| 1 | [MNIST CNN](./1_MNIST_CNN/) | MNIST (10 digits) | 99.35% acc | Scratch CNN, Early Stopping | ✅ Done |
| 2 | [CIFAR-10 CNN](./2_CIFAR10_CNN/) | CIFAR-10 (10 classes) | 81.04% acc | BatchNorm, LR Scheduler | ✅ Done |
| 3 | [Sports Classification](./3_Transfer_Learning/) | Sports (100 classes) | 96–98% acc | ResNet18 Transfer Learning | ✅ Done |
| 4 | [RNN Shakespeare](./4_RNN_Shakespeare/) | Tiny Shakespeare | 6.33 PPL | Vanilla RNN, Gradient Clipping | ✅ Done |
| 5 | [LSTM Sentiment](./5_LSTM_Sentiment/) | IMDB (movie reviews) | 84.48% acc (v2 GloVe) | LSTM, Packed Sequence, GloVe Freeze-vs-Trainable | ✅ Done |

---

## Visual Summaries

Browser-friendly summary pages are available through GitHub Pages:

- [Summary Hub](https://seungminnam.github.io/DL-Projects/)
- [MNIST CNN Summary](https://seungminnam.github.io/DL-Projects/1_MNIST_CNN/summary.html)
- [CIFAR-10 CNN Summary](https://seungminnam.github.io/DL-Projects/2_CIFAR10_CNN/summary.html)
- [Sports Classification Summary](https://seungminnam.github.io/DL-Projects/3_Transfer_Learning/summary.html)
- [RNN Shakespeare Summary](https://seungminnam.github.io/DL-Projects/4_RNN_Shakespeare/summary.html)
- [LSTM Sentiment Summary](https://seungminnam.github.io/DL-Projects/5_LSTM_Sentiment/summary.html)

The project `README.md` files remain the main written documentation. These summary pages are supplementary visual walkthroughs.
If these links do not load yet, enable GitHub Pages in the repository settings using the `main` branch and the repository root (`/`) as the source.

---

## Learning Progression

```
1_MNIST_CNN         → CNN basics, training loop, Early Stopping          → 99.35%
2_CIFAR10_CNN       → BatchNorm, LR Scheduler, scratch CNN ceiling       → 81.04%
3_Transfer_Learning → ResNet18 freeze → fine-tune, 100-class real data   → 96–98%
4_RNN_Shakespeare   → Vanilla RNN, hidden states, BPTT, text generation  → 6.33 PPL
5_LSTM_Sentiment    → LSTM, GloVe init/freeze-vs-trainable, sentiment     → 84.48%
```

The narrative: scratch CNNs hit a ceiling at ~81% on CIFAR-10. Transfer Learning with a pretrained ResNet18 broke through to 96–98% on a harder 100-class problem. Then we moved beyond images to **sequential data** — a vanilla RNN learned to generate Shakespeare character-by-character, but its short-term memory limitations motivated the next step: an LSTM that reads full IMDB reviews and classifies sentiment at the sequence level.

Each project answers:
- What limitation did the previous project hit?
- What new technique addresses it?
- How much did it improve, and why?

---

## Near-Term Direction

The next planned steps for this repo are:

- move into a **Transformer from scratch** project
- then add a small **LLM fine-tuning** project

This repo will stay focused on model-learning and experiment-driven deep learning work.
More systems-heavy LLM work such as **vLLM serving**, **RAG pipelines**, and
full app-style workflows will likely live in a separate repo later so the
portfolio story here stays clean.

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
- LSTM sequence classification with learned and pretrained word embeddings
- Packed sequences and dynamic padding for variable-length text
- GloVe initialization for NLP transfer-style experiments
- Frozen-vs-trainable comparison for pretrained embedding adaptation
- Hugging Face `datasets` for NLP pipelines

---

## Setup

```bash
git clone https://github.com/your-username/DL-Projects.git
cd DL-Projects
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Each project auto-downloads its dataset on first run. See individual project READMEs for details.

---

## Background

Built after reading:
- *Deep Learning from Scratch* (DLFS) Ch. 1–7
- *Hands-On Machine Learning* (HOML) up to the Deep Learning chapters

**Environment**: Google Colab (T4 GPU) for development/experimentation, Mac M4 Air (MPS) for local training and script validation.
