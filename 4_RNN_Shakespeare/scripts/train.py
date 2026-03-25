"""
Train a character-level vanilla RNN on Tiny Shakespeare.

Demonstrates:
- Sequential processing with hidden states
- Backpropagation Through Time (BPTT)
- Gradient clipping for vanilla RNN stability
- Temperature-controlled text generation
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from model import ShakespeareRNN
from utils import (get_device, get_dataloaders, train_one_epoch, evaluate,
                   generate_text, plot_history, plot_training_samples)

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(RESULTS_DIR, "best_model.pth")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Path to a .pth checkpoint to resume training from")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # Data — must come before model (we need vocab_size from the dataset)
    train_loader, val_loader, test_loader, chars, char_to_idx, idx_to_char = \
        get_dataloaders(batch_size=64, seq_len=100)

    # Model
    model = ShakespeareRNN(vocab_size=len(chars), hidden_size=512).to(device)

    if args.resume_from:
        model.load_state_dict(torch.load(args.resume_from, map_location=device, weights_only=True))
        print(f"Loaded checkpoint: {args.resume_from}")

    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()  # Same as CNN projects — 65 classes = 65 characters
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Hyperparams
    EPOCHS = 50
    PATIENCE = 5

    # History lists — perplexity replaces accuracy (no accuracy in text generation)
    train_losses, val_losses = [], []
    train_ppls, val_ppls = [], []

    # Training samples across epochs (for visualization)
    training_samples = {}

    # Early stopping state
    # float("inf") so that any real loss is an improvement on the first epoch (if it's 0.0, it won't beat any loss)
    best_val_loss = float("inf")
    counter = 0

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_ppl = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_ppl = evaluate(model, val_loader, criterion, device)

        train_losses.append(tr_loss); val_losses.append(vl_loss)
        train_ppls.append(tr_ppl);    val_ppls.append(vl_ppl)

        print(f"Epoch [{epoch}/{EPOCHS}]")
        print(f"  Train Loss: {tr_loss:.4f} | Train PPL: {tr_ppl:.2f}")
        print(f"  Val   Loss: {vl_loss:.4f} | Val   PPL: {vl_ppl:.2f}")

        # Generate a sample to watch the RNN learn over time
        sample = generate_text(model, device, "The ", 50,
                               temperature=1.0, char_to_idx=char_to_idx, idx_to_char=idx_to_char)
        print(f"  Sample: {sample}")
        training_samples[epoch] = sample

        # Early stopping based on val loss (not accuracy — lower loss = better)
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            counter = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  ✓ Best model saved (val loss: {best_val_loss:.4f})")
        else:
            counter += 1
            print(f"  patience: {counter}/{PATIENCE}")
            if counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}. Best Val Loss: {best_val_loss:.4f}")
                break

    # ── Final Evaluation ─────────────────────────────────────────────────────
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True))
    print("\nBest model loaded!")

    test_loss, test_ppl = evaluate(model, test_loader, criterion, device)
    print(f"Final Test Loss: {test_loss:.4f} | Test Perplexity: {test_ppl:.2f}")

    # ── Generate at different temperatures ────────────────────────────────────
    # Low temp = safe/repetitive, mid = balanced, high = creative/chaotic
    print("\n" + "=" * 60)
    print("Text Generation at Different Temperatures")
    print("=" * 60)

    sample_outputs = []
    for temp in [0.5, 1.0, 1.5]:
        text = generate_text(model, device, "ROMEO:\n", 500,
                             temperature=temp, char_to_idx=char_to_idx, idx_to_char=idx_to_char)
        header = f"\n{'─' * 40}\nTemperature = {temp}\n{'─' * 40}\n"
        print(header + text)
        sample_outputs.append(header + text)

    # Save generated samples to file
    samples_path = os.path.join(RESULTS_DIR, "sample_outputs.txt")
    with open(samples_path, "w") as f:
        f.write("\n".join(sample_outputs))
    print(f"\nSaved → {samples_path}")

    # ── Plots ────────────────────────────────────────────────────────────────
    plot_history(train_losses, val_losses, train_ppls, val_ppls,
                 save_path=os.path.join(RESULTS_DIR, "loss_perplexity_curves.png"))

    plot_training_samples(training_samples,
                          save_path=os.path.join(RESULTS_DIR, "training_samples.png"))


if __name__ == "__main__":
    main()
