import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim

from model import SentimentLSTM

from utils import (get_device, get_dataloaders, load_glove, train_one_epoch, evaluate, 
                    plot_history, plot_confusion_matrix, save_wrong_predictions_report)

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR    = os.path.join(PROJECT_DIR, "data")
GLOVE_DIR   = os.path.join(DATA_DIR, "glove")
GLOVE_PATH  = os.path.join(GLOVE_DIR, "glove.6B.100d.txt")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results", "v2_glove")
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
    train_loader, val_loader, test_loader, word_to_idx, idx_to_word = get_dataloaders()

    embedding_matrix = load_glove(GLOVE_PATH, word_to_idx, embed_dim=100)

    #model: Load pretrained GloVe vectors, then initialize the embedding layer from them.
    model = SentimentLSTM(
        vocab_size=len(word_to_idx),
        embed_dim=100,
        hidden_size=256,
        embedding_matrix=embedding_matrix,
        freeze_embeddings=False,
    ).to(device)

    if args.resume_from:
        model.load_state_dict(torch.load(args.resume_from, map_location=device, weights_only=True))
        print(f"Loaded checkpoint: {args.resume_from}")

    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")

    #criterion / optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    #training hyperparameters config
    EPOCHS = 15
    PATIENCE = 3

    # History lists for loss/accuracy tracking across epochs
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    # Best model tracking 
    # Early stopping state
    # float("inf") so that any real loss is an improvement on the first epoch (if it's 0.0, it won't beat any loss)
    best_val_loss = float("inf")
    counter = 0 
    run_start = time.perf_counter() #save start time for entire training script

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.perf_counter() #save start time for each epoch
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, device)
        epoch_time = time.perf_counter() - epoch_start #measure elapsed time for each epoch (train + val)

        train_losses.append(tr_loss); val_losses.append(vl_loss)
        train_accs.append(tr_acc); val_accs.append(vl_acc)

        print(f"Epoch [{epoch}/{EPOCHS}]")
        print(f"  Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc*100:.2f}%")
        print(f"  Val   Loss: {vl_loss:.4f} | Val   Acc: {vl_acc*100:.2f}%")
        print(f"  Epoch Time: {epoch_time:.1f}s") 

        # Early stopping based on val loss (not accuracy — lower loss = better)
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            counter = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  ✓ Best model saved (val loss: {best_val_loss:.4f})")
        else:
            counter+=1
            print(f"  patience: {counter}/{PATIENCE}")
            if counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}. Best Val Loss: {best_val_loss:.4f}")
                break

    # ── Final Evaluation ─────────────────────────────────────────────────────
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True))
    print("\nBest model loaded!")

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Final Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}%")
    print(f"Total Run Time: {time.perf_counter() - run_start:.1f}s") #total run time including test evaluation. 

    # ── Output Artifacts ─────────────────────────────────────────────────────
    plot_history(train_losses, val_losses, train_accs, val_accs,
                save_path=os.path.join(RESULTS_DIR, "loss_acc_curves.png"))
    
    plot_confusion_matrix(model, test_loader, device,
                save_path=os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    
    save_wrong_predictions_report(model, test_loader, device, idx_to_word,
                save_path=os.path.join(RESULTS_DIR, "wrong_predictions.md"))
    
if __name__ == "__main__":
    main()
