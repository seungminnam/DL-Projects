import os
import math
import urllib.request
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend: saves plots to file without opening a window
import matplotlib.pyplot as plt

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)                    # 4_RNN_Shakespeare/
DATA_DIR    = os.path.join(PROJECT_DIR, "data")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def get_device():
    """Detect best available device (MPS > CUDA > CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _download_shakespeare():
    """Download Tiny Shakespeare to DATA_DIR if not already present.
    Returns the entire text as one big string.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    filepath = os.path.join(DATA_DIR, "input.txt")
    if not os.path.exists(filepath):
        print("Downloading Tiny Shakespeare...")
        urllib.request.urlretrieve(SHAKESPEARE_URL, filepath)
        print(f"Saved → {filepath}")
    with open(filepath, "r") as f:
        return f.read()


def _build_vocab(text):
    """Build character-level vocabulary from text.
    Returns (chars, char_to_idx, idx_to_char).
    """
    # sorted() makes the mapping deterministic — without it, set() gives
    # a random order every time, and yesterday's model wouldn't match today's vocab
    chars = sorted(set(text))
    # Example: ['\n', ' ', '!', ..., 'A', 'B', ..., 'a', 'b', ..., 'z'] (~65 chars)

    char_to_idx = {ch: i for i, ch in enumerate(chars)}  # 'A' → 10, 'B' → 11, ...
    idx_to_char = {i: ch for ch, i in char_to_idx.items()}  # 10 → 'A', 11 → 'B', ...
    return chars, char_to_idx, idx_to_char


# In CNN projects, torchvision.datasets.MNIST handled everything — loading,
# transforms, (image, label) pairs. For raw text, we build our own Dataset.
class ShakespeareDataset(Dataset):
    """
    Character-level dataset: slices encoded text into non-overlapping chunks.

    Each sample is an (input, target) pair where target is shifted by 1 character:
        input  = text[i*seq_len : (i+1)*seq_len]       # chars 0..99
        target = text[i*seq_len+1 : (i+1)*seq_len+1]   # chars 1..100

    This shift teaches the RNN to predict the next character at every position:
        Input:  T  h  e     k  i  n  g
        Target: h  e     k  i  n  g
    """
    def __init__(self, encoded_text, seq_len):
        self.data = encoded_text  # Full text as a tensor of indices: [15, 46, 43, ...]
        self.seq_len = seq_len  # 100 characters per sample
        # (len - 1) because target is shifted by 1: last sample needs one extra char
        self.num_samples = (len(encoded_text) - 1) // seq_len

    # PyTorch's DataLoader requires exactly two methods:
    #   __len__()          — "how many samples do you have?"
    #   __getitem__(idx)   — "give me sample number idx"
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = self.data[start : start + self.seq_len]          # (seq_len,) input
        y = self.data[start + 1 : start + self.seq_len + 1]  # (seq_len,) target shifted by 1
        return x, y


def get_dataloaders(batch_size=64, seq_len=100):
    """
    Download Shakespeare, build vocab, split 80/10/10, return loaders + vocab info.
    Returns (train_loader, val_loader, test_loader, chars, char_to_idx, idx_to_char).
    """
    text = _download_shakespeare()
    chars, char_to_idx, idx_to_char = _build_vocab(text)

    # Encode entire text as tensor of indices: "The " → [20, 45, 41, 1]
    # dtype=torch.long because CrossEntropyLoss requires 64-bit integer targets
    encoded = torch.tensor([char_to_idx[ch] for ch in text], dtype=torch.long)

    # Split: 80/10/10 (contiguous, not random — shuffling would destroy sequences)
    n = len(encoded)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    train_data = encoded[:train_end]
    val_data = encoded[train_end:val_end]
    test_data = encoded[val_end:]

    train_dataset = ShakespeareDataset(train_data, seq_len)
    val_dataset = ShakespeareDataset(val_data, seq_len)
    test_dataset = ShakespeareDataset(test_data, seq_len)

    # shuffle=True for train: randomize which chunks we see each epoch
    # shuffle=False for val/test: consistent evaluation every time
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Vocab size: {len(chars)} | Text length: {n:,}")
    print(f"Train: {len(train_dataset):,} | Val: {len(val_dataset):,} | Test: {len(test_dataset):,} sequences")

    return train_loader, val_loader, test_loader, chars, char_to_idx, idx_to_char


# ── Training & Evaluation ────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device, max_grad_norm=5.0) -> tuple[float, float]:
    """Run one training epoch. Returns (avg_loss, perplexity)."""
    # 1. Train mode  2. Loop: forward → loss → backward → clip → step  3. Return metrics
    model.train()
    total_loss = 0.0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        logits, _ = model(inputs)  # Don't need hidden state during training

        # Reshape for CrossEntropyLoss (expects (N, C) not (N, seq_len, C))
        # logits: (N, seq_len, vocab_size) → (N*seq_len, vocab_size)
        # targets: (N, seq_len) → (N*seq_len,)
        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        loss.backward()
        # Gradient clipping: W_hh gets multiplied ~100 times in BPTT,
        # causing gradients to explode. This caps them at max_grad_norm.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    perplexity = math.exp(min(avg_loss, 100))  # Cap to avoid overflow
    return avg_loss, perplexity


def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    """Evaluate model on a dataset. Returns (avg_loss, perplexity)."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():  # No gradients needed — just measuring performance
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            logits, _ = model(inputs)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    perplexity = math.exp(min(avg_loss, 100))
    return avg_loss, perplexity

def generate_text(model, device, seed_text, length, temperature, char_to_idx, idx_to_char):                                                                             
    """Generate text character-by-character using the trained model."""
    model.eval()                                                                          
                  
    # Step 1: Encode seed text as tensor of indices    
    # "The " -> [[20, 45, 41, 1]]
    # Double brackets [[]] because model expects (N, seq_len) - N=1 for single sample                                   
    input_idx = torch.tensor(
        [[char_to_idx[ch] for ch in seed_text]], dtype=torch.long
    ).to(device)
                                                                                            
    # Step 2: Prime -- feed entire seed to build up hidden state                       
    hidden = None         
    generated = list(seed_text)                                                                
                                                                                            
    with torch.no_grad():
        # Run seed through model - only care about the hidden state
        # and the LAST timestep's logits (prediction for what comes after seed)                                            
        logits, hidden = model(input_idx, hidden)

        # Apply temperature to last timestep's logits
        # logits[:, -1, :] -> grab only the last timestep: (1, vocab_size)                                                
        logits = logits[:, -1, :] / temperature  # Apply temperature                                                 
        probs = F.softmax(logits, dim=-1)   # Softmax   

        # Sample one character from the probability distribution                                                        
        current_idx = torch.multinomial(probs, num_samples=1)  # Sample one character  (1,1)                                        
        generated.append(idx_to_char[current_idx.item()])
                                                                                            
        # Step 3: Keep generating one char at a time (remaining characters)
        for _ in range(length - 1):                                                       
            # Feed current_idx + hidden → get next logits + updated hidden   
            # Feed the last geenrated character + carry forward hidden state
            logits, hidden = model(current_idx, hidden)

            # Same process: temperature -> softmax -> sample
            logits = logits[:, -1, :] / temperature                                             
            probs = F.softmax(logits, dim=-1)                                                                              
            current_idx = torch.multinomial(probs, num_samples=1)                                
            generated.append(idx_to_char[current_idx.item()])

    return "".join(generated)  

def plot_history(train_losses, val_losses, train_ppls, val_ppls, save_path=None):
    """Plot loss and perplexity curves (replaces loss+accuracy from CNN projects)."""
    epochs = range(1, len(train_losses) + 1)                                              
   
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))                                 
                  
    ax1.plot(epochs, train_losses, label="Train Loss", marker="o")                        
    ax1.plot(epochs, val_losses,   label="Val Loss",   marker="o")
    ax1.set_title("Loss per Epoch")                                                       
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")                                                                
    ax1.legend()
    ax1.grid(True)                                                                        
                  
    ax2.plot(epochs, train_ppls, label="Train Perplexity", marker="o")                    
    ax2.plot(epochs, val_ppls,   label="Val Perplexity",   marker="o")
    ax2.set_title("Perplexity per Epoch")                                                 
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Perplexity")
    ax2.legend()                                                                          
    ax2.grid(True)
                                                                                            
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")                                                     
    plt.close()
                                                                                            
                                                                                            
def plot_training_samples(samples_dict, save_path=None):
    """Display generated text samples from different epochs."""                           
    fig, ax = plt.subplots(figsize=(12, max(4, len(samples_dict) * 1.2)))                 
    ax.axis("off")                                                                        
    ax.set_title("Generated Text Across Training", fontsize=14, fontweight="bold", pad=20)
                                                                                            
    y = 0.95    

    for epoch, text in sorted(samples_dict.items()):                                      
        display_text = text[:80].replace("\n", "\\n")                                     
        ax.text(0.02, y, f"Epoch {epoch:>3d}:", fontsize=10, fontweight="bold",
                fontfamily="monospace", transform=ax.transAxes, verticalalignment="top")  
        ax.text(0.12, y, display_text, fontsize=9,
                fontfamily="monospace", transform=ax.transAxes, verticalalignment="top")  
        y -= 1.0 / (len(samples_dict) + 1)
                                                                                            
    plt.tight_layout()

    if save_path:                                                                         
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.close()        

# Smoke test: verify the dataset pipeline works
if __name__ == "__main__":
    train_loader, val_loader, test_loader, chars, char_to_idx, idx_to_char = get_dataloaders()
    x, y = next(iter(train_loader))
    print(f"Input shape:  {x.shape}")    # Expected: (64, 100)
    print(f"Target shape: {y.shape}")    # Expected: (64, 100)
    print(f"First input chars:  {[idx_to_char[i.item()] for i in x[0][:20]]}")
    print(f"First target chars: {[idx_to_char[i.item()] for i in y[0][:20]]}")    

    