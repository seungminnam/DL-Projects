import os
import math
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)                    # 5_LSTM_Sentiment/
DATA_DIR    = os.path.join(PROJECT_DIR, "data")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

# Special tokens — reserved indices at the start of vocabulary
PAD_IDX = 0  # Padding: fills short sequences to fixed length
UNK_IDX = 1  # Unknown: replaces words not in our vocabulary


def get_device():
    """Detect best available device (MPS > CUDA > CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def build_vocab(texts, max_size=25000):
    """
    Build word-to-index mapping from training texts.

    Neural networks can't read text — they only understand numbers.
    word_to_idx is the conversion table:  "the" → 2, "movie" → 4, ...
    Later, a review like "the movie was great" becomes [2, 4, 87, 156]
    — a tensor of integers we can feed into nn.Embedding.

    Same idea as Shakespeare's char_to_idx, but at word level (25,000 vs 65).

    Why build from training data only? Same reason we fit a scaler on train only
    in ML — the model shouldn't "see" test vocabulary during training.

    Returns (word_to_idx, idx_to_word).
    """
    # TODO: Step 1 — Count every word across all reviews (use Counter)
    #       Hint: lowercase and split on whitespace
    counter = Counter()

    for text in texts:
        counter.update(text.lower().split())

    # TODO: Step 2 — Keep only the most common words (top max_size)
    most_common_words = counter.most_common(max_size)

    # TODO: Step 3 — Build word_to_idx dict
    #       Reserve index 0 for <pad>, 1 for <unk>
    #       Then assign indices 2, 3, 4... to the most common words
    word_to_idx = {"<pad>": PAD_IDX, "<unk>": UNK_IDX}  # indices 0 and 1 taken

    for word, _ in most_common_words:
        word_to_idx[word] = len(word_to_idx) #from index 2 (len = 2)

    # TODO: Step 4 — Build idx_to_word (reverse mapping)
    idx_to_word = {i: w for w, i in word_to_idx.items()}  # 2 → 'the', 3 → 'a', ...

    # TODO: Print vocab size and return (word_to_idx, idx_to_word)
    print(f"Vocabulary: {len(word_to_idx):,} words (top {max_size} + 2 special tokens)")
    return word_to_idx, idx_to_word

def encode_texts(texts, word_to_idx, max_len=256):
    """
    Tokenize texts and convert to padded index tensors.

    word_to_idx (from build_vocab) creates the lookup table,
    this function uses it to convert raw text → integer tensors.

    Pipeline:
        build_vocab(train_texts)  →  word_to_idx  →  encode_texts(texts, word_to_idx)

    NLP equivalent of image transforms in CNN projects:
    transforms.Resize(224) made all images the same size;
    here we first pad/truncate to a global ceiling (max_len=256).
    Later, our custom collate_fn will trim each batch down to only the
    longest real review in that batch, so we avoid wasting compute on extra pad.

    Each review becomes a fixed-length tensor of word indices:
        "this movie was great" → [45, 892, 23, 156, 0, 0, 0, ...]
                                                      ^^^ padding

    - Words not in vocab → UNK_IDX (1)
    - Sequences longer than max_len → truncated
    - Sequences shorter than max_len → padded with PAD_IDX (0)
    """
    # TODO: For each text:
    #   1. Lowercase and split into tokens
    #   2. Convert each token to its index (use .get() with UNK_IDX as default)
    #   3. Truncate to max_len OR pad with PAD_IDX to max_len
    #   4. Collect all encoded sequences
    encoded = []

    for text in texts:
        tokens = text.lower().split()
        indices = [word_to_idx.get(w, UNK_IDX) for w in tokens]

        # TODO: truncate or pad to max_len
        if len(indices) >= max_len:
            indices = indices[:max_len] #slicing: truncate to max_len 
        else:
            indices = indices + [PAD_IDX] * (max_len - len(indices)) #pad to max_len
        
        encoded.append(indices) #collect indices to encoded

    # TODO: Return as torch.tensor(..., dtype=torch.long)
    return torch.tensor(encoded, dtype=torch.long)


def collate_trim_to_batch_max(batch):
    """
    Stack a batch, then trim away extra right-padding beyond the longest
    real review in that batch.

    Why this helps:
    - The dataset is stored with a simple global max_len=256 for convenience.
    - But many batches do not actually need all 256 timesteps.
    - If the longest real review in a batch is length 143, we can slice the
      batch from (N, 256) down to (N, 143) before embedding/LSTM.

    This is a lightweight form of dynamic padding:
    simple to reason about, faster than always using 256, and fully compatible
    with pack_padded_sequence inside the model.
    """
    batch_x, batch_y = zip(*batch) 
    batch_x = torch.stack(batch_x)  # (N, 256) from the stored dataset
    batch_y = torch.stack(batch_y)  # (N,)

    lengths = (batch_x != PAD_IDX).sum(dim=1)
    lengths = torch.clamp(lengths, min=1)
    batch_max_len = int(lengths.max().item()) #get the longest review 
    #e.g. [3, 2, 5'] -> batch_max_len = 5

    # Keep only the timesteps this batch actually uses. - where trim happens
    batch_x = batch_x[:, :batch_max_len].contiguous()

    #trimmed batch
    return batch_x, batch_y


def get_dataloaders(batch_size=64, max_len=256, max_vocab=25000):
    """
    Download IMDB via Hugging Face, build vocab, encode, return DataLoaders.

    Returns (train_loader, val_loader, test_loader, word_to_idx, idx_to_word).
    """

    # TODO: Step 1 — Load IMDB dataset
    #       ds = load_dataset("imdb")
    dataset = load_dataset("imdb")

    # TODO: Step 2 — Extract texts and labels from train and test splits
    # Hint: Hugging Face dataset works like a dictionary of dictionaries: 
    # ds["train"]["text"]  # list of 25,000 review strings
    # ds["train"]["label"] # list of 25,000 labels (0 or 1) 

    train_texts  = dataset["train"]["text"] # list of 25,000 review strings (from train)
    train_labels = dataset["train"]["label"] # list of 25,000 labels (0 or 1) 
    test_texts   = dataset["test"]["text"] 
    test_labels  = dataset["test"]["label"]

    # TODO: Step 3 — Build vocabulary from training texts only
    word_to_idx, idx_to_word = build_vocab(train_texts, max_size=max_vocab)

    # TODO: Step 4 — Encode all texts to padded index tensors
    # use the same word_to_idx (built from training data) for both
    # test data uses the training vocabulary, just like in ML you fit a scaler on train and transform test with it
    train_enc = encode_texts(train_texts, word_to_idx, max_len)
    test_enc  = encode_texts(test_texts, word_to_idx, max_len)

    # TODO: Step 5 — Convert labels to float tensors (BCEWithLogitsLoss needs float)
    train_labels_t = torch.tensor(train_labels, dtype=torch.float32)
    test_labels_t = torch.tensor(test_labels, dtype=torch.float32)

    # TODO: Step 6 — Split training set → 20k train / 5k validation (stratified) -- IMDB dataset has no valid set. 
    #       Use sklearn.model_selection.train_test_split
    #       stratify=train_labels to keep pos/neg ratio balanced (50/50)
    #       random_state=42 for reproducibility

    train_idx, val_idx = train_test_split(
        range(len(train_enc)),      # indices 0..24999
        test_size=5000,             # 5k for validation
        random_state=42,            # reproducible
        stratify=train_labels       # keep pos/neg ratio balanced
    )

    # TODO: Step 7 — Create TensorDatasets and DataLoaders
    #       shuffle=True for train, False for val/test
    train_dataset = TensorDataset(train_enc[train_idx], train_labels_t[train_idx])
    val_dataset   = TensorDataset(train_enc[val_idx],   train_labels_t[val_idx])
    test_dataset  = TensorDataset(test_enc, test_labels_t)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_trim_to_batch_max, #using the custom trimming function
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_trim_to_batch_max,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_trim_to_batch_max,
    )

    # TODO: Print sizes for each set and return
    print(f"Train: {len(train_dataset):,} | Val: {len(val_dataset):,} | Test: {len(test_dataset):,}")
    return train_loader, val_loader, test_loader, word_to_idx, idx_to_word

def load_glove(glove_path, word_to_idx, embed_dim=100):
    """
    Build an embedding matrix from a GloVe file.

    For each word in our vocabulary, if GloVe has a vector for it,
    use that vector. Otherwise, initialize randomly.

    This is the text equivalent of loading pretrained ResNet weights
    in Project 3 — we start with knowledge someone else learned.

    Returns torch tensor of shape (vocab_size, embed_dim).
    """
    vocab_size = len(word_to_idx) #row size of embedding row (e.g. 25,002 vocab -> 25,002 em row)
    

    # Step 1: start with random vectors for eveyr vocab word
    embedding_matrix = np.random.normal(
        loc=0.0,
        scale=0.6,
        size=(vocab_size, embed_dim),
    ).astype(np.float32)
    # known word -> pretrained vector
    # unknown/missing word -> random vector

    # Step 2. keeping padding as a zero vector (<pad> is not an actual vocab)
    embedding_matrix[PAD_IDX] = np.zeros(embed_dim, dtype=np.float32)

    found = 0

    # Step 3. read the GloVe text file line by line
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            #split line by space
            parts = line.strip().split()

            # Skip empty / malformed lines
            if not parts:
                continue
            
            # e.g. the 0.418 0.24968 -0.41242 ...
            # parts[0] = "the"
            # parts[1:] = remaining numbers
            word = parts[0]
            vector = parts[1:]

            # Step 4: only keep rows with the expected embedding dimension 
            # we're using 100d GloVe -> skip lines that the vector length isn't 100
            if len(vector) != embed_dim:
                continue

            # Step 5: if this GloVe word exists in our vocab, overwrite that row
            # among enormous GloVe vocab, we only needs vocab that is in our word
            # instead of saving the entire GloVe, picking the word we need in our vocab row. 
            if word in word_to_idx:
                idx = word_to_idx[word]
                #vector is still a String list -> us asarray() to make it float array
                embedding_matrix[idx] = np.asarray(vector, dtype=np.float32)
                found += 1
    
    coverage = found / vocab_size * 100
    print(f"GloVe matched {found:,} / {vocab_size:,} words ({coverage:.2f}%)")

    # Return torch tensor so model.py can use it directly with from_pretrained()
    return torch.tensor(embedding_matrix, dtype=torch.float32)


# ── Training & Evaluation ────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device,
                    max_grad_norm=1.0) -> tuple[float, float]:
    """
    Run one training epoch. Returns (avg_loss, accuracy).

    Key differences from Shakespeare's train_one_epoch:
    - Returns accuracy instead of perplexity (classification, not generation)
    - Gentler gradient clipping (1.0 vs 5.0) — LSTM gates reduce exploding gradients
    - squeeze(1) on logits to match label shape: (N, 1) → (N,)
    - Accuracy uses sigmoid > 0.5 threshold (not argmax like CNN projects)
    """

    model.train()
    total_loss = 0.0 
    total_correct = 0 
    total_samples = 0
    
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device) #input feature (N, max_len) batch token sequence (review text -> integer indices)
        batch_y = batch_y.to(device) #label batch (N, ) (positive or negative sentiment)
        
        optimizer.zero_grad()
        #forward pass
        logits = model(batch_x) #raw score (N, 1)
        logits = logits.squeeze(1) #(N, 1) -> (N,)
        #logits & labels should have the same shape for BCEWithLogitsLoss (N,) (binary classification)
        loss = criterion(logits, batch_y) #BCEWithLogitsLoss(logits, labels)
        
        loss.backward() #backward pass (compute gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm) #gradient clipping to prevent exploding gradients
        optimizer.step() #update parameters using gradients

        probs = torch.sigmoid(logits) #transform raw score to probability of positive sentiment (0-1) - for accuracy calculation
        preds = (probs > 0.5).float() #convert probability to binary prediction (0 or 1) 

        batch_size = batch_y.size(0) #number of samples in the batch
        total_loss += loss.item() * batch_size #accumulate loss for each batch
        total_correct += (preds == batch_y).sum().item() # count correct predictions (binary classification)
        total_samples += batch_size #accumulate batch size (total number of samples)

    avg_loss = total_loss / total_samples # average loss per sample
    accuracy = total_correct / total_samples #accuracy for the entire epoch
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    """Evaluate model on a dataset. Returns (avg_loss, accuracy).
    
    No backward / clip / optimizer.step() 
    Use torch.no_grad() to disable gradient calculation"""

    model.eval()
    total_loss = 0.0 # Total loss for the entire epoch
    total_correct = 0 # Total correct predictions for the entire epoch
    total_samples = 0 # Total number of samples for the entire epoch

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # Forward pass
            logits = model(batch_x)
            logits = logits.squeeze(1)

            # Compute loss
            loss = criterion(logits, batch_y) #compare logits (N,) and labels (N,) for BCEWithLogitsLoss

            probs = torch.sigmoid(logits) # Transform raw score to probability of positive sentiment (0-1) - for accuracy calculation
            preds = (probs > 0.5).float() # Convert probability to binary prediction (0 or 1) 

            batch_size = batch_y.size(0) # Number of samples in the batch
            total_loss += loss.item() * batch_size # Accumulate loss for each batch
            total_correct += (preds == batch_y).sum().item() # Count correct predictions (binary classification)
            total_samples += batch_size # Accumulate batch size (total number of samples)

    avg_loss = total_loss / total_samples # Average loss per sample
    accuracy = total_correct / total_samples # Accuracy for the entire epoch

    return avg_loss, accuracy


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_history(train_losses, val_losses, train_accs, val_accs, save_path=None):
    """Plot loss and accuracy curves (same 2-panel style as CNN projects)."""
    epochs = range(1, len(train_losses) + 1)
    train_accs_pct = [acc * 100 for acc in train_accs]
    val_accs_pct = [acc * 100 for acc in val_accs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_losses, label='Train Loss', marker='o')
    ax1.plot(epochs, val_losses,   label='Val Loss',   marker='o')
    ax1.set_title('Loss per Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, train_accs_pct, label='Train Acc', marker='o')
    ax2.plot(epochs, val_accs_pct,   label='Val Acc',   marker='o')
    ax2.set_title('Accuracy per Epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved → {save_path}")
    plt.close()


def plot_confusion_matrix(model, loader, device, class_names=None, save_path=None):
    """Plot 2x2 confusion matrix for binary sentiment classification."""
    cm = np.zeros((2,2), dtype=int)

    if class_names is None:
        class_names = ["Negative", "Positive"]
    
    model.eval() 
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)

            logits = model(batch_x).squeeze(1) # (N, )
            probs = torch.sigmoid(logits) # (N, )

            #thresholding 
            preds = (probs > 0.5).long().cpu() # 0 or 1
            labels = (batch_y).long().cpu() # 0 or 1
            
            # Confusion matrix indexing:
            # cm[0][0] = true negative,  cm[0][1] = false positive
            # cm[1][0] = false negative, cm[1][1] = true positive
            for t, p in zip(labels.numpy(), preds.numpy()):
                # t = true label 
                # p = predicted label
                cm[t][p] += 1
    
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(cm, cmap="Blues") #Matrix heatmap
    plt.colorbar(im, ax=ax)

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.close()


def save_wrong_predictions_report(model, loader, device, idx_to_word, n=10, save_path=None):
    """
    Save a small markdown report of misclassified reviews with true label,
    predicted label, and model confidence.

    Why this function matters:
    - Confusion matrix tells us the overall error pattern.
    - Wrong-prediction examples show the actual reviews the model struggles with.
    - This is often the most useful artifact for README analysis because we can inspect
      whether errors come from mixed sentiment, sarcasm, very short reviews, etc.
    - It is more useful to separate false positives from
      false negatives than to mix them together. That makes it easier to explain
      whether the model is overly optimistic, overly conservative, or confused by
      a specific style of review.

    Important implementation details:
    - Reviews are stored as padded index sequences, so we remove tokens after PAD_IDX.
      Otherwise the reconstructed text would contain meaningless <pad> tail tokens.
    - We convert token ids back to words using idx_to_word so humans can read them.
    - We show only the first part of long reviews to keep the figure readable.
      The goal is quick qualitative inspection, not printing the entire review.
    - Confidence should match the predicted class:
        if pred = Positive -> use P(positive)
        if pred = Negative -> use 1 - P(positive)
      This makes the displayed number read naturally as "how confident was the model
      in the label it actually predicted?"
    """
    per_type = max(1, n // 2)
    false_positives = []
    false_negatives = []

    class_names = ["Negative", "Positive"]

    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x_device = batch_x.to(device)

            logits = model(batch_x_device).squeeze(1)   # (N,)
            probs = torch.sigmoid(logits).cpu()         # probability of positive class
            preds = (probs > 0.5).long()                # predicted label: 0 or 1
            labels = batch_y.long()                     # true label: 0 or 1

            wrong_mask = preds.ne(labels) #ne = not equal (!=) outputs Boolean
            wrong_indices = wrong_mask.nonzero(as_tuple=False).squeeze(1)

            for idx in wrong_indices.tolist():
                token_ids = batch_x[idx].tolist()
                #batch_x[idx] = token index for one review

                # Remove right-padding and convert token ids back to words.
                words = []
                for token_id in token_ids:
                    if token_id == PAD_IDX:
                        break
                    words.append(idx_to_word.get(token_id, "<unk>"))

                text = " ".join(words)

                # Keep the figure readable: show only the beginning of long reviews.
                preview_words = words[:40]
                preview_text = " ".join(preview_words)
                if len(words) > 40:
                    preview_text += " ..."

                # Confidence should reflect the predicted class, not always P(positive).
                # If predicted positive -> confidence = prob
                # If predicted negative -> confidence = 1 - prob
                confidence = probs[idx].item() if preds[idx].item() == 1 else 1.0 - probs[idx].item()

                example = {
                    "text": preview_text if preview_text else text,
                    "true": class_names[labels[idx].item()],
                    "pred": class_names[preds[idx].item()],
                    "conf": confidence,
                }

                # Split errors by direction so we can inspect both:
                # false positive  = true negative, predicted positive
                # false negative  = true positive, predicted negative
                if labels[idx].item() == 0 and preds[idx].item() == 1:
                    if len(false_positives) < per_type:
                        false_positives.append(example)
                elif labels[idx].item() == 1 and preds[idx].item() == 0:
                    if len(false_negatives) < per_type:
                        false_negatives.append(example)

                if len(false_positives) >= per_type and len(false_negatives) >= per_type:
                    break

            if len(false_positives) >= per_type and len(false_negatives) >= per_type:
                break

    def build_section(title, examples):
        lines = [f"## {title}", ""]
        if not examples:
            lines.append("No examples collected.")
            lines.append("")
            return lines

        for i, example in enumerate(examples, start=1):
            lines.append(
                f"{i}. **True:** {example['true']} | **Pred:** {example['pred']} "
                f"| **Confidence:** {example['conf']:.2f}"
            )
            lines.append(f"   - Review: {example['text']}")
            lines.append("")
        return lines

    lines = [
        "# Wrong Prediction Report",
        "",
        f"- False Positives collected: {len(false_positives)}",
        f"- False Negatives collected: {len(false_negatives)}",
        f"- Requested total examples: {n}",
        "",
        "False positive means a negative review predicted as positive.",
        "False negative means a positive review predicted as negative.",
        "",
    ]
    lines.extend(build_section("False Positives (Negative -> Positive)", false_positives))
    lines.extend(build_section("False Negatives (Positive -> Negative)", false_negatives))
    report = "\n".join(lines)

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Saved -> {save_path}")

    return report


# ── Smoke Test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_loader, val_loader, test_loader, word_to_idx, idx_to_word = get_dataloaders()
    x, y = next(iter(train_loader))
    print(f"Input shape:  {x.shape}")    # Expected: (64, batch_max_len)
    print(f"Labels shape: {y.shape}")    # Expected: (64,)
    print(f"Label values: {y[:10]}")     # Mix of 0.0 and 1.0
    print(f"First 10 words: {[idx_to_word.get(i.item(), '?') for i in x[0][:10]]}")
