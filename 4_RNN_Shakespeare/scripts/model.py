import torch
import torch.nn as nn
import torch.nn.functional as F


class ShakespeareRNN(nn.Module):
    """
    Character-level vanilla RNN for text generation.

    Pipeline:
        Input (N, seq_len)           — character indices
        → One-hot (N, seq_len, 65)   — one-hot encoded
        → RNN (N, seq_len, 512)      — hidden state at each timestep
        → Linear (N, seq_len, 65)    — logits for next character
    """
    def __init__(self, vocab_size, hidden_size=512):
        super().__init__()
        self.vocab_size = vocab_size  # Needed in forward() for one-hot encoding
        self.hidden_size = hidden_size

        # RNN: at each timestep, takes a 65-dim one-hot vector + previous hidden state
        #       → outputs a new 512-dim hidden state
        self.rnn = nn.RNN(
            input_size=vocab_size,    # 65 (one-hot size)
            hidden_size=hidden_size,  # 512
            num_layers=1,             # Single layer vanilla RNN
            batch_first=True,         # Input shape: (N, seq_len, input_size)
        )

        # Linear: maps 512-dim hidden state → 65 logits (one per character)
        self.fc = nn.Linear(hidden_size, vocab_size)  # (N, seq_len, 512) → (N, seq_len, 65)

    def forward(self, x, hidden=None):
        """
        x: (N, seq_len) — character indices
        hidden: (1, N, hidden_size) — optional initial hidden state
        Returns: (logits, hidden)
        """
        # Step 1: One-hot encode input indices
        # (N, seq_len) → (N, seq_len, vocab_size)
        x = F.one_hot(x, num_classes=self.vocab_size).float()

        # Step 2: RNN forward — pass one-hot + previous hidden state
        # out: (N, seq_len, hidden_size) — hidden state at EVERY timestep
        # hidden: (1, N, hidden_size) — FINAL hidden state only
        out, hidden = self.rnn(x, hidden)

        # Step 3: Project each hidden state to character logits
        # (N, seq_len, hidden_size) → (N, seq_len, vocab_size)
        logits = self.fc(out)

        return logits, hidden


# Smoke test: verify model shapes
if __name__ == "__main__":
    vocab_size = 65
    model = ShakespeareRNN(vocab_size=vocab_size, hidden_size=512)
    print(model)

    # Random character indices: batch=2, seq_len=100, values 0-64
    dummy = torch.randint(0, vocab_size, (2, 100))
    logits, hidden = model(dummy)

    print(f"Input shape:  {dummy.shape}")   # (2, 100)
    print(f"Output shape: {logits.shape}")  # (2, 100, 65)
    print(f"Hidden shape: {hidden.shape}")  # (1, 2, 512)

    total = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total:,}")
