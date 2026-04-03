import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class SentimentLSTM(nn.Module):
    """
    Binary sentiment classifier with Embedding -> LSTM -> Linear.

    Pipeline:
        Input (N, seq_len_in_batch)        -- word indices
        -> Embedding (N, seq_len_in_batch, 100) -- dense word vectors
        -> Packed LSTM                    -- skip padded timesteps
        -> Final hidden (N, 256)          -- summary of entire review
        -> Linear (N, 1)                  -- raw sentiment logit
    """

    def __init__(self, vocab_size, embed_dim=100, hidden_size=256):
        super().__init__()
        self.vocab_size = vocab_size #number of unique words in the vocabulary (include <pad> & <unk>)
        self.embed_dim = embed_dim #dimension of the word vectors (in this case, 100 numbers describing a single word)
        self.hidden_size = hidden_size #dimension of the hidden state ("memory")
        self.pad_idx = 0  # Keep padding token index explicit for readability in forward()

        # Embedding layer: maps each word index to a dense vector.
        # Input:  (N, seq_len_in_batch)
        # Output: (N, seq_len_in_batch, embed_dim)
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0,  # <pad> token should not learn useful semantics
        )

        # LSTM layer: reads the review left-to-right and updates memory over time.
        # Input to the raw module can be:
        #   - padded embeddings: (N, seq_len_in_batch, embed_dim)
        #   - or a PackedSequence after pack_padded_sequence(...)
        # In our forward() we use the packed form, and we mainly care about:
        #   h_n shape = (num_layers, N, hidden_size)
        self.lstm = nn.LSTM(
            input_size=embed_dim, #one time step input vector size (100 in this case)
            hidden_size=hidden_size, #in this case, LSTM updates 256-dim hidden state for every word. 
            num_layers=1, #single-layer LSTM 
            batch_first=True, #N (batch) comes first
        )

        # Final classifier layer: turns the review summary vector into one logit.
        # Input:  (N, hidden_size) <- entire review summary vector (N, 256) 
        # Output: (N, 1) <- one logit score 
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        x: (N, seq_len_in_batch) -- right-padded word-index sequences

        Returns:
            logits: (N, 1) -- raw score for positive sentiment

        x                               # (N, seq_len_in_batch)
        -> self.embedding(x)            # (N, seq_len_in_batch, 100)
        -> pack_padded_sequence(...)    # packed real tokens only
        -> self.lstm(...)               # h_n: (1, N, 256)
        -> h_n[-1]                      # (N, 256)
        -> self.fc(...)                 # (N, 1)

        """
        # Count how many real (non-pad) tokens each review has.
        # Example:
        #   [45, 892, 23, 156, 0, 0, 0] -> length 4
        #
        # We compute this BEFORE embedding, while x is still integer word indices.
        lengths = (x != self.pad_idx).sum(dim=1)

        # Safety guard: in practice reviews should not be empty, but clamp avoids
        # invalid zero-length sequences if a sample somehow contains only padding.
        lengths = torch.clamp(lengths, min=1)

        # Step 1: word indices -> dense word vectors
        # (N, seq_len_in_batch) -> (N, seq_len_in_batch, embed_dim)
        x = self.embedding(x)

        # Step 2: pack the batch so the LSTM processes only real tokens.
        #
        # Why this is the standard fix:
        #   Our reviews have different lengths, so we pad short reviews with 0s.
        #   Without packing, the LSTM would keep stepping through those padding
        #   tokens, which can blur the final summary for short reviews.
        #
        # Packing tells PyTorch:
        #   "For each review, only the first `lengths[i]` timesteps are real.
        #    Ignore the padded tail."
        #
        # enforce_sorted=False lets us keep the normal DataLoader order instead
        # of sorting each batch by length ourselves.
        packed_x = pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        # Step 3: run the packed sequence through the LSTM.
        # Because padding is skipped, h_n[-1] now corresponds to the last REAL
        # token of each review, not the last padded timestep.
        _, (h_n, _) = self.lstm(packed_x)

        # Final review summary: (N, hidden_size)
        final_hidden = h_n[-1]

        # Step 4: review summary -> one raw sentiment score
        # (N, hidden_size) -> (N, 1)
        logits = self.fc(final_hidden)

        # Step 5: return raw logits (NOT sigmoid probabilities).
        # BCEWithLogitsLoss will apply sigmoid internally in a numerically stable way.
        return logits

# Smoke test: catch shape/forward errors early before wiring the full training loop.
if __name__ == "__main__":  # Only runs when model.py itself is executed.
    vocab_size = 10002
    model = SentimentLSTM(vocab_size=vocab_size)
    print(model)

    # Random word indices with realistic right-padding.
    # Review 0 uses 180 real tokens, review 1 uses 64 real tokens.
    dummy = torch.randint(2, vocab_size, (2, 256))
    dummy[0, 180:] = 0
    dummy[1, 64:] = 0
    logits = model(dummy)

    print(f"Input shape:  {dummy.shape}")   # (2, 256)
    print(f"Output shape: {logits.shape}")  # (2, 1)

    total = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total:,}")
