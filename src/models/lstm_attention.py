from __future__ import annotations

import torch
import torch.nn as nn


class LSTMAttentionClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_classes: int,
        pad_idx: int = 0,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.attn_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_score = nn.Linear(hidden_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        outputs, _ = self.lstm(emb)

        scores = self.attn_score(torch.tanh(self.attn_proj(outputs))).squeeze(-1)
        mask = x.ne(self.pad_idx)
        scores = scores.masked_fill(~mask, -1e9)

        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), outputs).squeeze(1)
        logits = self.fc(self.dropout(context))
        return logits
