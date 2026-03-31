from __future__ import annotations

import torch
import torch.nn as nn


class BiLSTMClassifier(nn.Module):
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
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        _, (h_n, _) = self.lstm(emb)
        h_forward = h_n[-2]
        h_backward = h_n[-1]
        h_cat = torch.cat([h_forward, h_backward], dim=1)
        logits = self.fc(self.dropout(h_cat))
        return logits
