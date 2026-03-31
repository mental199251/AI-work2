from __future__ import annotations

import torch.nn as nn

from src.models.bilstm import BiLSTMClassifier
from src.models.lstm import LSTMClassifier
from src.models.lstm_attention import LSTMAttentionClassifier
from src.models.rnn import RNNClassifier


MODEL_REGISTRY = {
    "rnn": RNNClassifier,
    "lstm": LSTMClassifier,
    "bilstm": BiLSTMClassifier,
    "lstm_attention": LSTMAttentionClassifier,
}


def build_model(
    model_name: str,
    vocab_size: int,
    embedding_dim: int,
    hidden_dim: int,
    num_classes: int,
    pad_idx: int = 0,
    dropout: float = 0.3,
) -> nn.Module:
    key = model_name.lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model: {model_name}")

    model_cls = MODEL_REGISTRY[key]
    return model_cls(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        pad_idx=pad_idx,
        dropout=dropout,
    )
