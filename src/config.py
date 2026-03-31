from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainConfig:
    dataset: str = "imdb"
    data_path: str = "data/raw/imdb.csv"
    text_col: str = "text"
    label_col: str = "label"
    language: str = "en"

    max_vocab_size: int = 30000
    min_freq: int = 2
    max_len: int = 300

    embedding_dim: int = 128
    hidden_dim: int = 128
    dropout: float = 0.3

    batch_size: int = 64
    epochs: int = 10
    learning_rate: float = 1e-3
    early_stop_patience: int = 3

    seed: int = 42
