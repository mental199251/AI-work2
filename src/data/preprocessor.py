from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    import jieba  # type: ignore
except Exception:
    jieba = None

try:
    from nltk.tokenize import wordpunct_tokenize
except Exception:
    wordpunct_tokenize = None


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


@dataclass
class DataBundle:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    vocab: Dict[str, int]
    id_to_label: Dict[int, str]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def num_classes(self) -> int:
        return len(self.id_to_label)


class TextPreprocessor:
    def __init__(
        self,
        language: str,
        max_vocab_size: int,
        min_freq: int,
        max_len: int,
        stopwords_path: Optional[str] = None,
    ) -> None:
        self.language = language.lower()
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.max_len = max_len
        self.stopwords = self._load_stopwords(stopwords_path)
        self.vocab: Dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}

    @staticmethod
    def _load_stopwords(stopwords_path: Optional[str]) -> set:
        if not stopwords_path:
            return set()
        path = Path(stopwords_path)
        if not path.exists():
            return set()
        return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}

    def _clean_en(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"<.*?>", " ", text)
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _clean_zh(self, text: str) -> str:
        text = re.sub(r"\s+", "", text)
        text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _tokenize(self, text: str) -> List[str]:
        if self.language == "en":
            text = self._clean_en(text)
            if wordpunct_tokenize is not None:
                tokens = wordpunct_tokenize(text)
            else:
                tokens = text.split()
        else:
            text = self._clean_zh(text)
            if jieba is not None:
                tokens = jieba.lcut(text)
            else:
                tokens = list(text.replace(" ", ""))

        if self.stopwords:
            tokens = [tok for tok in tokens if tok not in self.stopwords]

        tokens = [tok for tok in tokens if tok.strip()]
        return tokens

    def build_vocab(self, tokenized_texts: List[List[str]]) -> None:
        counter = Counter()
        for tokens in tokenized_texts:
            counter.update(tokens)

        candidate_items = [
            (token, freq)
            for token, freq in counter.most_common(self.max_vocab_size)
            if freq >= self.min_freq
        ]

        self.vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        for token, _ in candidate_items:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)

    def encode_and_pad(self, tokenized_texts: List[List[str]]) -> np.ndarray:
        unk_id = self.vocab[UNK_TOKEN]
        pad_id = self.vocab[PAD_TOKEN]
        encoded = np.full((len(tokenized_texts), self.max_len), pad_id, dtype=np.int64)

        for i, tokens in enumerate(tokenized_texts):
            ids = [self.vocab.get(tok, unk_id) for tok in tokens[: self.max_len]]
            encoded[i, : len(ids)] = ids
        return encoded

    def fit_transform(self, train_texts: List[str], all_texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        train_tokens = [self._tokenize(t) for t in train_texts]
        self.build_vocab(train_tokens)

        all_tokens = [self._tokenize(t) for t in all_texts]
        all_encoded = self.encode_and_pad(all_tokens)
        train_encoded = self.encode_and_pad(train_tokens)
        return train_encoded, all_encoded


def _read_imdb_from_csv(csv_path: str, text_col: str, label_col: str) -> Tuple[List[str], List[str]]:
    df = pd.read_csv(csv_path)
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"CSV must contain columns: {text_col}, {label_col}")
    texts = df[text_col].astype(str).tolist()
    labels = df[label_col].astype(str).tolist()
    return texts, labels


def _read_imdb_from_acl_dir(root_dir: str) -> Tuple[List[str], List[str]]:
    root = Path(root_dir)
    texts: List[str] = []
    labels: List[str] = []

    label_map = {"pos": "positive", "neg": "negative"}
    for split in ["train", "test"]:
        for raw_label, norm_label in label_map.items():
            data_dir = root / split / raw_label
            if not data_dir.exists():
                continue
            for txt_file in data_dir.glob("*.txt"):
                try:
                    texts.append(txt_file.read_text(encoding="utf-8", errors="ignore"))
                    labels.append(norm_label)
                except Exception:
                    continue

    if not texts:
        raise ValueError("No IMDB texts found in ACL directory structure.")

    return texts, labels


def load_imdb_texts(data_path: str, text_col: str, label_col: str) -> Tuple[List[str], List[str]]:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")

    if path.is_file():
        return _read_imdb_from_csv(str(path), text_col, label_col)
    return _read_imdb_from_acl_dir(str(path))


def load_thucnews_texts(data_dir: str, max_samples_per_class: Optional[int] = None) -> Tuple[List[str], List[str]]:
    root = Path(data_dir)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"THUCNews directory not found: {data_dir}")

    texts: List[str] = []
    labels: List[str] = []

    class_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if not class_dirs:
        raise ValueError("No class directories found in THUCNews path.")

    for cls_dir in class_dirs:
        files = sorted(cls_dir.glob("*.txt"))
        if max_samples_per_class is not None:
            files = files[:max_samples_per_class]
        for txt_file in files:
            try:
                texts.append(txt_file.read_text(encoding="utf-8", errors="ignore"))
                labels.append(cls_dir.name)
            except Exception:
                continue

    if not texts:
        raise ValueError("No text samples loaded from THUCNews directory.")

    return texts, labels


def _encode_labels(labels: List[str]) -> Tuple[np.ndarray, Dict[int, str]]:
    unique_labels = sorted(set(labels))
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    id_to_label = {i: label for label, i in label_to_id.items()}
    y = np.array([label_to_id[label] for label in labels], dtype=np.int64)
    return y, id_to_label


def _split_indices(y: np.ndarray, random_state: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_idx = np.arange(len(y))
    stratify = y if len(np.unique(y)) > 1 else None

    try:
        train_idx, temp_idx = train_test_split(
            all_idx,
            test_size=0.30,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError:
        train_idx, temp_idx = train_test_split(
            all_idx,
            test_size=0.30,
            random_state=random_state,
            stratify=None,
        )

    stratify_temp = y[temp_idx] if len(np.unique(y[temp_idx])) > 1 else None
    try:
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=2 / 3,
            random_state=random_state,
            stratify=stratify_temp,
        )
    except ValueError:
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=2 / 3,
            random_state=random_state,
            stratify=None,
        )

    return train_idx, val_idx, test_idx


def build_data_bundle(
    dataset: str,
    data_path: str,
    language: str,
    text_col: str,
    label_col: str,
    max_vocab_size: int,
    min_freq: int,
    max_len: int,
    random_state: int,
    stopwords_path: Optional[str] = None,
    max_samples_per_class: Optional[int] = None,
) -> DataBundle:
    dataset = dataset.lower()

    if dataset == "imdb":
        texts, labels = load_imdb_texts(data_path, text_col, label_col)
    elif dataset == "thucnews":
        texts, labels = load_thucnews_texts(data_path, max_samples_per_class=max_samples_per_class)
    else:
        raise ValueError("dataset must be one of: imdb, thucnews")

    y, id_to_label = _encode_labels(labels)
    train_idx, val_idx, test_idx = _split_indices(y, random_state=random_state)

    train_texts = [texts[i] for i in train_idx]
    all_texts = texts

    preprocessor = TextPreprocessor(
        language=language,
        max_vocab_size=max_vocab_size,
        min_freq=min_freq,
        max_len=max_len,
        stopwords_path=stopwords_path,
    )

    _, all_encoded = preprocessor.fit_transform(train_texts=train_texts, all_texts=all_texts)

    x_train = all_encoded[train_idx]
    y_train = y[train_idx]
    x_val = all_encoded[val_idx]
    y_val = y[val_idx]
    x_test = all_encoded[test_idx]
    y_test = y[test_idx]

    return DataBundle(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        vocab=preprocessor.vocab,
        id_to_label=id_to_label,
    )
