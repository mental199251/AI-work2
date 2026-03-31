from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader


@torch.no_grad()
def predict(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        logits = model(batch_x)
        preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()

        y_pred.extend(preds)
        y_true.extend(batch_y.numpy().tolist())

    return np.array(y_true), np.array(y_pred)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)

    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    return {
        "accuracy": float(acc),
        "precision_weighted": float(p_weighted),
        "recall_weighted": float(r_weighted),
        "f1_weighted": float(f1_weighted),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),
    }
