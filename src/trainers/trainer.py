from __future__ import annotations

import copy
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean().item()


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_acc = 0.0

    for batch_x, batch_y in tqdm(dataloader, desc="Train", leave=False):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_acc += _accuracy(logits, batch_y)

    epoch_loss = running_loss / max(len(dataloader), 1)
    epoch_acc = running_acc / max(len(dataloader), 1)
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    running_acc = 0.0

    for batch_x, batch_y in tqdm(dataloader, desc="Val", leave=False):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        logits = model(batch_x)
        loss = criterion(logits, batch_y)

        running_loss += loss.item()
        running_acc += _accuracy(logits, batch_y)

    epoch_loss = running_loss / max(len(dataloader), 1)
    epoch_acc = running_acc / max(len(dataloader), 1)
    return epoch_loss, epoch_acc


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    epochs: int,
    early_stop_patience: int = 3,
) -> Tuple[Dict[str, list], Dict[str, torch.Tensor]]:
    history: Dict[str, list] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        val_loss, val_acc = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch}/{epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    return history, best_state
