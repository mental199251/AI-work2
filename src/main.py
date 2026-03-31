from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import TextDataset
from src.data.preprocessor import PAD_TOKEN, build_data_bundle
from src.evaluate import compute_metrics, predict
from src.models import build_model
from src.trainers.trainer import train_model
from src.utils.io_utils import ensure_dir, save_json
from src.utils.seed import set_seed
from src.utils.visualization import plot_confusion_matrix, plot_metric_comparison, plot_train_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Text Classification: RNN/LSTM/BiLSTM/LSTM+Attention")

    parser.add_argument("--dataset", type=str, default="imdb", choices=["imdb", "thucnews"])
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset file/folder")
    parser.add_argument("--text_col", type=str, default="text", help="Text column for CSV datasets")
    parser.add_argument("--label_col", type=str, default="label", help="Label column for CSV datasets")
    parser.add_argument("--language", type=str, default="en", choices=["en", "zh"])
    parser.add_argument("--stopwords_path", type=str, default=None)
    parser.add_argument("--max_samples_per_class", type=int, default=None)

    parser.add_argument("--max_vocab_size", type=int, default=30000)
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--max_len", type=int, default=300)

    parser.add_argument("--models", type=str, default="rnn,lstm,bilstm,lstm_attention")
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--early_stop_patience", type=int, default=3)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    parser.add_argument("--output_dir", type=str, default="outputs")

    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_dataloaders(bundle, batch_size: int):
    train_dataset = TextDataset(bundle.x_train, bundle.y_train)
    val_dataset = TextDataset(bundle.x_val, bundle.y_val)
    test_dataset = TextDataset(bundle.x_test, bundle.y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def parse_model_list(models_str: str) -> List[str]:
    models = [m.strip().lower() for m in models_str.split(",") if m.strip()]
    if not models:
        raise ValueError("No models specified.")
    return models


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = get_device(args.device)
    print(f"Using device: {device}")

    ensure_dir(args.output_dir)
    for sub in ["models", "figures", "metrics", "history"]:
        ensure_dir(os.path.join(args.output_dir, sub))

    print("Loading and preprocessing dataset...")
    bundle = build_data_bundle(
        dataset=args.dataset,
        data_path=args.data_path,
        language=args.language,
        text_col=args.text_col,
        label_col=args.label_col,
        max_vocab_size=args.max_vocab_size,
        min_freq=args.min_freq,
        max_len=args.max_len,
        random_state=args.seed,
        stopwords_path=args.stopwords_path,
        max_samples_per_class=args.max_samples_per_class,
    )

    print(
        f"Data ready: train={len(bundle.x_train)}, val={len(bundle.x_val)}, test={len(bundle.x_test)}, "
        f"vocab_size={bundle.vocab_size}, classes={bundle.num_classes}"
    )

    save_json(
        {
            "id_to_label": bundle.id_to_label,
            "vocab_size": bundle.vocab_size,
            "train_size": int(len(bundle.x_train)),
            "val_size": int(len(bundle.x_val)),
            "test_size": int(len(bundle.x_test)),
        },
        os.path.join(args.output_dir, "metrics", "dataset_info.json"),
    )

    train_loader, val_loader, test_loader = prepare_dataloaders(bundle, args.batch_size)

    model_names = parse_model_list(args.models)
    results = []

    pad_idx = bundle.vocab.get(PAD_TOKEN, 0)

    for model_name in model_names:
        print("=" * 70)
        print(f"Training model: {model_name}")

        model = build_model(
            model_name=model_name,
            vocab_size=bundle.vocab_size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_classes=bundle.num_classes,
            pad_idx=pad_idx,
            dropout=args.dropout,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        criterion = nn.CrossEntropyLoss()

        history, best_state = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epochs=args.epochs,
            early_stop_patience=args.early_stop_patience,
        )

        model.load_state_dict(best_state)

        model_path = os.path.join(args.output_dir, "models", f"{model_name}.pt")
        torch.save(best_state, model_path)

        history_path = os.path.join(args.output_dir, "history", f"{model_name}_history.json")
        save_json(history, history_path)

        curve_path = os.path.join(args.output_dir, "figures", f"{model_name}_curves.png")
        plot_train_history(history, curve_path, title=model_name.upper())

        y_true, y_pred = predict(model, test_loader, device)
        metric = compute_metrics(y_true, y_pred)
        metric["model"] = model_name
        results.append(metric)

        cm_path = os.path.join(args.output_dir, "figures", f"{model_name}_confusion_matrix.png")
        label_names = [bundle.id_to_label[i] for i in range(bundle.num_classes)]
        plot_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            labels=label_names,
            output_path=cm_path,
            title=f"{model_name.upper()} Confusion Matrix",
        )

        print(f"Test metrics ({model_name}): {metric}")

    results_df = pd.DataFrame(results)
    results_df = results_df[
        [
            "model",
            "accuracy",
            "precision_weighted",
            "recall_weighted",
            "f1_weighted",
            "precision_macro",
            "recall_macro",
            "f1_macro",
        ]
    ]

    metric_csv_path = os.path.join(args.output_dir, "metrics", "model_comparison.csv")
    results_df.to_csv(metric_csv_path, index=False, encoding="utf-8-sig")

    compare_fig_path = os.path.join(args.output_dir, "figures", "model_comparison.png")
    plot_metric_comparison(results_df, compare_fig_path)

    print("=" * 70)
    print("Experiment completed. Metrics summary:")
    print(results_df.to_string(index=False))
    print(f"\nSaved metrics to: {metric_csv_path}")
    print(f"Saved figure to: {compare_fig_path}")


if __name__ == "__main__":
    main()
