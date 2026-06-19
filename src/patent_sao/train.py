from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             matthews_corrcoef, precision_recall_curve, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import RobertaModel, RobertaTokenizerFast, get_linear_schedule_with_warmup

from .model import FocalLoss, SAO_BC, SAOTripleDataset
from .utils import save_json, set_seed


def _vocab(cluster_labels: dict[str, str]):
    entity_keys = sorted(key for key in cluster_labels if key.startswith(("s", "o")))
    relation_keys = sorted(key for key in cluster_labels if key.startswith("a"))
    return {key: index for index, key in enumerate(entity_keys)}, {key: index for index, key in enumerate(relation_keys)}


def _batch_forward(model, batch, device):
    return model(batch["head_text"], batch["rel_text"], batch["tail_text"],
                 batch["head_ids"].to(device), batch["rel_ids"].to(device), batch["tail_ids"].to(device))

@torch.no_grad()
def _predict(model, loader, device):
    model.eval()
    logits, labels = [], []
    for batch in tqdm(loader, desc="Evaluate"):
        output, _ = _batch_forward(model, batch, device)
        logits.extend(output.cpu().numpy())
        labels.extend(batch["label"].numpy())
    return 1 / (1 + np.exp(-np.asarray(logits))), np.asarray(labels, dtype=int)


def _best_threshold(labels, probabilities):
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    if not len(thresholds):
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def train_and_evaluate(dataset_pkl: str | Path, labels_pkl: str | Path, model_path: str | Path,
                       metrics_path: str | Path, predictions_path: str | Path, curve_path: str | Path,
                       settings: dict, seed: int = 42) -> dict:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with Path(dataset_pkl).open("rb") as file:
        X, y = pickle.load(file)
    with Path(labels_pkl).open("rb") as file:
        cluster_labels = {key: value for key, value in pickle.load(file).items() if isinstance(key, str)}
    entity2idx, relation2idx = _vocab(cluster_labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=settings["test_size"], random_state=seed)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, stratify=y_train, test_size=settings["validation_fraction_of_train"], random_state=seed
    )
    datasets = [SAOTripleDataset(a, b, cluster_labels, entity2idx, relation2idx)
                for a, b in ((X_train, y_train), (X_valid, y_valid), (X_test, y_test))]
    pin = device.type == "cuda"
    train_loader = DataLoader(datasets[0], batch_size=settings["batch_size"], shuffle=True,
                              num_workers=settings["num_workers"], pin_memory=pin)
    valid_loader = DataLoader(datasets[1], batch_size=settings["batch_size"] * 2, shuffle=False,
                              num_workers=settings["num_workers"], pin_memory=pin)
    test_loader = DataLoader(datasets[2], batch_size=settings["batch_size"] * 2, shuffle=False,
                             num_workers=settings["num_workers"], pin_memory=pin)

    tokenizer = RobertaTokenizerFast.from_pretrained(settings["model_name"])
    tokenizer.add_special_tokens({"additional_special_tokens": ["[S]", "[A]", "[O]"]})
    encoder = RobertaModel.from_pretrained(settings["model_name"])
    encoder.resize_token_embeddings(len(tokenizer))
    model = SAO_BC(
        encoder, tokenizer, len(entity2idx), len(relation2idx), encoder.config.hidden_size,
        settings["max_tokens"], settings["dropout"], settings["alpha_initial"], settings["embedding_dim"],
        settings["conv_out_channels"], settings["conv_kernel_size"], settings["conv_height"],
        settings["fc_hidden"], settings["input_dropout"], settings["feature_dropout"], settings["hidden_dropout"],
    ).to(device)
    criterion = FocalLoss(settings["focal_alpha"], settings["focal_gamma"])

    model.freeze_semantic_scorer()
    structural_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    structural_optimizer = AdamW(structural_parameters, lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    for epoch in range(settings["structural_only_epochs"]):
        model.train()
        for batch in tqdm(train_loader, desc=f"Structural Scorer epoch {epoch + 1}"):
            structural_optimizer.zero_grad()
            logits = model.forward_structural(batch["head_ids"].to(device), batch["rel_ids"].to(device), batch["tail_ids"].to(device))
            loss = criterion(logits, batch["label"].to(device))
            loss.backward()
            structural_optimizer.step()

    model.unfreeze_semantic_scorer()
    optimizer = AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    total_steps = max(1, len(train_loader) * settings["joint_epochs"] // settings["gradient_accumulation_steps"])
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * settings["warmup_ratio"]), total_steps)
    best_loss, stale = float("inf"), 0
    model_file = Path(model_path); model_file.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(settings["joint_epochs"]):
        model.train(); optimizer.zero_grad(); running = 0.0
        for step, batch in enumerate(tqdm(train_loader, desc=f"Joint epoch {epoch + 1}"), start=1):
            logits, _ = _batch_forward(model, batch, device)
            loss = criterion(logits, batch["label"].to(device)) / settings["gradient_accumulation_steps"]
            loss.backward(); running += loss.item()
            if step % settings["gradient_accumulation_steps"] == 0 or step == len(train_loader):
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
        model.eval(); validation_loss = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                logits, _ = _batch_forward(model, batch, device)
                validation_loss += criterion(logits, batch["label"].to(device)).item()
        validation_loss /= max(1, len(valid_loader))
        if validation_loss < best_loss - settings["min_delta"]:
            best_loss, stale = validation_loss, 0
            torch.save(model.state_dict(), model_file)
        else:
            stale += 1
            if stale >= settings["patience"]:
                break

    model.load_state_dict(torch.load(model_file, map_location=device, weights_only=True))
    probabilities, labels = _predict(model, test_loader, device)
    threshold = _best_threshold(labels, probabilities) if settings["threshold"] == "auto" else float(settings["threshold"])
    predictions = (probabilities >= threshold).astype(int)
    metrics = {
        "threshold": threshold,
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_macro": float(precision_score(labels, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(labels, predictions, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "test_samples": int(len(labels)),
    }
    save_json(metrics, metrics_path)
    Path(predictions_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"label": labels, "probability": probabilities, "prediction": predictions}).to_csv(predictions_path, index=False)
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    plt.figure(figsize=(8, 6)); plt.plot(recall, precision); plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall Curve"); plt.tight_layout(); Path(curve_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(curve_path, dpi=150); plt.close()
    return metrics
