import os
import json
import uuid
import time
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, log_loss
from sklearn.inspection import permutation_importance
from torch.utils.data import DataLoader, TensorDataset
import joblib

MODEL_ROOT = "models"
os.makedirs(MODEL_ROOT, exist_ok=True)

# Model registry: maps model_id -> metadata (on-disk persisted)
def _model_dir(model_id: str) -> str:
    return os.path.join(MODEL_ROOT, model_id)

def _save_metadata(model_id: str, metadata: Dict[str, Any]):
    path = os.path.join(_model_dir(model_id), "metadata.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

def _load_metadata(model_id: str) -> Dict[str, Any]:
    path = os.path.join(_model_dir(model_id), "metadata.json")
    if not os.path.exists(path):
        raise FileNotFoundError("Metadata not found for model " + model_id)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_model_state(model_id: str, model: nn.Module):
    os.makedirs(_model_dir(model_id), exist_ok=True)
    path = os.path.join(_model_dir(model_id), "model.pt")
    torch.save(model.state_dict(), path)

def _load_model_state(model: nn.Module, model_id: str):
    path = os.path.join(_model_dir(model_id), "model.pt")
    if not os.path.exists(path):
        raise FileNotFoundError("Model state not found for " + model_id)
    model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))

def _save_scaler(model_id: str, scaler):
    joblib.dump(scaler, os.path.join(_model_dir(model_id), "scaler.joblib"))

def _load_scaler(model_id: str):
    path = os.path.join(_model_dir(model_id), "scaler.joblib")
    if os.path.exists(path):
        return joblib.load(path)
    return None

def _build_mlp(input_dim: int, output_dim: int, layers: List[int], activation: str = "relu", dropout: float = 0.0) -> nn.Module:
    modules = []
    last = input_dim
    for i, h in enumerate(layers):
        modules.append(nn.Linear(last, h))
        if activation == "relu":
            modules.append(nn.ReLU())
        elif activation == "tanh":
            modules.append(nn.Tanh())
        if dropout and dropout > 0:
            modules.append(nn.Dropout(dropout))
        last = h
    modules.append(nn.Linear(last, output_dim))
    return nn.Sequential(*modules)

def _evaluate_model(model: nn.Module, X: np.ndarray, y: np.ndarray, batch_size: int = 64) -> Dict[str, Any]:
    model.eval()
    dataset = TensorDataset(torch.tensor(X.astype(np.float32)), torch.tensor(y.astype(np.int64)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds = []
    probs = []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            prob = torch.softmax(logits, dim=1).cpu().numpy()
            p = np.argmax(prob, axis=1)
            preds.extend(p.tolist())
            probs.extend(prob.tolist())
    preds = np.array(preds)
    acc = float(accuracy_score(y, preds))
    cm = confusion_matrix(y, preds).tolist()
    report = classification_report(y, preds, output_dict=True, zero_division=0)
    # return probabilities for log_loss if possible (use one-hot if needed)
    return {"accuracy": acc, "confusion_matrix": cm, "report": report, "predictions": preds.tolist(), "probabilities": probs}

def create_train_model(
    dataset_path: str,
    layers: Optional[List[int]] = None,
    epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 32,
    test_size: float = 0.2,
    scaler_type: str = "standard",
    activation: str = "relu",
    dropout: float = 0.0,
    auto_search: bool = False
) -> Dict[str, Any]:
    """
    Load CSV (last column is label), train MLP, evaluate, save model+metadata, return metadata including model_id.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError("Dataset not found: " + dataset_path)

    df = pd.read_csv(dataset_path)
    if df.shape[1] < 2:
        raise ValueError("Dataset must have at least one feature column and one label column")

    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values
    # convert labels to integers
    unique_labels, y_int = np.unique(y, return_inverse=True)
    y = y_int.astype(np.int64)
    input_dim = X.shape[1]
    output_dim = len(unique_labels)

    # scaling
    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = None

    if scaler is not None:
        X = scaler.fit_transform(X)

    # train/test split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y if len(unique_labels) > 1 else None)

    # default layers
    if layers is None:
        layers = [64, 32]

    # optional simple AutoML: try a few layer configurations / lrs and pick best by val accuracy
    if auto_search:
        search_space = [
            ([64, 32], 1e-3),
            ([32, 16], 1e-3),
            ([64, 64], 5e-4),
            ([16, 16], 1e-2)
        ]
        best = {"acc": -1, "layers": layers, "lr": lr, "state": None}
        for cand_layers, cand_lr in search_space:
            model = _build_mlp(input_dim, output_dim, cand_layers, activation=activation, dropout=dropout)
            _train_loop(model, X_train, y_train, X_val, y_val, epochs=3, lr=cand_lr, batch_size=batch_size)  # quick 3 epochs for search
            stats = _evaluate_model(model, X_val, y_val)
            if stats["accuracy"] > best["acc"]:
                best["acc"] = stats["accuracy"]
                best["layers"] = cand_layers
                best["lr"] = cand_lr
                best["state"] = model.state_dict()
        # use best
        layers = best["layers"]
        lr = best["lr"]
        model = _build_mlp(input_dim, output_dim, layers, activation=activation, dropout=dropout)
        model.load_state_dict(best["state"])
        # continue training full epochs on chosen config
        _train_loop(model, X_train, y_train, X_val, y_val, epochs=epochs, lr=lr, batch_size=batch_size)
    else:
        model = _build_mlp(input_dim, output_dim, layers, activation=activation, dropout=dropout)
        _train_loop(model, X_train, y_train, X_val, y_val, epochs=epochs, lr=lr, batch_size=batch_size)

    # final evaluation
    train_stats = _evaluate_model(model, X_train, y_train)
    val_stats = _evaluate_model(model, X_val, y_val)

    # permutation importance (on validation set) - using sklearn wrapper
    try:
        def _predict_fn(X_np):
            model.eval()
            with torch.no_grad():
                logits = model(torch.tensor(X_np.astype(np.float32)))
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                return probs
        r = permutation_importance(lambda X: np.argmax(_predict_fn(X), axis=1), X_val, y_val, n_repeats=5, random_state=42, n_jobs=1)
        perm_importances = {"mean": r.importances_mean.tolist(), "std": r.importances_std.tolist()}
    except Exception:
        perm_importances = None

    # metadata & save
    model_id = str(uuid.uuid4())
    os.makedirs(_model_dir(model_id), exist_ok=True)

    metadata = {
        "model_id": model_id,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "dataset": os.path.abspath(dataset_path),
        "n_rows": int(df.shape[0]),
        "n_features": int(input_dim),
        "label_values": unique_labels.tolist(),
        "layers": layers,
        "activation": activation,
        "dropout": dropout,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "test_size": test_size,
        "scaler_type": scaler_type,
        "train_metrics": train_stats,
        "val_metrics": val_stats,
        "permutation_importance": perm_importances
    }

    # persist model and scaler & metadata
    _save_model_state(model_id, model)
    if scaler is not None:
        _save_scaler(model_id, scaler)
    _save_metadata(model_id, metadata)

    return metadata

def _train_loop(model: nn.Module, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, epochs: int = 10, lr: float = 1e-3, batch_size: int = 32):
    model.train()
    dataset = TensorDataset(torch.tensor(X_train.astype(np.float32)), torch.tensor(y_train.astype(np.int64)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        running = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running += loss.item()
        # optionally print or log per-epoch info (kept minimal here)
    return

def load_model_metadata(model_id: str) -> Dict[str, Any]:
    return _load_metadata(model_id)

def predict_model(model_id: str, X_values: List[List[float]]) -> Dict[str, Any]:
    """
    Load model & scaler from disk, run prediction. Returns dict with predictions and probabilities.
    """
    meta = _load_metadata(model_id)
    model = _build_mlp(meta["n_features"], len(meta["label_values"]), meta["layers"], activation=meta.get("activation","relu"), dropout=meta.get("dropout",0.0))
    _load_model_state(model, model_id)
    scaler = _load_scaler(model_id)
    X_np = np.array(X_values, dtype=np.float32)
    if scaler is not None:
        X_np = scaler.transform(X_np)
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_np))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1).tolist()
    return {"predictions": preds, "probabilities": probs.tolist(), "metadata": meta}
