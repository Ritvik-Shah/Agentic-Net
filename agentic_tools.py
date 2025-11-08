import os
import uuid
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

MODEL_REGISTRY = {}

def create_train_model(dataset_path: str, layers: list = None, epochs: int = 10, lr: float = 0.001) -> str:
    """
    Creates and trains a PyTorch model on a CSV dataset.
    Assumes the last column is the target variable.
    """
    # Verify file exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    # Load dataset
    df = pd.read_csv(dataset_path)
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(np.int64)

    input_dim = X.shape[1]
    num_classes = len(np.unique(y))
    layers = layers or [32, 32]

    # Build model dynamically
    modules = []
    last_dim = input_dim
    for dim in layers:
        modules.append(nn.Linear(last_dim, dim))
        modules.append(nn.ReLU())
        last_dim = dim
    modules.append(nn.Linear(last_dim, num_classes))
    model = nn.Sequential(*modules)

    # Data setup
    dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(dataloader):.4f}")

    # Save trained model to registry
    model_id = str(uuid.uuid4())
    MODEL_REGISTRY[model_id] = model
    print(f"✅ Model trained and saved with ID: {model_id}")
    return model_id


def predict_model(model_id: str, X_values: list) -> list:
    """
    Run prediction using a trained model.
    """
    if model_id not in MODEL_REGISTRY:
        raise ValueError(f"Model ID {model_id} not found in registry.")
    model = MODEL_REGISTRY[model_id]
    model.eval()

    X_tensor = torch.tensor(np.array(X_values, dtype=np.float32))
    with torch.no_grad():
        logits = model(X_tensor)
        preds = torch.argmax(logits, dim=1).tolist()
    return preds
