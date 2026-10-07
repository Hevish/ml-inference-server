"""Train a small PyTorch classifier and save it for serving.

Kept deliberately simple: the point of this project is the serving
integration, not the model. Trains a 2-layer MLP on the Iris dataset
and saves the weights plus the normalisation statistics the server
needs at inference time.
"""

import json
import torch
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from model import IrisNet, FEATURE_NAMES, CLASS_NAMES


def main() -> None:
    torch.manual_seed(0)

    # Load data.
    data = load_iris()
    X, y = data.data.astype("float32"), data.target.astype("int64")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )

    # Standardise features; keep the stats so the server applies the same transform.
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    X_train_n = (X_train - mean) / std
    X_test_n = (X_test - mean) / std

    Xtr = torch.from_numpy(X_train_n)
    ytr = torch.from_numpy(y_train)

    model = IrisNet()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(200):
        optimiser.zero_grad()
        logits = model(Xtr)
        loss = loss_fn(logits, ytr)
        loss.backward()
        optimiser.step()

    # Report held-out accuracy so the number in the README is reproducible.
    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(X_test_n)).argmax(dim=1)
        acc = (preds == torch.from_numpy(y_test)).float().mean().item()
    print(f"Test accuracy: {acc:.3f}")

    # Persist weights and the preprocessing stats together.
    torch.save(model.state_dict(), "model.pt")
    with open("preprocess.json", "w") as f:
        json.dump(
            {
                "mean": mean.tolist(),
                "std": std.tolist(),
                "feature_names": FEATURE_NAMES,
                "class_names": CLASS_NAMES,
            },
            f,
            indent=2,
        )
    print("Saved model.pt and preprocess.json")


if __name__ == "__main__":
    main()
