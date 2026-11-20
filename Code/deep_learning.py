"""Deep-learning companion script (optional).

A small convolutional network for the 8x8 scikit-learn digits dataset.  PyTorch
is optional; if it is not installed (or cannot load), the script prints
installation instructions and exits cleanly.  Run from the repository root:

    python Code/deep_learning.py
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset
    except (ImportError, OSError, RuntimeError) as exc:
        print(f"PyTorch could not be loaded ({exc}). Install it with:\n")
        print("    pip install torch torchvision\n")
        print("Then re-run: python Code/deep_learning.py")
        return

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    device_name = args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        device_name = "cpu"
    device = torch.device(
        device_name if device_name != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    digits = load_digits()
    X = digits.data.reshape(-1, 1, 8, 8).astype(np.float32)
    X = X / max(float(X.max()), 1.0)
    y = digits.target.astype(np.int64)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256)

    # ------------------------------------------------------------------
    # 2. Model
    # ------------------------------------------------------------------
    class SmallCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.fc = nn.Linear(32 * 2 * 2, 10)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            return self.fc(x.flatten(1))

    model = SmallCNN().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    # 3. Training loop
    # ------------------------------------------------------------------
    history = {"train_loss": [], "test_loss": [], "test_acc": []}
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimiser.step()
            running += loss.item() * len(xb)
        train_loss = running / len(train_ds)

        model.eval()
        test_loss, correct = 0.0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                test_loss += criterion(logits, yb).item() * len(xb)
                correct += (logits.argmax(dim=1) == yb).sum().item()
        test_loss /= len(test_ds)
        test_acc = correct / len(test_ds)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        print(f"epoch {epoch + 1:2d}  train loss={train_loss:.4f}  "
              f"test loss={test_loss:.4f}  test acc={test_acc:.4f}")

    if args.no_plots:
        return

    # ------------------------------------------------------------------
    # 4. Confusion matrix and learning curves
    # ------------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_test).to(device))
        predictions = logits.argmax(dim=1).cpu().numpy()

    try:
        from sklearn.metrics import ConfusionMatrixDisplay
        fig_cm, ax_cm = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(
            y_test, predictions, cmap="Blues", ax=ax_cm
        )
        ax_cm.set_title("CNN confusion matrix (digits)")
        fig_cm.tight_layout()
    except Exception as exc:
        print(f"Could not draw the confusion matrix: {exc}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["test_loss"], label="test")
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy", title="Loss curves")
    axes[0].legend()

    axes[1].plot(history["test_acc"], color="green")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy", title="Test accuracy")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
