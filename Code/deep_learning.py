"""Deep-learning companion script (optional).

A small convolutional network for the 8x8 scikit-learn digits dataset.  PyTorch
is optional; if it is not installed, the script prints installation instructions
and exits cleanly.  Run from the repository root:

    python Code/deep_learning.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def main() -> None:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print("PyTorch is not installed. Install it with:\n")
        print("    pip install torch torchvision\n")
        print("Then re-run: python Code/deep_learning.py")
        return

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    digits = load_digits()
    X = digits.data.reshape(-1, 1, 8, 8).astype(np.float32) / 16.0
    y = digits.target.astype(np.int64)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
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

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            x = x.flatten(1)
            return self.fc(x)

    model = SmallCNN().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    # 3. Training loop
    # ------------------------------------------------------------------
    history = {"train_loss": [], "test_loss": [], "test_acc": []}
    for epoch in range(15):
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

    # ------------------------------------------------------------------
    # 4. Confusion matrix and sample predictions
    # ------------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_test).to(device))
        predictions = logits.argmax(dim=1).cpu().numpy()

    try:
        from sklearn.metrics import ConfusionMatrixDisplay
        ConfusionMatrixDisplay.from_predictions(y_test, predictions, cmap="Blues")
        plt.title("CNN confusion matrix (digits)")
        plt.tight_layout()
        plt.show()
    except Exception:
        pass

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
