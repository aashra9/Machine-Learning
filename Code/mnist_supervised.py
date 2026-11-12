"""MNIST Supervised Learning Example

Demonstrates supervised learning on handwritten digit classification.
Uses scikit-learn's fetch_openml for MNIST (falls back to load_digits offline).
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

plt.rcParams["figure.figsize"] = (10, 6)
np.random.seed(42)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
USE_MNIST = True
try:
    from sklearn.datasets import fetch_openml

    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X, y = mnist["data"], mnist["target"]
    y = y.astype(np.uint8)
    print("Loaded MNIST from OpenML")
except Exception as exc:
    USE_MNIST = False
    from sklearn.datasets import load_digits

    digits = load_digits()
    X, y = digits.data, digits.target.astype(np.uint8)
    print(f"Fell back to load_digits: {exc}")

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Pixel range: [{X.min()}, {X.max()}]")
print(f"Classes: {np.unique(y)}")
print(f"Class counts:\n{np.bincount(y)}")

# ---------------------------------------------------------------------------
# 2. Visualize sample digits
# ---------------------------------------------------------------------------
img_side = int(np.sqrt(X.shape[1]))

fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for digit in range(10):
    idx = np.where(y == digit)[0][0]
    img = X[idx].reshape(img_side, img_side)
    ax = axes[digit // 5, digit % 5]
    ax.imshow(img, cmap="gray")
    ax.set_title(f"Digit {digit}")
    ax.axis("off")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# 3. Preprocess and split
# ---------------------------------------------------------------------------
# Scale pixels to [0, 1]
X = X / X.max()

# Stratified train/test split
n_test = 10000 if USE_MNIST and len(y) >= 20000 else 0.2
n_train = 60000 if USE_MNIST and len(y) >= 70000 else None

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=n_test,
    train_size=n_train,
    random_state=42,
    stratify=y,
)

print(f"Train: {X_train.shape}, {y_train.shape}")
print(f"Test:  {X_test.shape},  {y_test.shape}")

# ---------------------------------------------------------------------------
# 4. Train and compare classifiers
# ---------------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, n_jobs=-1, random_state=42
    ),
    "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    "SVM (RBF)": SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=None, n_jobs=-1, random_state=42
    ),
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"{name:18s}  test accuracy = {acc:.4f}")

# ---------------------------------------------------------------------------
# Plot accuracy comparison
# ---------------------------------------------------------------------------
names = list(results.keys())
accs = list(results.values())

plt.bar(names, accs, color="steelblue")
plt.ylim(0, 1)
plt.ylabel("Test accuracy")
plt.title("MNIST Classifier Comparison")
plt.xticks(rotation=15)
for i, v in enumerate(accs):
    plt.text(i, v + 0.01, f"{v:.4f}", ha="center")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# 5. Detailed evaluation of the best model
# ---------------------------------------------------------------------------
best_name = max(results, key=results.get)
best_model = models[best_name]
y_pred = best_model.predict(X_test)

print(f"\nBest model: {best_name} (accuracy = {results[best_name]:.4f})\n")
print(classification_report(y_test, y_pred, digits=4))

# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------
disp = ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    cmap="Blues",
    colorbar=False,
)
plt.title(f"Confusion Matrix - {best_name}")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# 6. Inspect misclassified digits
# ---------------------------------------------------------------------------
wrong = np.where(y_pred != y_test)[0]
print(f"Misclassified: {len(wrong)} / {len(y_test)} test samples")

n_show = min(10, len(wrong))
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, idx in enumerate(wrong[:n_show]):
    img = X_test[idx].reshape(img_side, img_side)
    ax = axes[i // 5, i % 5]
    ax.imshow(img, cmap="gray")
    ax.set_title(f"True: {y_test[idx]}  Pred: {y_pred[idx]}", color="red")
    ax.axis("off")
plt.tight_layout()
plt.show()

print("\nDone!")
