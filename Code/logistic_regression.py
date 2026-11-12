"""Logistic regression and classification metrics companion script.

Covers binary logistic regression, regularisation, ROC/AUC, calibration, and
multiclass softmax regression.  Run from the repository root:

    python Code/logistic_regression.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from sklearn.calibration import CalibrationDisplay
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    log_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Binary classification: breast cancer
    # ------------------------------------------------------------------
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.25, stratify=data.target, random_state=42
    )

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=5000, random_state=42),
    )
    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    print("Binary logistic regression (breast cancer)")
    print(classification_report(y_test, pred, target_names=data.target_names, digits=4))
    print(f"  log loss = {log_loss(y_test, proba):.4f}")
    print(f"  ROC AUC  = {roc_auc_score(y_test, proba):.4f}")

    fpr, tpr, _ = roc_curve(y_test, proba)

    # ------------------------------------------------------------------
    # 2. Effect of regularisation on coefficients
    # ------------------------------------------------------------------
    Cs = np.logspace(-3, 3, 25)
    coef_paths = []
    for C in Cs:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=C, max_iter=5000, random_state=42),
        ).fit(X_train, y_train)
        coef_paths.append(model[-1].coef_.ravel())
    coef_paths = np.asarray(coef_paths)

    # ------------------------------------------------------------------
    # 3. Multiclass softmax regression: digits
    # ------------------------------------------------------------------
    digits = load_digits()
    Xd_train, Xd_test, yd_train, yd_test = train_test_split(
        digits.data, digits.target, test_size=0.25, stratify=digits.target,
        random_state=42,
    )
    multi = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, random_state=42),
    ).fit(Xd_train, yd_train)
    yd_pred = multi.predict(Xd_test)
    print("\nMulticlass logistic regression (digits)")
    print(f"  accuracy = {multi.score(Xd_test, yd_test):.4f}")
    print(f"  log loss = {log_loss(yd_test, multi.predict_proba(Xd_test)):.4f}")

    # ------------------------------------------------------------------
    # 4. Plots
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, display_labels=data.target_names, cmap="Blues",
        colorbar=False, ax=axes[0, 0],
    )
    axes[0, 0].set_title("Confusion matrix (threshold 0.5)")

    axes[0, 1].plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc_score(y_test, proba):.3f}")
    axes[0, 1].plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    axes[0, 1].set(xlabel="False positive rate", ylabel="True positive rate",
                   title="ROC curve")
    axes[0, 1].legend(loc="lower right")

    CalibrationDisplay.from_estimator(clf, X_test, y_test, n_bins=8, ax=axes[1, 0])
    axes[1, 0].set_title("Calibration curve")

    for j in range(coef_paths.shape[1]):
        axes[1, 1].plot(Cs, coef_paths[:, j], linewidth=1, alpha=0.7)
    axes[1, 1].set_xscale("log")
    axes[1, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 1].set(xlabel="C (inverse regularisation strength)", ylabel="Coefficient",
                   title="Coefficient paths")

    fig.tight_layout()
    plt.show()

    # A quick demonstration of the 0.5 threshold trade-off
    print("\nThreshold sweep (precision / recall):")
    for threshold in [0.2, 0.5, 0.8]:
        pred_t = (proba >= threshold).astype(int)
        tp = np.sum((pred_t == 1) & (y_test == 1))
        fp = np.sum((pred_t == 1) & (y_test == 0))
        fn = np.sum((pred_t == 0) & (y_test == 1))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        print(f"  threshold {threshold:.1f}: precision={precision:.3f} recall={recall:.3f}")


if __name__ == "__main__":
    main()
