"""Logistic regression and classification metrics companion script.

Covers binary logistic regression, regularisation paths, ROC/AUC, calibration,
threshold trade-offs, and multiclass softmax regression.  Run from the
repository root:

    python Code/logistic_regression.py
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibrationDisplay
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


def fit_binary(X: np.ndarray, y: np.ndarray, C: float = 1.0) -> Pipeline:
    """Fit a scaled binary logistic-regression model."""
    if C <= 0:
        raise ValueError("C must be positive")
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, max_iter=5000, random_state=42),
    ).fit(X, y)


def coefficient_paths(
    X: np.ndarray, y: np.ndarray, Cs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the regularisation strengths and the fitted coefficient matrix."""
    paths = [fit_binary(X, y, C=C)[-1].coef_.ravel() for C in Cs]
    return np.asarray(Cs), np.asarray(paths)


def threshold_metrics(
    y_true: np.ndarray, proba: np.ndarray, threshold: float
) -> dict[str, float]:
    """Compute the confusion counts, precision, and recall at a threshold."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1]")
    predicted = (np.asarray(proba) >= threshold).astype(int)
    truth = np.asarray(y_true).astype(int)
    tp = int(np.sum((predicted == 1) & (truth == 1)))
    fp = int(np.sum((predicted == 1) & (truth == 0)))
    fn = int(np.sum((predicted == 0) & (truth == 1)))
    tn = int(np.sum((predicted == 0) & (truth == 0)))
    return {
        "tp": float(tp), "fp": float(fp), "fn": float(fn), "tn": float(tn),
        "precision": float(precision_score(truth, predicted, zero_division=0)),
        "recall": float(recall_score(truth, predicted, zero_division=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--c-min", type=float, default=1e-3)
    parser.add_argument("--c-max", type=float, default=1e3)
    parser.add_argument("--n-c", type=int, default=25)
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=[0.2, 0.5, 0.8])
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Binary classification: breast cancer
    # ------------------------------------------------------------------
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.25, stratify=data.target, random_state=42
    )

    clf = fit_binary(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    auc = roc_auc_score(y_test, proba)

    print("Binary logistic regression (breast cancer)")
    print(classification_report(y_test, pred, target_names=data.target_names, digits=4))
    print(f"  log loss = {log_loss(y_test, clf.predict_proba(X_test)):.4f}")
    print(f"  ROC AUC  = {auc:.4f}")

    fpr, tpr, _ = roc_curve(y_test, proba)

    # ------------------------------------------------------------------
    # 2. Threshold sweep
    # ------------------------------------------------------------------
    print("\nThreshold sweep (precision / recall):")
    for threshold in args.thresholds:
        metrics = threshold_metrics(y_test, proba, threshold)
        print(f"  threshold {threshold:.1f}: precision={metrics['precision']:.3f}"
              f"  recall={metrics['recall']:.3f}")

    # ------------------------------------------------------------------
    # 3. Regularisation and multiclass softmax regression
    # ------------------------------------------------------------------
    Cs = np.logspace(np.log10(args.c_min), np.log10(args.c_max), args.n_c)
    Cs, coef_paths_values = coefficient_paths(X_train, y_train, Cs)

    digits = load_digits()
    Xd_train, Xd_test, yd_train, yd_test = train_test_split(
        digits.data, digits.target, test_size=0.25, stratify=digits.target,
        random_state=42,
    )
    multi = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=5000, random_state=42)
    ).fit(Xd_train, yd_train)
    print("\nMulticlass logistic regression (digits)")
    print(f"  accuracy = {multi.score(Xd_test, yd_test):.4f}")
    print(f"  log loss = {log_loss(yd_test, multi.predict_proba(Xd_test)):.4f}")

    if args.no_plots:
        return

    # ------------------------------------------------------------------
    # 4. Plots
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, display_labels=data.target_names, cmap="Blues",
        colorbar=False, ax=axes[0, 0],
    )
    axes[0, 0].set_title("Confusion matrix (threshold 0.5)")

    axes[0, 1].plot(fpr, tpr, linewidth=2, label=f"AUC = {auc:.3f}")
    axes[0, 1].plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    axes[0, 1].set(xlabel="False positive rate", ylabel="True positive rate",
                   title="ROC curve")
    axes[0, 1].legend(loc="lower right")

    CalibrationDisplay.from_estimator(clf, X_test, y_test, n_bins=8, ax=axes[1, 0])
    axes[1, 0].set_title("Calibration curve")

    for j in range(coef_paths_values.shape[1]):
        axes[1, 1].plot(Cs, coef_paths_values[:, j], linewidth=1, alpha=0.7)
    axes[1, 1].set_xscale("log")
    axes[1, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 1].set(xlabel="C (inverse regularisation strength)", ylabel="Coefficient",
                   title="Coefficient paths")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
