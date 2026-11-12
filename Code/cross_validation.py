"""Model assessment companion script.

Demonstrates k-fold CV, leave-one-out CV, nested CV, learning curves, and
imputation pipelines.  Run from the repository root:

    python Code/cross_validation.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    LeaveOneOut,
    StratifiedKFold,
    cross_val_score,
    learning_curve,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_model(**kwargs) -> object:
    """A scaled logistic-regression pipeline."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, random_state=42, **kwargs),
    )


def main() -> None:
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ------------------------------------------------------------------
    # 1. k-fold versus leave-one-out cross-validation
    # ------------------------------------------------------------------
    cv5 = cross_val_score(make_model(), X_train, y_train,
                          cv=StratifiedKFold(5, shuffle=True, random_state=42))
    cv10 = cross_val_score(make_model(), X_train, y_train,
                           cv=StratifiedKFold(10, shuffle=True, random_state=42))

    # LOOCV is expensive; use a small subset to keep the example quick.
    rng = np.random.default_rng(42)
    subset = rng.choice(len(X_train), size=80, replace=False)
    loo = cross_val_score(make_model(), X_train[subset], y_train[subset],
                          cv=LeaveOneOut())

    print("Cross-validation accuracy")
    print(f"  5-fold   mean={cv5.mean():.4f}  sd={cv5.std():.4f}")
    print(f"  10-fold  mean={cv10.mean():.4f}  sd={cv10.std():.4f}")
    print(f"  LOOCV    mean={loo.mean():.4f}  sd={loo.std():.4f}  (n=80 subset)")

    # ------------------------------------------------------------------
    # 2. Nested cross-validation
    # ------------------------------------------------------------------
    param_grid = {"logisticregression__C": [0.01, 0.1, 1.0, 10.0, 100.0]}
    outer = StratifiedKFold(5, shuffle=True, random_state=1)
    inner = StratifiedKFold(4, shuffle=True, random_state=2)
    nested_scores = []
    chosen_C = []
    for train_idx, valid_idx in outer.split(X_train, y_train):
        search = GridSearchCV(
            make_model(), param_grid, cv=inner, scoring="accuracy", n_jobs=-1
        )
        search.fit(X_train[train_idx], y_train[train_idx])
        score = search.score(X_train[valid_idx], y_train[valid_idx])
        nested_scores.append(score)
        chosen_C.append(search.best_params_["logisticregression__C"])
    print("\nNested cross-validation")
    print(f"  outer scores = {np.round(nested_scores, 4)}")
    print(f"  mean         = {np.mean(nested_scores):.4f}")
    print(f"  chosen C     = {chosen_C}")

    # ------------------------------------------------------------------
    # 3. Learning curve
    # ------------------------------------------------------------------
    sizes, train_scores, valid_scores = learning_curve(
        make_model(C=1.0), X_train, y_train,
        train_sizes=np.linspace(0.1, 1.0, 8),
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring="accuracy", n_jobs=-1,
    )

    # ------------------------------------------------------------------
    # 4. Imputation: compare mean and k-NN imputation
    # ------------------------------------------------------------------
    X_missing = X_train.copy()
    mask = rng.random(X_missing.shape) < 0.10
    X_missing[mask] = np.nan
    X_test_missing = X_test.copy()
    test_mask = rng.random(X_test_missing.shape) < 0.10
    X_test_missing[test_mask] = np.nan

    imputers = {
        "mean": SimpleImputer(strategy="mean"),
        "median": SimpleImputer(strategy="median"),
        "k-NN": KNNImputer(n_neighbors=5),
    }
    imputation_scores = {}
    for name, imputer in imputers.items():
        pipeline = make_pipeline(
            imputer,
            StandardScaler(),
            LogisticRegression(max_iter=5000, random_state=42),
        )
        score = cross_val_score(
            pipeline, X_missing, y_train,
            cv=StratifiedKFold(5, shuffle=True, random_state=42),
        ).mean()
        imputation_scores[name] = score
        pipeline.fit(X_missing, y_train)
        test_score = pipeline.score(X_test_missing, y_test)
        print(f"\n{name:6s} imputation: CV={score:.4f}  test={test_score:.4f}")

    # ------------------------------------------------------------------
    # 5. Plots
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].boxplot([cv5, cv10], tick_labels=["5-fold", "10-fold"])
    axes[0].set(ylabel="Accuracy", title="Repeated-fold variability")

    axes[1].plot(sizes, train_scores.mean(axis=1), marker="o", label="Training")
    axes[1].plot(sizes, valid_scores.mean(axis=1), marker="o", label="Validation")
    axes[1].fill_between(sizes, train_scores.mean(axis=1) - train_scores.std(axis=1),
                         train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.15)
    axes[1].fill_between(sizes, valid_scores.mean(axis=1) - valid_scores.std(axis=1),
                         valid_scores.mean(axis=1) + valid_scores.std(axis=1), alpha=0.15)
    axes[1].set(xlabel="Training examples", ylabel="Accuracy", title="Learning curve")
    axes[1].legend()

    axes[2].bar(list(imputation_scores), list(imputation_scores.values()),
                color="steelblue")
    axes[2].set(ylim=(0.7, 1.0), ylabel="CV accuracy", title="Imputation strategies")
    for i, value in enumerate(imputation_scores.values()):
        axes[2].text(i, value + 0.005, f"{value:.3f}", ha="center", fontsize=9)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
