"""Decision trees, random forests, and gradient boosting companion script.

Run from the repository root:

    python Code/trees.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree


def main() -> None:
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    cv = StratifiedKFold(5, shuffle=True, random_state=42)

    # ------------------------------------------------------------------
    # 1. Compare tree-based models
    # ------------------------------------------------------------------
    models = {
        "Decision tree (depth 3)": DecisionTreeClassifier(max_depth=3, random_state=42),
        "Decision tree (full)": DecisionTreeClassifier(random_state=42),
        "Random forest": RandomForestClassifier(
            n_estimators=300, max_features="sqrt", n_jobs=-1, random_state=42
        ),
        "Gradient boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=2, random_state=42
        ),
    }
    print("Cross-validated accuracy")
    cv_scores = {}
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy",
                                 n_jobs=-1)
        cv_scores[name] = scores
        print(f"  {name:24s} {scores.mean():.4f} +/- {scores.std():.4f}")

    # ------------------------------------------------------------------
    # 2. Tree complexity: validation curve over depth
    # ------------------------------------------------------------------
    depths = np.arange(1, 13)
    train_acc, valid_acc = [], []
    for depth in depths:
        tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
        tree.fit(X_train, y_train)
        train_acc.append(tree.score(X_train, y_train))
        valid_acc.append(tree.score(X_test, y_test))

    # ------------------------------------------------------------------
    # 3. Feature importance
    # ------------------------------------------------------------------
    forest = models["Random forest"].fit(X_train, y_train)
    impurity_importance = forest.feature_importances_
    perm = permutation_importance(
        forest, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
    )
    order = np.argsort(impurity_importance)[-10:]

    print(f"\nRandom forest test accuracy = {forest.score(X_test, y_test):.4f}")
    print("Top features by impurity importance:")
    for idx in order[::-1]:
        print(f"  {data.feature_names[idx]:22s} {impurity_importance[idx]:.4f}")

    # ------------------------------------------------------------------
    # 4. Plots
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    plot_tree(
        DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_train, y_train),
        feature_names=data.feature_names,
        class_names=data.target_names,
        filled=True,
        rounded=True,
        fontsize=6,
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("Decision tree (depth 3)")

    axes[0, 1].plot(depths, train_acc, marker="o", label="Training")
    axes[0, 1].plot(depths, valid_acc, marker="o", label="Test")
    axes[0, 1].set(xlabel="Maximum depth", ylabel="Accuracy",
                   title="Tree complexity")
    axes[0, 1].legend()

    axes[1, 0].barh(np.arange(len(order)), impurity_importance[order],
                    color="steelblue")
    axes[1, 0].set_yticks(np.arange(len(order)))
    axes[1, 0].set_yticklabels([data.feature_names[i] for i in order], fontsize=8)
    axes[1, 0].set(xlabel="Impurity importance", title="Top features")

    perm_order = np.argsort(perm.importances_mean)[-10:]
    axes[1, 1].barh(np.arange(len(perm_order)),
                    perm.importances_mean[perm_order],
                    xerr=perm.importances_std[perm_order], color="darkorange")
    axes[1, 1].set_yticks(np.arange(len(perm_order)))
    axes[1, 1].set_yticklabels([data.feature_names[i] for i in perm_order], fontsize=8)
    axes[1, 1].set(xlabel="Permutation importance", title="Top features (test set)")

    fig.tight_layout()
    plt.show()

    # Confusion matrix of the forest
    ConfusionMatrixDisplay.from_estimator(
        forest, X_test, y_test, display_labels=data.target_names, cmap="Blues"
    )
    plt.title("Random forest confusion matrix")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
