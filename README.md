# Machine Learning Notes & Code

Repository for machine learning lecture notes (LaTeX) and runnable Python
companions.

## Structure

```
Machine Learning/
├── .gitignore                       # LaTeX + Python build artifacts
├── Code/                            # Python companions, one per major topic
│   ├── linear_regression.py         # OLS, normal equations, diagnostics
│   ├── logistic_regression.py       # Binary/multiclass logistic, ROC, calibration
│   ├── regularization.py            # Ridge, LASSO, elastic net, coefficient paths
│   ├── cross_validation.py          # k-fold, LOOCV, nested CV, imputation
│   ├── svm.py                       # Margins, kernels, grid search
│   ├── trees.py                     # CART, random forests, gradient boosting
│   ├── gradient_descent.py          # GD, SGD, momentum, Adam
│   ├── unsupervised.py              # k-means, hierarchical, DBSCAN, PCA, t-SNE
│   ├── neural_networks.py           # Perceptron and MLP from scratch
│   ├── deep_learning.py             # Optional PyTorch CNN (digits)
│   ├── probabilistic_models.py      # GMM/EM and HMM Viterbi
│   ├── reinforcement_learning.py    # Value iteration, Q-learning, SARSA
│   ├── learning_theory.py           # Bias-variance, learning curves, double descent
│   ├── linear_regression.ipynb      # Notebook version of the regression worksheet
│   ├── knn_tutorial.ipynb           # Detailed k-NN tutorial: scratch, boundaries, tuning
│   ├── knn_datasets_worksheet.ipynb # k-NN worksheet on Iris, MNIST, and ImageNet
│   ├── knn_iris.ipynb               # Focused k-NN worksheet on Iris
│   ├── knn_mnist.ipynb              # Focused k-NN worksheet on MNIST
│   ├── knn_imagenet.ipynb           # Focused k-NN worksheet on ImageNet-style images
│   ├── mnist_supervised.py          # MNIST classifier comparison script
│   ├── mnist_supervised.ipynb       # Notebook version of the MNIST example
│   └── requirements.txt
└── Lecture Notes/                   # LaTeX lecture notes
    ├── Lectures/
    │   ├── Lectures.tex             # Main notes document
    │   ├── latexmkrc                # latexmk configuration
    │   └── build/                   # Compilation output (ignored by git)
    └── Files/
        ├── Style/
        │   └── course.sty           # Custom LaTeX style (trimmed for ML)
        └── Bibliography/
            └── References.bib       # ML references
```

## LaTeX notes

Compile with `latexmk` from `Lecture Notes/Lectures/`:

```bash
cd "Lecture Notes/Lectures"
latexmk -pdf Lectures.tex
```

Requires a TeX distribution (TeX Live / MacTeX) with `biblatex`, `biber`,
`imakeidx`, `tcolorbox`, `scrlayer-scrpage`, `hyperref`, `tikz`, `pgfplots`,
`listings`, `epigraph`, and `microtype`.

The notes cover:

1. Resources and mathematical/probability background
2. Learning paradigms and supervised-learning foundations
3. Linear regression and logistic regression
4. Regularization, model assessment, cross-validation, and imputation
5. Support vector machines, trees, and ensembles
6. Gradient descent and optimization
7. Unsupervised learning, neural networks, and deep learning
8. Probabilistic models, reinforcement learning, and statistical learning theory

## Python companions

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r Code/requirements.txt
```

Run any script from the repository root, for example:

```bash
python Code/linear_regression.py
python Code/svm.py
python Code/unsupervised.py
```

Each script is self-contained and uses only NumPy, SciPy, matplotlib, and
scikit-learn. `Code/deep_learning.py` additionally requires PyTorch; if PyTorch
is not installed, it prints installation instructions and exits cleanly.

For the notebooks:

```bash
jupyter notebook Code/linear_regression.ipynb
jupyter notebook Code/knn_tutorial.ipynb
jupyter notebook Code/knn_datasets_worksheet.ipynb
jupyter notebook Code/knn_iris.ipynb
jupyter notebook Code/knn_mnist.ipynb
jupyter notebook Code/knn_imagenet.ipynb
jupyter notebook Code/mnist_supervised.ipynb
```

The k-NN notebook is a self-contained tutorial: it implements classification and
regression from scratch, visualises decision boundaries, studies scaling, tunes `k` and
the distance metric with cross-validation, and applies the pipeline to the breast-cancer
dataset. It uses only NumPy, matplotlib, and scikit-learn.

The datasets worksheet runs the same k-NN workflow on Iris, MNIST, and an image dataset.
MNIST is fetched from OpenML and falls back to `load_digits` when offline. The ImageNet
section expects a local copy of the class folders:

```bash
export IMAGENET_ROOT=/path/to/imagenet/train
jupyter notebook Code/knn_datasets_worksheet.ipynb
```

If no local folder is present, the notebook generates a small synthetic image dataset so
that every cell still runs. Loading a real image folder additionally requires Pillow
(`pip install pillow`).

Three focused notebooks contain the same workflow for a single dataset each:

- `Code/knn_iris.ipynb` — scaling, cross-validated tuning, confusion matrix, and a
  two-feature decision-boundary plot.
- `Code/knn_mnist.ipynb` — OpenML loader with a `load_digits` fallback, image gallery,
  raw pixels versus PCA, and error inspection. Set `KNN_FORCE_DIGITS=1` (or
  `DOWNLOAD_MNIST = False`) to skip the download.
- `Code/knn_imagenet.ipynb` — configurable `IMAGENET_ROOT`, a Pillow folder loader with a
  synthetic fallback, PCA comparison, and a distance-concentration experiment.

```bash
KNN_FORCE_DIGITS=1 jupyter notebook Code/knn_mnist.ipynb
IMAGENET_ROOT=/path/to/imagenet/train jupyter notebook Code/knn_imagenet.ipynb
```

The MNIST notebook fetches data from OpenML and falls back to scikit-learn's
`load_digits` if offline.
