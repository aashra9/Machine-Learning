# Machine Learning Notes & Code

Repository for machine learning lecture notes (LaTeX) and runnable Python
companions.

## Structure

```
.
├── .gitignore                       # LaTeX + Python build artifacts
├── .github/workflows/ci.yml         # pytest + ruff on Python 3.10-3.12
├── compose.yml                      # opencode / kilo dev containers
├── Makefile                         # install, test, lint, format, notebooks
├── pyproject.toml                   # dependencies, pytest and ruff config
├── Code/
│   ├── linear_regression.py         # OLS, bootstrap CIs, residual diagnostics
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
│   ├── information_theory.py        # Entropy, cross-entropy, KL/JS, Bayesian updating
│   ├── knn.py                       # k-NN from scratch, tuning, decision regions
│   ├── knn_utils.py                 # Shared k-NN pipelines, loaders, plotting
│   ├── plotting_utils.py            # Shared decision-boundary plotting
│   ├── mnist_supervised.py          # MNIST classifier comparison script
│   ├── requirements.txt
│   ├── calculus.ipynb               # Derivatives, integrals, Jacobians, optimisation
│   ├── voronoi_diagram.ipynb        # Voronoi cells and 1-NN decision boundaries
│   ├── distance_metrics.ipynb       # All nine metrics: points, strings, sets, distributions
│   ├── probability_distances.ipynb  # Hellinger, Wasserstein, total variation
│   ├── multivariate_gaussian.ipynb  # Covariance, conditioning, Mahalanobis distance
│   ├── entropy.ipynb                # Entropy, surprisal, joint/conditional entropy
│   ├── cross_entropy.ipynb          # Cross-entropy loss, softmax, logistic training
│   ├── kl_divergence.ipynb          # KL divergence, cross-entropy and maximum likelihood
│   ├── js_divergence.ipynb          # Symmetric bounded divergence and mutual information
│   ├── bayes_rule.ipynb             # Priors, posteriors, naive Bayes, Gaussian conjugacy
│   ├── linear_regression.ipynb      # Notebook version of the regression worksheet
│   ├── knn_tutorial.ipynb           # k-NN: scratch, boundaries, tuning
│   ├── knn_datasets_worksheet.ipynb # k-NN on Iris, MNIST and ImageNet
│   ├── knn_iris.ipynb               # Focused k-NN worksheet on Iris
│   ├── knn_mnist.ipynb              # Focused k-NN worksheet on MNIST
│   ├── knn_imagenet.ipynb           # Focused k-NN worksheet on ImageNet-style images
│   ├── knn_lemons_oranges.ipynb     # k-NN on a synthetic/real citrus image set
│   └── mnist_supervised.ipynb       # Notebook version of the MNIST example
├── tests/                           # pytest suite for pure helpers and notebooks
└── Lecture Notes/
    ├── Lectures/
    │   ├── Lectures.tex             # Main notes document
    │   ├── latexmkrc                # latexmk configuration
    │   └── build/                   # Compilation output (ignored by git)
    └── Files/
        ├── Diagrams/                # Voronoi .dat files used in the notes
        ├── Style/course.sty
        └── Bibliography/References.bib
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
9. Information theory: surprisal, entropy, cross-entropy, KL divergence, and Bayes' rule

## Python companions

Install dependencies from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r Code/requirements.txt
```

Run any script from the repository root:

```bash
python Code/linear_regression.py
python Code/svm.py
python Code/unsupervised.py
python Code/knn.py --no-plots
python Code/information_theory.py --no-plots
```

Each script is self-contained apart from two small shared modules in `Code/`
(`plotting_utils.py` for decision boundaries, `knn_utils.py` for k-NN helpers).
Most scripts expose an `argparse` CLI — try `--help` for options such as
`--no-plots`, `--epochs`, or `--seed`. All dependencies are NumPy, SciPy,
matplotlib, scikit-learn, SymPy and Pillow. `Code/deep_learning.py` additionally
requires PyTorch; if it is not installed, it prints installation instructions and
exits cleanly.

## Notebooks

Open any notebook with Jupyter from the repository root:

```bash
jupyter notebook Code/knn_tutorial.ipynb
```

- `calculus.ipynb` — numerical derivatives, Riemann sums, Jacobians, optimisation.
- `voronoi_diagram.ipynb` — builds Voronoi diagrams and shows that 1-NN decision
  boundaries are exactly the edges between differently labelled cells.
- `distance_metrics.ipynb` — detailed tour of all nine metrics in the notes:
  Euclidean, Manhattan, Chebyshev, Minkowski, Hamming, Jaccard, Hellinger,
  Wasserstein and total variation, with unit balls, axiom checks, value tables,
  CDF-area intuition, and metric-dependent k-NN boundaries.
- `probability_distances.ipynb` — Hellinger, 1-Wasserstein and total variation distances,
  CDF-area interpretation, and why Wasserstein detects small location shifts.
- `multivariate_gaussian.ipynb` — affine transformations, contour ellipses, marginals,
  conditionals, and Mahalanobis distance.
- `entropy.ipynb` — Shannon entropy, binary entropy curve surprisal and code length,
  joint/conditional entropy, mutual information, and differential entropy.
- `cross_entropy.ipynb` — binary and multiclass cross-entropy, stable softmax,
  logistic-regression training, and why cross-entropy beats squared error.
- `kl_divergence.ipynb` — KL divergence for Bernoulli and Gaussian laws, Gibbs'
  inequality, asymmetry, cross-entropy, and KL minimisation as maximum likelihood.
- `js_divergence.ipynb` — Jensen-Shannon divergence, symmetry and boundedness, KL
  comparison, and its interpretation as mutual information.
- `bayes_rule.ipynb` — priors, likelihoods, evidence and posteriors, medical testing,
  naive Bayes in log odds, Gaussian conjugacy, and MAP as regularised MLE.
- `linear_regression.ipynb` — OLS worksheet with residual diagnostics.
- `knn_tutorial.ipynb` — self-contained k-NN tutorial: classification and
  regression from scratch, decision boundaries, scaling, cross-validation,
  distance metrics, the curse of dimensionality, and a breast-cancer case study.
- `knn_datasets_worksheet.ipynb` — the same workflow on Iris, MNIST and images.
- `knn_iris.ipynb`, `knn_mnist.ipynb`, `knn_imagenet.ipynb` — focused versions,
  one dataset each.
- `knn_lemons_oranges.ipynb` — k-NN on a synthetic citrus dataset with an
  optional real-image loader.
- `mnist_supervised.ipynb` — notebook version of the MNIST classifier comparison.

### Dataset options

MNIST is fetched from OpenML and falls back to scikit-learn's `load_digits` when
offline. For the focused MNIST notebook, set `KNN_FORCE_DIGITS=1` (or
`DOWNLOAD_MNIST = False`) to skip the download entirely:

```bash
KNN_FORCE_DIGITS=1 jupyter notebook Code/knn_mnist.ipynb
```

The ImageNet worksheets look for a local class-folder tree. Set `IMAGENET_ROOT`
to use real images; without it a small synthetic image dataset is generated so
every cell still runs. Loading real images requires Pillow:

```bash
export IMAGENET_ROOT=/path/to/imagenet/train
IMAGENET_ROOT=/path/to/imagenet/train jupyter notebook Code/knn_imagenet.ipynb
```

The citrus notebook reads `lemons_oranges/` relative to the working directory
unless you edit `DATA_ROOT`.

## Tests and tooling

Install the development extras and run the checks:

```bash
pip install -e ".[dev]"
pytest
ruff check Code tests
```

Or use the Makefile shortcuts: `make install`, `make test`, `make lint`,
`make format`. The pytest suite covers the pure numeric helpers, input
validation, and `nbformat.validate` on every notebook. GitHub Actions runs the
same checks on Python 3.10, 3.11 and 3.12.

`compose.yml` provides two development containers (`opencode` and `kilo`) with
the repository mounted at `/workspace`:

```bash
docker compose run --rm opencode
```
