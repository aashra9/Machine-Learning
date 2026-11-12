# Machine Learning Notes & Code

Repository for machine learning lecture notes (LaTeX) and supervised-learning examples (Python).

## Structure

```
Machine Learning/
├── .gitignore                  # LaTeX + Python build artifacts
├── Code/                       # Python supervised-learning worksheets
│   ├── linear_regression.ipynb # Simple & multiple linear regression (OLS, sklearn)
│   ├── mnist_supervised.ipynb  # MNIST digit classification (logistic, k-NN, SVM, RF)
│   └── mnist_supervised.py     # Standalone script version of the MNIST notebook
└── Lecture Notes/              # LaTeX lecture notes
    ├── Lectures/
    │   ├── Lectures.tex        # Main notes document
    │   ├── latexmkrc           # latexmk configuration
    │   └── build/              # Compilation output (ignored by git)
    └── Files/
        ├── Style/
        │   └── course.sty      # Custom LaTeX style (trimmed for ML)
        └── Bibliography/
            └── References.bib  # ML references
```

## LaTeX notes

Compile with `latexmk` from `Lecture Notes/Lectures/`:

```bash
cd "Lecture Notes/Lectures"
latexmk -pdf Lectures.tex
```

Requires a TeX distribution (TeX Live / MacTeX) with `biblatex`, `biber`, `imakeidx`, `tcolorbox`, `scrlayer-scrpage`, `hyperref`, `tikz`, `epigraph`, and `microtype`.

## Python notebooks

Install dependencies:

```bash
pip install scikit-learn matplotlib numpy jupyter
```

Run a notebook:

```bash
jupyter notebook Code/mnist_supervised.ipynb
```

Or run the standalone script:

```bash
python Code/mnist_supervised.py
```

The MNIST notebook fetches data from OpenML and falls back to scikit-learn's `load_digits` if offline.
