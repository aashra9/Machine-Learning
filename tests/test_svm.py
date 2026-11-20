import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from svm import plot_boundary


def test_plot_boundary_draws_margin_and_data() -> None:
    X, y = make_classification(
        n_samples=40, n_features=2, n_redundant=0,
        n_informative=2, random_state=0,
    )
    model = SVC(kernel="linear").fit(X, y)
    fig, ax = plt.subplots()
    plot_boundary(ax, model, X, y, "linear SVM", resolution=20)
    assert ax.get_title() == "linear SVM"
    assert len(ax.collections) >= 3  # filled regions, contour levels, and points
    assert ax.get_legend() is not None  # support vectors are highlighted
    plt.close(fig)
