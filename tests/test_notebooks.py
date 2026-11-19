from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")


NOTEBOOKS = sorted((Path(__file__).resolve().parents[1] / "Code").glob("*.ipynb"))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_is_valid(path: Path) -> None:
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
