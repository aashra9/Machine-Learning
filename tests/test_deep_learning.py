import sys

from deep_learning import main


def test_main_reports_missing_torch(monkeypatch, capsys) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setattr(sys, "argv", ["deep_learning.py", "--no-plots"])
    main()
    output = capsys.readouterr().out
    assert "PyTorch could not be loaded" in output
    assert "pip install torch torchvision" in output
