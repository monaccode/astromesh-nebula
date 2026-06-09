import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("demo_app", str(Path("demo/app.py")))
demo_app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo_app)


def test_classify_text_constrains_output(monkeypatch):
    monkeypatch.setattr(demo_app, "_predict", lambda text: "claramente negativo")
    assert demo_app.classify_text("la acción se desploma") == "negativo"


def test_classify_text_reports_invalid(monkeypatch):
    monkeypatch.setattr(demo_app, "_predict", lambda text: "ni idea")
    assert demo_app.classify_text("texto raro") == "no clasificable"
