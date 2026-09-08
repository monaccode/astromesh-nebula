import json
from importlib.resources import files
from pathlib import Path

from nebula.catalog import compile_to_file, validate_lock

REPO = Path(__file__).resolve().parents[1]


def test_shipped_lock_is_present_and_valid():
    lock_text = (files("nebula") / "catalog.lock.json").read_text(encoding="utf-8")
    lock = json.loads(lock_text)
    validate_lock(lock)  # must not raise
    assert lock["schema_version"] == "1"
    assert any(m["name"] == "centinela-sentiment" for m in lock["models"])


def test_shipped_lock_has_no_drift(tmp_path):
    """The committed nebula/catalog.lock.json must equal a fresh compile of catalog/*.yaml."""
    fresh = tmp_path / "fresh.json"
    compile_to_file(
        str(REPO / "catalog" / "centinela.yaml"),
        str(REPO / "catalog" / "datasets.yaml"),
        str(fresh),
    )
    committed = json.loads((REPO / "nebula" / "catalog.lock.json").read_text(encoding="utf-8"))
    assert json.loads(fresh.read_text(encoding="utf-8")) == committed
