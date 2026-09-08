import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location("catalog_cli", str(Path("scripts/09_catalog.py")))
catalog_cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(catalog_cli)

FIX = "tests/fixtures"


def test_compile_to_file_writes_valid_deterministic_lock(tmp_path):
    out = tmp_path / "catalog.lock.json"
    lock = catalog_cli.compile_to_file(
        f"{FIX}/catalog_models.yaml", f"{FIX}/catalog_datasets.yaml", str(out)
    )
    on_disk = json.loads(out.read_text(encoding="utf-8"))
    assert on_disk == lock
    assert on_disk["schema_version"] == "1"
    # Deterministic: recompiling yields byte-identical output.
    text_first = out.read_text(encoding="utf-8")
    catalog_cli.compile_to_file(
        f"{FIX}/catalog_models.yaml", f"{FIX}/catalog_datasets.yaml", str(out)
    )
    assert out.read_text(encoding="utf-8") == text_first
