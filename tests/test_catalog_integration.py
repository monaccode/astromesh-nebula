import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("catalog_cli", str(Path("scripts/09_catalog.py")))
catalog_cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(catalog_cli)


def test_real_catalog_compiles_and_validates(tmp_path):
    out = tmp_path / "catalog.lock.json"
    lock = catalog_cli.compile_to_file("catalog/centinela.yaml", "catalog/datasets.yaml", str(out))
    assert lock["schema_version"] == "1"
    model = lock["models"][0]
    assert model["name"] == "centinela-sentiment"
    assert model["aliases"] == {"prod": "v0.1"}
    assert "v0.1" in model["revisions"]
