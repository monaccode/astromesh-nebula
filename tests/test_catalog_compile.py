import pytest

from nebula.catalog import CatalogError, compile_catalog, load_catalog

FIX = "tests/fixtures"


def _load():
    return load_catalog(f"{FIX}/catalog_models.yaml", f"{FIX}/catalog_datasets.yaml")


def test_compile_produces_lock_with_schema_version():
    lock = compile_catalog(*_load())
    assert lock["schema_version"] == "1"
    names = [m["name"] for m in lock["models"]]
    assert names == ["centinela-sentiment"]


def test_compile_keys_revisions_by_version():
    lock = compile_catalog(*_load())
    model = lock["models"][0]
    assert set(model["revisions"]) == {"v0.1"}
    assert model["aliases"] == {"prod": "v0.1"}


def test_alias_to_missing_revision_raises():
    models, datasets = _load()
    models["centinela-sentiment"].aliases["staging"] = "v9.9"
    with pytest.raises(CatalogError, match="v9.9"):
        compile_catalog(models, datasets)


def test_alias_to_unpassed_gate_raises():
    models, datasets = _load()
    models["centinela-sentiment"].revisions["v0.1"].gate = "failed"
    with pytest.raises(CatalogError, match="gate"):
        compile_catalog(models, datasets)


def test_revision_with_unknown_dataset_raises():
    models, datasets = _load()
    models["centinela-sentiment"].revisions["v0.1"].dataset = "ghost@1"
    with pytest.raises(CatalogError, match="ghost@1"):
        compile_catalog(models, datasets)


def test_contract_mismatch_between_model_and_dataset_raises():
    models, datasets = _load()
    datasets["centinela-sentiment-es@3"].contract = {"labels": ["a", "b"]}
    with pytest.raises(CatalogError, match="contract"):
        compile_catalog(models, datasets)


def test_compile_is_deterministic():
    lock_a = compile_catalog(*_load())
    lock_b = compile_catalog(*_load())
    assert lock_a == lock_b
