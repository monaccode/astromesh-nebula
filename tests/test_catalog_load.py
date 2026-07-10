from nebula.catalog import load_catalog, ModelEntry, DatasetEntry, Revision

FIX = "tests/fixtures"


def test_load_returns_models_and_datasets_by_key():
    models, datasets = load_catalog(f"{FIX}/catalog_models.yaml", f"{FIX}/catalog_datasets.yaml")
    assert set(models) == {"centinela-sentiment"}
    assert set(datasets) == {"centinela-sentiment-es@3"}
    assert isinstance(models["centinela-sentiment"], ModelEntry)
    assert isinstance(datasets["centinela-sentiment-es@3"], DatasetEntry)


def test_revisions_are_keyed_by_version():
    models, _ = load_catalog(f"{FIX}/catalog_models.yaml", f"{FIX}/catalog_datasets.yaml")
    revs = models["centinela-sentiment"].revisions
    assert set(revs) == {"v0.1"}
    assert isinstance(revs["v0.1"], Revision)
    assert revs["v0.1"].gate == "passed"


def test_dataset_fields_parsed():
    _, datasets = load_catalog(f"{FIX}/catalog_models.yaml", f"{FIX}/catalog_datasets.yaml")
    ds = datasets["centinela-sentiment-es@3"]
    assert ds.contract == {"labels": ["positivo", "neutral", "negativo"], "validation": "constrain_label"}
    assert ds.kind == "classifier"


def test_empty_yaml_files_yield_empty_catalog(tmp_path):
    empty_models = tmp_path / "models.yaml"
    empty_datasets = tmp_path / "datasets.yaml"
    empty_models.write_text("# no models yet\n", encoding="utf-8")
    empty_datasets.write_text("", encoding="utf-8")
    models, datasets = load_catalog(str(empty_models), str(empty_datasets))
    assert models == {}
    assert datasets == {}
