import pytest
from jsonschema import ValidationError

from nebula.catalog import compile_catalog, load_catalog, validate_lock

FIX = "tests/fixtures"


def test_compiled_lock_validates_against_schema():
    lock = compile_catalog(*load_catalog(f"{FIX}/catalog_models.yaml", f"{FIX}/catalog_datasets.yaml"))
    validate_lock(lock)  # must not raise


def test_bad_gate_enum_fails_schema():
    lock = compile_catalog(*load_catalog(f"{FIX}/catalog_models.yaml", f"{FIX}/catalog_datasets.yaml"))
    lock["models"][0]["revisions"]["v0.1"]["gate"] = "approved"
    with pytest.raises(ValidationError):
        validate_lock(lock)


def test_missing_required_field_fails_schema():
    lock = compile_catalog(*load_catalog(f"{FIX}/catalog_models.yaml", f"{FIX}/catalog_datasets.yaml"))
    del lock["models"][0]["hf_repo"]
    with pytest.raises(ValidationError):
        validate_lock(lock)
