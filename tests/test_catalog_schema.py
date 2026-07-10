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


def _valid_lock() -> dict:
    return {
        "schema_version": "1",
        "models": [
            {
                "name": "centinela-sentiment",
                "kind": "classifier",
                "task": "text-classification",
                "vertical": "finanzas",
                "hf_repo": "astromesh/Centinela-Qwen3-4B",
                "contract": {"labels": ["positivo", "neutral", "negativo"]},
                "aliases": {"prod": "v0.1"},
                "revisions": {
                    "v0.1": {
                        "version": "v0.1",
                        "sha": "abc123",
                        "eval": {"macro_f1": 0.9},
                        "gate": "passed",
                        "formats": ["safetensors"],
                        "dataset": "centinela-sentiment-es@1",
                        "train_config_hash": "sha256:deadbeef",
                        "base_model": "Qwen/Qwen3-4B",
                    }
                },
            }
        ],
    }


def test_valid_lock_passes():
    validate_lock(_valid_lock())  # must not raise


def test_unknown_model_key_rejected():
    lock = _valid_lock()
    lock["models"][0]["surprise"] = "nope"
    with pytest.raises(ValidationError):
        validate_lock(lock)


def test_unknown_revision_key_rejected():
    lock = _valid_lock()
    lock["models"][0]["revisions"]["v0.1"]["surprise"] = "nope"
    with pytest.raises(ValidationError):
        validate_lock(lock)
