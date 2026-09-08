import importlib.util
from pathlib import Path
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location("scout_cli", str(Path("scripts/00_scout.py")))
scout_cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scout_cli)


def test_params_b_from_id_parses_suffix():
    assert scout_cli._params_b_from_id("Qwen/Qwen3-4B") == 4.0
    assert scout_cli._params_b_from_id("meta-llama/Llama-3.1-8b-instruct") == 8.0
    assert scout_cli._params_b_from_id("some/model-without-size") is None


def test_to_candidate_maps_model_info_fields():
    info = SimpleNamespace(
        id="Qwen/Qwen3-4B",
        pipeline_tag="text-generation",
        downloads=1234,
        likes=56,
        tags=["es", "text-generation", "license:apache-2.0"],
        card_data={"license": "apache-2.0", "language": ["es", "en"]},
    )
    cand = scout_cli.to_candidate(info)
    assert cand.model_id == "Qwen/Qwen3-4B"
    assert cand.license == "apache-2.0"
    assert cand.params_b == 4.0
    assert "es" in cand.languages
    assert cand.downloads == 1234


def test_to_candidate_reads_non_dict_card_data():
    # Real huggingface_hub ModelCardData is not a dict but exposes .get().
    class FakeCardData:
        def __init__(self, data):
            self._d = data

        def get(self, key, default=None):
            return self._d.get(key, default)

    info = SimpleNamespace(
        id="meta-llama/Llama-3.1-8B",
        pipeline_tag="text-generation",
        downloads=10,
        likes=1,
        tags=["text-generation"],
        card_data=FakeCardData({"license": "mit", "language": ["es"]}),
    )
    cand = scout_cli.to_candidate(info)
    assert cand.license == "mit"
    assert cand.languages == ["es"]
