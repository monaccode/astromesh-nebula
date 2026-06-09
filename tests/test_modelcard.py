from nebula.modelcard import render_model_card


def test_render_model_card_has_yaml_header_and_injected_eval():
    card = render_model_card(
        repo_id="astromesh/Centinela-Qwen3-4B",
        base_model="Qwen/Qwen3-4B",
        dataset_id="NickyNicky/Finance_sentiment_and_topic_classification_Translation_English_to_Spanish_v1",
        eval_table="| Metric | Base | Centinela-4B |\n|---|---|---|\n| macro-F1 | 0.61 | 0.88 |",
        n_train=1000,
        n_eval=250,
        base_model_id="Qwen/Qwen3-4B",
        dataset_hash="abc123",
        config_path="configs/train-4b.yaml",
        commit="deadbeef",
    )
    assert card.startswith("---\n")
    assert "license: apache-2.0" in card
    assert "base_model: Qwen/Qwen3-4B" in card
    assert "language:" in card
    assert "pipeline_tag: text-generation" in card
    assert "macro-F1 | 0.61 | 0.88" in card
    assert "abc123" in card  # reproducibility block
    assert "deadbeef" in card
