from nebula.config import load_eval_config, load_train_config


def test_eval_config_has_threshold_and_labels():
    cfg = load_eval_config("configs/eval.yaml")
    assert cfg["task"] == "sentiment_analysis"
    assert cfg["labels"] == ["positivo", "neutral", "negativo"]
    assert cfg["thresholds"]["macro_f1"] == 0.85
    assert cfg["thresholds"]["must_beat_base"] is True


def test_train_config_has_qlora_hyperparams():
    cfg = load_train_config("configs/train-4b.yaml")
    assert cfg["base_model"] == "Qwen/Qwen3-4B"
    assert cfg["lora"]["r"] == 16
    assert cfg["lora"]["alpha"] == 16
    assert cfg["train"]["learning_rate"] == 2e-4
    assert 1 <= cfg["train"]["epochs"] <= 3
