from nebula.labels import LABELS, EN_TO_ES, SYSTEM_PROMPT_ES


def test_label_set_is_the_three_spanish_classes():
    assert LABELS == ["positivo", "neutral", "negativo"]


def test_english_to_spanish_mapping_covers_dataset_answers():
    assert EN_TO_ES == {"positive": "positivo", "neutral": "neutral", "negative": "negativo"}
    # every mapped value must be a valid label
    assert set(EN_TO_ES.values()) == set(LABELS)


def test_system_prompt_is_spanish_and_lists_the_labels():
    p = SYSTEM_PROMPT_ES.lower()
    assert "sentimiento" in p
    for label in LABELS:
        assert label in p
