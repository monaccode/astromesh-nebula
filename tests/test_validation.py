from nebula.validation import constrain_label


def test_exact_match_returns_label():
    assert constrain_label("positivo") == "positivo"


def test_is_case_and_whitespace_insensitive():
    assert constrain_label("  NEGATIVO\n") == "negativo"


def test_extracts_label_from_a_sentence():
    assert constrain_label("El sentimiento es claramente negativo.") == "negativo"


def test_returns_none_when_no_label_present():
    assert constrain_label("no lo sé") is None


def test_returns_none_for_none_input():
    assert constrain_label(None) is None


def test_prefers_first_occurring_label_in_text():
    # "positivo" appears before "negativo" -> positivo wins
    assert constrain_label("tono positivo, no negativo") == "positivo"
