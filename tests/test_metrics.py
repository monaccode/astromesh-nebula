from nebula.metrics import accuracy, macro_f1, classification_summary

LABELS = ["positivo", "neutral", "negativo"]


def test_accuracy_perfect():
    gold = ["positivo", "negativo", "neutral"]
    pred = ["positivo", "negativo", "neutral"]
    assert accuracy(gold, pred) == 1.0


def test_accuracy_counts_none_as_wrong():
    gold = ["positivo", "negativo"]
    pred = ["positivo", None]
    assert accuracy(gold, pred) == 0.5


def test_macro_f1_perfect_is_one():
    gold = ["positivo", "negativo", "neutral"]
    pred = ["positivo", "negativo", "neutral"]
    assert macro_f1(gold, pred, LABELS) == 1.0


def test_macro_f1_handles_a_missed_class():
    # neutral never predicted -> its F1 is 0, dragging the macro average down
    gold = ["positivo", "neutral", "negativo"]
    pred = ["positivo", "positivo", "negativo"]
    f1 = macro_f1(gold, pred, LABELS)
    assert 0.0 < f1 < 1.0


def test_summary_reports_invalid_rate():
    gold = ["positivo", "negativo"]
    pred = ["positivo", None]
    summary = classification_summary(gold, pred, LABELS)
    assert summary["accuracy"] == 0.5
    assert summary["invalid_rate"] == 0.5
    assert "macro_f1" in summary
    assert set(summary["per_label"].keys()) == set(LABELS)
