import json

from nebula.evaluate import load_examples, evaluate_classifier
from nebula.labels import LABELS


class FakeClassifier:
    """Returns canned raw outputs in order; lets us test scoring without a model."""

    def __init__(self, outputs):
        self._outputs = list(outputs)
        self._i = 0

    def predict(self, text: str) -> str:
        out = self._outputs[self._i]
        self._i += 1
        return out


def test_load_examples_reads_eval_jsonl(tmp_path):
    path = tmp_path / "eval.jsonl"
    rec = {
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "La bolsa cae."},
            {"role": "assistant", "content": "negativo"},
        ]
    }
    path.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    examples = load_examples(str(path))
    assert examples == [{"text": "La bolsa cae.", "gold": "negativo"}]


def test_evaluate_classifier_scores_and_constrains():
    examples = [
        {"text": "sube", "gold": "positivo"},
        {"text": "cae", "gold": "negativo"},
        {"text": "estable", "gold": "neutral"},
    ]
    # raw outputs: a sentence (constrained to positivo), exact, and garbage (-> invalid)
    clf = FakeClassifier(["es positivo", "negativo", "no sé"])
    result = evaluate_classifier(clf, examples, LABELS)
    assert result["n"] == 3
    assert result["accuracy"] == 2 / 3
    assert result["invalid_rate"] == 1 / 3
    assert result["predictions"] == ["positivo", "negativo", None]
