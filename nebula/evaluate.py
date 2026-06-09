"""Evaluation harness: run a Classifier over eval examples and score it."""

import json
from typing import Protocol

from nebula.metrics import classification_summary
from nebula.validation import constrain_label


class Classifier(Protocol):
    def predict(self, text: str) -> str: ...


def load_examples(path: str) -> list[dict]:
    examples: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            messages = json.loads(line)["messages"]
            text = next(m["content"] for m in messages if m["role"] == "user")
            gold = next(m["content"] for m in messages if m["role"] == "assistant")
            examples.append({"text": text, "gold": gold})
    return examples


def evaluate_classifier(clf: Classifier, examples: list[dict], labels: list[str]) -> dict:
    gold = [ex["gold"] for ex in examples]
    pred = [constrain_label(clf.predict(ex["text"]), labels) for ex in examples]
    summary = classification_summary(gold, pred, labels)
    summary["predictions"] = pred
    return summary
