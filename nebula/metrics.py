"""Classification metrics over (gold, pred) lists. `pred` may contain None (invalid)."""


def accuracy(gold: list[str], pred: list[str | None]) -> float:
    if not gold:
        return 0.0
    correct = sum(1 for g, p in zip(gold, pred) if p is not None and g == p)
    return correct / len(gold)


def _f1_for_label(gold: list[str], pred: list[str | None], label: str) -> float:
    tp = sum(1 for g, p in zip(gold, pred) if p == label and g == label)
    fp = sum(1 for g, p in zip(gold, pred) if p == label and g != label)
    fn = sum(1 for g, p in zip(gold, pred) if p != label and g == label)
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def macro_f1(gold: list[str], pred: list[str | None], labels: list[str]) -> float:
    if not labels:
        return 0.0
    return sum(_f1_for_label(gold, pred, lab) for lab in labels) / len(labels)


def classification_summary(
    gold: list[str], pred: list[str | None], labels: list[str]
) -> dict:
    invalid = sum(1 for p in pred if p is None)
    return {
        "n": len(gold),
        "accuracy": accuracy(gold, pred),
        "macro_f1": macro_f1(gold, pred, labels),
        "invalid_rate": invalid / len(gold) if gold else 0.0,
        "per_label": {lab: _f1_for_label(gold, pred, lab) for lab in labels},
    }
