"""Deterministic output-constraining: map raw model text to a valid label or None.

This is the classification-flavored version of Nebula's non-negotiable rule that the
model never emits free-form answers for structured tasks. Anything outside the closed
label set is rejected (returns None) so callers can retry or count it as invalid.
"""

from nebula.labels import LABELS


def constrain_label(raw: str | None, labels: list[str] = LABELS) -> str | None:
    if not raw:
        return None
    text = raw.strip().lower()
    if text in labels:
        return text
    # Otherwise pick the label that appears earliest in the text.
    best: str | None = None
    best_pos = len(text) + 1
    for label in labels:
        pos = text.find(label)
        if 0 <= pos < best_pos:
            best, best_pos = label, pos
    return best
