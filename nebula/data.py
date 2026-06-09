"""Transform source rows into Spanish chat records and split them deterministically."""

import random
from collections import defaultdict

from nebula.labels import EN_TO_ES, SYSTEM_PROMPT_ES


def row_to_chat_record(row: dict) -> dict | None:
    if row.get("task_type") != "sentiment_analysis":
        return None
    answer = (row.get("answer") or "").strip().lower()
    if answer not in EN_TO_ES:
        return None
    text = (row.get("user_prompt_es") or "").strip()
    if not text:
        return None
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_ES},
            {"role": "user", "content": text},
            {"role": "assistant", "content": EN_TO_ES[answer]},
        ]
    }


def record_label(record: dict) -> str:
    return record["messages"][2]["content"]


def stratified_split(
    records: list[dict], eval_frac: float, seed: int
) -> tuple[list[dict], list[dict]]:
    by_label: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_label[record_label(rec)].append(rec)

    rng = random.Random(seed)
    train: list[dict] = []
    eval_set: list[dict] = []
    for label in sorted(by_label):
        group = by_label[label][:]
        rng.shuffle(group)
        n_eval = round(len(group) * eval_frac)
        eval_set.extend(group[:n_eval])
        train.extend(group[n_eval:])
    rng.shuffle(train)
    rng.shuffle(eval_set)
    return train, eval_set
