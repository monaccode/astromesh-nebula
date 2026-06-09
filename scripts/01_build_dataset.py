"""Build Centinela's Spanish sentiment dataset from the NickyNicky source.

Usage:
    uv run python scripts/01_build_dataset.py \
        --dataset NickyNicky/Finance_sentiment_and_topic_classification_Translation_English_to_Spanish_v1 \
        --out data/centinela --eval-frac 0.2 --seed 42
"""

import argparse
import hashlib
import json
import os

from nebula.data import record_label, row_to_chat_record, stratified_split
from nebula.labels import LABELS


def build_split_files(rows, out_dir: str, eval_frac: float, seed: int) -> dict:
    records = [r for r in (row_to_chat_record(row) for row in rows) if r is not None]
    # dedup by Spanish text to prevent leakage from duplicate source rows
    seen: set[str] = set()
    deduped = []
    for rec in records:
        text = rec["messages"][1]["content"]
        if text not in seen:
            seen.add(text)
            deduped.append(rec)

    train, eval_set = stratified_split(deduped, eval_frac=eval_frac, seed=seed)

    os.makedirs(out_dir, exist_ok=True)
    _write_jsonl(os.path.join(out_dir, "train.jsonl"), train)
    _write_jsonl(os.path.join(out_dir, "eval.jsonl"), eval_set)
    with open(os.path.join(out_dir, "labels.json"), "w", encoding="utf-8") as fh:
        json.dump(LABELS, fh, ensure_ascii=False)

    digest = _hash_records(train + eval_set)
    stats = {
        "n_train": len(train),
        "n_eval": len(eval_set),
        "dataset_hash": digest,
        "label_counts": _label_counts(deduped),
    }
    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)
    return stats


def _write_jsonl(path: str, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _label_counts(records: list[dict]) -> dict:
    counts: dict[str, int] = {lab: 0 for lab in LABELS}
    for rec in records:
        counts[record_label(rec)] += 1
    return counts


def _hash_records(records: list[dict]) -> str:
    h = hashlib.sha256()
    for rec in records:
        h.update(json.dumps(rec, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    return h.hexdigest()[:12]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default="data/centinela")
    ap.add_argument("--eval-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset(args.dataset, split="train")
    stats = build_split_files(
        list(ds), args.out, eval_frac=args.eval_frac, seed=args.seed
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
