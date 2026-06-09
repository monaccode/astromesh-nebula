"""Evaluate base Qwen3-4B vs Centinela-4B on the held-out Spanish eval set.

Usage:
    uv run python scripts/05_evaluate.py \
        --base Qwen/Qwen3-4B --tuned ./export/merged \
        --eval data/centinela/eval.jsonl --out eval/results/results.md
"""

import argparse
import os

from nebula.config import load_eval_config
from nebula.evaluate import evaluate_classifier, load_examples
from nebula.labels import LABELS


def format_results_table(base: dict, tuned: dict) -> str:
    rows = [
        ("accuracy", "accuracy"),
        ("macro-F1", "macro_f1"),
        ("invalid rate", "invalid_rate"),
    ]
    lines = ["| Metric | Base | Centinela-4B |", "|---|---|---|"]
    for label, key in rows:
        lines.append(f"| {label} | {base[key]:.2f} | {tuned[key]:.2f} |")
    return "\n".join(lines)


def gate_passes(tuned: dict, base: dict, threshold: float, must_beat_base: bool) -> bool:
    if tuned["macro_f1"] < threshold:
        return False
    if must_beat_base and tuned["macro_f1"] <= base["macro_f1"]:
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--tuned", required=True)
    ap.add_argument("--eval", default="data/centinela/eval.jsonl")
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("--out", default="eval/results/results.md")
    ap.add_argument("--frontier", action="store_true", help="(disabled in v0.1)")
    args = ap.parse_args()

    from nebula.hf_classifier import HFChatClassifier

    cfg = load_eval_config(args.config)
    examples = load_examples(args.eval)

    base_res = evaluate_classifier(HFChatClassifier(args.base), examples, LABELS)
    tuned_res = evaluate_classifier(HFChatClassifier(args.tuned), examples, LABELS)

    table = format_results_table(base_res, tuned_res)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("# Centinela-4B evaluation\n\n" + table + "\n")

    passed = gate_passes(
        tuned_res,
        base_res,
        threshold=cfg["thresholds"]["macro_f1"],
        must_beat_base=cfg["thresholds"]["must_beat_base"],
    )
    print(table)
    print(f"\nGATE: {'PASS' if passed else 'FAIL'}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
