"""Compile the Centinela catalog source YAML into a validated catalog.lock.json.

    uv run python scripts/09_catalog.py compile \
        --models catalog/centinela.yaml --datasets catalog/datasets.yaml \
        --out catalog.lock.json
"""

import argparse
import json

from nebula.catalog import compile_catalog, load_catalog, validate_lock


def compile_to_file(models_path: str, datasets_path: str, out_path: str) -> dict:
    models, datasets = load_catalog(models_path, datasets_path)
    lock = compile_catalog(models, datasets)
    validate_lock(lock)
    text = json.dumps(lock, sort_keys=True, ensure_ascii=False, indent=2)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    return lock


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compile")
    c.add_argument("--models", default="catalog/centinela.yaml")
    c.add_argument("--datasets", default="catalog/datasets.yaml")
    c.add_argument("--out", default="catalog.lock.json")
    args = ap.parse_args()

    if args.cmd == "compile":
        lock = compile_to_file(args.models, args.datasets, args.out)
        print(f"wrote {args.out} ({len(lock['models'])} models)")


if __name__ == "__main__":
    main()
