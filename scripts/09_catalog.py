"""Compile the Centinela catalog source YAML into a validated catalog.lock.json.

    uv run python scripts/09_catalog.py compile \
        --models catalog/centinela.yaml --datasets catalog/datasets.yaml \
        --out catalog.lock.json
"""

import argparse

from nebula.catalog import compile_to_file


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
