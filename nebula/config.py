"""Load YAML configs for evaluation and training."""

import yaml


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_eval_config(path: str) -> dict:
    return _load(path)


def load_train_config(path: str) -> dict:
    return _load(path)
