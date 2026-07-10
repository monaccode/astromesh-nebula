"""Centinela model/dataset catalog: parse source YAML into typed entries.

The catalog is the machine contract between the foundry (Nebula) and the
runtime (Astromesh). Source YAML is human-editable; the compiled lock
(see compile_catalog) is the machine truth, like uv.lock vs pyproject.toml.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import yaml


class CatalogError(Exception):
    """Raised when the catalog violates a hard invariant."""


@dataclass
class DatasetEntry:
    id: str
    vertical: str
    task: str
    kind: str
    contract: dict
    language: str
    size: int
    hash: str


@dataclass
class Revision:
    version: str
    sha: str
    eval: dict
    gate: str
    formats: list[str]
    dataset: str
    train_config_hash: str
    base_model: str
    scout_rationale: str = ""


@dataclass
class ModelEntry:
    name: str
    kind: str
    task: str
    vertical: str
    hf_repo: str
    aliases: dict
    revisions: dict[str, Revision]
    contract: dict


def _read_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_catalog(
    models_path: str, datasets_path: str
) -> tuple[dict[str, ModelEntry], dict[str, DatasetEntry]]:
    """Parse the source YAML files into typed entries keyed by name/id."""
    datasets: dict[str, DatasetEntry] = {}
    for raw in _read_yaml(datasets_path).get("datasets", []):
        ds = DatasetEntry(**raw)
        datasets[ds.id] = ds

    models: dict[str, ModelEntry] = {}
    for raw in _read_yaml(models_path).get("models", []):
        revisions = {r["version"]: Revision(**r) for r in raw.pop("revisions", [])}
        model = ModelEntry(revisions=revisions, **raw)
        models[model.name] = model
    return models, datasets


def compile_catalog(
    models: dict[str, ModelEntry], datasets: dict[str, DatasetEntry]
) -> dict:
    """Validate the catalog invariants and emit the deterministic lock dict."""
    out_models = []
    for name in sorted(models):
        model = models[name]

        for alias, version in model.aliases.items():
            rev = model.revisions.get(version)
            if rev is None:
                raise CatalogError(
                    f"{name}: alias '{alias}' points to missing revision '{version}'"
                )
            if rev.gate != "passed":
                raise CatalogError(
                    f"{name}: alias '{alias}' -> '{version}' has gate '{rev.gate}', "
                    "only gate 'passed' may be aliased"
                )

        for version, rev in model.revisions.items():
            ds = datasets.get(rev.dataset)
            if ds is None:
                raise CatalogError(
                    f"{name}:{version}: references unknown dataset '{rev.dataset}'"
                )
            if ds.contract != model.contract:
                raise CatalogError(
                    f"{name}:{version}: dataset '{rev.dataset}' contract does not "
                    f"match model contract"
                )

        out_models.append(
            {
                "name": model.name,
                "kind": model.kind,
                "task": model.task,
                "vertical": model.vertical,
                "hf_repo": model.hf_repo,
                "contract": model.contract,
                "aliases": dict(sorted(model.aliases.items())),
                "revisions": {v: asdict(model.revisions[v]) for v in sorted(model.revisions)},
            }
        )

    return {"schema_version": "1", "models": out_models}
