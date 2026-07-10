# Centinela Catálogo + Scout — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construir en `astromesh-nebula` el catálogo de modelos/datasets (fuente `catalog/*.yaml` → compilado `catalog.lock.json`) y el Base-Model Scout, que juntos forman el contrato MLOps que consumirá Astromesh.

**Architecture:** Lógica pura en `nebula/catalog.py` y `nebula/scout.py` (dataclasses + funciones, sin red, unit-tested). CLIs finas en `scripts/09_catalog.py` (compila y valida el lock) y `scripts/00_scout.py` (consulta la HF Hub API y rankea candidatos). El lock se valida contra un JSON Schema versionado que será el guardián del contrato cruzado entre repos.

**Tech Stack:** Python 3.11+, `uv`, `pyyaml`, `jsonschema` (nueva dep), `huggingface_hub` (ya presente), `pytest`, `ruff`.

## Global Constraints

- Python `>=3.11` (verbatim de `pyproject.toml`).
- `ruff` line-length `100`, target `py311`.
- Lógica pura en `nebula/`, unit-tested; CLIs finas en `scripts/`.
- Sin acceso a red en tests unitarios — la lógica pura recibe datos por parámetro (inyección).
- Conventional commits.
- Tests importan desde `nebula.*` como los existentes (p.ej. `from nebula.validation import constrain_label`).
- `catalog.lock.json` es determinista: `json.dumps(..., sort_keys=True, ensure_ascii=False, indent=2)`.

---

### Task 1: Modelos de datos + loader del catálogo

**Files:**
- Create: `nebula/catalog.py`
- Test: `tests/test_catalog_load.py`
- Create (fixture): `tests/fixtures/catalog_models.yaml`, `tests/fixtures/catalog_datasets.yaml`

**Interfaces:**
- Produces:
  - `class CatalogError(Exception)`
  - `@dataclass DatasetEntry(id: str, vertical: str, task: str, kind: str, contract: dict, language: str, size: int, hash: str)`
  - `@dataclass Revision(version: str, sha: str, eval: dict, gate: str, formats: list[str], dataset: str, train_config_hash: str, base_model: str, scout_rationale: str = "")`
  - `@dataclass ModelEntry(name: str, kind: str, task: str, vertical: str, hf_repo: str, aliases: dict, revisions: dict[str, Revision], contract: dict)`
  - `load_catalog(models_path: str, datasets_path: str) -> tuple[dict[str, ModelEntry], dict[str, DatasetEntry]]`

- [ ] **Step 1: Write the fixture files**

`tests/fixtures/catalog_datasets.yaml`:
```yaml
datasets:
  - id: centinela-sentiment-es@3
    vertical: finanzas
    task: text-classification
    kind: classifier
    language: es
    size: 4840
    hash: sha256:aaa
    contract:
      labels: [positivo, neutral, negativo]
      validation: constrain_label
```

`tests/fixtures/catalog_models.yaml`:
```yaml
models:
  - name: centinela-sentiment
    kind: classifier
    task: text-classification
    vertical: finanzas
    hf_repo: astromesh/Centinela-Qwen3-4B
    contract:
      labels: [positivo, neutral, negativo]
      validation: constrain_label
    aliases:
      prod: v0.1
    revisions:
      - version: v0.1
        sha: abc123
        gate: passed
        base_model: Qwen/Qwen3-4B
        dataset: centinela-sentiment-es@3
        train_config_hash: sha256:deadbeef
        formats: [safetensors, gguf]
        eval:
          macro_f1: 0.87
          invalid_rate: 0.01
```

- [ ] **Step 2: Write the failing test**

`tests/test_catalog_load.py`:
```python
from nebula.catalog import load_catalog, ModelEntry, DatasetEntry, Revision

FIX = "tests/fixtures"


def test_load_returns_models_and_datasets_by_key():
    models, datasets = load_catalog(f"{FIX}/catalog_models.yaml", f"{FIX}/catalog_datasets.yaml")
    assert set(models) == {"centinela-sentiment"}
    assert set(datasets) == {"centinela-sentiment-es@3"}
    assert isinstance(models["centinela-sentiment"], ModelEntry)
    assert isinstance(datasets["centinela-sentiment-es@3"], DatasetEntry)


def test_revisions_are_keyed_by_version():
    models, _ = load_catalog(f"{FIX}/catalog_models.yaml", f"{FIX}/catalog_datasets.yaml")
    revs = models["centinela-sentiment"].revisions
    assert set(revs) == {"v0.1"}
    assert isinstance(revs["v0.1"], Revision)
    assert revs["v0.1"].gate == "passed"


def test_dataset_fields_parsed():
    _, datasets = load_catalog(f"{FIX}/catalog_models.yaml", f"{FIX}/catalog_datasets.yaml")
    ds = datasets["centinela-sentiment-es@3"]
    assert ds.contract == {"labels": ["positivo", "neutral", "negativo"], "validation": "constrain_label"}
    assert ds.kind == "classifier"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_catalog_load.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nebula.catalog'`

- [ ] **Step 4: Write minimal implementation**

`nebula/catalog.py`:
```python
"""Centinela model/dataset catalog: parse source YAML into typed entries.

The catalog is the machine contract between the foundry (Nebula) and the
runtime (Astromesh). Source YAML is human-editable; the compiled lock
(see compile_catalog) is the machine truth, like uv.lock vs pyproject.toml.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_catalog_load.py -v`
Expected: PASS (3 passed)

- [ ] **Step 6: Lint**

Run: `uv run ruff check nebula/catalog.py tests/test_catalog_load.py`
Expected: no errors

- [ ] **Step 7: Commit**

```bash
git add nebula/catalog.py tests/test_catalog_load.py tests/fixtures/catalog_models.yaml tests/fixtures/catalog_datasets.yaml
git commit -m "feat: catalog data models and YAML loader"
```

---

### Task 2: Compilador del catálogo + reglas duras

**Files:**
- Modify: `nebula/catalog.py` (add `compile_catalog`)
- Test: `tests/test_catalog_compile.py`

**Interfaces:**
- Consumes: `load_catalog`, `ModelEntry`, `DatasetEntry`, `CatalogError` (Task 1).
- Produces: `compile_catalog(models: dict[str, ModelEntry], datasets: dict[str, DatasetEntry]) -> dict` — returns the lock dict with top-level `schema_version` and `models` list; raises `CatalogError` on any rule violation.

Hard rules enforced:
1. Every alias target version must exist in that model's `revisions`.
2. Every aliased revision must have `gate == "passed"`.
3. Every revision's `dataset` must exist in `datasets`.
4. The referenced dataset's `contract` must equal the model's `contract`.

- [ ] **Step 1: Write the failing tests**

`tests/test_catalog_compile.py`:
```python
import pytest

from nebula.catalog import CatalogError, compile_catalog, load_catalog

FIX = "tests/fixtures"


def _load():
    return load_catalog(f"{FIX}/catalog_models.yaml", f"{FIX}/catalog_datasets.yaml")


def test_compile_produces_lock_with_schema_version():
    lock = compile_catalog(*_load())
    assert lock["schema_version"] == "1"
    names = [m["name"] for m in lock["models"]]
    assert names == ["centinela-sentiment"]


def test_compile_keys_revisions_by_version():
    lock = compile_catalog(*_load())
    model = lock["models"][0]
    assert set(model["revisions"]) == {"v0.1"}
    assert model["aliases"] == {"prod": "v0.1"}


def test_alias_to_missing_revision_raises():
    models, datasets = _load()
    models["centinela-sentiment"].aliases["staging"] = "v9.9"
    with pytest.raises(CatalogError, match="v9.9"):
        compile_catalog(models, datasets)


def test_alias_to_unpassed_gate_raises():
    models, datasets = _load()
    models["centinela-sentiment"].revisions["v0.1"].gate = "failed"
    with pytest.raises(CatalogError, match="gate"):
        compile_catalog(models, datasets)


def test_revision_with_unknown_dataset_raises():
    models, datasets = _load()
    models["centinela-sentiment"].revisions["v0.1"].dataset = "ghost@1"
    with pytest.raises(CatalogError, match="ghost@1"):
        compile_catalog(models, datasets)


def test_contract_mismatch_between_model_and_dataset_raises():
    models, datasets = _load()
    datasets["centinela-sentiment-es@3"].contract = {"labels": ["a", "b"]}
    with pytest.raises(CatalogError, match="contract"):
        compile_catalog(models, datasets)


def test_compile_is_deterministic():
    lock_a = compile_catalog(*_load())
    lock_b = compile_catalog(*_load())
    assert lock_a == lock_b
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_catalog_compile.py -v`
Expected: FAIL with `ImportError: cannot import name 'compile_catalog'`

- [ ] **Step 3: Add the implementation**

Append to `nebula/catalog.py`:
```python
from dataclasses import asdict


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_catalog_compile.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Lint**

Run: `uv run ruff check nebula/catalog.py tests/test_catalog_compile.py`
Expected: no errors

- [ ] **Step 6: Commit**

```bash
git add nebula/catalog.py tests/test_catalog_compile.py
git commit -m "feat: catalog compiler with hard invariant checks"
```

---

### Task 3: JSON Schema del lock + validación

**Files:**
- Create: `nebula/catalog.schema.json`
- Modify: `nebula/catalog.py` (add `validate_lock`)
- Modify: `pyproject.toml` (add `jsonschema` dependency)
- Test: `tests/test_catalog_schema.py`

**Interfaces:**
- Consumes: `compile_catalog` (Task 2).
- Produces: `validate_lock(lock: dict) -> None` — raises `jsonschema.ValidationError` if the lock does not conform to `nebula/catalog.schema.json`.

- [ ] **Step 1: Add the dependency**

Edit `pyproject.toml`, in `[project].dependencies`, add `"jsonschema>=4.0"`:
```toml
dependencies = [
    "datasets>=2.19",
    "huggingface_hub>=0.24",
    "pyyaml>=6.0",
    "jsonschema>=4.0",
]
```
Then run: `uv sync --extra dev`
Expected: resolves and installs `jsonschema`.

- [ ] **Step 2: Write the schema file**

`nebula/catalog.schema.json`:
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Centinela catalog.lock",
  "type": "object",
  "required": ["schema_version", "models"],
  "properties": {
    "schema_version": { "type": "string" },
    "models": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["name", "kind", "task", "vertical", "hf_repo", "contract", "aliases", "revisions"],
        "properties": {
          "name": { "type": "string" },
          "kind": { "type": "string", "enum": ["classifier", "extractor", "instruct"] },
          "task": { "type": "string" },
          "vertical": { "type": "string" },
          "hf_repo": { "type": "string" },
          "contract": { "type": "object" },
          "aliases": {
            "type": "object",
            "additionalProperties": { "type": "string" }
          },
          "revisions": {
            "type": "object",
            "additionalProperties": {
              "type": "object",
              "required": ["version", "sha", "eval", "gate", "formats", "dataset", "train_config_hash", "base_model"],
              "properties": {
                "version": { "type": "string" },
                "sha": { "type": "string" },
                "eval": { "type": "object" },
                "gate": { "type": "string", "enum": ["passed", "failed", "pending"] },
                "formats": { "type": "array", "items": { "type": "string" } },
                "dataset": { "type": "string" },
                "train_config_hash": { "type": "string" },
                "base_model": { "type": "string" },
                "scout_rationale": { "type": "string" }
              }
            }
          }
        }
      }
    }
  }
}
```

- [ ] **Step 3: Write the failing test**

`tests/test_catalog_schema.py`:
```python
import pytest
from jsonschema import ValidationError

from nebula.catalog import compile_catalog, load_catalog, validate_lock

FIX = "tests/fixtures"


def test_compiled_lock_validates_against_schema():
    lock = compile_catalog(*load_catalog(f"{FIX}/catalog_models.yaml", f"{FIX}/catalog_datasets.yaml"))
    validate_lock(lock)  # must not raise


def test_bad_gate_enum_fails_schema():
    lock = compile_catalog(*load_catalog(f"{FIX}/catalog_models.yaml", f"{FIX}/catalog_datasets.yaml"))
    lock["models"][0]["revisions"]["v0.1"]["gate"] = "approved"
    with pytest.raises(ValidationError):
        validate_lock(lock)


def test_missing_required_field_fails_schema():
    lock = compile_catalog(*load_catalog(f"{FIX}/catalog_models.yaml", f"{FIX}/catalog_datasets.yaml"))
    del lock["models"][0]["hf_repo"]
    with pytest.raises(ValidationError):
        validate_lock(lock)
```

- [ ] **Step 4: Run test to verify it fails**

Run: `uv run pytest tests/test_catalog_schema.py -v`
Expected: FAIL with `ImportError: cannot import name 'validate_lock'`

- [ ] **Step 5: Add the implementation**

Append to `nebula/catalog.py`:
```python
import json
from pathlib import Path

import jsonschema

_SCHEMA_PATH = Path(__file__).with_name("catalog.schema.json")


def validate_lock(lock: dict) -> None:
    """Validate a compiled lock dict against the versioned JSON Schema."""
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(instance=lock, schema=schema)
```

Also ensure `nebula/catalog.schema.json` ships with the package: edit `pyproject.toml` under `[tool.hatch.build.targets.wheel]`:
```toml
[tool.hatch.build.targets.wheel]
packages = ["nebula"]

[tool.hatch.build.targets.wheel.force-include]
"nebula/catalog.schema.json" = "nebula/catalog.schema.json"
```

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run pytest tests/test_catalog_schema.py -v`
Expected: PASS (3 passed)

- [ ] **Step 7: Lint**

Run: `uv run ruff check nebula/catalog.py tests/test_catalog_schema.py`
Expected: no errors

- [ ] **Step 8: Commit**

```bash
git add nebula/catalog.py nebula/catalog.schema.json tests/test_catalog_schema.py pyproject.toml uv.lock
git commit -m "feat: versioned JSON Schema for catalog.lock plus validate_lock"
```

---

### Task 4: CLI `scripts/09_catalog.py` + target Makefile

**Files:**
- Create: `scripts/09_catalog.py`
- Modify: `Makefile` (add `catalog` target)
- Test: `tests/test_catalog_cli.py`

**Interfaces:**
- Consumes: `load_catalog`, `compile_catalog`, `validate_lock` (Tasks 1-3).
- Produces: CLI `compile` that writes a deterministic `catalog.lock.json`.
  Callable form: `python scripts/09_catalog.py compile --models <path> --datasets <path> --out <path>`.
  Reusable function `compile_to_file(models_path: str, datasets_path: str, out_path: str) -> dict`.

- [ ] **Step 1: Write the failing test**

The scripts are numbered (`09_...`), which are not importable module names. Load them inline with `importlib.util` — the exact convention already used by `tests/test_push_to_hub.py`.

`tests/test_catalog_cli.py`:
```python
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location("catalog_cli", str(Path("scripts/09_catalog.py")))
catalog_cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(catalog_cli)

FIX = "tests/fixtures"


def test_compile_to_file_writes_valid_deterministic_lock(tmp_path):
    out = tmp_path / "catalog.lock.json"
    lock = catalog_cli.compile_to_file(
        f"{FIX}/catalog_models.yaml", f"{FIX}/catalog_datasets.yaml", str(out)
    )
    on_disk = json.loads(out.read_text(encoding="utf-8"))
    assert on_disk == lock
    assert on_disk["schema_version"] == "1"
    # Deterministic: recompiling yields byte-identical output.
    text_first = out.read_text(encoding="utf-8")
    catalog_cli.compile_to_file(
        f"{FIX}/catalog_models.yaml", f"{FIX}/catalog_datasets.yaml", str(out)
    )
    assert out.read_text(encoding="utf-8") == text_first
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_catalog_cli.py -v`
Expected: FAIL (script file / `compile_to_file` does not exist)

- [ ] **Step 3: Write the CLI**

`scripts/09_catalog.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_catalog_cli.py -v`
Expected: PASS

- [ ] **Step 5: Add the Makefile target**

Edit `Makefile` — add `catalog` to `.PHONY` and this target at the end:
```make
catalog:
	uv run python scripts/09_catalog.py compile \
		--models catalog/centinela.yaml --datasets catalog/datasets.yaml --out catalog.lock.json
```

- [ ] **Step 6: Lint**

Run: `uv run ruff check scripts/09_catalog.py tests/test_catalog_cli.py`
Expected: no errors

- [ ] **Step 7: Commit**

```bash
git add scripts/09_catalog.py tests/test_catalog_cli.py Makefile
git commit -m "feat: catalog compile CLI and make target"
```

---

### Task 5: Scout — ranking puro de candidatos

**Files:**
- Create: `nebula/scout.py`
- Test: `tests/test_scout.py`

**Interfaces:**
- Produces:
  - `@dataclass Constraints(max_params_b: float, allowed_licenses: list[str], language: str, pipeline_tag: str = "text-generation")`
  - `@dataclass Candidate(model_id: str, license: str | None, params_b: float | None, languages: list[str], pipeline_tag: str | None, downloads: int, likes: int, leaderboard_score: float | None = None)`
  - `@dataclass ScoredCandidate(candidate: Candidate, score: float, rationale: str)`
  - `rank_candidates(candidates: list[Candidate], constraints: Constraints) -> list[ScoredCandidate]` — filters by hard constraints, then returns survivors sorted by descending score (tie-break by `model_id`). Unknown (`None`) attributes pass the filter (best-effort).

Filter rules: drop if `license` not in `allowed_licenses`; drop if `params_b is not None and params_b > max_params_b`; drop if `pipeline_tag is not None and pipeline_tag != constraints.pipeline_tag`; drop if `languages and language not in languages`.

- [ ] **Step 1: Write the failing tests**

`tests/test_scout.py`:
```python
from nebula.scout import Candidate, Constraints, rank_candidates

K = Constraints(max_params_b=8.0, allowed_licenses=["apache-2.0", "mit"], language="es")


def _cand(model_id, license="apache-2.0", params_b=4.0, languages=None,
          pipeline_tag="text-generation", downloads=100, likes=10, leaderboard_score=None):
    return Candidate(model_id, license, params_b, languages or ["es"], pipeline_tag,
                     downloads, likes, leaderboard_score)


def test_drops_disallowed_license():
    out = rank_candidates([_cand("a", license="cc-by-nc-4.0")], K)
    assert out == []


def test_drops_oversized_model():
    out = rank_candidates([_cand("a", params_b=70.0)], K)
    assert out == []


def test_drops_wrong_pipeline_tag():
    out = rank_candidates([_cand("a", pipeline_tag="image-classification")], K)
    assert out == []


def test_drops_model_without_target_language():
    out = rank_candidates([_cand("a", languages=["en", "fr"])], K)
    assert out == []


def test_unknown_attributes_pass_filter():
    out = rank_candidates([_cand("a", params_b=None, languages=[], pipeline_tag=None)], K)
    assert [s.candidate.model_id for s in out] == ["a"]


def test_ranks_higher_leaderboard_first():
    a = _cand("low", leaderboard_score=0.60)
    b = _cand("high", leaderboard_score=0.90)
    out = rank_candidates([a, b], K)
    assert [s.candidate.model_id for s in out] == ["high", "low"]


def test_is_deterministic_with_tie_break_by_model_id():
    a = _cand("zzz", downloads=100, likes=10, leaderboard_score=0.5)
    b = _cand("aaa", downloads=100, likes=10, leaderboard_score=0.5)
    out = rank_candidates([a, b], K)
    assert [s.candidate.model_id for s in out] == ["aaa", "zzz"]


def test_rationale_is_populated():
    out = rank_candidates([_cand("a")], K)
    assert "apache-2.0" in out[0].rationale
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_scout.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nebula.scout'`

- [ ] **Step 3: Write the implementation**

`nebula/scout.py`:
```python
"""Base-Model Scout: rank Hugging Face base-model candidates for a Centinela model.

Pure ranking logic — callers (scripts/00_scout.py) fetch raw model records from
the HF Hub API and map them to Candidate before calling rank_candidates. No network
here so the ranking stays deterministic and unit-testable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Constraints:
    max_params_b: float
    allowed_licenses: list[str]
    language: str
    pipeline_tag: str = "text-generation"


@dataclass
class Candidate:
    model_id: str
    license: str | None
    params_b: float | None
    languages: list[str]
    pipeline_tag: str | None
    downloads: int
    likes: int
    leaderboard_score: float | None = None


@dataclass
class ScoredCandidate:
    candidate: Candidate
    score: float
    rationale: str


def _passes(c: Candidate, k: Constraints) -> bool:
    if c.license not in k.allowed_licenses:
        return False
    if c.params_b is not None and c.params_b > k.max_params_b:
        return False
    if c.pipeline_tag is not None and c.pipeline_tag != k.pipeline_tag:
        return False
    if c.languages and k.language not in c.languages:
        return False
    return True


def _score(c: Candidate) -> float:
    leaderboard = c.leaderboard_score if c.leaderboard_score is not None else 0.0
    popularity = math.log10(c.downloads + 1) + math.log10(c.likes + 1)
    return 1.5 * leaderboard + 0.5 * popularity


def _rationale(c: Candidate, k: Constraints) -> str:
    size = f"{c.params_b}B" if c.params_b is not None else "size unknown"
    return (
        f"license={c.license}, {size} <= {k.max_params_b}B budget, "
        f"lang '{k.language}' ok, downloads={c.downloads}, likes={c.likes}"
    )


def rank_candidates(
    candidates: list[Candidate], constraints: Constraints
) -> list[ScoredCandidate]:
    """Filter by hard constraints, then sort survivors by descending score."""
    survivors = [c for c in candidates if _passes(c, constraints)]
    scored = [
        ScoredCandidate(candidate=c, score=_score(c), rationale=_rationale(c, constraints))
        for c in survivors
    ]
    scored.sort(key=lambda s: (-s.score, s.candidate.model_id))
    return scored
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_scout.py -v`
Expected: PASS (8 passed)

- [ ] **Step 5: Lint**

Run: `uv run ruff check nebula/scout.py tests/test_scout.py`
Expected: no errors

- [ ] **Step 6: Commit**

```bash
git add nebula/scout.py tests/test_scout.py
git commit -m "feat: base-model scout ranking logic"
```

---

### Task 6: CLI `scripts/00_scout.py` + target Makefile

**Files:**
- Create: `scripts/00_scout.py`
- Modify: `Makefile` (add `scout` target)
- Test: `tests/test_scout_cli.py`

**Interfaces:**
- Consumes: `Candidate`, `Constraints`, `rank_candidates` (Task 5).
- Produces:
  - `to_candidate(model_info) -> Candidate` — maps a `huggingface_hub` `ModelInfo`-like object to a `Candidate` (best-effort field extraction; lives in the thin CLI, not unit-tested against the real API).
  - `_params_b_from_id(model_id: str) -> float | None` — heuristic param count from a model id suffix like `-4B` / `-7b`.
  - CLI: `python scripts/00_scout.py --task text-classification --vertical finanzas --max-params-b 8 --language es`.

- [ ] **Step 1: Write the failing test**

`tests/test_scout_cli.py`:
```python
import importlib.util
from pathlib import Path
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location("scout_cli", str(Path("scripts/00_scout.py")))
scout_cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scout_cli)


def test_params_b_from_id_parses_suffix():
    assert scout_cli._params_b_from_id("Qwen/Qwen3-4B") == 4.0
    assert scout_cli._params_b_from_id("meta-llama/Llama-3.1-8b-instruct") == 8.0
    assert scout_cli._params_b_from_id("some/model-without-size") is None


def test_to_candidate_maps_model_info_fields():
    info = SimpleNamespace(
        id="Qwen/Qwen3-4B",
        pipeline_tag="text-generation",
        downloads=1234,
        likes=56,
        tags=["es", "text-generation", "license:apache-2.0"],
        card_data={"license": "apache-2.0", "language": ["es", "en"]},
    )
    cand = scout_cli.to_candidate(info)
    assert cand.model_id == "Qwen/Qwen3-4B"
    assert cand.license == "apache-2.0"
    assert cand.params_b == 4.0
    assert "es" in cand.languages
    assert cand.downloads == 1234
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_scout_cli.py -v`
Expected: FAIL (script file / functions do not exist)

- [ ] **Step 3: Write the CLI**

`scripts/00_scout.py`:
```python
"""Base-Model Scout CLI: query the HF Hub for candidate base models and rank them.

    uv run python scripts/00_scout.py --task text-classification --vertical finanzas \
        --language es --max-params-b 8 --top 5

Prints a ranked shortlist with rationale. The human picks one to set as base_model
in configs/train-<name>.yaml.
"""

import argparse
import re

from huggingface_hub import HfApi

from nebula.scout import Candidate, Constraints, rank_candidates

_SIZE_RE = re.compile(r"[-_/](\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])")


def _params_b_from_id(model_id: str) -> float | None:
    match = _SIZE_RE.search(model_id)
    return float(match.group(1)) if match else None


def _license_from(info) -> str | None:
    card = getattr(info, "card_data", None) or {}
    if isinstance(card, dict) and card.get("license"):
        return card["license"]
    for tag in getattr(info, "tags", []) or []:
        if tag.startswith("license:"):
            return tag.split(":", 1)[1]
    return None


def _languages_from(info) -> list[str]:
    card = getattr(info, "card_data", None) or {}
    lang = card.get("language") if isinstance(card, dict) else None
    if isinstance(lang, str):
        return [lang]
    return list(lang) if lang else []


def to_candidate(info) -> Candidate:
    return Candidate(
        model_id=info.id,
        license=_license_from(info),
        params_b=_params_b_from_id(info.id),
        languages=_languages_from(info),
        pipeline_tag=getattr(info, "pipeline_tag", None),
        downloads=getattr(info, "downloads", 0) or 0,
        likes=getattr(info, "likes", 0) or 0,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="text-classification", help="downstream task (for display)")
    ap.add_argument("--vertical", default="", help="business vertical (for display)")
    ap.add_argument("--language", default="es")
    ap.add_argument("--max-params-b", type=float, default=8.0)
    ap.add_argument("--pipeline-tag", default="text-generation", help="base capability to fine-tune from")
    ap.add_argument("--licenses", nargs="+", default=["apache-2.0", "mit"])
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    api = HfApi()
    raw = api.list_models(
        pipeline_tag=args.pipeline_tag,
        language=args.language,
        sort="downloads",
        direction=-1,
        limit=200,
        cardData=True,
    )
    candidates = [to_candidate(info) for info in raw]
    constraints = Constraints(
        max_params_b=args.max_params_b,
        allowed_licenses=args.licenses,
        language=args.language,
        pipeline_tag=args.pipeline_tag,
    )
    ranked = rank_candidates(candidates, constraints)

    print(f"# Scout: task={args.task} vertical={args.vertical} language={args.language}")
    print(f"# {len(ranked)} candidates within constraints (max {args.max_params_b}B, {args.licenses})\n")
    for i, s in enumerate(ranked[: args.top], 1):
        print(f"{i}. {s.candidate.model_id}  (score={s.score:.2f})")
        print(f"   {s.rationale}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_scout_cli.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Add the Makefile target**

Edit `Makefile` — add `scout` to `.PHONY` and this target:
```make
scout:
	uv run python scripts/00_scout.py --task $(TASK) --vertical $(VERTICAL) \
		--language es --max-params-b $(MAXB) --top 5
```
And near the top with the other defaults, add:
```make
TASK ?= text-classification
VERTICAL ?= finanzas
MAXB ?= 8
```

- [ ] **Step 6: Lint**

Run: `uv run ruff check scripts/00_scout.py tests/test_scout_cli.py`
Expected: no errors

- [ ] **Step 7: Commit**

```bash
git add scripts/00_scout.py tests/test_scout_cli.py Makefile
git commit -m "feat: base-model scout CLI over HF Hub"
```

---

### Task 7: Seed del catálogo real + prueba de integración

**Files:**
- Create: `catalog/centinela.yaml`
- Create: `catalog/datasets.yaml`
- Test: `tests/test_catalog_integration.py`
- Modify: `.gitignore` (ignore the generated `catalog.lock.json` at repo root)

**Interfaces:**
- Consumes: `compile_to_file` (Task 4). Proves the real catalog compiles and validates end-to-end.

- [ ] **Step 1: Write the real dataset catalog**

`catalog/datasets.yaml` (values mirror the existing Centinela sentiment model; `size` and `hash` are placeholders to be replaced with the real dataset stats when known — see Step 5):
```yaml
datasets:
  - id: centinela-sentiment-es@1
    vertical: finanzas
    task: text-classification
    kind: classifier
    language: es
    size: 0
    hash: "sha256:REPLACE_WITH_REAL_DATASET_HASH"
    contract:
      labels: [positivo, neutral, negativo]
      validation: constrain_label
```

- [ ] **Step 2: Write the real model catalog**

`catalog/centinela.yaml`:
```yaml
models:
  - name: centinela-sentiment
    kind: classifier
    task: text-classification
    vertical: finanzas
    hf_repo: astromesh/Centinela-Qwen3-4B
    contract:
      labels: [positivo, neutral, negativo]
      validation: constrain_label
    aliases:
      prod: v0.1
    revisions:
      - version: v0.1
        sha: "REPLACE_WITH_REAL_HF_REVISION_SHA"
        gate: passed
        base_model: Qwen/Qwen3-4B
        dataset: centinela-sentiment-es@1
        train_config_hash: "sha256:REPLACE_WITH_REAL_TRAIN_CONFIG_HASH"
        formats: [safetensors, gguf]
        eval:
          macro_f1: 0.0
          invalid_rate: 0.0
```
> Note: `sha`, `train_config_hash`, `size`, `hash`, and `eval` scores are seeded as explicit placeholders. They get their real values wired in sub-project 2 (when the eval-gate and push_to_hub emit them). The point of this task is that the *structure* compiles and validates.

- [ ] **Step 3: Write the integration test**

`tests/test_catalog_integration.py`:
```python
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("catalog_cli", str(Path("scripts/09_catalog.py")))
catalog_cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(catalog_cli)


def test_real_catalog_compiles_and_validates(tmp_path):
    out = tmp_path / "catalog.lock.json"
    lock = catalog_cli.compile_to_file("catalog/centinela.yaml", "catalog/datasets.yaml", str(out))
    assert lock["schema_version"] == "1"
    model = lock["models"][0]
    assert model["name"] == "centinela-sentiment"
    assert model["aliases"] == {"prod": "v0.1"}
    assert "v0.1" in model["revisions"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_catalog_integration.py -v`
Expected: PASS (the real catalog compiles because `v0.1` has `gate: passed` and its dataset exists with a matching contract)

- [ ] **Step 5: Ignore the generated lock**

Append to `.gitignore`:
```
# Generated catalog lock (compiled from catalog/*.yaml)
/catalog.lock.json
```
> The lock is a build artifact published as a release asset by CI, not committed. (Placeholders in the seed YAML from Steps 1-2 are tracked and replaced in sub-project 2.)

- [ ] **Step 6: Run the full suite + lint**

Run: `uv run pytest -v`
Expected: all pass, including the new catalog and scout tests.

Run: `uv run ruff check nebula/ tests/ scripts/`
Expected: no errors

- [ ] **Step 7: Commit**

```bash
git add catalog/centinela.yaml catalog/datasets.yaml tests/test_catalog_integration.py .gitignore
git commit -m "feat: seed Centinela catalog with the sentiment model"
```

---

## Self-Review

**Spec coverage (§ del spec de diseño → task):**
- §1 contrato `catalog.lock.json` + reglas del compilador → Tasks 1, 2, 3.
- §2·1-2 `catalog/*.yaml` + lock → Tasks 1, 7.
- §2·3 `nebula catalog` CLI (`scripts/09_catalog.py`) → Task 4.
- §2·4 Base-Model Scout (`scripts/00_scout.py`) → Tasks 5, 6.
- §5 dataset-first (dataset debe existir + contrato coincide) → Task 2 (reglas 3-4), Task 7.
- §7 JSON Schema versionado como guardián del contrato → Task 3 (el gemelo en el repo astromesh es sub-proyecto 2).
- §7 tests deterministas sin red → Tasks 2, 5 (inyección de datos).

Fuera de alcance de este plan (correctamente, son sub-proyectos 2-4): reconciler, provider Centinela, doc generator, bots de PR, el gemelo del JSON Schema en astromesh.

**Placeholder scan:** Los `REPLACE_WITH_REAL_*` en Task 7 son valores de datos semilla intencionales (no placeholders de plan): cada uno tiene una nota de cuándo/quién los reemplaza (sub-proyecto 2) y el test verifica que la *estructura* compila. Todo paso de código muestra código completo.

**Type consistency:** `load_catalog` → `compile_catalog` → `validate_lock` → `compile_to_file` encadenan tipos consistentes (`dict[str, ModelEntry]`, `dict[str, DatasetEntry]`, lock `dict`). `Candidate`/`Constraints`/`rank_candidates` consistentes entre Task 5 (definición) y Task 6 (consumo). Los tests de scripts numerados cargan el módulo inline con `importlib.util` (Tasks 4, 6, 7), siguiendo la convención de `tests/test_push_to_hub.py`.
