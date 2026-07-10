# Centinela Runtime — Reconciler + Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Hacer consumibles los modelos Centinela desde los agentes de Astromesh: un reconciler compile-only (`bindings.yaml` + `catalog.lock.json` → `providers.yaml`) y un provider de doble fachada (capability tipada + shim routable), reusando el mismo contrato de validación que la foundry.

**Architecture:** `nebula` (foundry) publica el contrato (schema + lock + `validation`) dentro de su wheel. `astromesh` (runtime) lo consume: un módulo de lógica pura `astromesh/centinela/reconcile.py` (con CLI `astromeshctl centinela reconcile`) traduce despliegues declarados a config de provider, y `astromesh/providers/centinela.py` sirve esas entradas contra un HF Inference Endpoint (TGI, OpenAI-compatible) aplicando `nebula.validation.constrain_label`. El shim se registra en `engine.py` (camino vivo de agentes) y `factory.py` (BYOK).

**Tech Stack:** Python (nebula ≥3.11, astromesh ≥3.12), `httpx`, `pydantic`, `typer`, `pytest` + `respx` (astromesh) / `pytest` (nebula), `hatchling`, `uv`, `jsonschema`.

**Spec:** `docs/superpowers/specs/2026-07-10-centinela-runtime-provider-design.md`.

## Global Constraints

- **Dos repos.** Tasks 1–2 corren en `/Users/fulfaro/monaccode/astromesh-nebula` (foundry). Tasks 3–8 corren en `/Users/fulfaro/monaccode/astromesh` (runtime). Cada commit aterriza en su repo. Cada tarea empieza con `cd` al repo correcto.
- **Versiones / lint.** nebula: `requires-python >=3.11`, ruff `line-length = 100`. astromesh: `requires-python >=3.12`, ruff `line-length = 100`.
- **Conventional commits.** Un commit por tarea. Terminar el mensaje de commit con `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- **TDD estricto.** Test primero, verlo fallar, implementación mínima, verlo pasar, commit.
- **Sin red en unit tests.** nebula: sin red (regla existente). astromesh: HTTP mockeado con `respx` + `httpx.Response`; `asyncio_mode = "auto"` (los tests async NO llevan `@pytest.mark.asyncio`).
- **Lock determinista** (nebula): `json.dumps(..., sort_keys=True, ensure_ascii=False, indent=2)` (ya lo hace `scripts/09_catalog.py`).
- **`ProviderProtocol` es estructural.** El provider implementa los 6 métodos (`complete`, `stream`, `health_check`, `supports_tools`, `supports_vision`, `estimated_cost`); **no hereda** de `ProviderProtocol`.
- **Convención `usage`:** claves `input_tokens`/`output_tokens`.
- **Contrato compartido, no duplicado.** El label set sale de `contract["labels"]`; la validación es `nebula.validation.constrain_label` (el mismo código del eval-gate). No se reimplementa validación en astromesh.
- **Sin ciclo core→adk.** No importar `astromesh_adk` dentro de `astromesh/providers/`. La fachada tipada se expone como método `classify()` que devuelve un modelo Pydantic; el envoltorio `@tool` lo hace el autor del agente.
- **Compile-only.** El reconciler nunca instancia ni llama a un cliente HF.

## File Structure

**nebula (`astromesh-nebula`):**
- Modify: `nebula/catalog.schema.json` — `additionalProperties: false` en ambos niveles.
- Create (generado, commiteado): `nebula/catalog.lock.json` — contrato compilado, shippeado en el wheel.
- Modify: `pyproject.toml` — force-include del lock, `datasets` a extras.
- Modify: `Makefile` — target que compila el lock a `nebula/catalog.lock.json`.
- Test: `tests/test_catalog_schema.py` (existente, se le agregan casos), `tests/test_catalog_lock_shipped.py` (nuevo).

**astromesh (`astromesh`):**
- Modify: `pyproject.toml` — dep + source de `astromesh-nebula`.
- Create: `astromesh/centinela/__init__.py`, `astromesh/centinela/reconcile.py` — lógica pura del reconciler.
- Create: `astromesh/providers/centinela.py` — cliente + fachada tipada + shim routable.
- Create: `astromesh-node/src/astromesh_node/cli/commands/centinela.py` — CLI `reconcile`.
- Modify: `astromesh-node/src/astromesh_node/cli/plugin.py` — registrar el subcomando.
- Modify: `astromesh/runtime/engine.py` — rama `centinela` en `build_candidate_provider`.
- Modify: `astromesh/providers/factory.py` — rama `centinela` en `create_provider`.
- Create: `config/centinela/bindings.yaml` — seed de despliegues.
- Test: `tests/test_centinela_contract.py`, `tests/test_centinela_reconcile.py`, `tests/test_centinela_provider.py`, `tests/test_centinela_wiring.py`, `astromesh-node/tests/test_centinela_cli.py`.

---

### Task 1: [nebula] Endurecer el JSON Schema (`additionalProperties: false`)

**Files:**
- Modify: `nebula/catalog.schema.json`
- Test: `tests/test_catalog_schema.py`

**Interfaces:**
- Consumes: `nebula.catalog.validate_lock(lock: dict) -> None` (existente; valida contra el schema, lanza `jsonschema.ValidationError`).
- Produces: schema que rechaza claves desconocidas en el objeto modelo y en el objeto revisión.

- [ ] **Step 1: Escribir los tests que fallan**

En `tests/test_catalog_schema.py`, agregar (ajustá el import si el archivo ya tiene helpers para construir un lock válido; si no, usá este fixture mínimo):

```python
import pytest
from jsonschema import ValidationError

from nebula.catalog import validate_lock


def _valid_lock() -> dict:
    return {
        "schema_version": "1",
        "models": [
            {
                "name": "centinela-sentiment",
                "kind": "classifier",
                "task": "text-classification",
                "vertical": "finanzas",
                "hf_repo": "astromesh/Centinela-Qwen3-4B",
                "contract": {"labels": ["positivo", "neutral", "negativo"]},
                "aliases": {"prod": "v0.1"},
                "revisions": {
                    "v0.1": {
                        "version": "v0.1",
                        "sha": "abc123",
                        "eval": {"macro_f1": 0.9},
                        "gate": "passed",
                        "formats": ["safetensors"],
                        "dataset": "centinela-sentiment-es@1",
                        "train_config_hash": "sha256:deadbeef",
                        "base_model": "Qwen/Qwen3-4B",
                    }
                },
            }
        ],
    }


def test_valid_lock_passes():
    validate_lock(_valid_lock())  # must not raise


def test_unknown_model_key_rejected():
    lock = _valid_lock()
    lock["models"][0]["surprise"] = "nope"
    with pytest.raises(ValidationError):
        validate_lock(lock)


def test_unknown_revision_key_rejected():
    lock = _valid_lock()
    lock["models"][0]["revisions"]["v0.1"]["surprise"] = "nope"
    with pytest.raises(ValidationError):
        validate_lock(lock)
```

- [ ] **Step 2: Correr los tests para verlos fallar**

Run: `cd /Users/fulfaro/monaccode/astromesh-nebula && uv run pytest tests/test_catalog_schema.py::test_unknown_model_key_rejected tests/test_catalog_schema.py::test_unknown_revision_key_rejected -v`
Expected: FAIL (hoy el schema permite claves extra → no lanza `ValidationError`).

- [ ] **Step 3: Agregar `additionalProperties: false` al schema**

En `nebula/catalog.schema.json`, agregar `"additionalProperties": false` a: (a) el objeto de cada modelo (el `items` de `models`, después de `"properties": { ... }`), y (b) el objeto de cada revisión (el `additionalProperties` que hoy describe la revisión pasa a tener también `"additionalProperties": false` dentro de ese sub-objeto). Queda así:

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
        "additionalProperties": false,
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
              "additionalProperties": false,
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

- [ ] **Step 4: Correr toda la suite de schema para verla pasar**

Run: `cd /Users/fulfaro/monaccode/astromesh-nebula && uv run pytest tests/test_catalog_schema.py -v`
Expected: PASS (incluidos los 3 casos nuevos y los existentes; el lock real usa solo claves conocidas).

- [ ] **Step 5: Verificar que la compilación real sigue validando**

Run: `cd /Users/fulfaro/monaccode/astromesh-nebula && uv run python scripts/09_catalog.py compile --models catalog/centinela.yaml --datasets catalog/datasets.yaml --out /Users/fulfaro/.claude/jobs/867b91d4/tmp/lock_check.json`
Expected: sale sin error (el seed no tiene claves extra). Si falla con `ValidationError`, hay una clave espuria en `catalog/*.yaml` o en `compile_catalog` — corregirla antes de seguir.

- [ ] **Step 6: Commit**

```bash
cd /Users/fulfaro/monaccode/astromesh-nebula
git add nebula/catalog.schema.json tests/test_catalog_schema.py
git commit -m "feat: reject unknown keys in catalog schema (additionalProperties false)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: [nebula] Shippear el lock en el wheel + adelgazar deps base

**Files:**
- Create: `nebula/catalog.lock.json` (generado, commiteado)
- Modify: `pyproject.toml`
- Modify: `Makefile`
- Test: `tests/test_catalog_lock_shipped.py`

**Interfaces:**
- Consumes: `scripts/09_catalog.py compile` (existente) → escribe el lock determinista.
- Produces: `nebula/catalog.lock.json` legible por `importlib.resources.files("nebula") / "catalog.lock.json"`; deps base de nebula sin `datasets` (solo `huggingface_hub`, `pyyaml`, `jsonschema`).

- [ ] **Step 1: Escribir el test de deriva (drift) que falla**

Crear `tests/test_catalog_lock_shipped.py`:

```python
import json
from importlib.resources import files
from pathlib import Path

from nebula.catalog import compile_to_file, validate_lock

REPO = Path(__file__).resolve().parents[1]


def test_shipped_lock_is_present_and_valid():
    lock_text = (files("nebula") / "catalog.lock.json").read_text(encoding="utf-8")
    lock = json.loads(lock_text)
    validate_lock(lock)  # must not raise
    assert lock["schema_version"] == "1"
    assert any(m["name"] == "centinela-sentiment" for m in lock["models"])


def test_shipped_lock_has_no_drift(tmp_path):
    """The committed nebula/catalog.lock.json must equal a fresh compile of catalog/*.yaml."""
    fresh = tmp_path / "fresh.json"
    compile_to_file(
        str(REPO / "catalog" / "centinela.yaml"),
        str(REPO / "catalog" / "datasets.yaml"),
        str(fresh),
    )
    committed = json.loads((REPO / "nebula" / "catalog.lock.json").read_text(encoding="utf-8"))
    assert json.loads(fresh.read_text(encoding="utf-8")) == committed
```

- [ ] **Step 2: Correr para verlo fallar**

Run: `cd /Users/fulfaro/monaccode/astromesh-nebula && uv run pytest tests/test_catalog_lock_shipped.py -v`
Expected: FAIL (no existe `nebula/catalog.lock.json` todavía → `FileNotFoundError`/`ModuleNotFound` en `files`).

- [ ] **Step 3: Generar el lock commiteable**

Run: `cd /Users/fulfaro/monaccode/astromesh-nebula && uv run python scripts/09_catalog.py compile --models catalog/centinela.yaml --datasets catalog/datasets.yaml --out nebula/catalog.lock.json`
Expected: crea `nebula/catalog.lock.json`. Verificar que quedó dentro del paquete (no en la raíz): `ls nebula/catalog.lock.json`.

- [ ] **Step 4: Force-include del lock en el wheel + `datasets` a extras**

En `pyproject.toml`, extender el bloque force-include:

```toml
[tool.hatch.build.targets.wheel.force-include]
"nebula/catalog.schema.json" = "nebula/catalog.schema.json"
"nebula/catalog.lock.json" = "nebula/catalog.lock.json"
```

Y adelgazar las deps base (mover `datasets` a los extras que lo usan; el paquete `nebula/` no lo importa — solo los scripts numerados, que declaran sus deps inline):

```toml
dependencies = [
    "huggingface_hub>=0.24",
    "pyyaml>=6.0",
    "jsonschema>=4.0",
]

[project.optional-dependencies]
eval = ["transformers>=4.44", "torch>=2.3", "accelerate>=0.33", "datasets>=2.19"]
demo = ["gradio>=4.0", "llama-cpp-python>=0.2.80"]
dev = ["pytest>=8.0", "ruff>=0.6", "datasets>=2.19"]
```

- [ ] **Step 5: Agregar el target de Makefile**

En `Makefile`, agregar:

```makefile
lock:  ## Compile the catalog lock shipped inside the wheel
	uv run python scripts/09_catalog.py compile \
		--models catalog/centinela.yaml --datasets catalog/datasets.yaml \
		--out nebula/catalog.lock.json
```

- [ ] **Step 6: Correr la suite completa (con dev extra) para verla pasar**

Run: `cd /Users/fulfaro/monaccode/astromesh-nebula && uv sync --extra dev && uv run pytest -v`
Expected: PASS, incluidos los 2 tests nuevos. Si algún test existente importaba `datasets` sin declararlo, ahora lo cubre el `dev` extra.

- [ ] **Step 7: Lint**

Run: `cd /Users/fulfaro/monaccode/astromesh-nebula && uv run ruff check nebula/ tests/ scripts/`
Expected: sin errores.

- [ ] **Step 8: Commit**

```bash
cd /Users/fulfaro/monaccode/astromesh-nebula
git add nebula/catalog.lock.json pyproject.toml Makefile tests/test_catalog_lock_shipped.py
git commit -m "feat: ship compiled catalog.lock.json in the wheel; slim base deps

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: [astromesh] Dependencia `nebula` + test twin del contrato cruzado

**Files:**
- Modify: `pyproject.toml` (repo `astromesh`, raíz)
- Test: `tests/test_centinela_contract.py`

**Interfaces:**
- Consumes: `nebula.catalog.validate_lock(lock: dict) -> None` desde el paquete instalado (mismo código/schema que la foundry).
- Produces: `astromesh` puede `import nebula.validation` y `import nebula.catalog`; guardián que rompe CI si el entendimiento del lock diverge del schema shippeado por nebula.

- [ ] **Step 1: Cablear la dependencia**

En `/Users/fulfaro/monaccode/astromesh/pyproject.toml`:

Agregar el extra opcional (bajo `[project.optional-dependencies]`):

```toml
centinela = ["astromesh-nebula>=0.1.0"]
```

Sumar `centinela` al agregador `all`:

```toml
all = [
    "astromesh[redis,postgres,sqlite,chromadb,qdrant,faiss,embeddings,onnx,observability,mcp,mesh,litellm,centinela]",
]
```

Agregar `astromesh-nebula` al dev group (para que `uv sync` lo instale y los tests puedan importarlo), junto a `astromesh-adk`:

```toml
[dependency-groups]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24.0",
    "pytest-cov>=5.0",
    "respx>=0.21.0",
    "ruff>=0.6.0",
    "pytest-benchmark>=4.0",
    "asgi-lifespan>=2.1.0",
    "pytest-mock>=3.15.1",
    "pre-commit>=3.5.0",
    "astromesh-adk",
    "astromesh-nebula",
]
```

Y pinnear la fuente (dev/CI editable; en Docker/deploy se resuelve el wheel publicado):

```toml
[tool.uv.sources]
astromesh-adk = { path = "astromesh-adk", editable = true }
astromesh-nebula = { path = "../astromesh-nebula", editable = true }
```

- [ ] **Step 2: Sincronizar e importar**

Run: `cd /Users/fulfaro/monaccode/astromesh && uv sync`
Expected: instala `astromesh-nebula` editable. Verificar: `uv run python -c "import nebula.validation, nebula.catalog; print('ok')"` → imprime `ok`.

- [ ] **Step 3: Escribir el test twin que falla**

Crear `tests/test_centinela_contract.py`:

```python
"""Cross-repo guardian: astromesh validates the SAME lock against nebula's shipped schema."""

import pytest
from jsonschema import ValidationError

from nebula.catalog import validate_lock


def _valid_lock() -> dict:
    return {
        "schema_version": "1",
        "models": [
            {
                "name": "centinela-sentiment",
                "kind": "classifier",
                "task": "text-classification",
                "vertical": "finanzas",
                "hf_repo": "astromesh/Centinela-Qwen3-4B",
                "contract": {"labels": ["positivo", "neutral", "negativo"]},
                "aliases": {"prod": "v0.1"},
                "revisions": {
                    "v0.1": {
                        "version": "v0.1",
                        "sha": "abc123",
                        "eval": {"macro_f1": 0.9},
                        "gate": "passed",
                        "formats": ["safetensors"],
                        "dataset": "centinela-sentiment-es@1",
                        "train_config_hash": "sha256:deadbeef",
                        "base_model": "Qwen/Qwen3-4B",
                    }
                },
            }
        ],
    }


def test_astromesh_accepts_valid_lock():
    validate_lock(_valid_lock())  # must not raise


def test_astromesh_rejects_incompatible_lock():
    lock = _valid_lock()
    lock["models"][0]["revisions"]["v0.1"]["gate"] = "banana"  # not in enum
    with pytest.raises(ValidationError):
        validate_lock(lock)
```

- [ ] **Step 4: Correr para verlo pasar**

Run: `cd /Users/fulfaro/monaccode/astromesh && uv run pytest tests/test_centinela_contract.py -v`
Expected: PASS. (Este test depende de que nebula ya tenga `additionalProperties`/enums de las Tasks 1–2, pero acá solo ejercita gate-enum y validez básica.)

- [ ] **Step 5: Commit**

```bash
cd /Users/fulfaro/monaccode/astromesh
git add pyproject.toml tests/test_centinela_contract.py
git commit -m "feat: depend on astromesh-nebula; cross-repo lock schema twin test

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: [astromesh] Reconciler — lógica pura

**Files:**
- Create: `astromesh/centinela/__init__.py`
- Create: `astromesh/centinela/reconcile.py`
- Test: `tests/test_centinela_reconcile.py`

**Interfaces:**
- Consumes: un `lock: dict` con forma `{"schema_version","models":[{"name","kind","contract","aliases":{alias:version},"revisions":{version:{"gate","sha",...}}}]}`, y un `bindings: dict` con forma `{"spec":{"bindings":[{"model","alias","endpoint"}]}}`.
- Produces:
  - `reconcile(lock: dict, bindings: dict) -> dict[str, dict]` — mapa `provider_name -> entry` ordenado; `entry = {"type":"centinela","endpoint","models":[name],"kind","contract","revision","sha"}`.
  - `to_provider_config(providers: dict) -> dict` — envuelve en un doc `ProviderConfig`.
  - `class ReconcileError(ValueError)`.

- [ ] **Step 1: Escribir los tests que fallan**

Crear `tests/test_centinela_reconcile.py`:

```python
import pytest

from astromesh.centinela.reconcile import ReconcileError, reconcile, to_provider_config


def _lock() -> dict:
    return {
        "schema_version": "1",
        "models": [
            {
                "name": "centinela-sentiment",
                "kind": "classifier",
                "task": "text-classification",
                "vertical": "finanzas",
                "hf_repo": "astromesh/Centinela-Qwen3-4B",
                "contract": {"labels": ["positivo", "neutral", "negativo"]},
                "aliases": {"prod": "v0.1"},
                "revisions": {
                    "v0.1": {"version": "v0.1", "sha": "abc123", "gate": "passed"},
                    "v0.2": {"version": "v0.2", "sha": "def456", "gate": "pending"},
                },
            },
            {
                "name": "centinela-chat",
                "kind": "instruct",
                "task": "text-generation",
                "vertical": "finanzas",
                "hf_repo": "astromesh/Centinela-Chat",
                "contract": {},
                "aliases": {"prod": "v1"},
                "revisions": {"v1": {"version": "v1", "sha": "c0ffee", "gate": "passed"}},
            },
        ],
    }


def _bindings(model="centinela-sentiment", alias="prod"):
    return {
        "apiVersion": "astromesh/v1",
        "kind": "CentinelaBindings",
        "metadata": {"name": "default"},
        "spec": {
            "bindings": [
                {"model": model, "alias": alias, "endpoint": "https://ep.example.cloud"}
            ]
        },
    }


def test_reconcile_happy_path():
    out = reconcile(_lock(), _bindings())
    assert out == {
        "centinela-sentiment": {
            "type": "centinela",
            "endpoint": "https://ep.example.cloud",
            "models": ["centinela-sentiment"],
            "kind": "classifier",
            "contract": {"labels": ["positivo", "neutral", "negativo"]},
            "revision": "v0.1",
            "sha": "abc123",
        }
    }


def test_reconcile_unknown_model_raises():
    with pytest.raises(ReconcileError, match="unknown model"):
        reconcile(_lock(), _bindings(model="nope"))


def test_reconcile_unknown_alias_raises():
    with pytest.raises(ReconcileError, match="alias"):
        reconcile(_lock(), _bindings(alias="staging"))


def test_reconcile_non_passed_gate_raises():
    lock = _lock()
    lock["models"][0]["aliases"]["prod"] = "v0.2"  # gate pending
    with pytest.raises(ReconcileError, match="gate"):
        reconcile(lock, _bindings())


def test_reconcile_instruct_kind_raises():
    with pytest.raises(ReconcileError, match="hf_tgi"):
        reconcile(_lock(), _bindings(model="centinela-chat"))


def test_reconcile_is_sorted():
    bindings = _bindings()
    bindings["spec"]["bindings"].insert(
        0, {"model": "centinela-sentiment", "alias": "prod", "endpoint": "https://ep.example.cloud"}
    )
    out = reconcile(_lock(), bindings)
    assert list(out.keys()) == sorted(out.keys())


def test_to_provider_config_wraps():
    providers = reconcile(_lock(), _bindings())
    doc = to_provider_config(providers)
    assert doc["apiVersion"] == "astromesh/v1"
    assert doc["kind"] == "ProviderConfig"
    assert doc["spec"]["providers"] == providers
```

- [ ] **Step 2: Correr para verlos fallar**

Run: `cd /Users/fulfaro/monaccode/astromesh && uv run pytest tests/test_centinela_reconcile.py -v`
Expected: FAIL (`ModuleNotFoundError: astromesh.centinela`).

- [ ] **Step 3: Implementar la lógica pura**

Crear `astromesh/centinela/__init__.py`:

```python
"""Centinela runtime glue: reconcile foundry contract + bindings into provider config."""
```

Crear `astromesh/centinela/reconcile.py`:

```python
"""Compile Centinela deployment bindings + the foundry catalog lock into provider config.

Compile-only: this never touches HF infrastructure. Given the machine contract
(`catalog.lock.json`, produced by the astromesh-nebula foundry) and the operator's
`bindings.yaml`, produce the `type: centinela` entries for `config/providers.yaml`.
Output is deterministic (sorted) so it can be snapshot-tested.
"""

from __future__ import annotations

_SERVED_KINDS = {"classifier", "extractor"}


class ReconcileError(ValueError):
    """A binding cannot be reconciled against the catalog lock."""


def reconcile(lock: dict, bindings: dict) -> dict:
    """Map each binding to a Centinela provider entry, validating against the lock."""
    models = {m["name"]: m for m in lock.get("models", [])}
    out: dict[str, dict] = {}

    for b in bindings.get("spec", {}).get("bindings", []):
        name = b["model"]
        alias = b["alias"]

        model = models.get(name)
        if model is None:
            raise ReconcileError(f"binding references unknown model '{name}'")

        version = model["aliases"].get(alias)
        if version is None:
            raise ReconcileError(f"{name}: alias '{alias}' not found in catalog")

        rev = model["revisions"][version]
        if rev["gate"] != "passed":
            raise ReconcileError(
                f"{name}:{version} has gate '{rev['gate']}', only 'passed' may be served"
            )

        if model["kind"] not in _SERVED_KINDS:
            raise ReconcileError(
                f"{name} kind '{model['kind']}' is not served by the Centinela provider; "
                "instruct models route through hf_tgi"
            )

        out[name] = {
            "type": "centinela",
            "endpoint": b["endpoint"],
            "models": [name],
            "kind": model["kind"],
            "contract": model["contract"],
            "revision": version,
            "sha": rev["sha"],
        }

    return dict(sorted(out.items()))


def to_provider_config(providers: dict) -> dict:
    """Wrap reconciled provider entries in a ProviderConfig document."""
    return {
        "apiVersion": "astromesh/v1",
        "kind": "ProviderConfig",
        "metadata": {"name": "centinela-generated"},
        "spec": {"providers": providers},
    }
```

- [ ] **Step 4: Correr para verlos pasar**

Run: `cd /Users/fulfaro/monaccode/astromesh && uv run pytest tests/test_centinela_reconcile.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Lint**

Run: `cd /Users/fulfaro/monaccode/astromesh && uv run ruff check astromesh/centinela/ tests/test_centinela_reconcile.py`
Expected: sin errores.

- [ ] **Step 6: Commit**

```bash
cd /Users/fulfaro/monaccode/astromesh
git add astromesh/centinela/ tests/test_centinela_reconcile.py
git commit -m "feat: Centinela reconciler pure logic (compile-only)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: [astromesh] Reconciler — CLI `astromeshctl centinela reconcile` + seed

**Files:**
- Create: `astromesh-node/src/astromesh_node/cli/commands/centinela.py`
- Modify: `astromesh-node/src/astromesh_node/cli/plugin.py`
- Create: `config/centinela/bindings.yaml`
- Test: `astromesh-node/tests/test_centinela_cli.py`

**Interfaces:**
- Consumes: `astromesh.centinela.reconcile.reconcile`, `.to_provider_config` (Task 4); el lock shippeado por nebula vía `importlib.resources`.
- Produces: comando typer `reconcile(bindings, out)` bajo el grupo `centinela`; helper `_load_lock() -> dict` (monkeypatcheable en tests).

- [ ] **Step 1: Escribir el test del CLI que falla**

Crear `astromesh-node/tests/test_centinela_cli.py`:

```python
from pathlib import Path

import yaml

from astromesh_node.cli.commands import centinela

LOCK = {
    "schema_version": "1",
    "models": [
        {
            "name": "centinela-sentiment",
            "kind": "classifier",
            "task": "text-classification",
            "vertical": "finanzas",
            "hf_repo": "astromesh/Centinela-Qwen3-4B",
            "contract": {"labels": ["positivo", "neutral", "negativo"]},
            "aliases": {"prod": "v0.1"},
            "revisions": {"v0.1": {"version": "v0.1", "sha": "abc123", "gate": "passed"}},
        }
    ],
}

BINDINGS = {
    "apiVersion": "astromesh/v1",
    "kind": "CentinelaBindings",
    "metadata": {"name": "default"},
    "spec": {
        "bindings": [
            {"model": "centinela-sentiment", "alias": "prod", "endpoint": "https://ep.example.cloud"}
        ]
    },
}


def test_reconcile_command_writes_provider_config(tmp_path, monkeypatch):
    monkeypatch.setattr(centinela, "_load_lock", lambda: LOCK)
    bindings_path = tmp_path / "bindings.yaml"
    bindings_path.write_text(yaml.safe_dump(BINDINGS))
    out_path = tmp_path / "providers.centinela.yaml"

    centinela.reconcile_command(bindings=str(bindings_path), out=str(out_path))

    doc = yaml.safe_load(out_path.read_text())
    assert doc["kind"] == "ProviderConfig"
    entry = doc["spec"]["providers"]["centinela-sentiment"]
    assert entry["type"] == "centinela"
    assert entry["endpoint"] == "https://ep.example.cloud"
    assert entry["contract"]["labels"] == ["positivo", "neutral", "negativo"]


def test_plugin_registers_centinela():
    import typer

    from astromesh_node.cli import plugin

    app = typer.Typer()
    plugin.register(app)
    names = {g.name for g in app.registered_groups}
    assert "centinela" in names
```

- [ ] **Step 2: Correr para verlo fallar**

Run: `cd /Users/fulfaro/monaccode/astromesh && uv run pytest astromesh-node/tests/test_centinela_cli.py -v`
Expected: FAIL (`ImportError: cannot import name 'centinela'`).

- [ ] **Step 3: Implementar el comando**

Crear `astromesh-node/src/astromesh_node/cli/commands/centinela.py`:

```python
"""astromeshctl centinela — reconcile Centinela bindings into provider config."""

import json
from importlib.resources import files
from pathlib import Path

import typer
import yaml

from astromesh.centinela.reconcile import reconcile, to_provider_config
from astromesh_node.cli.output import console

app = typer.Typer(help="Centinela model provider management.")


def _load_lock() -> dict:
    """Read the compiled catalog lock shipped inside the astromesh-nebula wheel."""
    text = (files("nebula") / "catalog.lock.json").read_text(encoding="utf-8")
    return json.loads(text)


@app.command("reconcile")
def reconcile_command(
    bindings: str = typer.Option(
        "./config/centinela/bindings.yaml", "--bindings", help="Path to bindings.yaml"
    ),
    out: str = typer.Option(
        "./config/providers.centinela.yaml", "--out", help="Output ProviderConfig path"
    ),
) -> None:
    """Compile bindings + catalog lock into provider config (compile-only; no HF calls)."""
    lock = _load_lock()
    bindings_doc = yaml.safe_load(Path(bindings).read_text())
    providers = reconcile(lock, bindings_doc)
    doc = to_provider_config(providers)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(yaml.safe_dump(doc, sort_keys=True, allow_unicode=True))
    console.print(f"[green]Reconciled[/green] {len(providers)} Centinela provider(s) -> {out}")
```

- [ ] **Step 4: Registrar el subcomando en el plugin**

Modificar `astromesh-node/src/astromesh_node/cli/plugin.py`:

```python
"""Node CLI plugin — registers init/validate/config/centinela into astromeshctl."""

import typer

from astromesh_node.cli.commands import centinela, config, init, validate


def register(app: typer.Typer) -> None:
    """Plugin entry point — called by astromeshctl plugin discovery."""
    app.add_typer(config.app, name="config")
    app.add_typer(centinela.app, name="centinela")
    app.command("init")(init.init_command)
    app.command("validate")(validate.validate_command)
```

- [ ] **Step 5: Crear el seed de bindings**

Crear `config/centinela/bindings.yaml`:

```yaml
apiVersion: astromesh/v1
kind: CentinelaBindings
metadata:
  name: default
spec:
  bindings:
    - model: centinela-sentiment
      alias: prod
      endpoint: "https://REPLACE_WITH_REAL_HF_ENDPOINT.endpoints.huggingface.cloud"
```

- [ ] **Step 6: Correr para verlo pasar**

Run: `cd /Users/fulfaro/monaccode/astromesh && uv run pytest astromesh-node/tests/test_centinela_cli.py -v`
Expected: PASS (2 tests).

- [ ] **Step 7: Lint**

Run: `cd /Users/fulfaro/monaccode/astromesh && uv run ruff check astromesh-node/src/astromesh_node/cli/commands/centinela.py astromesh-node/src/astromesh_node/cli/plugin.py astromesh-node/tests/test_centinela_cli.py`
Expected: sin errores.

- [ ] **Step 8: Commit**

```bash
cd /Users/fulfaro/monaccode/astromesh
git add astromesh-node/src/astromesh_node/cli/commands/centinela.py astromesh-node/src/astromesh_node/cli/plugin.py config/centinela/bindings.yaml astromesh-node/tests/test_centinela_cli.py
git commit -m "feat: astromeshctl centinela reconcile command + bindings seed

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: [astromesh] Provider — cliente + fachada tipada (`classify`)

**Files:**
- Create: `astromesh/providers/centinela.py`
- Test: `tests/test_centinela_provider.py`

**Interfaces:**
- Consumes: `nebula.validation.constrain_label(raw: str | None, labels: list[str]) -> str | None`; HF Endpoint OpenAI-compatible en `POST {endpoint}/v1/chat/completions`.
- Produces:
  - `class SentimentResult(BaseModel)` — `label: str | None`, `valid: bool`, `raw: str`, `score: float | None = None`.
  - `class _CentinelaEndpointClient` — `__init__(config)`, `async classify(text) -> SentimentResult`, `async health_check() -> bool`. Config: `endpoint`, `model`, `timeout`, `contract` (con `labels`), `invalid_policy` (`"mark"`|`"retry"`), `max_retries`.

**Nota de diseño (resuelve la pregunta abierta del spec §4):** los modelos Centinela son generativos (Qwen3-4B) servidos por TGI (OpenAI-compatible). La generación no expone probabilidad por etiqueta → `score` queda `None` bajo este serving. `constrain_label` mapea el texto generado al label set cerrado; fuera del set → `valid=False`. `score` se mantiene como campo opcional por si más adelante el endpoint expone logprobs/text-classification.

- [ ] **Step 1: Escribir los tests que fallan**

Crear `tests/test_centinela_provider.py`:

```python
from __future__ import annotations

import httpx
import respx

from astromesh.providers.centinela import SentimentResult, _CentinelaEndpointClient

CONTRACT = {"labels": ["positivo", "neutral", "negativo"]}


def _chat_response(content: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


@respx.mock
async def test_classify_valid_label():
    respx.post("http://ep.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_chat_response("positivo"))
    )
    client = _CentinelaEndpointClient({"endpoint": "http://ep.test", "contract": CONTRACT})
    result = await client.classify("las ganancias del trimestre subieron")

    assert isinstance(result, SentimentResult)
    assert result.label == "positivo"
    assert result.valid is True
    assert result.score is None


@respx.mock
async def test_classify_out_of_set_is_invalid():
    respx.post("http://ep.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_chat_response("no tengo idea"))
    )
    client = _CentinelaEndpointClient({"endpoint": "http://ep.test", "contract": CONTRACT})
    result = await client.classify("texto ambiguo")

    assert result.label is None
    assert result.valid is False
    assert result.raw == "no tengo idea"


@respx.mock
async def test_classify_retry_policy_reclassifies():
    route = respx.post("http://ep.test/v1/chat/completions")
    route.side_effect = [
        httpx.Response(200, json=_chat_response("ninguna")),
        httpx.Response(200, json=_chat_response("negativo")),
    ]
    client = _CentinelaEndpointClient(
        {"endpoint": "http://ep.test", "contract": CONTRACT, "invalid_policy": "retry", "max_retries": 2}
    )
    result = await client.classify("la deuda creció")

    assert result.label == "negativo"
    assert result.valid is True
    assert route.call_count == 2


@respx.mock
async def test_health_check_healthy_and_unhealthy():
    respx.get("http://ep.test/health").mock(return_value=httpx.Response(200))
    client = _CentinelaEndpointClient({"endpoint": "http://ep.test", "contract": CONTRACT})
    assert await client.health_check() is True

    respx.get("http://ep.test/health").mock(return_value=httpx.Response(503))
    assert await client.health_check() is False
```

- [ ] **Step 2: Correr para verlos fallar**

Run: `cd /Users/fulfaro/monaccode/astromesh && uv run pytest tests/test_centinela_provider.py -v`
Expected: FAIL (`ModuleNotFoundError: astromesh.providers.centinela`).

- [ ] **Step 3: Implementar cliente + fachada tipada**

Crear `astromesh/providers/centinela.py`:

```python
"""Centinela provider — typed classifier capability + routable chat shim.

Centinela classifiers/extractors are served on HF Inference Endpoints running TGI
(OpenAI-compatible `/v1/chat/completions`). This module wraps that endpoint.

Typed facade: `classify()` returns a Pydantic `SentimentResult`. It applies the SAME
`nebula.validation.constrain_label` used by the foundry eval-gate, so a served model can
never emit a label outside the closed contract set. Expose it to an ADK agent by wrapping
`classify` in an `@tool` from the agent side (kept out of core to avoid a core->adk cycle):

    from astromesh_adk import tool
    from pydantic import BaseModel

    class TextInput(BaseModel):
        text: str

    @tool("Clasifica sentimiento financiero")
    async def centinela_sentiment(inp: TextInput) -> SentimentResult:
        return await client.classify(inp.text)

Routable facade: `CentinelaProvider` implements ProviderProtocol so the model_router can
select Centinela (estimated_cost approx 0 -> preferred under cost_optimized).
"""

from __future__ import annotations

import time
from typing import Any, AsyncIterator

import httpx
from pydantic import BaseModel

from nebula.validation import constrain_label

from .base import CompletionChunk, CompletionResponse


class SentimentResult(BaseModel):
    """Typed classifier output. `score` is None under generative (TGI) serving."""

    label: str | None
    valid: bool
    raw: str
    score: float | None = None


def _build_system_prompt(labels: list[str]) -> str:
    joined = ", ".join(labels)
    return (
        "Clasificá el texto. Respondé únicamente con una sola de estas etiquetas, "
        f"sin explicación ni puntuación: {joined}."
    )


class _CentinelaEndpointClient:
    """httpx core over a Centinela HF Inference Endpoint (TGI, OpenAI-compatible)."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self.endpoint: str = (config.get("endpoint") or "http://localhost:8080").rstrip("/")
        self.model: str = config.get("model", "centinela")
        self.timeout: float = float(config.get("timeout", 30.0))
        contract = config.get("contract") or {}
        self.labels: list[str] = list(contract.get("labels", []))
        self.invalid_policy: str = config.get("invalid_policy", "mark")
        self.max_retries: int = int(config.get("max_retries", 1))
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(base_url=self.endpoint, timeout=self.timeout)
        return self._client

    async def _raw_label(self, text: str) -> str:
        client = await self._get_client()
        messages = [
            {"role": "system", "content": _build_system_prompt(self.labels)},
            {"role": "user", "content": text},
        ]
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": self.model, "messages": messages, "stream": False},
        )
        resp.raise_for_status()
        data = resp.json()
        return (data["choices"][0].get("message", {}).get("content", "") or "").strip()

    async def classify(self, text: str) -> SentimentResult:
        attempts = self.max_retries if self.invalid_policy == "retry" else 1
        raw = ""
        label: str | None = None
        for _ in range(max(1, attempts)):
            raw = await self._raw_label(text)
            label = constrain_label(raw, self.labels)
            if label is not None:
                break
        return SentimentResult(label=label, valid=label is not None, raw=raw, score=None)

    async def health_check(self) -> bool:
        try:
            client = await self._get_client()
            resp = await client.get("/health")
            return resp.status_code == 200
        except Exception:
            return False
```

- [ ] **Step 4: Correr para verlos pasar**

Run: `cd /Users/fulfaro/monaccode/astromesh && uv run pytest tests/test_centinela_provider.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Lint**

Run: `cd /Users/fulfaro/monaccode/astromesh && uv run ruff check astromesh/providers/centinela.py tests/test_centinela_provider.py`
Expected: sin errores.

- [ ] **Step 6: Commit**

```bash
cd /Users/fulfaro/monaccode/astromesh
git add astromesh/providers/centinela.py tests/test_centinela_provider.py
git commit -m "feat: Centinela endpoint client + typed classify facade

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: [astromesh] Provider — shim routable `CentinelaProvider`

**Files:**
- Modify: `astromesh/providers/centinela.py`
- Test: `tests/test_centinela_provider.py`

**Interfaces:**
- Consumes: `_CentinelaEndpointClient`, `SentimentResult` (Task 6); `CompletionResponse`, `CompletionChunk` de `.base`.
- Produces: `class CentinelaProvider` con los 6 métodos de `ProviderProtocol`. `complete` mapea el último msg `user` → `classify` → `content=label`, estructura en `metadata`. `estimated_cost` → 0.0.

- [ ] **Step 1: Escribir los tests que fallan**

Agregar a `tests/test_centinela_provider.py`:

```python
from astromesh.providers.base import CompletionResponse  # add to existing imports
from astromesh.providers.centinela import CentinelaProvider  # add to existing imports


@respx.mock
async def test_provider_complete_maps_label_to_content():
    respx.post("http://ep.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_chat_response("negativo"))
    )
    provider = CentinelaProvider(
        {"endpoint": "http://ep.test", "model": "centinela-sentiment", "contract": CONTRACT}
    )
    result = await provider.complete([{"role": "user", "content": "la empresa entró en default"}])

    assert isinstance(result, CompletionResponse)
    assert result.provider == "centinela"
    assert result.content == "negativo"
    assert result.model == "centinela-sentiment"
    assert result.metadata["label"] == "negativo"
    assert result.metadata["valid"] is True
    assert result.usage == {"input_tokens": 0, "output_tokens": 0}
    assert result.cost == 0.0


@respx.mock
async def test_provider_stream_yields_single_done_chunk():
    respx.post("http://ep.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_chat_response("neutral"))
    )
    provider = CentinelaProvider({"endpoint": "http://ep.test", "contract": CONTRACT})
    chunks = [c async for c in provider.stream([{"role": "user", "content": "sin cambios"}])]

    assert len(chunks) == 1
    assert chunks[0].content == "neutral"
    assert chunks[0].done is True
    assert chunks[0].provider == "centinela"


def test_provider_capabilities_and_cost():
    provider = CentinelaProvider({"endpoint": "http://ep.test", "contract": CONTRACT})
    assert provider.supports_tools() is False
    assert provider.supports_vision() is False
    assert provider.estimated_cost("centinela", 1000, 1000) == 0.0
```

- [ ] **Step 2: Correr para verlos fallar**

Run: `cd /Users/fulfaro/monaccode/astromesh && uv run pytest tests/test_centinela_provider.py -k "provider" -v`
Expected: FAIL (`ImportError: cannot import name 'CentinelaProvider'`).

- [ ] **Step 3: Implementar el shim**

Agregar al final de `astromesh/providers/centinela.py`:

```python
class CentinelaProvider:
    """Routable ProviderProtocol shim over the Centinela typed capability.

    Structural implementation (does not inherit ProviderProtocol). `estimated_cost`
    reports ~0 so the model_router prefers Centinela under the cost_optimized strategy.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._client = _CentinelaEndpointClient(config)
        self.model = self._client.model

    async def complete(self, messages: list[dict], **kwargs: Any) -> CompletionResponse:
        model = kwargs.pop("model", self.model)
        text = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                text = m.get("content", "") or ""
                break

        start = time.perf_counter()
        result = await self._client.classify(text)
        latency_ms = (time.perf_counter() - start) * 1000

        return CompletionResponse(
            content=result.label or "",
            model=model,
            provider="centinela",
            usage={"input_tokens": 0, "output_tokens": 0},
            latency_ms=latency_ms,
            cost=0.0,
            metadata={
                "label": result.label,
                "valid": result.valid,
                "raw": result.raw,
                "score": result.score,
            },
        )

    async def stream(self, messages: list[dict], **kwargs: Any) -> AsyncIterator[CompletionChunk]:
        resp = await self.complete(messages, **kwargs)
        yield CompletionChunk(
            content=resp.content, model=resp.model, provider="centinela", done=True
        )

    async def health_check(self) -> bool:
        return await self._client.health_check()

    def supports_tools(self) -> bool:
        return False

    def supports_vision(self) -> bool:
        return False

    def estimated_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        return 0.0
```

- [ ] **Step 4: Correr toda la suite del provider para verla pasar**

Run: `cd /Users/fulfaro/monaccode/astromesh && uv run pytest tests/test_centinela_provider.py -v`
Expected: PASS (7 tests en total).

- [ ] **Step 5: Verificar conformidad estructural con el protocolo**

Run: `cd /Users/fulfaro/monaccode/astromesh && uv run python -c "from astromesh.providers.base import ProviderProtocol; from astromesh.providers.centinela import CentinelaProvider; print(isinstance(CentinelaProvider({'endpoint':'http://x','contract':{'labels':[]}}), ProviderProtocol))"`
Expected: imprime `True` (`ProviderProtocol` es `@runtime_checkable`).

- [ ] **Step 6: Lint + commit**

Run: `cd /Users/fulfaro/monaccode/astromesh && uv run ruff check astromesh/providers/centinela.py tests/test_centinela_provider.py`
Expected: sin errores.

```bash
cd /Users/fulfaro/monaccode/astromesh
git add astromesh/providers/centinela.py tests/test_centinela_provider.py
git commit -m "feat: CentinelaProvider routable shim (ProviderProtocol)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 8: [astromesh] Registro del provider en `engine.py` + `factory.py`

**Files:**
- Modify: `astromesh/runtime/engine.py`
- Modify: `astromesh/providers/factory.py`
- Test: `tests/test_centinela_wiring.py`

**Interfaces:**
- Consumes: `CentinelaProvider` (Task 7); `build_candidate_provider(block: dict)` (existente); `create_provider(provider_name, api_key)` (existente).
- Produces: `source: "centinela"` en un bloque de agente construye un `CentinelaProvider`; `create_provider("centinela", ...)` idem.

- [ ] **Step 1: Escribir los tests que fallan**

Crear `tests/test_centinela_wiring.py`:

```python
from astromesh.providers.centinela import CentinelaProvider
from astromesh.providers.factory import create_provider
from astromesh.runtime.engine import build_candidate_provider


def test_build_candidate_provider_centinela():
    provider = build_candidate_provider(
        {
            "source": "centinela",
            "endpoint": "https://ep.example.cloud",
            "model": "centinela-sentiment",
            "contract": {"labels": ["positivo", "neutral", "negativo"]},
        }
    )
    assert isinstance(provider, CentinelaProvider)
    assert provider.model == "centinela-sentiment"
    assert provider._client.endpoint == "https://ep.example.cloud"
    assert provider._client.labels == ["positivo", "neutral", "negativo"]


def test_factory_create_provider_centinela():
    provider = create_provider("centinela", "")
    assert isinstance(provider, CentinelaProvider)
```

- [ ] **Step 2: Correr para verlos fallar**

Run: `cd /Users/fulfaro/monaccode/astromesh && uv run pytest tests/test_centinela_wiring.py -v`
Expected: FAIL (`build_candidate_provider` devuelve `None` para `centinela`; `create_provider` lanza `ValueError`).

- [ ] **Step 3: Agregar la rama en `engine.py`**

En `astromesh/runtime/engine.py`, dentro de `build_candidate_provider`, insertar este bloque **antes** del `return None` final (después de la rama `litellm`):

```python
    if source == "centinela":
        from astromesh.providers.centinela import CentinelaProvider

        base = (block.get("endpoint") or "http://localhost:8080").rstrip("/")
        return CentinelaProvider(
            config={
                "endpoint": base,
                "model": model or "centinela",
                "contract": block.get("contract") or {},
                "invalid_policy": block.get("invalid_policy", "mark"),
                "max_retries": int(block.get("max_retries", 1)),
            }
        )
```

- [ ] **Step 4: Agregar la rama en `factory.py`**

En `astromesh/providers/factory.py`, en `create_provider`, agregar una rama antes del `else`:

```python
    elif provider_name == "centinela":
        from astromesh.providers.centinela import CentinelaProvider

        return CentinelaProvider(config={})
```

- [ ] **Step 5: Correr para verlos pasar**

Run: `cd /Users/fulfaro/monaccode/astromesh && uv run pytest tests/test_centinela_wiring.py -v`
Expected: PASS (2 tests).

- [ ] **Step 6: Regresión — suite de providers + reconciler + contrato**

Run: `cd /Users/fulfaro/monaccode/astromesh && uv run pytest tests/test_centinela_provider.py tests/test_centinela_reconcile.py tests/test_centinela_contract.py tests/test_centinela_wiring.py tests/test_providers.py -v`
Expected: PASS (nada roto en la suite de providers existente).

- [ ] **Step 7: Lint + commit**

Run: `cd /Users/fulfaro/monaccode/astromesh && uv run ruff check astromesh/runtime/engine.py astromesh/providers/factory.py tests/test_centinela_wiring.py`
Expected: sin errores.

```bash
cd /Users/fulfaro/monaccode/astromesh
git add astromesh/runtime/engine.py astromesh/providers/factory.py tests/test_centinela_wiring.py
git commit -m "feat: register Centinela provider in engine + BYOK factory

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- §1 contrato cruzado (schema twin + `additionalProperties`) → Tasks 1, 3. ✓
- §1/§7 lock en el wheel + fuente única → Task 2. ✓
- §2 `bindings.yaml` (CRD, alias explícito) → Task 5 (seed) + Task 4 (consumo). ✓
- §3 reconciler compile-only (reglas duras, determinista, sin cliente HF) → Tasks 4, 5. ✓
- §4 provider doble fachada (tipada + routable, validación compartida, `estimated_cost≈0`, `supports_*`=False) → Tasks 6, 7. ✓
- §5 registro engine.py + factory.py + entrada `providers.yaml` generada → Tasks 8, 5. ✓
- §6 testing (respx, snapshot reconciler, twin) → distribuido en cada task. ✓
- §7 empaquetado (nebula publicable, `datasets` a extra, dep+source en astromesh) → Tasks 2, 3. ✓
- Ítems diferidos: `additionalProperties:false` → Task 1 (resuelto); orden de revisiones → esquivado por alias explícito (Task 4/5), sigue diferido; `leaderboard_score` → fuera de alcance. ✓

**Placeholder scan:** El único literal `REPLACE_WITH_REAL_HF_ENDPOINT` está en el seed `bindings.yaml` (Task 5), intencional — es un valor de operación que se completa al provisionar el endpoint real, no un placeholder del plan. Sin `TBD`/`TODO`/“handle edge cases” en pasos.

**Type consistency:** `SentimentResult(label, valid, raw, score)` idéntico en Tasks 6–7. `reconcile`/`to_provider_config`/`ReconcileError` idénticos entre Task 4 (definición) y Task 5 (consumo). `CentinelaProvider(config)` con `_client.endpoint`/`_client.labels`/`.model` consistente entre Tasks 6, 7, 8. `constrain_label(raw, labels)` usado con la firma real de nebula. `usage` con `input_tokens`/`output_tokens` en todos lados.

**Nota de orden cruzado:** Task 3 (astromesh) depende de que Tasks 1–2 (nebula) estén commiteadas y de que `uv sync` en astromesh resuelva `astromesh-nebula` editable. Ejecutar en orden 1→2→3→4→5→6→7→8.
