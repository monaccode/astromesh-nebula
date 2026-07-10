# Centinela MLOps — Foundry → Hosting → Consumo

**Fecha:** 2026-07-10
**Estado:** Diseño aprobado (brainstorming)
**Repos afectados:** `astromesh-nebula` (foundry), `astromesh` (runtime)

## Objetivo

Definir el proceso MLOps end-to-end para construir la familia de modelos **Centinela**,
publicarlos en Hugging Face, hostearlos para consumo interno y consumirlos desde los agentes
de Astromesh vía un provider dedicado.

Meta de negocio: **abaratar el consumo de tokens** para las PYMES del ecosistema Fainansu y
para el propio ecosistema Astromesh, ofreciendo modelos chicos especializados que descargan
trabajo de los LLMs caros (GPT/Claude) en tareas acotadas.

## Decisiones de diseño (fijadas en brainstorming)

| Eje | Decisión |
|-----|----------|
| Naturaleza de los modelos | **Familia mixta**: clasificadores/extractores (salida estructurada validada) + instruct de chat |
| Hosting | **HF Inference Endpoints** (gestionado, autoescala, pago por hora de GPU) |
| Gobernanza de promoción | **GitOps con registry y aliases** (staging/prod), dos loops |
| Frontera foundry/runtime | **Enfoque C**: catálogo en Nebula, binding de despliegue en Astromesh |
| Interfaz de clasificadores | **Ambas fachadas**: capability tipada (ADK) + shim routable (model_router) |
| Selección de base | **Dataset-first + Base-Model Scout** (no hay modelo sin dataset alineado) |

## Principios rectores

- **Separación foundry/runtime.** Nebula es *where models are born* (calidad, linaje, artefacto).
  Astromesh es el runtime (cómo/cuándo servir, SLA, provider). El único acoplamiento es el catálogo.
- **Dataset-first.** Ningún modelo entra al pipeline sin un dataset registrado y alineado a su uso.
- **Contrato compartido, no duplicado.** La validación determinista (`nebula.validation`) es el mismo
  código en el eval-gate (foundry) y en el provider (runtime).
- **Todo auditable en git.** Qué revisión sirve prod, qué gate pasó, con qué base y qué hiperparámetros:
  todo vive en archivos versionados.
- **YAGNI.** Los instruct reusan `hf_tgi` (ya OpenAI-compatible). El provider Centinela se construye solo
  para los kinds que el molde chat no cubre.

---

## 1. Arquitectura end-to-end

Dos loops GitOps unidos por un único artefacto de contrato: `catalog.lock.json`.

```
┌─────────────────────── NEBULA (foundry) ───────────────────────┐
│  dataset → scout → train(HF Jobs) → merge → quantize → EVAL-GATE│
│      │                                                          │
│      └─ push_to_hub: artifacts + card + tag  →  HF Hub repo     │
│      └─ actualiza  catalog/centinela.yaml  (alias → revisión)   │
└───────────────────────────────┬────────────────────────────────┘
                                 │  publica catalog.lock.json (release asset / tag)
                                 ▼
┌─────────────────────── ASTROMESH (runtime) ────────────────────┐
│  registry/bindings.yaml  (alias → endpoint + provider + SLA)    │
│      ├─ CI reconciler: provisiona/actualiza HF Inference Endpoint│
│      ├─ genera  config/providers.yaml  (instruct → hf_tgi)      │
│      ├─ Centinela provider (typed facade + routable shim)       │
│      └─ docs-site: fichas de modelo generadas del catálogo      │
│                          │                                       │
│                          ▼   agents (ADK) consumen              │
└─────────────────────────────────────────────────────────────────┘
```

### El contrato: `catalog.lock.json`

Emitido por nebula (compilado de `catalog/centinela.yaml`), consumido por astromesh. Es a
`catalog/*.yaml` lo que `uv.lock` es a `pyproject.toml`: la fuente editable es humana, el `.lock`
es la verdad de máquina.

```jsonc
{
  "schema_version": "1",
  "models": [{
    "name": "centinela-sentiment",
    "kind": "classifier",                 // classifier | extractor | instruct
    "task": "text-classification",
    "vertical": "finanzas",
    "hf_repo": "astromesh/Centinela-Qwen3-4B",
    "aliases": { "prod": "v0.1", "staging": "v0.2-rc1" },
    "revisions": {
      "v0.1": {
        "sha": "abc123",
        "eval": { "macro_f1": 0.87, "invalid_rate": 0.01 },
        "gate": "passed",
        "formats": ["safetensors", "gguf"],
        "dataset": "centinela-sentiment-es@3",
        "train_config_hash": "sha256:…",   // base_model + hiperparámetros + dataset_hash
        "base_model": "Qwen/Qwen3-4B",
        "scout_rationale": "apache-2.0, 4B ≤ budget, es-fuerte, mejor macro_f1 del shortlist"
      }
    },
    "contract": { "labels": ["positivo","neutral","negativo"], "validation": "constrain_label" }
  }]
}
```

Reglas del compilador (verificadas por tests, ver §7):
1. Un alias no puede apuntar a una revisión con `gate ≠ passed`.
2. El `dataset` de una revisión debe existir en `catalog/datasets.yaml` y su `contract` debe
   coincidir con el `contract` del modelo.
3. La compilación yaml→lock es determinista.

---

## 2. Componentes

**N** = vive en nebula · **A** = vive en astromesh.

| # | Componente | Repo | Responsabilidad | Depende de |
|---|-----------|------|-----------------|-----------|
| 1 | `catalog/centinela.yaml` + `catalog/datasets.yaml` (fuente) | N | Verdad editable de modelos y datasets | — |
| 2 | `catalog.lock.json` (compilado) | N | Contrato de máquina entre repos | 1, eval-gate |
| 3 | `nebula catalog` CLI (`scripts/09_catalog.py`) | N | Compila yaml→lock, valida reglas, publica el lock | huggingface_hub |
| 4 | Base-Model Scout (`scripts/00_scout.py`) | N | Recomienda bases de HF Hub por task+dataset+restricciones | huggingface_hub |
| 5 | `registry/bindings.yaml` | A | Liga aliases del catálogo a endpoint + provider + SLA | 2 |
| 6 | Endpoint reconciler (CI job) | A | Provisiona/actualiza HF Endpoints, genera `providers.yaml` | 5, huggingface_hub |
| 7 | **Centinela provider** (`astromesh/providers/centinela.py`) | A | Fachada dual sobre el endpoint; aplica `contract` | base.py, nebula.validation |
| 8 | Validación compartida (`nebula.validation`, `nebula.labels`) | N→A | `constrain_label` idéntico en eval y en provider | — |
| 9 | Doc generator (`docs-site/scripts/gen-models.mjs`) | A | `catalog.lock.json` → páginas `models/*.md` + sidebar | 2 |

Decisiones destacadas:

- **(8) Validación como dependencia compartida.** `nebula` ya es un paquete `hatchling`. Astromesh lo
  importa como dependencia y usa `nebula.validation`/`nebula.labels`. El contrato del catálogo se
  garantiza con el mismo código que corrió el eval-gate.
- **(9) Fichas de modelo generadas, no escritas.** El docs-site (Astro Starlight) genera una página por
  modelo desde `catalog.lock.json` bajo una nueva sección **"Models"** del sidebar (nombre, kind, task,
  alias prod/staging, eval, labels del contrato, snippet de uso). Corre en el CI de docs → la web nunca
  se desincroniza de prod.

---

## 3. Flujo de promoción MLOps (dos loops, dos gates)

### Loop 1 — Foundry: "¿esta revisión es buena?"

```
1. Trigger retrain (workflow_dispatch / cron / nuevo dataset)
2. HF Job: build → train → merge → quantize → EVAL-GATE
     (macro_f1 ≥ umbral, invalid_rate ≤ umbral, no regresión vs prod)
     ✗ falla → nada se publica.  ✓ pasa ↓
3. push_to_hub: artefactos + card + tag → HF repo revision (sha)
4. PR automático "catalog: <modelo> <rc>":
     - añade la revisión a catalog/centinela.yaml con eval + gate:passed
     - mueve alias staging → <rc>   (prod NO se toca)
5. Merge → CI compila catalog.lock.json → publica como release asset/tag
```

**Gate de calidad** (eval-gate): única puerta hacia `staging`, vive en foundry, automático.

### Loop 2 — Runtime: "¿cómo y cuándo la sirvo?"

```
6. Bot detecta nuevo catalog.lock.json → PR "bindings: bump <modelo> staging → <rc>"
7. Merge → Endpoint reconciler (CI):
     - crea/actualiza el HF Inference Endpoint de STAGING a la revisión sha
     - regenera config/providers.yaml
     - health-check + smoke test del provider contra staging
8. Validación en staging (agentes de prueba / eval online / canary)
9. Promoción a PROD = PR humano en nebula que mueve alias prod → <version>
     → recompila lock → astromesh PR bumpea binding prod → reconciler actualiza endpoint prod
10. Doc generator corre en CI de docs → fichas web reflejan prod=<version>
```

**Gate de release** (promoción a prod): **aprobación humana** vía PR en nebula. Nadie sirve prod
sin que un humano mueva el alias.

### Propiedades

- **Dos gates, dos dueños.** Calidad (eval, automático, foundry) → staging. Release (humano, foundry) → prod.
- **Rollback = revertir un commit.** Prod apunta a un alias; volver atrás es un PR que reapunta el alias.
- **Trazabilidad total.** `catalog.lock.json` (qué sha/eval en prod) + `bindings.yaml` (qué endpoint/SLA),
  ambos en git con historia.
- **Arranque manual.** Los pasos 6 y 9 empiezan como PRs manuales; el bot se automatiza cuando el flujo
  esté rodado (reduce piezas móviles al principio).

---

## 4. Anatomía de un entrenamiento Centinela

### Parámetros que toma el pipeline (de `configs/train-<name>.yaml`)

| Grupo | Parámetro | Ejemplo (4B) | Controla |
|-------|-----------|--------------|----------|
| Base | `base_model` | `Qwen/Qwen3-4B` | Modelo del que parte QLoRA |
| | `max_seq_length` | `1024` | Largo máx de secuencia (VRAM + cobertura de inputs) |
| Datos | `train_path` / `dataset` | `train.jsonl` | Dataset conversacional (columna `messages`) |
| LoRA | `r` / `alpha` | `16 / 16` | Capacidad del adapter |
| | `dropout` | `0.0` | Regularización |
| | `target_modules` | q,k,v,o,gate,up,down | Matrices con adapter |
| Optim | `epochs` | `2` | Pasadas sobre el dataset |
| | `learning_rate` | `2e-4` | Típico QLoRA |
| | `per_device_batch_size` × `gradient_accumulation_steps` | `2 × 8 = 16` | Batch efectivo |
| | `gradient_checkpointing` | `true` | Cambia cómputo por VRAM |
| | `seed` | `42` | Reproducibilidad |

Decisiones automáticas del código (no del yaml): precisión (`bf16` si la GPU lo soporta, si no `fp16`)
y `load_in_4bit=True` (QLoRA: base en 4-bit, adapter en fp16/bf16).

### Local → HF Jobs: qué cambia

Los hiperparámetros **no cambian**. Cambia el entorno de ejecución y aparece el eje **flavor** (GPU).

```bash
hf jobs uv run --flavor a10g-large scripts/02_train.py -- \
    --config configs/train-4b.yaml --train data/centinela/train.jsonl --out export/adapter
```

| Eje | Local | HF Jobs |
|-----|-------|---------|
| GPU | RTX 2060, 12GB, sm_75 | `--flavor` (a10g-large 24GB, a100-large 80GB…) |
| Precisión | fp16 (Turing sin bf16) | bf16 (Ampere+) |
| Batch | achicado por 12GB | subir `per_device_batch_size` con más VRAM |
| Deps | venv local | PEP 723 inline (`# /// script`), resueltas por `uv` en el job |
| Auth | login local | `HF_TOKEN` inyectado al job |
| Costo | electricidad | **por minuto de GPU** (plan PRO/Team); flavor = palanca costo↔velocidad |

El script ya es hardware-adaptivo (detecta bf16, usa gradient checkpointing), así que HF Jobs no
requiere cambios de lógica — solo elegir flavor y, opcionalmente, subir el batch.

### Condiciones para un modelo propio nuevo

Cada decisión es un campo de `configs/train-<name>.yaml` / `eval-<name>.yaml`:

1. **Kind / task** — clasificador, extractor o instruct → formato del dataset y `contract`.
2. **Base model** — lo elige el Scout (§5) bajo restricciones; determina VRAM → flavor.
3. **Dataset + contrato** — `messages`/labels + reglas de validación (`constrain_label`).
4. **max_seq_length** — según largo real de los inputs de la tarea.
5. **LoRA r/alpha/target_modules** — capacidad según riqueza de la tarea.
6. **epochs / lr / batch efectivo** — presupuesto de optimización.
7. **Flavor** — GPU/costo del job, derivado de (2) y (4).
8. **Umbrales de eval** (`eval-<name>.yaml`) — `macro_f1`, `must_beat_base`, `invalid_rate` → puerta a staging.

**Reproducibilidad:** el `catalog.lock.json` registra `train_config_hash` (base_model + hiperparámetros +
dataset_hash) por revisión, para que "qué condiciones produjeron esta revisión" sea auditable.

---

## 5. Dataset-first & Base-Model Scout

**Regla dura:** ningún modelo entra al pipeline sin un dataset registrado y alineado a su uso.

- **`catalog/datasets.yaml`** — cada dataset declara: `vertical`, `task/kind`, `contract`
  (labels cerrados o schema de extracción), `language`, tamaño, `hash`. El `train-<name>.yaml`
  referencia un dataset por id. El compilador del catálogo rechaza un modelo cuyo dataset no exista
  o cuyo contrato no coincida.

- **Base-Model Scout (`scripts/00_scout.py`)** — dado *(vertical + task/kind + perfil del dataset +
  restricciones)*, consulta la HF Hub API y devuelve una **shortlist rankeada** con justificación.
  El humano elige; la elección + el porqué se guardan en el linaje (`base_model` + `scout_rationale`).

  Restricciones duras del scout:
  - **Licencia comercialmente usable** (apache-2.0 / mit) — obligatorio (releases Apache-2.0).
  - **Presupuesto de tamaño** (≤ N params) — objetivo de costo/latencia PYME.
  - **Idioma** (`es` / multilingüe con español fuerte).
  - **Task fit** (`pipeline_tag`, score de leaderboard si existe).
  - Ranking compuesto: task fit → score → descargas/recencia.

  El scout es asesor, no oráculo: propone, el humano confirma.

### Roadmap de verticales (apéndice)

- **Fase 1 (ahora):** tareas administrativas repetitivas, gestión de finanzas, gestión contable.
- **Fase 2:** atención al cliente, ventas, gestión de pedidos.

Cada vertical → uno o más modelos Centinela, cada uno con su dataset alineado.

---

## 6. Diseño del provider Centinela

**YAGNI:** los Centinela **instruct** reusan `hf_tgi_provider` (ya OpenAI-compatible). El provider
Centinela se construye solo para los kinds `classifier`/`extractor`.

Arquitectura (`astromesh/providers/centinela.py`) — un core con dos fachadas:

```
              _CentinelaEndpointClient (httpx)
              URL del HF Endpoint + kind + contract (del binding)
                     │
        ┌────────────┴─────────────┐
        ▼                          ▼
 FACHADA TIPADA (capability)   FACHADA ROUTABLE (shim chat)
 classify(text)->Result        CentinelaProvider(ProviderProtocol)
 extract(text,schema)->dict    complete(messages)->CompletionResponse
 • aplica nebula.validation    • último msg user → typed → label/JSON en content
 • label + score + valid       • estructura en metadata
 → expuesta al ADK como tool   → uniforme con el model_router
```

Decisiones:

1. **Validación = mismo código en ambos lados.** Ambas fachadas llaman a
   `nebula.validation.constrain_label` contra el `contract`. Salida fuera del label set → política
   configurable (`retry N` o `valid=False`).
2. **`estimated_cost` como palanca de ahorro.** El provider reporta costo marginal ~0 (endpoint
   amortizado) vs. costo por token de GPT/Claude. Con `cost_optimized` el `model_router` prefiere
   Centinela para las tareas que cubre — el objetivo de abaratar tokens.
3. **Registro declarativo.** Nuevo `type: centinela` en `factory.py` + `providers.yaml`, generado por el
   reconciler desde `bindings.yaml`. El ADK obtiene la capability tipada; el router obtiene el shim.
4. **`supports_tools()` = False** para classifier/extractor (ellos son la herramienta). Los instruct
   soportan tools vía `hf_tgi`.

---

## 7. Testing y verificación

**Nebula (foundry):**
- **Compilador de catálogo** — tests de las reglas duras: alias→`gate:passed`, dataset existe y contrato
  coincide, compilación determinista.
- **Scout** — dado un fixture de la HF Hub API, filtrado y ranking deterministas. Sin red en tests.
- **Validación** — ya cubierta (`constrain_label`); ancla del contrato.
- **Eval-gate** — test de calidad del modelo (macro_f1, must_beat_base, invalid_rate); corre en el HF Job.

**Astromesh (runtime):**
- **Provider Centinela** — fachada tipada (validación, política de invalid), shim routable (mapeo
  input→typed→content, metadata), `estimated_cost`, `health_check`, registro en `factory.py`.
- **Reconciler** — dado `catalog.lock.json` + `bindings.yaml` fixtures, genera el `providers.yaml`
  esperado (snapshot test, cliente HF mockeado).
- **Doc generator** — snapshot: fixture lock → páginas `.md` + sidebar esperados.

**Contrato cruzado (crítico):**
- **JSON Schema de `catalog.lock.json` versionado**, con un test en *cada* repo que valida el mismo
  fixture. Un cambio incompatible rompe CI en ambos lados antes de runtime.

**Online:**
- **Smoke test de staging** (Loop 2, paso 7) ejerce el endpoint real con el provider antes de habilitar
  validación online.

---

## 8. Alcance y decomposición de implementación

El diseño abarca dos repos; conviene implementarlo en sub-proyectos secuenciales, cada uno con su
propio ciclo spec → plan → implementación:

1. **Foundry — catálogo + scout** (nebula): `catalog/*.yaml`, `scripts/09_catalog.py`,
   `scripts/00_scout.py`, JSON Schema del lock, tests. Habilita el contrato.
2. **Runtime — reconciler + provider** (astromesh): `bindings.yaml`, reconciler CI,
   `providers/centinela.py`, registro en factory, `nebula` como dependencia, tests.
3. **Docs — generador de fichas** (astromesh): `gen-models.mjs`, sección "Models" en el sidebar.
4. **Automación — bots de PR** (ambos): automatizar pasos 6 y 9 una vez rodado el flujo manual.

Orden recomendado: 1 → 2 → 3 → 4. El sub-proyecto 1 es prerequisito de todo (define el contrato).

## Fuera de alcance (por ahora)

- Self-hosting en infra propia (Ollama/vLLM/k8s): descartado a favor de HF Endpoints. Reevaluable si el
  costo por hora supera el break-even vs. GPU propia.
- Promoción automática a prod: prod siempre requiere aprobación humana en esta fase.
- A/B testing / canary automático de modelos en prod: futuro, sobre la base de staging + bindings.
