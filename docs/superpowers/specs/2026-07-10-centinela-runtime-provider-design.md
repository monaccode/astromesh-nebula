# Centinela Runtime — Reconciler + Provider (Sub-proyecto 2)

**Fecha:** 2026-07-10
**Estado:** Diseño aprobado (brainstorming)
**Repo primario:** `astromesh` (runtime). Depende de `astromesh-nebula` (foundry, publicado).
**Spec padre:** `2026-07-10-centinela-mlops-design.md` (arquitectura end-to-end). Este documento es el
spec enfocado del **sub-proyecto 2** de esa decomposición (§8 del padre).

## Objetivo

Hacer que los modelos Centinela publicados por la foundry sean **consumibles desde los agentes de
Astromesh**: un provider dedicado con dos fachadas (capability tipada para el ADK + shim routable para
el `model_router`) y un **reconciler compile-only** que traduce los despliegues declarados
(`bindings.yaml`) más el contrato de la foundry (`catalog.lock.json`) en configuración de provider
(`providers.yaml`). Meta de negocio: que el router prefiera Centinela en las tareas que cubre y así
**abarate el consumo de tokens** frente a los LLMs caros.

## Decisiones de diseño (fijadas en brainstorming, 2026-07-10)

| Eje | Decisión |
|-----|----------|
| Alcance del reconciler | **Solo compilar config.** Asume que el HF Inference Endpoint ya existe; la provisión viva de GPUs queda para un sub-proyecto de infra aparte. Todo testeable sin tocar infra ni gastar GPU-hora. |
| Fachada tipada | **ADK `@tool` con Pydantic.** Reusa el mecanismo tipado existente (`astromesh-adk/.../tools.py`); no se agrega una abstracción nueva al ADK. |
| Dependencia `nebula` en astromesh | **Publicar `astromesh-nebula` a índice privado** (fallback `git+https@tag`). Contrato compartido, no duplicado; reproducible en Docker/CI; pinneado por versión. |
| Registro del shim routable | **`engine.py` + `factory.py`.** Cablear `centinela` en `build_candidate_provider()` (camino vivo de agentes) y en `create_provider()` (override BYOK). El reconciler escribe la entrada `type: centinela` en `providers.yaml`. |
| Ubicación del reconciler | **Comando de `astromesh-node`** (`astromesh-node centinela reconcile`), con la lógica pura en un módulo aparte. |
| Fuente del lock en runtime | **Dentro del wheel de `nebula`.** El build de nebula compila e incluye `catalog.lock.json`; astromesh lo lee con `importlib.resources`. Fuente única versionada. |

## Realidad del código (verificada, corrige supuestos del spec padre)

Exploración del repo `astromesh` (paths abajo). Puntos que **cambian** lo que el spec padre asumía:

- **`ProviderProtocol` es estructural** (`typing.Protocol`, `@runtime_checkable`) en
  `astromesh/providers/base.py`. Los providers **no heredan**: implementan los 6 métodos
  (`complete`, `stream`, `health_check`, `supports_tools`, `supports_vision`, `estimated_cost`).
  Convención de `usage`: claves `input_tokens`/`output_tokens`. Constructor `__init__(self, config: dict | None = None)`.
- **No hay un único punto de registro.** Hay **3 sitios disjuntos**: (a) `build_candidate_provider()`
  en `astromesh/runtime/engine.py` — el **camino vivo** que resuelve el provider desde el `spec.model`
  de cada agente; (b) `PROVIDER_REGISTRY` en `astromesh-adk/.../providers.py` (ADK); (c)
  `create_provider()` en `astromesh/providers/factory.py` (override BYOK). El ladder de `engine.py`
  **hoy ni siquiera incluye** `hf_tgi`/`vllm`/`onnx`. El spec padre decía solo "`factory.py` +
  `providers.yaml`"; eso deja el shim a medio cablear. Por eso este sub-proyecto toca **engine.py + factory.py**.
- **El ADK no tiene una clase base de "capability con salida estructurada"** separada del chat. El
  patrón tipado real es un `@tool` cuyo parámetro/retorno es un modelo Pydantic
  (`_generate_schema`, duck-typing de `model_json_schema`/`model_validate`).
- **`astromesh` no es un uv workspace.** Los sub-paquetes se pegan con `[tool.uv.sources]` editable
  path deps (patrón `astromesh-adk = { path = "astromesh-adk", editable = true }`). Un path editable
  a `../astromesh-nebula` **no es reproducible en Docker/CI** → de ahí la decisión de publicar.
- **`nebula.validation.constrain_label(raw, labels)`** ya existe y toma una **lista** de labels; el
  `contract` del catálogo es un `dict` con `labels: [...]` y `validation: constrain_label`. El provider
  extrae `contract["labels"]` y se lo pasa a `constrain_label`.
- **No existe `reconciler` ni `bindings`** en el repo: es un concepto genuinamente nuevo. Patrones de
  registro/loop existentes para inspirarse: `ToolLoader`, `ToolRegistry`, `ModelRegistry`, `mesh/manager.py`.
- Tests de provider: `respx` + `httpx.Response`, `asyncio_mode = "auto"` (sin `@pytest.mark.asyncio`).

Paths clave: `astromesh/providers/base.py`, `.../hf_tgi_provider.py` (template HTTP),
`.../onnx_provider.py` (template in-process/fallback stream), `astromesh/runtime/engine.py`
(`build_candidate_provider` ~L92-148, registro ~L181-215), `astromesh/providers/factory.py`,
`astromesh/core/model_router.py`, `config/providers.yaml`, `astromesh-node/.../cli/commands/`,
`tests/test_providers.py`, `pyproject.toml` + `astromesh-adk/pyproject.toml`.

## Principios rectores (heredados del padre)

- **Separación foundry/runtime.** El único acoplamiento es el contrato (`catalog.lock.json` + schema +
  `validation`), que viaja publicado desde nebula.
- **Contrato compartido, no duplicado.** El provider llama al *mismo* `nebula.validation.constrain_label`
  que el eval-gate de la foundry.
- **Compile-only.** El reconciler no toca infra viva; la provisión de endpoints es otro sub-proyecto.
- **YAGNI.** Los Centinela **instruct** reusan `hf_tgi` (ya OpenAI-compatible). El provider Centinela se
  construye solo para `classifier`/`extractor`.

---

## 1. Contrato cruzado (el guardián)

**nebula publicable.** El build del wheel de `astromesh-nebula` compila e incluye `catalog.lock.json`
como package-data, junto al `catalog.schema.json` que ya va `force-include`. Se publica a índice
privado (fallback `git+https@<tag>`). Así el lock y el schema son **un único artefacto versionado**.

**astromesh consume por `importlib.resources`.** Lee `nebula/catalog.lock.json` y
`nebula/catalog.schema.json` del paquete instalado — sin copiar archivos entre repos.

**Twin del schema (crítico).** Un test en `astromesh` valida el *mismo* fixture de `catalog.lock.json`
contra el `catalog.schema.json` shippeado por nebula. Un cambio incompatible del contrato rompe CI en
**ambos** repos antes de runtime.

**Ítem diferido que aterriza acá:** agregar `additionalProperties: false` (en ambos niveles: modelo y
revisión) al `nebula/catalog.schema.json`, ahora que el twin lo ejercita en dos lados. Se hace en el
repo nebula como parte de este sub-proyecto.

## 2. `bindings.yaml` (registro de despliegues, en astromesh)

Estilo CRD para uniformar con `providers.yaml`. **Thin:** referencia modelo + alias; la revisión, el
`kind` y el `contract` se resuelven del lock, no se repiten.

```yaml
apiVersion: astromesh/v1
kind: CentinelaBindings
metadata: { name: default }
spec:
  bindings:
    - model: centinela-sentiment
      alias: prod
      endpoint: "https://xxxx.endpoints.huggingface.cloud"
```

Al referenciar **alias explícito** (no "latest"), este sub-proyecto **esquiva el ítem diferido de orden
de revisiones** (`v0.10 < v0.2`): no se toca hasta que aparezca un selector "latest". Límite consciente
documentado.

Ubicación propuesta: `config/centinela/bindings.yaml` (junto a `config/providers.yaml`).

## 3. Reconciler (lógica pura + CLI, compile-only)

Espejo de la convención de nebula: lógica pura testeable + CLI fino.

- **Lógica pura:** `astromesh/centinela/reconcile.py`.
- **CLI:** comando de `astromesh-node` → `astromesh-node centinela reconcile`.
- **Entrada:** `catalog.lock.json` (leído del paquete `nebula` vía `importlib.resources`) + `bindings.yaml`.
- **Salida:** entradas `type: centinela` para `config/providers.yaml` (endpoint, `models`, `contract`
  embebido del lock). Salida **determinista** (mismo orden/serialización) → snapshot-testable.
- **Reglas duras** (falla ruidoso si no se cumplen):
  1. Cada `(model, alias)` del binding resuelve a una revisión existente en el lock.
  2. Esa revisión tiene `gate: passed`.
  3. `kind ∈ {classifier, extractor}` (un binding cuyo modelo es `instruct` se rechaza con mensaje que
     apunta a `hf_tgi`; fuera de alcance de este provider).
  4. El `contract` del modelo (del lock) se adjunta a la entrada del provider.
- **Compile-only:** el cliente HF **nunca** se instancia ni se llama. No hay red en el reconciler.

## 4. Provider `astromesh/providers/centinela.py` (core + 2 fachadas)

Un core `_CentinelaEndpointClient` (httpx; conoce endpoint URL + `kind` + `contract`) con dos fachadas:

**Fachada tipada (ADK `@tool`).** Modelos Pydantic — p.ej. `SentimentResult(label: str, score: float,
valid: bool)`. Aplica `nebula.validation.constrain_label(raw, contract["labels"])`. Política de salida
inválida **configurable**: `retry N` o `valid=False`. Se expone al ADK como herramienta tipada; el
agente la llama directamente (no pasa por el router).

**Fachada routable (`CentinelaProvider`).** Implementa los 6 métodos de `ProviderProtocol`
(estructural, no hereda):

- `complete(messages, **kwargs) -> CompletionResponse`: toma el último mensaje `user` → fachada tipada
  → pone el `label`/JSON en `content` y la estructura completa (`label`, `score`, `valid`) en
  `metadata`. `usage` con `input_tokens`/`output_tokens`. `provider="centinela"`.
- `stream(...)`: envuelve `complete` en un único chunk `done=True` (fallback estándar para backends no
  streaming, como `onnx_provider`).
- `health_check() -> bool`: GET al `/health` del endpoint.
- `supports_tools() -> bool`: **False** (el clasificador *es* la herramienta).
- `supports_vision() -> bool`: **False**.
- `estimated_cost(model, input_tokens, output_tokens) -> float`: **≈0** (endpoint amortizado, costo
  marginal ~nulo) para que la estrategia `cost_optimized` del router lo prefiera — palanca de ahorro.

## 5. Cableado de registro

- **`build_candidate_provider()` en `astromesh/runtime/engine.py`:** nueva rama `centinela` → construye
  `CentinelaProvider(config)` desde el `spec.model` del agente (`endpoint`, `model`, `contract`). Esto
  lo hace routable desde el YAML del agente (`source: centinela`). Es el camino **vivo**.
- **`create_provider()` en `astromesh/providers/factory.py`:** nueva rama `centinela` (override BYOK).
- **`providers.yaml`:** el reconciler escribe la entrada `type: centinela`.
- **ADK `PROVIDER_REGISTRY`: fuera de alcance** (decisión de brainstorming: solo engine.py + factory.py).

## 6. Testing (convención astromesh)

`respx` + `httpx.Response`, `asyncio_mode = "auto"`.

- **Provider:** fachada tipada (validación contra `contract["labels"]`, política de inválido
  `retry`/`valid=False`); shim routable (mapeo input→typed→`content`+`metadata`); `estimated_cost≈0`;
  `health_check` healthy/unhealthy; `supports_tools/vision` = False; construcción vía `build_candidate_provider`
  y `create_provider`.
- **Reconciler:** snapshot — fixtures `catalog.lock.json` + `bindings.yaml` → `providers.yaml` esperado;
  reglas duras (alias inexistente, `gate != passed`, `kind` instruct) fallan con mensaje claro; cliente
  HF **no** instanciado.
- **Twin del schema (cross-repo):** mismo fixture de lock validado contra el `catalog.schema.json`
  shippeado por nebula; gemelo del test que ya vive en nebula.

## 7. Empaquetado

**nebula (repo `astromesh-nebula`):**
- Versión ya existe (`0.1.0`).
- Agregar al build la **compilación + inclusión de `catalog.lock.json`** en el wheel (package-data,
  como el schema).
- Publicar wheel a índice privado (o `git+https@<tag>`).
- Agregar `additionalProperties: false` al schema (§1).

**astromesh (repo `astromesh`):**
- Agregar dep `astromesh-nebula>=0.1.0` (+ `[tool.uv.sources]` para dev/CI: editable path a
  `../astromesh-nebula`; en Docker/deploy: wheel publicado).
- Si `nebula` necesitara backend pesado opcional, va como extra (no dep dura).

## 8. Orden de implementación sugerido

1. **nebula:** incluir lock en el wheel + `additionalProperties: false` en schema (habilita el consumo).
2. **astromesh:** dep + source de `nebula`; test twin del schema (guardián primero).
3. **Reconciler** (`reconcile.py` + comando `astromesh-node`) con snapshot tests.
4. **Provider** (`centinela.py`, 2 fachadas) con tests `respx`.
5. **Cableado** en `engine.py` + `factory.py` con tests de construcción.
6. **`bindings.yaml`** seed + `providers.yaml` generado de ejemplo.

## Ítems diferidos — estado tras este sub-proyecto

- `additionalProperties: false` en el schema → **resuelto acá** (§1/§7).
- Orden de revisiones (`v0.10 < v0.2`) → **sigue diferido**, legítimamente esquivado por usar alias
  explícito en los bindings (§2). Debe resolverse **antes** de introducir cualquier selector "latest".
- `leaderboard_score` del scout → **sigue diferido** (es de `scripts/00_scout.py`, ortogonal a este
  sub-proyecto).

## Fuera de alcance (por ahora)

- **Provisión viva de HF Inference Endpoints** (crear/actualizar/escalar GPUs): sub-proyecto de infra
  aparte. El reconciler es compile-only.
- **Registro en el ADK `PROVIDER_REGISTRY`** (strings `centinela/model`): diferido; hoy engine.py + factory.py.
- **Doc generator** (`gen-models.mjs`, sección "Models"): es el sub-proyecto 3.
- **Bots de PR** que automatizan los pasos GitOps: sub-proyecto 4.
- **Selector "latest"** de revisiones: requiere resolver el orden diferido antes.
