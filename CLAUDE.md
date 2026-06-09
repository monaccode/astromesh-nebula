# CLAUDE.md — Astromesh Nebula

Open-model foundry. Pure logic lives in `nebula/` (unit-tested); thin CLIs in `scripts/01..08`.
GPU work (train/merge/quantize) runs on HF Jobs. Python 3.12+/3.11+, `uv`, `ruff` (line 100),
`pytest`. Conventional commits. First family: Centinela (Spanish financial sentiment).

Build/test: `uv sync --extra dev` · `uv run pytest -v` · `uv run ruff check nebula/ tests/ scripts/`
