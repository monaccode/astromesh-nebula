# Contributing to Astromesh Nebula

1. Fork and branch from `develop`.
2. `uv sync --extra dev`, then `uv run pytest -v` and `uv run ruff check`.
3. Conventional commits (`feat:`, `fix:`, `chore:`, `docs:`, `test:`).
4. A model change must not regress the eval tier threshold (`configs/eval.yaml`).
