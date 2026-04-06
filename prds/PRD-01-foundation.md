# PRD-01: Foundation

> Status: TODO
> Module: 33_TDE

## Objective
Set up the project skeleton: venv, pyproject.toml, directory structure, TOML
configs, linting, and basic test infrastructure.

## Deliverables
- [x] pyproject.toml with hatchling backend, torch cu128
- [x] configs/paper.toml -- paper-faithful hyperparameters
- [x] configs/debug.toml -- quick smoke test config
- [x] src/anima_tde/__init__.py
- [x] tests/ directory with conftest.py
- [x] anima_module.yaml
- [ ] .venv created and synced
- [ ] ruff passes on all files

## Acceptance Criteria
- `uv sync` succeeds
- `uv run ruff check src/ tests/` passes
- `uv run pytest tests/ -x` collects tests (even if 0 pass)
- Directory structure matches ANIMA conventions
