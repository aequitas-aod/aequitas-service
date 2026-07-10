include .env

CHECK_DIRS := .

# ─── Environment ────────────────────────────────────────────────────────────

install:
	poetry install

install-dev:
	poetry install --with dev

# ─── Dependency management ───────────────────────────────────────────────────
# Usage: make add pkg=<package>            (e.g. make add pkg=httpx)
# Usage: make add pkg="httpx>=0.27"
add:
	poetry add $(pkg)

# Usage: make add-dev pkg=<package>        (e.g. make add-dev pkg=pytest)
add-dev:
	poetry add --group dev $(pkg)

remove:
	poetry remove $(pkg)

remove-dev:
	poetry remove --group dev $(pkg)

update:
	poetry update

update-pkg:
	poetry update $(pkg)

show:
	poetry show --tree

# ─── Code quality ────────────────────────────────────────────────────────────

format:
	poetry run black $(CHECK_DIRS)

typecheck:
	poetry run mypy temlops/

test:
	poetry run pytest

test-cov:
	poetry run pytest --cov=library --cov-report=term-missing

# ─── Build & publish ─────────────────────────────────────────────────────────

build:
	poetry build

publish:
	poetry publish

publish-test:
	poetry publish --repository testpypi

# ─── Runtime ─────────────────────────────────────────────────────────────────

run_api:
	poetry run python -m library.api.api --port 8000 --host 0.0.0.0

.PHONY: install install-dev add add-dev remove remove-dev update update-pkg show \
        format typecheck test test-cov build publish publish-test run_api
