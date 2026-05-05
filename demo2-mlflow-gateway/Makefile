VENV := $(shell pwd)/.venv

# ── Local dev ──────────────────────────────────────────────────────────────────

install:
	python3 -m venv $(VENV)
	VIRTUAL_ENV=$(VENV) poetry install --with dev

shell:
	VIRTUAL_ENV=$(VENV) poetry shell

# ── Docker ────────────────────────────────────────────────────────────────────

up:
	docker compose up -d
	@echo ""
	@echo "Services starting... (~2 min for model pulls)"
	@echo "  Gateway UI    →  http://localhost:5050"
	@echo "  MLflow UI     →  http://localhost:5001"

down:
	docker compose down

logs:
	docker compose logs -f

status:
	docker compose ps

# ── Run demos ─────────────────────────────────────────────────────────────────

demo:
	docker compose run --rm demo-app python -m app.demo_routes

eval:
	docker compose run --rm demo-app python -m app.rag_eval

notebook:
	VIRTUAL_ENV=$(VENV) poetry run jupyter notebook notebooks/

# ── Open UIs ──────────────────────────────────────────────────────────────────

ui:
	@echo "Opening MLflow Tracking UI → http://localhost:5001"
	open http://localhost:5001 2>/dev/null || xdg-open http://localhost:5001 2>/dev/null || echo "Navigate to: http://localhost:5001"

routes:
	@echo "Gateway endpoints:"
	@curl -s http://localhost:5050/api/2.0/gateway/routes/ | python3 -m json.tool

.PHONY: install shell up down logs status demo eval notebook ui routes
