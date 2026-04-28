# VS Code / Claude Code inherit a stale VIRTUAL_ENV from the Homebrew Python.
# Overriding it here ensures Poetry uses the project's own .venv.
VENV := $(shell pwd)/.venv

install:
	python3 -m venv $(VENV)
	VIRTUAL_ENV=$(VENV) poetry install --with dev

shell:
	VIRTUAL_ENV=$(VENV) poetry shell

run:
	VIRTUAL_ENV=$(VENV) poetry run python -m app.main

notebook:
	VIRTUAL_ENV=$(VENV) poetry run jupyter notebook notebooks/

.PHONY: install shell run notebook
