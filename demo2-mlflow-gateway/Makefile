VENV := $(shell pwd)/.venv

install:
	python3 -m venv $(VENV)
	VIRTUAL_ENV=$(VENV) poetry install --with dev

shell:
	VIRTUAL_ENV=$(VENV) poetry shell

run:
	VIRTUAL_ENV=$(VENV) poetry run python -m app.demo_routes

notebook:
	VIRTUAL_ENV=$(VENV) poetry run jupyter notebook notebooks/

.PHONY: install shell run notebook
