# Usa el venv del proyecto si existe (./.venv o ../.venv); si no, el python del PATH.
PYTHON ?= $(firstword $(wildcard .venv/bin/python ../.venv/bin/python) python)

.PHONY: fuente test entrenar

fuente:
	$(PYTHON) scripts/crear_fuente.py

test:
	$(PYTHON) -m pytest

entrenar:
	$(PYTHON) -m bicis.entrenar
