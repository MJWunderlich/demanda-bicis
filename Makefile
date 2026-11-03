# Usa el venv del proyecto si existe (./.venv o ../.venv); si no, el python del PATH.
PYTHON ?= $(firstword $(wildcard .venv/bin/python ../.venv/bin/python) python)

# Servidor MLflow visto desde la laptop; dentro de compose Airflow usa http://mlflow:5000.
# MLFLOW_PORT es el puerto publicado en la laptop (el mismo que usa docker compose).
MLFLOW_PORT ?= 5000
MLFLOW_TRACKING_URI ?= http://localhost:$(MLFLOW_PORT)

.PHONY: fuente test entrenar

fuente:
	$(PYTHON) scripts/crear_fuente.py

test:
	$(PYTHON) -m pytest

entrenar:
	PYTHONPATH=src MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI) $(PYTHON) -m bicis.entrenar
