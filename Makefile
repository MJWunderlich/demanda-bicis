# Variables locales (AIRFLOW_UID, MLFLOW_PORT); ver .env.ejemplo.
-include .env

# Usa el venv del proyecto si existe (./.venv o ../.venv); si no, el python del PATH.
PYTHON ?= $(firstword $(wildcard .venv/bin/python ../.venv/bin/python) python)

# Servidor MLflow visto desde la laptop; dentro de compose Airflow usa http://mlflow:5000.
# MLFLOW_PORT es el puerto publicado en la laptop (el mismo que usa docker compose).
MLFLOW_PORT ?= 5000
MLFLOW_TRACKING_URI ?= http://localhost:$(MLFLOW_PORT)

.PHONY: env fuente test test-dag entrenar entrenar-docker up down logs

# Crea .env con el uid del usuario si no existe (Airflow escribe logs con ese dueño).
env:
	@test -f .env || { printf 'AIRFLOW_UID=%s\nMLFLOW_PORT=%s\n' "$$(id -u)" "$(MLFLOW_PORT)" > .env; cat .env; }

fuente:
	$(PYTHON) scripts/crear_fuente.py

test:
	$(PYTHON) -m pytest

# Las pruebas del DAG necesitan Airflow: corren dentro de la imagen.
test-dag:
	docker compose run --rm airflow-cli python -m pytest tests/test_dag.py

entrenar:
	PYTHONPATH=src MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI) $(PYTHON) -m bicis.entrenar

# Entrena con las mismas versiones que cargan el modelo en Airflow.
entrenar-docker:
	docker compose run --rm airflow-cli python -m bicis.entrenar

up: env
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f --tail=100
