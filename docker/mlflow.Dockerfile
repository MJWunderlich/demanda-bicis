# Servidor de tracking y Registry de MLflow, con la misma versión que requirements.txt.
FROM python:3.12-slim

RUN pip install --no-cache-dir mlflow==3.17.2

# Backend SQLite y artefactos en el volumen /mlflow (montado desde ./mlflow).
RUN mkdir -p /mlflow/artifacts
WORKDIR /mlflow
EXPOSE 5000
