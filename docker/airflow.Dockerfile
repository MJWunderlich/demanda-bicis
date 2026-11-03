# Imagen de Airflow para el laboratorio. Parte de la imagen oficial de
# Airflow 3 con Python 3.12 e instala las MISMAS versiones de requirements.txt
# que usa el entrenamiento, para que el modelo se cargue con el scikit-learn
# con el que se entrenó (sección 4.5 del diseño).
FROM apache/airflow:3.3.2-python3.12

# git hace falta para que `bicis.entrenar` lea el commit al entrenar dentro
# de la imagen (docker compose run --rm airflow-cli python -m bicis.entrenar).
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
USER airflow

COPY requirements.txt /opt/bicis/requirements.txt
RUN pip install --no-cache-dir -r /opt/bicis/requirements.txt
