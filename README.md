# Demanda de bicicletas

Pipeline diario que predice la demanda por hora (`cnt`) de Capital Bikeshare y guarda un plan de bicicletas por hora. Laboratorio 1 de Product Development, Maestría en Data Science, Universidad Galileo.

Este README se completa en la fase final. Por ahora documenta cómo obtener los datos y cómo levantar los servicios.

## Datos

`data/raw/hour.csv` está versionado con DVC contra un remoto local, que el evaluador no puede alcanzar. Para reproducir:

1. Descargar el UCI Bike Sharing Dataset: <https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset>.
2. Extraer `hour.csv` en `data/raw/hour.csv`.
3. Comprobar que coincide con la versión registrada: `dvc status` responde que está al día (md5 en `data/raw/hour.csv.dvc`).

El archivo tiene 17,379 filas, del 2011-01-01 al 2012-12-31.

## Servicios (Airflow 3 y MLflow)

1. `make env` crea `.env` con `AIRFLOW_UID` (tu uid) y `MLFLOW_PORT`. En macOS el puerto 5000 suele ocuparlo AirPlay Receiver; en ese caso usa `MLFLOW_PORT=5001`.
2. `make up` (o `docker compose up -d`) construye las imágenes y levanta Postgres, Airflow (LocalExecutor) y MLflow.
   - Airflow: <http://localhost:8080>, usuario `airflow`, contraseña `airflow`.
   - MLflow: `http://localhost:$MLFLOW_PORT`. Dentro de compose Airflow lo ve como `http://mlflow:5000`.
3. El DAG `prediccion_diaria` nace pausado. Actívalo en la UI y dispáralo con la fecha lógica explícita (por ejemplo `2012-01-02`), o desde la terminal:

   ```
   docker compose exec airflow-scheduler airflow dags unpause prediccion_diaria
   docker compose exec airflow-scheduler airflow dags trigger prediccion_diaria --logical-date 2012-01-02T00:00:00+00:00
   ```

4. Backfill de enero de 2012 (31 ejecuciones, 741 filas en `salidas/predicciones.db`; tres días de ese mes tienen 23 horas en la fuente):

   ```
   docker compose exec airflow-scheduler airflow backfill create --dag-id prediccion_diaria \
     --from-date 2012-01-01 --to-date 2012-01-31 --max-active-runs 4 --reprocess-behavior completed
   ```

El repo se monta en `/opt/bicis/proyecto`, la fuente en `/opt/bicis/fuente` (solo lectura) y los logs en `airflow/logs/`. Las pruebas del DAG necesitan Airflow y corren dentro de la imagen con `make test-dag`; `make entrenar-docker` entrena con exactamente las mismas versiones que cargan el modelo en Airflow.
