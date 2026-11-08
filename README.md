# Demanda de bicicletas

Pipeline diario que predice la demanda por hora (`cnt`) de Capital Bikeshare y guarda un plan de bicicletas por hora. Laboratorio 1 de Product Development, Maestría en Data Science, Universidad Galileo.

[![tests](https://github.com/MJWunderlich/demanda-bicis/actions/workflows/tests.yml/badge.svg)](https://github.com/MJWunderlich/demanda-bicis/actions/workflows/tests.yml)

## El problema

Un operador de bicicletas compartidas (Capital Bikeshare, Washington D.C.) necesita saber cada día cuántas bicicletas por hora tiene que tener disponibles. Faltar una bicicleta cuesta 3 y sobrar una cuesta 1 (`configs/costos.yaml`). El sistema:

1. Predice la demanda por hora de cada día con el modelo campeón.
2. Ajusta la predicción con un factor de seguridad elegido para minimizar ese costo: `plan = ceil(max(pred, 0) × (1 + factor))`.
3. Guarda el plan del día de forma automática, sin intervención manual, con la versión del modelo que lo produjo.

Datos: UCI Bike Sharing Dataset, `hour.csv`, 17,379 filas del 2011-01-01 al 2012-12-31.

## Arquitectura

```
                        ┌───────────────────────── docker compose ─────────────────────────┐
 data/raw/hour.csv      │                                                                   │
   (DVC)                │  postgres ◄── scheduler / dag-processor / api-server (:8080)      │
     │                  │                     │  LocalExecutor                              │
     ▼                  │                     ▼                                             │
 scripts/crear_fuente   │   DAG prediccion_diaria (diario, logical_date)                    │
     │                  │   extraer → limpiar → transformar → predecir → almacenar          │
     ▼                  │      ▲                                  │           │             │
 fuente/bicis.db ───────┼──────┘ (montaje :ro)                    │           ▼             │
                        │                       models:/demanda-bicis@champion             │
                        │                                         │   salidas/predicciones.db
                        │  mlflow server (:5000) ◄────────────────┘                         │
                        │   backend SQLite · artefactos servidos por HTTP                   │
                        └───────────────────────────────────────────────────────────────────┘
 bicis/entrenar.py ──► registra runs y versiones en MLflow ──► alias @champion
```

- **Airflow 3.3.2** con `LocalExecutor` y Postgres, imagen propia (`docker/airflow.Dockerfile`) con las versiones exactas de `requirements.txt`, para que el modelo se cargue con el mismo scikit-learn con el que se entrenó. El repo se monta en `/opt/bicis/proyecto`, fuera de `/opt/airflow`.
- **MLflow 3.17.2** con backend SQLite y artefactos servidos por HTTP (`--serve-artifacts`). Dentro de compose Airflow lo ve como `http://mlflow:5000`.
- **`src/bicis/`**: la lógica. El DAG solo llama a estas funciones y pasa rutas por XCom; los datos van a Parquet en `trabajo/<fecha>/`.

| módulo | contrato |
|---|---|
| `extraer(fecha, ruta_fuente, dir_trabajo)` | filas del día lógico → `extraido.parquet` |
| `limpiar(ruta)` | falla si el día no tiene filas, faltan columnas, hay valores fuera de rango u horas repetidas → `limpio.parquet` |
| `transformar(ruta)` | deja solo `FEATURES` → `features.parquet` |
| `predecir(ruta)` | carga `models:/demanda-bicis@champion`, predice, aplica el factor → `predicciones.parquet` |
| `almacenar(ruta, ruta_db)` | borra e inserta las filas del día en una transacción (idempotente) |
| `costos.py` | costo por hora, plan con factor, elección del factor en validación |
| `entrenar.py` | escalera de modelos, runs anidados, linaje, registro del campeón |
| `compuerta.py` | reglas de promoción, función pura |

## Partición temporal

Cada conjunto tiene un solo papel. Las fechas viven en `params.yaml` y el split nunca es aleatorio.

| conjunto | fechas | papel |
|---|---|---|
| entrenamiento | 2011-01-01 a 2011-11-30 | ajusta el modelo |
| validación | 2011-12-01 a 2011-12-31 | elige el modelo y el factor de ajuste |
| evaluación | 2012, un mes a la vez | mide la deriva; ningún modelo lo vio |
| producción simulada | enero de 2012 (backfill) | el DAG predice día por día |

## Reglas anti-fuga

- Features permitidas, definidas una sola vez en `transformar.FEATURES`: `hr, weekday, workingday, holiday, season, mnth, weathersit, temp, atemp, hum, windspeed`.
- Nunca `casual` ni `registered` (`casual + registered = cnt`, es la etiqueta), ni `instant` (codifica el tiempo), ni `dteday` ni `yr` (el modelo no ve el año; la deriva se corrige reentrenando). La prueba de `transformar` lo verifica.
- Cada tarea procesa solo su fecha lógica (`logical_date`), nunca la fecha de hoy.
- El factor y el modelo se eligen en diciembre de 2011. 2012 solo se usa para evaluar.
- La firma del modelo en MLflow valida columnas y tipos al predecir: si `FEATURES` cambia, `predecir` falla con "Model is missing inputs" en vez de predecir mal.
- Semilla fija (`semilla: 42`) en todo lo que tiene aleatoriedad.

## Cómo reproducir

Requisitos: Docker con Compose v2, Python 3.12 o 3.13, `dvc` y `gh` opcionales.

1. **Datos.** `data/raw/hour.csv` está versionado con DVC contra un remoto local, que no es accesible fuera de esta máquina. Descarga el [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset), extrae `hour.csv` en `data/raw/hour.csv` y comprueba que coincide con el puntero: `dvc status` responde que está al día (md5 `d50bd5a6f55131e72a7bedc334e2fce1`). Con acceso al remoto basta `dvc pull`.
2. **Entorno.** `python -m venv .venv && .venv/bin/pip install -r requirements.txt`. El Makefile usa `.venv` o `../.venv` si existen.
3. **Fuente.** `make fuente` crea `fuente/bicis.db` (tabla `demanda_hora`, 17,379 filas).
4. **Pruebas.** `make test` (65 pruebas, sin MLflow ni Docker).
5. **Servicios.** `make env` escribe `.env` con `AIRFLOW_UID` y `MLFLOW_PORT`; en macOS el puerto 5000 suele ocuparlo AirPlay Receiver, por eso aquí se usó `MLFLOW_PORT=5001`. Luego `make up` (o `docker compose up -d`).
   - Airflow: <http://localhost:8080>, usuario `airflow`, contraseña `airflow`.
   - MLflow: `http://localhost:$MLFLOW_PORT`.
6. **Entrenar.** Con el árbol de git limpio (el `git_commit` que se registra tiene que existir): `make entrenar`, o `make entrenar-docker` para entrenar dentro de la imagen de Airflow. Registra la escalera en el experimento `demanda-bicis` y promueve al ganador a `demanda-bicis@champion`.
7. **Pipeline.** El DAG nace pausado. Actívalo y dispáralo con la fecha lógica explícita, o desde la terminal:

   ```
   docker compose exec airflow-scheduler airflow dags unpause prediccion_diaria
   docker compose exec airflow-scheduler airflow dags trigger prediccion_diaria --logical-date 2012-01-02T00:00:00+00:00
   ```

8. **Backfill de enero de 2012.** 31 ejecuciones, 741 filas en `salidas/predicciones.db`:

   ```
   docker compose exec airflow-scheduler airflow backfill create --dag-id prediccion_diaria \
     --from-date 2012-01-01 --to-date 2012-01-31 --max-active-runs 4 --reprocess-behavior completed
   ```

9. **Deriva.** `make evaluar` regenera la tabla y la figura de la sección siguiente.

Las pruebas del DAG necesitan Airflow y corren dentro de la imagen con `make test-dag`.

## Escalera de modelos

`bicis.entrenar` entrena cuatro escalones en orden, cada uno como un run de MLflow (los de HGB anidados bajo `busqueda-hgb`). Cada run registra parámetros, métricas, los tags de linaje `git_commit`, `features`, `huella_datos` y `dvc_md5`, los datasets de entrenamiento y validación, el modelo con firma y ejemplo de entrada, y los artefactos `factor.json`, `costo_vs_factor.png`, `mae_por_hora.png`, `real_vs_prediccion_val.png` e `importancias.png`.

Validación: diciembre de 2011, 741 horas. Entrenamiento: 7,904 horas.

| escalón | factor | MAE val | costo/hora val | latencia p95 (ms) | tamaño (MB) |
|---|---|---|---|---|---|
| **hgb-lr-0.05** (campeón) | 0.05 | 31.64 | **46.43** | 28.1 | 1.07 |
| hgb-lr-0.1 | 0.05 | 31.45 | 47.53 | 25.1 | 1.07 |
| hgb-lr-0.2 | 0.05 | 33.33 | 50.22 | 25.5 | 1.07 |
| random-forest | 0.05 | 34.06 | 50.94 | 30.1 | 203.46 |
| perfil (media por hora y tipo de día) | 0.05 | 43.71 | 58.44 | 0.1 | 0.002 |
| ridge (one-hot de hora y mes) | 0.20 | 48.13 | 81.05 | 1.2 | 0.003 |

El criterio de selección es el costo por hora en validación, no el MAE: `hgb-lr-0.1` tiene mejor MAE pero peor costo. Dos observaciones:

- `ridge` queda por debajo del perfil porque diciembre nunca aparece en entrenamiento y su columna one-hot de `mnth = 12` queda en cero.
- `random-forest` pesa 203 MB, más del doble del umbral de la compuerta.

**Factor de ajuste.** Para cada escalón se recorre la grilla 0.00 a 0.40 en pasos de 0.05 sobre validación y se elige el factor que minimiza el costo por hora; la curva queda como artefacto `costo_vs_factor.png`. Con costos 3 a 1 conviene planear algo más que la predicción, y 0.05 fue el mínimo para todos los modelos salvo `ridge` (0.20, porque subestima más).

El campeón es `demanda-bicis` versión 1 con alias `champion`. `predecir` lo carga solo por alias, nunca por ruta ni por número de versión, y cada fila guardada lleva `version_modelo = demanda-bicis/1` y `factor = 0.05`.

## Deriva: el campeón en producción

`scripts/evaluar_produccion.py` une `salidas/predicciones.db` con la demanda real de la fuente y compara enero de 2012 (producción, tras el backfill) con diciembre de 2011 (validación, recalculada con el mismo campeón y el mismo factor).

| métrica | diciembre 2011 (validación) | enero 2012 (producción) | cambio |
|---|---|---|---|
| filas (horas) | 741 | 741 | +0 |
| demanda media por hora | 117.8 | 130.6 | +12.71 (+11%) |
| MAE | 31.64 | 68.71 | +37.08 (+117%) |
| sesgo (predicción − real) | +15.20 | -67.76 | -82.96 |
| costo por hora | 46.43 | 194.89 | +148.46 (+320%) |

El sesgo pasa de +15 a −68 bicicletas por hora: el modelo subestima casi toda la demanda de enero de 2012, y ese sesgo explica casi todo el MAE. Con los costos del laboratorio, faltar cuesta tres veces más que sobrar, así que el costo por hora se cuadruplica. La causa es la deriva: la demanda media por hora pasó de 143.8 en 2011 a 234.7 en 2012 (+63 %), y el modelo no ve el año. Pero hay un efecto más fuerte: aprendió con un solo año que `mnth = 1` significa poca demanda, porque enero de 2011 promedió 56 bicicletas por hora con el sistema recién arrancado. Para enero de 2012, que promedia 131, predice 63. Esto es lo que justifica reentrenar con datos recientes.

![MAE por día en enero de 2012](docs/capturas/deriva_enero_2012.png)

El único día por debajo de la referencia de validación es el 21 de enero (nevada en Washington): la demanda real cayó al nivel que el modelo esperaba.

**Huracán Sandy.** El 2012-10-29 tiene una sola hora en la fuente y el 2012-10-30 tiene once. El DAG los procesa sin fallar (`limpiar` solo rechaza días con cero filas): 1 y 11 filas de plan respectivamente. Son días válidos, no errores de datos.

## Compuerta de promoción

`bicis.compuerta.compuerta(challenger, champion, umbrales)` es una función pura (no importa MLflow) que decide si un modelo nuevo reemplaza al campeón. Aplica cuatro reglas en orden y devuelve `(promover, motivo)`, donde el motivo nombra la primera regla que falla con sus números. Los umbrales viven en `params.yaml` bajo `compuerta`:

1. Features idénticas (misma lista, mismo orden).
2. `latencia_p95_ms` ≤ 200.
3. `tamano_modelo_mb` ≤ 100.
4. `costo_hora_eval` al menos 2 % más bajo que el del campeón.

Aplicada a la escalera, con las métricas de validación como evaluación y `hgb-lr-0.05` como campeón, ningún otro escalón se promovería:

| retador | decisión | motivo |
|---|---|---|
| hgb-lr-0.1 | no | mejora de costo −2.4 % (46.43 → 47.53), no alcanza el mínimo 2 % |
| hgb-lr-0.2 | no | mejora de costo −8.2 % |
| random-forest | no | tamaño 203.46 MB supera el máximo 100 |
| perfil | no | mejora de costo −25.9 % |
| ridge | no | mejora de costo −74.6 % |

El DAG de reentrenamiento (Nivel 3), que entrenaría un retador por mes y usaría esta compuerta para mover el alias, quedó fuera de esta entrega.

## Capturas

En `docs/capturas/`:

| archivo | qué muestra |
|---|---|
| `deriva_enero_2012.png` | MAE por día en enero de 2012 contra la referencia de validación |
| `mlflow_experimento.png` | tabla del experimento `demanda-bicis` con los cuatro escalones y los runs anidados de HGB |
| `mlflow_run_champion.png` | run `hgb-lr-0.05`: parámetros, métricas, tags de linaje y artefactos |
| `mlflow_registry.png` | modelo registrado `demanda-bicis`, versión 1 con alias `champion` |
| `airflow_grilla_enero.png` | grilla de `prediccion_diaria` con las 31 ejecuciones de enero en verde |
| `airflow_log_predecir.png` | log de la tarea `predecir` con la línea `version_modelo=demanda-bicis/1` |

## Verificación

- `make test`: 65 pruebas en verde, también en GitHub Actions (`.github/workflows/tests.yml`, Python 3.12 como la imagen de Airflow).
- Ejecución manual del 2012-01-02: 5 tareas en verde, 23 filas. Ese día tiene 23 horas en la fuente, no 24; lo mismo pasa el 10 y el 17 de enero, por eso el backfill deja 741 filas y no 744.
- Idempotencia: *Clear* del mismo día deja 23 filas con una nueva marca de tiempo, no 46.
- Backfill de enero: 31 ejecuciones en verde en 60 segundos, 741 filas.
- La recalculación de validación en `evaluar_produccion` reproduce el `mae_val` del run campeón (31.64).

## Limitaciones y trabajo futuro

- **Un año de entrenamiento.** El modelo asocia el mes con el nivel de demanda de 2011, y por eso subestima tanto en 2012. El reentrenamiento mensual con ventana móvil (Fase 7 del diseño, DAG `reentrenamiento` con `@task.branch` sobre la compuerta) es la respuesta y no se construyó.
- **Horas faltantes.** Si un día tiene menos de 24 horas en la fuente, el plan cubre solo las horas presentes. Completar las horas que faltan (y marcarlas como imputadas) daría un plan completo para la operación.
- **Puerto de MLflow.** El enunciado usa el 5000; en esta máquina lo ocupa AirPlay Receiver, por eso `MLFLOW_PORT` es configurable y aquí vale 5001.
- **Versiones de Python.** El campeón se entrenó en un venv con Python 3.13 y se carga en la imagen con Python 3.12 con los mismos paquetes fijados; MLflow avisa de la diferencia pero predice igual. `make entrenar-docker` entrena dentro de la imagen si se quiere eliminar el aviso.
- **Lab 2.** Servir el modelo con FastAPI (validación con Pydantic, `/health`, `/modelo`, `/v1/prediccion`) y empaquetarlo en Docker queda para el siguiente laboratorio.
