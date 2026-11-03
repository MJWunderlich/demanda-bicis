# Demanda de bicicletas

Pipeline diario que predice la demanda por hora (cnt) de Capital Bikeshare y guarda un plan de
bicicletas por hora. Laboratorio 1 de Product Development (Universidad Galileo).
Datos: UCI Bike Sharing Dataset, hour.csv (2011-01-01 a 2012-12-31, 17,379 filas).

## Comandos
- `make fuente`: crea fuente/bicis.db a partir de data/raw/hour.csv.
- `make test`: ejecuta pytest.
- `make entrenar`: ejecuta la escalera de modelos y registra los runs en MLflow.
- `docker compose up -d`: levanta Airflow (http://localhost:8080) y MLflow (http://localhost:5000).
- Backfill: `docker compose exec airflow-scheduler airflow backfill create --dag-id prediccion_diaria --from-date 2012-01-01 --to-date 2012-01-31 --max-active-runs 4`

## Estructura
- `src/bicis/`: extraer.py, limpiar.py, transformar.py, predecir.py, almacenar.py, entrenar.py, costos.py, compuerta.py
- `dags/`: prediccion_diaria.py (y reentrenamiento.py)
- `scripts/crear_fuente.py`: hour.csv → fuente/bicis.db (tabla demanda_hora)
- `configs/costos.yaml`: costo por faltante y sobrante · `params.yaml`: ventanas, semilla, hiperparámetros
- `tests/`: una prueba por módulo como mínimo

## Reglas del modelo (no negociables)
- Features permitidas: hr, weekday, workingday, holiday, season, mnth, weathersit, temp, atemp, hum, windspeed.
- NUNCA usar como feature: casual, registered (casual + registered = cnt, es fuga de la etiqueta),
  instant (codifica el tiempo), dteday ni yr (el modelo no ve el año; la deriva se corrige reentrenando).
- La lista de features vive en un solo lugar: `transformar.FEATURES`. Nadie más la redefine.
- El split es temporal y lo definen las fechas de params.yaml, nunca un split aleatorio.
- 2012 nunca se usa para elegir modelo ni factor: solo para evaluar.
- Semilla fija desde params.yaml en todo lo que tenga aleatoriedad.

## Reglas del pipeline
- Cada tarea recibe la fecha lógica (logical_date) y procesa SOLO ese día. Nunca usar la fecha de hoy.
- Entre tareas se pasan rutas o conteos por XCom; los datos van a disco (Parquet en trabajo/<fecha>/).
- limpiar falla con un error claro si el día no tiene filas o si faltan columnas o hay valores fuera de rango.
- almacenar es idempotente: borra las filas del día e inserta las nuevas en UNA transacción.
- predecir carga el modelo SOLO por alias: models:/demanda-bicis@champion. Nunca por ruta local ni por número de versión.
- Cada predicción guardada lleva version_modelo y factor.

## Reglas de trabajo
- Nunca trabajar en main. Una rama por tarea, PR con `gh pr create`, merge solo con las pruebas en verde.
- Commits pequeños con Conventional Commits (feat:, fix:, test:, docs:, chore:).
- Nunca hacer force push. Nunca commitear datos, .db, modelos ni mlruns: los datos van a DVC.
- Commit ANTES de entrenar: el git_commit que se registra en MLflow tiene que existir.
- Docstrings autocontenidos: cada docstring define los conceptos y términos que menciona
  (fecha lógica, idempotencia, alias, factor de ajuste, etc.), sin suponer que el lector conoce el resto del diseño.
- Al terminar cada tarea, resume qué cambió, cómo se verificó y qué queda pendiente.
