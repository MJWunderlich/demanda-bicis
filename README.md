# Demanda de bicicletas

Pipeline diario que predice la demanda por hora (`cnt`) de Capital Bikeshare y guarda un plan de bicicletas por hora. Laboratorio 1 de Product Development, Maestría en Data Science, Universidad Galileo.

Este README se completa en la fase final. Por ahora documenta cómo obtener los datos.

## Datos

`data/raw/hour.csv` está versionado con DVC contra un remoto local, que el evaluador no puede alcanzar. Para reproducir:

1. Descargar el UCI Bike Sharing Dataset: <https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset>.
2. Extraer `hour.csv` en `data/raw/hour.csv`.
3. Comprobar que coincide con la versión registrada: `dvc status` responde que está al día (md5 en `data/raw/hour.csv.dvc`).

El archivo tiene 17,379 filas, del 2011-01-01 al 2012-12-31.
