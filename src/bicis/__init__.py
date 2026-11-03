"""Paquete bicis: módulos del pipeline diario de demanda de bicicletas.

Cada módulo expone una función con el mismo nombre que el archivo
(extraer, limpiar, transformar, predecir, almacenar) y se usa tanto desde
las tareas del DAG de Airflow como desde las pruebas. La lógica vive aquí,
nunca en el DAG.
"""
