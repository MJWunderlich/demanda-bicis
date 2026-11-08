"""Predice la demanda por hora de un día con el modelo campeón y calcula el plan.

El "campeón" es la versión del modelo registrado ``demanda-bicis`` que lleva
el alias ``@champion`` en el Registry de MLflow. Un alias es una etiqueta
movible que apunta a una versión concreta; cargar por alias (y nunca por
ruta local ni por número de versión) es lo que permite promover un modelo
nuevo sin tocar el código del pipeline.

El "factor de ajuste" es el porcentaje que se añade a la predicción para
cubrir la asimetría de costos (faltar una bicicleta cuesta 3, sobrar una
cuesta 1). Se eligió en validación al entrenar y quedó guardado en el run
que produjo la versión campeona, como parámetro ``factor`` y como artefacto
``factor.json``; este módulo lo lee de ahí, así que el plan siempre usa el
factor del modelo que lo generó.

El "plan" es la cantidad de bicicletas a tener disponibles en cada hora:
``plan = ceil(max(prediccion, 0) * (1 + factor))``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from bicis.costos import plan_con_factor
from bicis.transformar import FEATURES

MODEL_URI_CHAMPION = "models:/demanda-bicis@champion"
NOMBRE_SALIDA = "predicciones.parquet"
COLUMNAS_SALIDA = ["fecha", "hr", "prediccion", "factor", "plan", "version_modelo"]

_URI_ALIAS = re.compile(r"^models:/(?P<nombre>[^/@]+)@(?P<alias>[^/@]+)$")


@dataclass
class ModeloCargado:
    """Lo que ``predecir`` necesita de MLflow, resuelto una sola vez.

    Attributes:
        modelo: Objeto con ``predict(X)`` que acepta un DataFrame con ``FEATURES``.
        version_modelo: Modelo registrado y número de versión que resolvió el
            alias, como texto ``"demanda-bicis/1"``. Así cada predicción
            guardada dice qué modelo y qué versión la produjo.
        factor: Factor de ajuste del run que entrenó esa versión.
    """

    modelo: object
    version_modelo: str
    factor: float


def cargar_champion(model_uri: str = MODEL_URI_CHAMPION) -> ModeloCargado:
    """Resuelve un alias del Registry y devuelve el modelo, su versión y su factor.

    Args:
        model_uri: URI con la forma ``models:/<nombre>@<alias>``. Cualquier
            otra forma (ruta local, ``models:/<nombre>/<version>``) se rechaza.

    Returns:
        Un ``ModeloCargado`` listo para predecir.

    Raises:
        ValueError: Si la URI no es un alias del Registry.
        mlflow.exceptions.MlflowException: Si el modelo o el alias no existen.
    """
    partes = _URI_ALIAS.match(model_uri)
    if not partes:
        raise ValueError(
            f"predecir solo carga modelos por alias (models:/<nombre>@<alias>); "
            f"se recibió {model_uri!r}"
        )

    # Import tardío: el DAG se parsea sin cargar mlflow, y las pruebas lo reemplazan.
    import mlflow
    from mlflow import MlflowClient

    cliente = MlflowClient()
    version = cliente.get_model_version_by_alias(partes["nombre"], partes["alias"])
    modelo = mlflow.pyfunc.load_model(f"models:/{partes['nombre']}/{version.version}")
    factor = _leer_factor(cliente, version.run_id)
    return ModeloCargado(modelo=modelo, version_modelo=f"{partes['nombre']}/{version.version}", factor=factor)


def _leer_factor(cliente, run_id: str) -> float:
    """Lee el factor del run: primero el parámetro ``factor``, si no, ``factor.json``."""
    run = cliente.get_run(run_id)
    valor = run.data.params.get("factor")
    if valor is None:
        ruta = cliente.download_artifacts(run_id, "factor.json")
        valor = json.loads(Path(ruta).read_text())["factor"]
    return float(valor)


def calcular_plan(prediccion: np.ndarray, factor: float) -> np.ndarray:
    """Aplica el factor y redondea hacia arriba: ``ceil(max(pred, 0) * (1 + factor))``.

    Delega en ``costos.plan_con_factor`` para que el plan que se guarda en
    producción sea el mismo cálculo con el que se eligió el factor.
    """
    return plan_con_factor(prediccion, factor).astype(int)


def predecir(ruta: str | Path, model_uri: str = MODEL_URI_CHAMPION) -> str:
    """Escribe ``predicciones.parquet`` con la predicción y el plan de cada hora.

    Args:
        ruta: Ruta a ``features.parquet``, producido por ``transformar``.
        model_uri: Alias del Registry a cargar. Por defecto el campeón.

    Returns:
        Ruta, como texto, de ``predicciones.parquet`` en la misma carpeta, con
        las columnas ``fecha, hr, prediccion, factor, plan, version_modelo``.

    Raises:
        ValueError: Si ``model_uri`` no es un alias.
        mlflow.exceptions.MlflowException: Si el modelo no existe o la firma
            registrada no acepta las columnas de entrada (por ejemplo, porque
            ``FEATURES`` cambió después de entrenar).
    """
    ruta = Path(ruta)
    features = pd.read_parquet(ruta)
    cargado = cargar_champion(model_uri)

    # La firma del modelo valida columnas y tipos; aquí solo se le pasa FEATURES en orden.
    prediccion = np.asarray(cargado.modelo.predict(features[FEATURES]), dtype=float).ravel()
    prediccion = np.maximum(prediccion, 0.0)

    salida_df = pd.DataFrame(
        {
            "fecha": features["fecha"].astype(str),
            "hr": features["hr"].astype(int),
            "prediccion": prediccion,
            "factor": cargado.factor,
            "plan": calcular_plan(prediccion, cargado.factor),
            "version_modelo": cargado.version_modelo,
        }
    )[COLUMNAS_SALIDA]

    salida = ruta.with_name(NOMBRE_SALIDA)
    salida_df.to_parquet(salida, index=False)
    return str(salida)
