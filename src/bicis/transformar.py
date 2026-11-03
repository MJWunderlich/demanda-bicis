"""Transforma el parquet limpio de un día en la tabla de features que consume el modelo.

Una "feature" es una columna de entrada del modelo. La lista ``FEATURES``
de este módulo es la única definición de qué columnas puede ver el modelo:
``entrenar`` la usa para construir la matriz de entrenamiento y ``predecir``
para construir la matriz de predicción, así que las dos coinciden por
construcción. Nadie más la redefine.

Quedan fuera a propósito ``casual`` y ``registered`` (su suma es ``cnt``,
la etiqueta, y usarlas sería fuga de la etiqueta), ``instant`` (numera las
filas y codifica el tiempo), ``dteday`` e ``yr`` (el modelo no ve el año;
la deriva entre 2011 y 2012 se corrige reentrenando, no con una feature).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

FEATURES: list[str] = [
    "hr",
    "weekday",
    "workingday",
    "holiday",
    "season",
    "mnth",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
]

PROHIBIDAS: list[str] = ["casual", "registered", "instant", "yr", "dteday", "cnt"]

NOMBRE_SALIDA = "features.parquet"


def transformar(ruta: str | Path) -> str:
    """Construye ``features.parquet`` con ``fecha``, ``hr`` y las columnas de ``FEATURES``.

    Lee el parquet limpio del día, renombra ``dteday`` a ``fecha`` (texto
    ``YYYY-MM-DD``) y conserva exactamente las columnas que el modelo puede
    ver. Ninguna columna prohibida sobrevive a este paso, de modo que aguas
    abajo no hay forma de que entre al modelo por accidente.

    Args:
        ruta: Ruta a ``limpio.parquet``, producido por ``limpiar``.

    Returns:
        Ruta, como texto, de ``features.parquet`` escrito en la misma carpeta.

    Raises:
        ValueError: Si falta alguna columna de ``FEATURES`` o la columna
            ``dteday``.
    """
    ruta = Path(ruta)
    df = pd.read_parquet(ruta)

    faltantes = [col for col in ["dteday", *FEATURES] if col not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas para construir las features: {faltantes}")

    # fecha e hr identifican la fila; FEATURES ya incluye hr, así que no se repite.
    features = df[["dteday", *FEATURES]].rename(columns={"dteday": "fecha"})
    features["fecha"] = features["fecha"].astype(str)
    features = features.sort_values("hr").reset_index(drop=True)

    salida = ruta.with_name(NOMBRE_SALIDA)
    features.to_parquet(salida, index=False)
    return str(salida)
