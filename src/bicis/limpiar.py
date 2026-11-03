"""Valida el parquet extraído de un día y lo deja listo para transformar.

"Limpiar" aquí significa comprobar que el día tiene sentido y detener el
pipeline con un error claro si no lo tiene. No se imputa ni se corrige
nada: un día sin filas, con columnas faltantes o con valores fuera de rango
es un problema de la fuente, y la tarea debe fallar para que el
orquestador lo reintente o alguien lo mire.

Un día con menos de 24 horas es válido. En hour.csv hay 76 días así, el
más extremo el 2012-10-29 con una sola hora (huracán Sandy). Solo se
rechaza el día sin ninguna fila.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bicis.transformar import FEATURES

NOMBRE_SALIDA = "limpio.parquet"

# Columnas que el resto del pipeline necesita. FEATURES ya contiene hr.
COLUMNAS_REQUERIDAS: list[str] = ["dteday", *FEATURES]

# Variables que el dataset entrega normalizadas al intervalo [0, 1].
NORMALIZADAS: list[str] = ["temp", "atemp", "hum", "windspeed"]


def limpiar(ruta: str | Path) -> str:
    """Comprueba el parquet extraído y escribe ``limpio.parquet`` junto a él.

    Args:
        ruta: Ruta a ``extraido.parquet``, producido por ``extraer``.

    Returns:
        Ruta, como texto, de ``limpio.parquet`` en la misma carpeta.

    Raises:
        ValueError: Si el día tiene 0 filas, faltan columnas requeridas,
            ``hr`` sale de 0 a 23, alguna variable normalizada sale de
            [0, 1], o hay horas duplicadas.
    """
    ruta = Path(ruta)
    df = pd.read_parquet(ruta)

    if len(df) == 0:
        raise ValueError(f"El día no tiene filas: {ruta}")

    faltantes = [col for col in COLUMNAS_REQUERIDAS if col not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas en {ruta}: {faltantes}")

    fuera = df.loc[~df["hr"].between(0, 23), "hr"].tolist()
    if fuera:
        raise ValueError(f"Valores de hr fuera de 0-23 en {ruta}: {fuera}")

    for col in NORMALIZADAS:
        malos = df.loc[~df[col].between(0.0, 1.0), col].tolist()
        if malos:
            raise ValueError(f"Valores de {col} fuera de [0, 1] en {ruta}: {malos}")

    duplicadas = df.loc[df["hr"].duplicated(), "hr"].tolist()
    if duplicadas:
        raise ValueError(f"Horas duplicadas en {ruta}: {duplicadas}")

    salida = ruta.with_name(NOMBRE_SALIDA)
    df.sort_values("hr").reset_index(drop=True).to_parquet(salida, index=False)
    return str(salida)
