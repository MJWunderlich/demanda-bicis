"""Guarda el plan de un día en la base de salidas, de forma idempotente.

"Idempotente" significa que ejecutar la tarea dos veces para el mismo día
deja el mismo resultado que ejecutarla una vez. Aquí se consigue borrando
las filas de esa fecha e insertando las nuevas dentro de una sola
transacción: si algo falla a medio camino, la base queda como estaba, y si
todo sale bien, el día tiene exactamente las filas de esta ejecución.

La tabla ``predicciones`` de ``salidas/predicciones.db`` guarda, por fecha y
hora, la predicción, el factor usado, el plan, la versión del modelo que lo
produjo y la marca de tiempo ``ejecutado_en`` (UTC, ISO 8601) de la
ejecución que escribió la fila.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

TABLA = "predicciones"
COLUMNAS_ENTRADA = ["fecha", "hr", "prediccion", "factor", "plan", "version_modelo"]

_CREAR_TABLA = f"""
CREATE TABLE IF NOT EXISTS {TABLA} (
    fecha          TEXT    NOT NULL,
    hr             INTEGER NOT NULL,
    prediccion     REAL    NOT NULL,
    factor         REAL    NOT NULL,
    plan           INTEGER NOT NULL,
    version_modelo TEXT    NOT NULL,
    ejecutado_en   TEXT    NOT NULL,
    PRIMARY KEY (fecha, hr)
)
"""


def almacenar(ruta: str | Path, ruta_db: str | Path) -> int:
    """Reemplaza en ``predicciones`` las filas del día con las del parquet.

    Args:
        ruta: Ruta a ``predicciones.parquet``, producido por ``predecir``.
        ruta_db: Ruta de ``predicciones.db``; se crea si no existe.

    Returns:
        Número de filas escritas para ese día.

    Raises:
        ValueError: Si el parquet mezcla más de una fecha o le faltan columnas.
    """
    df = pd.read_parquet(ruta)

    faltantes = [col for col in COLUMNAS_ENTRADA if col not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas en {ruta}: {faltantes}")
    fechas = df["fecha"].astype(str).unique().tolist()
    if len(fechas) != 1:
        raise ValueError(f"almacenar procesa un solo día; el parquet trae {fechas}")
    fecha = fechas[0]

    ejecutado_en = datetime.now(timezone.utc).isoformat(timespec="seconds")
    filas = [
        (
            fecha,
            int(r.hr),
            float(r.prediccion),
            float(r.factor),
            int(r.plan),
            str(r.version_modelo),
            ejecutado_en,
        )
        for r in df.itertuples(index=False)
    ]

    ruta_db = Path(ruta_db)
    ruta_db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(ruta_db)
    try:
        con.execute(_CREAR_TABLA)
        # Borrado e inserción en UNA transacción: o entra el día completo o no entra nada.
        with con:
            con.execute(f"DELETE FROM {TABLA} WHERE fecha = ?", (fecha,))
            con.executemany(
                f"INSERT INTO {TABLA} (fecha, hr, prediccion, factor, plan, "
                "version_modelo, ejecutado_en) VALUES (?, ?, ?, ?, ?, ?, ?)",
                filas,
            )
    finally:
        con.close()
    return len(filas)
