"""Extrae de la fuente las filas de un solo día y las deja en la carpeta de trabajo.

La "fuente" es la base SQLite ``fuente/bicis.db`` con la tabla
``demanda_hora`` (una fila por hora, columnas originales de hour.csv). La
"carpeta de trabajo" es el directorio donde cada ejecución del pipeline
deja sus archivos intermedios en Parquet, uno por día:
``<dir_trabajo>/<fecha>/``.

La fecha que se extrae es la fecha lógica de la ejecución (la fecha que el
orquestador quiere procesar, no la fecha de hoy). Procesar solo ese día es
lo que permite re-ejecutar o rellenar fechas pasadas con el mismo código.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pandas as pd

TABLA = "demanda_hora"
NOMBRE_SALIDA = "extraido.parquet"
_FORMATO_FECHA = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def extraer(fecha: str, ruta_fuente: str | Path, dir_trabajo: str | Path) -> str:
    """Copia a Parquet las filas de ``demanda_hora`` cuya ``dteday`` es ``fecha``.

    Args:
        fecha: Día a extraer, como texto ``YYYY-MM-DD``.
        ruta_fuente: Ruta de ``bicis.db``.
        dir_trabajo: Carpeta raíz de trabajo; se crea ``<dir_trabajo>/<fecha>/``.

    Returns:
        Ruta, como texto, de ``<dir_trabajo>/<fecha>/extraido.parquet``. El
        archivo puede tener 0 filas si el día no existe en la fuente; decidir
        si eso es un error le toca a ``limpiar``.

    Raises:
        FileNotFoundError: Si la fuente no existe.
        ValueError: Si ``fecha`` no tiene el formato ``YYYY-MM-DD``.
    """
    ruta_fuente = Path(ruta_fuente)
    if not ruta_fuente.exists():
        raise FileNotFoundError(f"No existe la fuente: {ruta_fuente}")
    if not _FORMATO_FECHA.match(fecha):
        raise ValueError(f"La fecha debe ser YYYY-MM-DD, se recibió {fecha!r}")

    with sqlite3.connect(ruta_fuente) as con:
        df = pd.read_sql_query(
            f"SELECT * FROM {TABLA} WHERE dteday = ? ORDER BY hr", con, params=(fecha,)
        )

    carpeta = Path(dir_trabajo) / fecha
    carpeta.mkdir(parents=True, exist_ok=True)
    salida = carpeta / NOMBRE_SALIDA
    df.to_parquet(salida, index=False)
    return str(salida)
