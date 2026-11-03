"""Crea la fuente de datos del pipeline: fuente/bicis.db a partir de data/raw/hour.csv.

La "fuente" es la base SQLite que simula el sistema operacional del que el
DAG extrae cada día. Contiene una sola tabla, ``demanda_hora``, con todas
las columnas originales de hour.csv (UCI Bike Sharing Dataset) y una fila
por hora registrada. La columna ``dteday`` se guarda como texto
``YYYY-MM-DD`` para que ``extraer`` pueda filtrar un día con una
comparación de cadenas, y lleva un índice porque esa es la única forma de
acceso del pipeline.

El script es re-ejecutable: cada corrida reemplaza la tabla completa, así
que correrlo dos veces deja la fuente idéntica a correrlo una vez.

Uso::

    python scripts/crear_fuente.py [--csv data/raw/hour.csv] [--db fuente/bicis.db]
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

TABLA = "demanda_hora"
INDICE = "idx_demanda_hora_dteday"


def crear_fuente(ruta_csv: Path, ruta_db: Path) -> int:
    """Lee el CSV de demanda por hora y lo carga en SQLite, reemplazando la tabla.

    Crea la carpeta de destino si no existe, escribe la tabla ``demanda_hora``
    con todas las columnas del CSV y construye un índice sobre ``dteday``.
    Reemplazar la tabla (en lugar de agregar filas) es lo que hace al script
    re-ejecutable: una segunda corrida no duplica datos.

    Args:
        ruta_csv: Ruta a hour.csv con las 17 columnas originales del dataset.
        ruta_db: Ruta del archivo SQLite a crear o reemplazar.

    Returns:
        Número de filas escritas en la tabla.

    Raises:
        FileNotFoundError: Si el CSV no existe.
    """
    if not ruta_csv.exists():
        raise FileNotFoundError(f"No existe el CSV de origen: {ruta_csv}")

    # dteday se lee como texto para que SQLite la guarde como TEXT y no como fecha.
    df = pd.read_csv(ruta_csv, dtype={"dteday": str})
    ruta_db.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(ruta_db) as con:
        df.to_sql(TABLA, con, if_exists="replace", index=False)
        # El índice se pierde al reemplazar la tabla; se recrea en cada corrida.
        con.execute(f"CREATE INDEX IF NOT EXISTS {INDICE} ON {TABLA}(dteday)")
    return len(df)


def resumir(ruta_db: Path) -> tuple[dict[str, int], int]:
    """Calcula el resumen que se imprime al terminar.

    Un "día con menos de 24 horas" es una fecha de ``dteday`` que aparece en
    menos de 24 filas. En hour.csv eso ocurre cuando el sistema no registró
    alquileres en alguna hora (por ejemplo, el huracán Sandy en octubre de
    2012). Esos días son válidos y el pipeline debe aceptarlos.

    Args:
        ruta_db: Ruta del archivo SQLite ya creado.

    Returns:
        Una tupla con el diccionario de filas por año (clave ``"2011"``,
        ``"2012"``...) y el número de días con menos de 24 horas.
    """
    with sqlite3.connect(ruta_db) as con:
        filas_por_anio = dict(
            con.execute(
                f"SELECT substr(dteday, 1, 4) AS anio, COUNT(*) FROM {TABLA} "
                "GROUP BY anio ORDER BY anio"
            ).fetchall()
        )
        (dias_cortos,) = con.execute(
            f"SELECT COUNT(*) FROM (SELECT dteday, COUNT(*) AS n FROM {TABLA} "
            "GROUP BY dteday) WHERE n < 24"
        ).fetchone()
    return filas_por_anio, dias_cortos


def main() -> None:
    """Punto de entrada de ``make fuente``: crea la base e imprime el resumen."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, default=Path("data/raw/hour.csv"))
    parser.add_argument("--db", type=Path, default=Path("fuente/bicis.db"))
    args = parser.parse_args()

    total = crear_fuente(args.csv, args.db)
    filas_por_anio, dias_cortos = resumir(args.db)

    print(f"Fuente creada en {args.db} con {total:,} filas en la tabla {TABLA}.")
    for anio, filas in filas_por_anio.items():
        print(f"  {anio}: {filas:,} filas")
    print(f"Días con menos de 24 horas: {dias_cortos}")


if __name__ == "__main__":
    main()
