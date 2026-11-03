"""Pruebas de scripts/crear_fuente.py sobre una fuente temporal de 3 filas."""

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from crear_fuente import INDICE, TABLA, crear_fuente, resumir

COLUMNAS = [
    "instant", "dteday", "season", "yr", "mnth", "hr", "holiday", "weekday",
    "workingday", "weathersit", "temp", "atemp", "hum", "windspeed",
    "casual", "registered", "cnt",
]


@pytest.fixture
def csv_tres_filas(tmp_path: Path) -> Path:
    """CSV con el esquema de hour.csv y 3 horas de un mismo día."""
    filas = [
        [1, "2011-01-01", 1, 0, 1, 0, 0, 6, 0, 1, 0.24, 0.2879, 0.81, 0.0, 3, 13, 16],
        [2, "2011-01-01", 1, 0, 1, 1, 0, 6, 0, 1, 0.22, 0.2727, 0.80, 0.0, 8, 32, 40],
        [3, "2011-01-01", 1, 0, 1, 2, 0, 6, 0, 1, 0.22, 0.2727, 0.80, 0.0, 5, 27, 32],
    ]
    ruta = tmp_path / "hour.csv"
    pd.DataFrame(filas, columns=COLUMNAS).to_csv(ruta, index=False)
    return ruta


def test_crea_tabla_con_esquema_original(csv_tres_filas: Path, tmp_path: Path):
    ruta_db = tmp_path / "fuente" / "bicis.db"

    escritas = crear_fuente(csv_tres_filas, ruta_db)

    assert escritas == 3
    with sqlite3.connect(ruta_db) as con:
        info = con.execute(f"PRAGMA table_info({TABLA})").fetchall()
        columnas = [fila[1] for fila in info]
        tipos = {fila[1]: fila[2].upper() for fila in info}
        indices = [fila[1] for fila in con.execute(f"PRAGMA index_list({TABLA})")]
        columnas_indice = [
            fila[2] for fila in con.execute(f"PRAGMA index_info({INDICE})")
        ]
        filas = con.execute(f"SELECT COUNT(*) FROM {TABLA}").fetchone()[0]

    assert columnas == COLUMNAS
    assert tipos["dteday"] == "TEXT"
    assert tipos["hr"] == "INTEGER"
    assert tipos["temp"] == "REAL"
    assert INDICE in indices
    assert columnas_indice == ["dteday"]
    assert filas == 3


def test_es_reejecutable(csv_tres_filas: Path, tmp_path: Path):
    ruta_db = tmp_path / "bicis.db"

    crear_fuente(csv_tres_filas, ruta_db)
    crear_fuente(csv_tres_filas, ruta_db)

    with sqlite3.connect(ruta_db) as con:
        filas = con.execute(f"SELECT COUNT(*) FROM {TABLA}").fetchone()[0]
    assert filas == 3, "la segunda corrida debe reemplazar la tabla, no duplicarla"


def test_resumen_cuenta_filas_por_anio_y_dias_cortos(csv_tres_filas: Path, tmp_path: Path):
    ruta_db = tmp_path / "bicis.db"
    crear_fuente(csv_tres_filas, ruta_db)

    filas_por_anio, dias_cortos = resumir(ruta_db)

    assert filas_por_anio == {"2011": 3}
    assert dias_cortos == 1


def test_falla_si_no_existe_el_csv(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        crear_fuente(tmp_path / "no_existe.csv", tmp_path / "bicis.db")
