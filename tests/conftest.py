"""Fixtures compartidas: una fuente SQLite temporal con días sintéticos."""

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

COLUMNAS = [
    "instant", "dteday", "season", "yr", "mnth", "hr", "holiday", "weekday",
    "workingday", "weathersit", "temp", "atemp", "hum", "windspeed",
    "casual", "registered", "cnt",
]

# Fila real de hour.csv: 2012-10-29, la única hora registrada ese día (huracán Sandy).
FILA_SANDY = [15884, "2012-10-29", 4, 1, 10, 0, 0, 1, 1, 3, 0.44, 0.4394, 0.88, 0.3582, 2, 20, 22]


def dia_completo(fecha: str, instant_inicial: int = 1, horas: int = 24) -> pd.DataFrame:
    """Construye un día sintético con ``horas`` filas consecutivas desde hr=0."""
    filas = []
    for i in range(horas):
        filas.append(
            [instant_inicial + i, fecha, 1, 0, 1, i, 0, 6, 0, 1,
             0.24, 0.2879, 0.81, 0.0, 3 + i, 13 + i, 16 + 2 * i]
        )
    return pd.DataFrame(filas, columns=COLUMNAS)


@pytest.fixture
def fuente_db(tmp_path: Path) -> Path:
    """Fuente con dos días completos (2011-01-01, 2011-01-02) y el día de Sandy."""
    df = pd.concat(
        [
            dia_completo("2011-01-01", 1),
            dia_completo("2011-01-02", 25),
            pd.DataFrame([FILA_SANDY], columns=COLUMNAS),
        ],
        ignore_index=True,
    )
    ruta = tmp_path / "fuente" / "bicis.db"
    ruta.parent.mkdir()
    with sqlite3.connect(ruta) as con:
        df.to_sql("demanda_hora", con, index=False)
        con.execute("CREATE INDEX idx_demanda_hora_dteday ON demanda_hora(dteday)")
    return ruta


@pytest.fixture
def dir_trabajo(tmp_path: Path) -> Path:
    return tmp_path / "trabajo"
