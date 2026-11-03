import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from bicis.almacenar import almacenar


def _predicciones(tmp_path: Path, fecha: str, horas: int = 24, version: str = "1") -> Path:
    df = pd.DataFrame(
        {
            "fecha": fecha,
            "hr": range(horas),
            "prediccion": [100.0 + h for h in range(horas)],
            "factor": 0.10,
            "plan": [111 + h for h in range(horas)],
            "version_modelo": version,
        }
    )
    ruta = tmp_path / fecha / "predicciones.parquet"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ruta, index=False)
    return ruta


def _filas(ruta_db: Path, fecha: str | None = None) -> list[tuple]:
    with sqlite3.connect(ruta_db) as con:
        if fecha is None:
            return con.execute("SELECT fecha, hr, version_modelo FROM predicciones ORDER BY fecha, hr").fetchall()
        return con.execute(
            "SELECT fecha, hr, version_modelo FROM predicciones WHERE fecha = ? ORDER BY hr", (fecha,)
        ).fetchall()


def test_escribe_las_filas_del_dia_y_crea_la_tabla(tmp_path: Path):
    ruta_db = tmp_path / "salidas" / "predicciones.db"

    escritas = almacenar(_predicciones(tmp_path, "2012-01-02"), ruta_db)

    assert escritas == 24
    with sqlite3.connect(ruta_db) as con:
        columnas = [c[1] for c in con.execute("PRAGMA table_info(predicciones)")]
        ejecutado_en = con.execute("SELECT DISTINCT ejecutado_en FROM predicciones").fetchall()
    assert columnas == ["fecha", "hr", "prediccion", "factor", "plan", "version_modelo", "ejecutado_en"]
    assert len(ejecutado_en) == 1


def test_es_idempotente_dos_veces_deja_24_filas(tmp_path: Path):
    ruta_db = tmp_path / "predicciones.db"
    ruta = _predicciones(tmp_path, "2012-01-02", version="1")

    almacenar(ruta, ruta_db)
    almacenar(_predicciones(tmp_path, "2012-01-02", version="2"), ruta_db)

    filas = _filas(ruta_db, "2012-01-02")
    assert len(filas) == 24, "la segunda corrida reemplaza, no agrega"
    assert {f[2] for f in filas} == {"2"}, "las filas son las de la última corrida"


def test_no_toca_las_filas_de_otro_dia(tmp_path: Path):
    ruta_db = tmp_path / "predicciones.db"
    almacenar(_predicciones(tmp_path, "2012-01-02", version="1"), ruta_db)
    antes = _filas(ruta_db, "2012-01-02")

    almacenar(_predicciones(tmp_path, "2012-01-03", horas=23, version="2"), ruta_db)
    almacenar(_predicciones(tmp_path, "2012-01-03", horas=23, version="3"), ruta_db)

    assert _filas(ruta_db, "2012-01-02") == antes
    assert len(_filas(ruta_db, "2012-01-03")) == 23
    assert len(_filas(ruta_db)) == 47


def test_rechaza_un_parquet_con_dos_fechas(tmp_path: Path):
    df = pd.concat(
        [pd.read_parquet(_predicciones(tmp_path, "2012-01-02")),
         pd.read_parquet(_predicciones(tmp_path, "2012-01-03"))]
    )
    ruta = tmp_path / "mezcla.parquet"
    df.to_parquet(ruta, index=False)
    with pytest.raises(ValueError, match="un solo día"):
        almacenar(ruta, tmp_path / "predicciones.db")
