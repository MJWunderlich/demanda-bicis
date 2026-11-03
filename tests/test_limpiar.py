from pathlib import Path

import pandas as pd
import pytest

from bicis.extraer import extraer
from bicis.limpiar import limpiar
from conftest import FILA_SANDY, COLUMNAS, dia_completo


def _parquet(tmp_path: Path, df: pd.DataFrame) -> Path:
    ruta = tmp_path / "extraido.parquet"
    df.to_parquet(ruta, index=False)
    return ruta


def test_dia_completo_pasa(fuente_db: Path, dir_trabajo: Path):
    ruta = limpiar(extraer("2011-01-01", fuente_db, dir_trabajo))
    assert Path(ruta).name == "limpio.parquet"
    assert len(pd.read_parquet(ruta)) == 24


def test_falla_en_dia_vacio(tmp_path: Path):
    with pytest.raises(ValueError, match="no tiene filas"):
        limpiar(_parquet(tmp_path, dia_completo("2011-01-01").iloc[0:0]))


def test_falla_si_faltan_columnas(tmp_path: Path):
    df = dia_completo("2011-01-01").drop(columns=["hum", "windspeed"])
    with pytest.raises(ValueError, match="Faltan columnas"):
        limpiar(_parquet(tmp_path, df))


def test_falla_con_hr_24(tmp_path: Path):
    df = dia_completo("2011-01-01")
    df.loc[0, "hr"] = 24
    with pytest.raises(ValueError, match="hr fuera de 0-23"):
        limpiar(_parquet(tmp_path, df))


def test_falla_con_variable_normalizada_fuera_de_rango(tmp_path: Path):
    df = dia_completo("2011-01-01")
    df.loc[5, "hum"] = 1.5
    with pytest.raises(ValueError, match="hum fuera de"):
        limpiar(_parquet(tmp_path, df))


def test_falla_con_horas_duplicadas(tmp_path: Path):
    df = dia_completo("2011-01-01")
    df.loc[1, "hr"] = 0
    with pytest.raises(ValueError, match="Horas duplicadas"):
        limpiar(_parquet(tmp_path, df))


def test_acepta_el_dia_de_sandy_con_una_sola_hora(fuente_db: Path, dir_trabajo: Path):
    ruta = limpiar(extraer("2012-10-29", fuente_db, dir_trabajo))
    df = pd.read_parquet(ruta)
    assert len(df) == 1
    assert df.iloc[0]["hr"] == FILA_SANDY[COLUMNAS.index("hr")]


FUENTE_REAL = Path(__file__).resolve().parents[1] / "fuente" / "bicis.db"


@pytest.mark.skipif(not FUENTE_REAL.exists(), reason="requiere make fuente")
def test_acepta_sandy_en_la_fuente_real(dir_trabajo: Path):
    df = pd.read_parquet(limpiar(extraer("2012-10-29", FUENTE_REAL, dir_trabajo)))
    assert len(df) == 1
