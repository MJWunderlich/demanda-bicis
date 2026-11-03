from pathlib import Path

import pandas as pd
import pytest

from bicis.extraer import extraer


def test_devuelve_solo_las_filas_de_la_fecha(fuente_db: Path, dir_trabajo: Path):
    ruta = extraer("2011-01-02", fuente_db, dir_trabajo)

    df = pd.read_parquet(ruta)
    assert Path(ruta) == dir_trabajo / "2011-01-02" / "extraido.parquet"
    assert len(df) == 24
    assert df["dteday"].unique().tolist() == ["2011-01-02"]
    assert df["hr"].tolist() == list(range(24))


def test_un_dia_ausente_produce_cero_filas(fuente_db: Path, dir_trabajo: Path):
    df = pd.read_parquet(extraer("2013-01-01", fuente_db, dir_trabajo))
    assert len(df) == 0


def test_falla_si_no_existe_la_fuente(tmp_path: Path, dir_trabajo: Path):
    with pytest.raises(FileNotFoundError):
        extraer("2011-01-01", tmp_path / "no_existe.db", dir_trabajo)


def test_rechaza_fechas_mal_formadas(fuente_db: Path, dir_trabajo: Path):
    with pytest.raises(ValueError):
        extraer("01/01/2011", fuente_db, dir_trabajo)
