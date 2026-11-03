from pathlib import Path

import pandas as pd
import pytest

from bicis.extraer import extraer
from bicis.limpiar import limpiar
from bicis.transformar import FEATURES, PROHIBIDAS, transformar
from conftest import dia_completo


def test_produce_exactamente_fecha_hr_y_features(fuente_db: Path, dir_trabajo: Path):
    ruta = transformar(limpiar(extraer("2011-01-01", fuente_db, dir_trabajo)))

    df = pd.read_parquet(ruta)
    assert Path(ruta).name == "features.parquet"
    assert set(df.columns) == {"fecha", "hr", *FEATURES}
    assert df.columns.tolist() == ["fecha", *FEATURES]
    assert len(df) == 24
    assert df["fecha"].unique().tolist() == ["2011-01-01"]


def test_nunca_incluye_columnas_prohibidas(fuente_db: Path, dir_trabajo: Path):
    df = pd.read_parquet(transformar(limpiar(extraer("2011-01-01", fuente_db, dir_trabajo))))
    for col in ["casual", "registered", "instant", "yr", "dteday", "cnt"]:
        assert col not in df.columns
        assert col in PROHIBIDAS
        assert col not in FEATURES


def test_features_son_las_once_del_diseno():
    assert FEATURES == [
        "hr", "weekday", "workingday", "holiday", "season", "mnth",
        "weathersit", "temp", "atemp", "hum", "windspeed",
    ]


def test_falla_si_falta_una_feature(tmp_path: Path):
    df = dia_completo("2011-01-01").drop(columns=["atemp"])
    ruta = tmp_path / "limpio.parquet"
    df.to_parquet(ruta, index=False)
    with pytest.raises(ValueError, match="atemp"):
        transformar(ruta)
