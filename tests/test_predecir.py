from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bicis import predecir as mod
from bicis.extraer import extraer
from bicis.limpiar import limpiar
from bicis.transformar import FEATURES, transformar


class ModeloFalso:
    """Imita un modelo pyfunc: exige las columnas de la firma y devuelve valores fijos."""

    def __init__(self, valores):
        self.valores = np.asarray(valores, dtype=float)

    def predict(self, X: pd.DataFrame):
        if list(X.columns) != FEATURES:
            raise RuntimeError("Model is missing inputs")
        return self.valores[: len(X)]


@pytest.fixture
def features(fuente_db: Path, dir_trabajo: Path) -> Path:
    return Path(transformar(limpiar(extraer("2011-01-01", fuente_db, dir_trabajo))))


def _falso(monkeypatch, valores, factor=0.10, version="7"):
    monkeypatch.setattr(
        mod,
        "cargar_champion",
        lambda model_uri=mod.MODEL_URI_CHAMPION: mod.ModeloCargado(
            modelo=ModeloFalso(valores), version_modelo=version, factor=factor
        ),
    )


def test_escribe_las_columnas_del_contrato(features: Path, monkeypatch):
    _falso(monkeypatch, valores=np.full(24, 100.0), factor=0.10, version="7")

    ruta = mod.predecir(features)

    df = pd.read_parquet(ruta)
    assert Path(ruta).name == "predicciones.parquet"
    assert df.columns.tolist() == ["fecha", "hr", "prediccion", "factor", "plan", "version_modelo"]
    assert len(df) == 24
    assert df["factor"].unique().tolist() == [0.10]
    assert df["version_modelo"].unique().tolist() == ["7"]
    # 100 * 1.10 = 110.00000000000001 -> ceil = 111; el redondeo hacia arriba es el contrato.
    assert df["plan"].unique().tolist() == [111]


def test_recorta_predicciones_negativas_y_redondea_hacia_arriba(features: Path, monkeypatch):
    valores = np.concatenate([[-5.0, 0.0, 10.2], np.full(21, 1.0)])
    _falso(monkeypatch, valores=valores, factor=0.0)

    df = pd.read_parquet(mod.predecir(features))

    assert df["prediccion"].iloc[:3].tolist() == [0.0, 0.0, 10.2]
    assert df["plan"].iloc[:3].tolist() == [0, 0, 11]


def test_calcular_plan():
    plan = mod.calcular_plan(np.array([-3.0, 0.0, 10.0, 10.4]), factor=0.20)
    assert plan.tolist() == [0, 0, 12, 13]


def test_solo_acepta_alias_del_registry():
    for uri in ["models:/demanda-bicis/3", "/tmp/modelo", "runs:/abc/model"]:
        with pytest.raises(ValueError, match="alias"):
            mod.cargar_champion(uri)


def test_falla_si_la_firma_no_coincide(features: Path, monkeypatch):
    _falso(monkeypatch, valores=np.full(24, 1.0))
    df = pd.read_parquet(features).drop(columns=["windspeed"])
    ruta = features.with_name("features_rotas.parquet")
    df.to_parquet(ruta, index=False)

    with pytest.raises(KeyError):
        mod.predecir(ruta)


def test_falla_si_el_modelo_no_existe(features: Path, monkeypatch):
    def sin_modelo(model_uri=mod.MODEL_URI_CHAMPION):
        raise LookupError("Registered model alias champion not found")

    monkeypatch.setattr(mod, "cargar_champion", sin_modelo)
    with pytest.raises(LookupError):
        mod.predecir(features)
