"""Pruebas de las piezas de entrenar.py que no necesitan MLflow."""

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bicis import entrenar
from bicis.transformar import FEATURES
from conftest import dia_completo

REPO = Path(__file__).resolve().parents[1]


def test_perfil_promedia_por_hora_y_tipo_de_dia():
    X = pd.DataFrame({"hr": [8, 8, 8, 9], "workingday": [1, 1, 0, 1]})
    y = np.array([100.0, 200.0, 30.0, 50.0])

    modelo = entrenar.PerfilEstimator().fit(X, y)
    pred = modelo.predict(pd.DataFrame({"hr": [8, 8, 9, 23], "workingday": [1, 0, 1, 1]}))

    assert pred.tolist() == [150.0, 30.0, 50.0, 95.0]  # 23/1 no existe: promedio global


def test_construir_modelo_cubre_los_cuatro_escalones_y_acepta_features():
    train = pd.concat([dia_completo("2011-01-01", 1), dia_completo("2011-01-03", 25)], ignore_index=True)
    X, y = train[FEATURES], train["cnt"].to_numpy(dtype=float)
    hiper = {
        "perfil": {},
        "ridge": {"alpha": 1.0},
        "random-forest": {"n_estimators": 5, "max_depth": 3, "n_jobs": 1},
        "hgb": {"learning_rate": 0.1, "max_iter": 5},
    }
    for tipo, h in hiper.items():
        modelo = entrenar.construir_modelo(tipo, h, semilla=42).fit(X, y)
        assert entrenar.construir_modelo(tipo, h, 42).fit(X, y).predict(X).shape == (48,)
        assert modelo.predict(X).shape == (48,)
    with pytest.raises(ValueError):
        entrenar.construir_modelo("svm", {}, 42)


def test_huella_datos_es_determinista_y_sensible_a_los_datos():
    a = dia_completo("2011-01-01")
    b = a.copy()
    b.loc[0, "cnt"] += 1
    assert entrenar.huella_datos(a) == entrenar.huella_datos(a.copy())
    assert entrenar.huella_datos(a) != entrenar.huella_datos(b)
    assert len(entrenar.huella_datos(a, b)) == 64


def test_dvc_md5_lee_el_puntero_del_repo():
    md5 = entrenar.dvc_md5(REPO / "data" / "raw" / "hour.csv.dvc")
    assert len(md5) == 32


def test_git_commit_aborta_con_arbol_sucio(tmp_path: Path):
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q",
                    "--allow-empty", "-m", "inicial"], cwd=tmp_path, check=True)
    assert len(entrenar.git_commit(tmp_path)) == 40

    (tmp_path / "pendiente.txt").write_text("x")
    with pytest.raises(SystemExit, match="sin commit"):
        entrenar.git_commit(tmp_path)


def test_latencia_y_tamano_devuelven_numeros_positivos():
    X = dia_completo("2011-01-01")[FEATURES]
    modelo = entrenar.PerfilEstimator().fit(X, np.ones(24))
    assert entrenar.latencia_p95_ms(modelo, X, repeticiones=5) > 0
    assert entrenar.tamano_modelo_mb(modelo) > 0
