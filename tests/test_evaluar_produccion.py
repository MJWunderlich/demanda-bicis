"""Pruebas de scripts/evaluar_produccion.py sin MLflow (la validación se omite)."""

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

import evaluar_produccion as ev
from conftest import dia_completo

COSTOS = {"faltante": 3.0, "sobrante": 1.0}


def _fuente_y_predicciones(tmp_path: Path) -> tuple[Path, Path]:
    """Fuente con dos días de 2012 y predicciones solo para el primero (24 h) y mitad del segundo."""
    fuente = tmp_path / "bicis.db"
    dias = pd.concat([dia_completo("2012-01-01", 1), dia_completo("2012-01-02", 25)], ignore_index=True)
    dias["cnt"] = 100  # demanda real constante
    with sqlite3.connect(fuente) as con:
        dias.to_sql("demanda_hora", con, index=False)

    predicciones = tmp_path / "predicciones.db"
    filas = [("2012-01-01", h, 90.0, 0.10, 99, "1", "t") for h in range(24)]
    filas += [("2012-01-02", h, 120.0, 0.10, 132, "1", "t") for h in range(12)]
    filas += [("2012-01-03", 0, 50.0, 0.10, 55, "1", "t")]  # sin real: no se evalúa
    with sqlite3.connect(predicciones) as con:
        con.execute(
            "CREATE TABLE predicciones (fecha TEXT, hr INTEGER, prediccion REAL, factor REAL, "
            "plan INTEGER, version_modelo TEXT, ejecutado_en TEXT, PRIMARY KEY (fecha, hr))"
        )
        con.executemany("INSERT INTO predicciones VALUES (?, ?, ?, ?, ?, ?, ?)", filas)
    return fuente, predicciones


def test_rango_del_mes():
    assert ev.rango_del_mes("2012-01") == ("2012-01-01", "2012-01-31")
    assert ev.rango_del_mes("2012-02") == ("2012-02-01", "2012-02-29")
    assert ev.nombre_mes("2012-02-10") == "febrero 2012"


def test_leer_produccion_une_solo_horas_con_real(tmp_path):
    fuente, predicciones = _fuente_y_predicciones(tmp_path)

    df = ev.leer_produccion(predicciones, fuente, "2012-01-01", "2012-01-31")

    assert len(df) == 36  # 24 + 12; la hora del 03 no tiene real
    assert list(df.columns) == ev.COLUMNAS_EVALUACION + ["version_modelo", "factor"]
    assert df["fecha"].tolist() == ["2012-01-01"] * 24 + ["2012-01-02"] * 12
    assert (df["real"] == 100).all()


def test_leer_produccion_falla_sin_filas(tmp_path):
    fuente, predicciones = _fuente_y_predicciones(tmp_path)
    with pytest.raises(ValueError, match="No hay predicciones"):
        ev.leer_produccion(predicciones, fuente, "2012-03-01", "2012-03-31")


def test_metricas_y_mae_por_dia(tmp_path):
    fuente, predicciones = _fuente_y_predicciones(tmp_path)
    df = ev.leer_produccion(predicciones, fuente, "2012-01-01", "2012-01-31")

    m = ev.metricas(df, COSTOS)

    # 24 h con error -10 y 12 h con error +20: MAE = (240 + 240) / 36
    assert m["filas"] == 36
    assert m["demanda_media"] == 100
    assert m["mae"] == pytest.approx((24 * 10 + 12 * 20) / 36)
    assert m["sesgo"] == pytest.approx((24 * -10 + 12 * 20) / 36)
    # plan 99 vs 100: falta 1 → 3; plan 132 vs 100: sobran 32 → 32
    assert m["costo_hora"] == pytest.approx((24 * 3 + 12 * 32) / 36)
    assert ev.mae_por_dia(df).to_dict() == {"2012-01-01": 10.0, "2012-01-02": 20.0}


def test_tabla_markdown_con_cambio():
    tabla = ev.tabla_markdown(
        {
            "dic (validación)": {"filas": 10, "demanda_media": 100.0, "mae": 30.0, "sesgo": 1.0, "costo_hora": 40.0},
            "ene (producción)": {"filas": 12, "demanda_media": 150.0, "mae": 60.0, "sesgo": -50.0, "costo_hora": 100.0},
        }
    )
    lineas = tabla.splitlines()
    assert lineas[0] == "| métrica | dic (validación) | ene (producción) | cambio |"
    assert "| MAE | 30.00 | 60.00 | +30.00 (+100%) |" in lineas
    assert "| sesgo (predicción − real) | +1.00 | -50.00 | -51.00 |" in lineas
    assert "| filas (horas) | 10 | 12 | +2 |" in lineas


def test_graficar_escribe_png(tmp_path):
    serie = pd.Series([10.0, 20.0, 15.0], index=["2012-01-01", "2012-01-02", "2012-01-03"])
    ruta = ev.graficar(serie, 12.0, "prueba", tmp_path / "sub" / "deriva.png")
    assert Path(ruta).stat().st_size > 1000


def test_main_sin_validacion(tmp_path, capsys):
    fuente, predicciones = _fuente_y_predicciones(tmp_path)
    costos = tmp_path / "costos.yaml"
    costos.write_text("faltante: 3\nsobrante: 1\n")

    codigo = ev.main(
        [
            "--mes", "2012-01",
            "--predicciones", str(predicciones),
            "--fuente", str(fuente),
            "--costos", str(costos),
            "--figura", str(tmp_path / "deriva.png"),
            "--tabla", str(tmp_path / "deriva.md"),
            "--sin-validacion",
        ]
    )

    assert codigo == 0
    salida = capsys.readouterr().out
    assert "| métrica | enero 2012 (producción) |" in salida
    assert "Campeón: 1 (modelo registrado/versión)" in salida
    assert (tmp_path / "deriva.png").exists()
    assert (tmp_path / "deriva.md").read_text().count("\n") >= 7
