"""Pruebas de la compuerta: un caso que pasa y uno que falla por cada regla."""

from pathlib import Path

import pytest

from bicis.compuerta import cargar_umbrales, compuerta
from bicis.transformar import FEATURES

REPO = Path(__file__).resolve().parents[1]
UMBRALES = {"latencia_p95_ms_max": 200, "tamano_modelo_mb_max": 100, "mejora_minima_costo": 0.02}


def champion(**cambios) -> dict:
    base = {"features": list(FEATURES), "latencia_p95_ms": 28.1, "tamano_modelo_mb": 1.07, "costo_hora_eval": 100.0}
    return {**base, **cambios}


def challenger(**cambios) -> dict:
    return champion(**{"costo_hora_eval": 90.0, **cambios})


def test_pasa_todas_las_reglas():
    promover, motivo = compuerta(challenger(), champion(), UMBRALES)
    assert promover is True
    assert "promover" in motivo and "+10.0%" in motivo


def test_pasa_justo_en_los_umbrales():
    reto = challenger(latencia_p95_ms=200, tamano_modelo_mb=100, costo_hora_eval=98.0)
    assert compuerta(reto, champion(), UMBRALES) == (
        True,
        "promover: costo_hora_eval 100.00 → 98.00 (+2.0%), latencia_p95_ms 200.0, tamano_modelo_mb 100.00",
    )


def test_regla_1_features_distintas():
    promover, motivo = compuerta(challenger(features=FEATURES[:-1]), champion(), UMBRALES)
    assert promover is False
    assert motivo.startswith("features distintas")


def test_regla_1_mismo_conjunto_otro_orden_falla():
    al_reves = list(reversed(FEATURES))
    promover, motivo = compuerta(challenger(features=al_reves), champion(), UMBRALES)
    assert promover is False
    assert "features distintas" in motivo


def test_regla_1_acepta_el_tag_de_mlflow_como_texto():
    reto = challenger(features=",".join(FEATURES))
    assert compuerta(reto, champion(), UMBRALES)[0] is True


def test_regla_2_latencia():
    promover, motivo = compuerta(challenger(latencia_p95_ms=200.1), champion(), UMBRALES)
    assert promover is False
    assert motivo == "latencia_p95_ms 200.1 supera el máximo 200"


def test_regla_3_tamano():
    # El random forest de la escalera (203 MB) falla aquí aunque tenga mejor costo.
    promover, motivo = compuerta(challenger(tamano_modelo_mb=203.46, costo_hora_eval=50.0), champion(), UMBRALES)
    assert promover is False
    assert motivo == "tamano_modelo_mb 203.46 supera el máximo 100"


def test_regla_4_mejora_insuficiente():
    promover, motivo = compuerta(challenger(costo_hora_eval=98.5), champion(), UMBRALES)
    assert promover is False
    assert motivo.startswith("mejora de costo_hora_eval +1.5%")
    assert "no alcanza el mínimo 2%" in motivo


def test_regla_4_empeora():
    promover, motivo = compuerta(challenger(costo_hora_eval=120.0), champion(), UMBRALES)
    assert promover is False
    assert "-20.0%" in motivo


def test_las_reglas_se_evaluan_en_orden():
    # Falla por features aunque también fallaría por latencia, tamaño y costo.
    reto = challenger(features=["hr"], latencia_p95_ms=999, tamano_modelo_mb=999, costo_hora_eval=999)
    assert compuerta(reto, champion(), UMBRALES)[1].startswith("features distintas")


def test_falta_una_metrica():
    reto = challenger()
    del reto["tamano_modelo_mb"]
    with pytest.raises(ValueError, match="challenger le faltan: tamano_modelo_mb"):
        compuerta(reto, champion(), UMBRALES)


def test_champion_con_costo_cero():
    with pytest.raises(ValueError, match="positivo"):
        compuerta(challenger(), champion(costo_hora_eval=0), UMBRALES)


def test_cargar_umbrales_de_params_yaml():
    umbrales = cargar_umbrales(REPO / "params.yaml")
    assert umbrales == {"latencia_p95_ms_max": 200.0, "tamano_modelo_mb_max": 100.0, "mejora_minima_costo": 0.02}


def test_cargar_umbrales_sin_seccion(tmp_path):
    ruta = tmp_path / "params.yaml"
    ruta.write_text("semilla: 42\n")
    with pytest.raises(ValueError, match="no tiene la sección 'compuerta'"):
        cargar_umbrales(ruta)
    ruta.write_text("compuerta:\n  latencia_p95_ms_max: 200\n")
    with pytest.raises(ValueError, match="faltan: tamano_modelo_mb_max, mejora_minima_costo"):
        cargar_umbrales(ruta)
