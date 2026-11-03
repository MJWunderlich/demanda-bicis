from pathlib import Path

import numpy as np

from bicis.costos import (
    cargar_costos,
    costo_hora,
    elegir_factor,
    grilla_factores,
    plan_con_factor,
)

COSTOS = {"faltante": 3.0, "sobrante": 1.0}
RUTA_COSTOS_REPO = Path(__file__).resolve().parents[1] / "configs" / "costos.yaml"


def test_costo_cero_cuando_el_plan_iguala_lo_real():
    real = np.array([10, 20, 30])
    assert costo_hora(real, real, COSTOS) == 0.0


def test_un_faltante_cuesta_tres_veces_un_sobrante():
    real = np.array([10.0])
    falta_una = costo_hora(real, np.array([9.0]), COSTOS)
    sobra_una = costo_hora(real, np.array([11.0]), COSTOS)
    assert falta_una == 3.0
    assert sobra_una == 1.0
    assert falta_una == 3 * sobra_una


def test_costo_es_promedio_por_hora():
    real = np.array([10.0, 10.0])
    plan = np.array([8.0, 12.0])  # faltan 2 (6) y sobran 2 (2): 8 / 2 horas
    assert costo_hora(real, plan, COSTOS) == 4.0


def test_cargar_costos_lee_el_yaml_del_repo():
    assert cargar_costos(RUTA_COSTOS_REPO) == {"faltante": 3.0, "sobrante": 1.0}


def test_grilla_va_de_0_a_040_en_pasos_de_005():
    grilla = grilla_factores()
    assert grilla[0] == 0.0 and grilla[-1] == 0.40 and len(grilla) == 9
    assert all(abs(b - a - 0.05) < 1e-9 for a, b in zip(grilla, grilla[1:]))


def test_plan_con_factor_recorta_y_redondea_hacia_arriba():
    assert plan_con_factor(np.array([-2.0, 10.0, 10.4, 100.0]), 0.10).tolist() == [0.0, 11.0, 12.0, 110.0]


def test_elegir_factor_devuelve_un_valor_de_la_grilla_y_la_curva():
    rng = np.random.default_rng(0)
    prediccion = rng.uniform(50, 300, size=200)
    real = prediccion * 1.12  # el modelo subestima un 12%: conviene un factor cercano a 0.10-0.15

    factor, curva = elegir_factor(real, prediccion, COSTOS)

    assert factor in grilla_factores()
    assert set(curva) == set(grilla_factores())
    assert curva[factor] == min(curva.values())
    assert factor > 0.0


def test_elegir_factor_prefiere_cero_si_el_modelo_ya_sobreestima():
    prediccion = np.full(50, 100.0)
    real = np.full(50, 80.0)
    factor, _ = elegir_factor(real, prediccion, COSTOS)
    assert factor == 0.0
