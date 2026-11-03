"""Costo de un plan de bicicletas y elección del factor de ajuste.

El "costo por hora" mide, en unidades de costo, cuánto se equivocó el plan
respecto de la demanda real, promediado sobre las horas evaluadas. Faltar
una bicicleta (demanda mayor que el plan) y que sobre una (plan mayor que
la demanda) no cuestan lo mismo: los dos precios viven en
``configs/costos.yaml`` y nunca se escriben en el código.

    costo_hora = promedio_h [ faltante · max(cnt_h − plan_h, 0)
                            + sobrante · max(plan_h − cnt_h, 0) ]

El "factor de ajuste" es el porcentaje que se suma a la predicción antes
de redondear hacia arriba para formar el plan. Como faltar cuesta más que
sobrar, conviene planear por encima de la predicción; cuánto por encima
se decide en validación: se prueba una grilla de factores y se queda el
que minimiza el costo por hora. El plan con factor ``f`` es
``ceil(max(pred, 0) · (1 + f))``, el mismo cálculo que hace ``predecir``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

RUTA_COSTOS = Path("configs/costos.yaml")


def cargar_costos(ruta: str | Path = RUTA_COSTOS) -> dict[str, float]:
    """Lee ``faltante`` y ``sobrante`` del YAML de costos.

    Args:
        ruta: Ruta al archivo YAML con las claves ``faltante`` y ``sobrante``.

    Returns:
        Diccionario ``{"faltante": float, "sobrante": float}``.

    Raises:
        KeyError: Si falta alguna de las dos claves.
    """
    datos = yaml.safe_load(Path(ruta).read_text())
    return {"faltante": float(datos["faltante"]), "sobrante": float(datos["sobrante"])}


def costo_hora(real: np.ndarray, plan: np.ndarray, costos: dict[str, float]) -> float:
    """Costo promedio por hora de un plan frente a la demanda real.

    Args:
        real: Demanda real (``cnt``) por hora.
        plan: Bicicletas planificadas por hora, mismo largo que ``real``.
        costos: Diccionario con ``faltante`` y ``sobrante``.

    Returns:
        Promedio sobre las horas de ``faltante·max(real−plan,0) + sobrante·max(plan−real,0)``.
    """
    real = np.asarray(real, dtype=float)
    plan = np.asarray(plan, dtype=float)
    faltan = np.maximum(real - plan, 0.0)
    sobran = np.maximum(plan - real, 0.0)
    return float(np.mean(costos["faltante"] * faltan + costos["sobrante"] * sobran))


def plan_con_factor(prediccion: np.ndarray, factor: float) -> np.ndarray:
    """Plan de bicicletas para una predicción y un factor: ``ceil(max(pred, 0)·(1 + f))``.

    El producto se redondea a 6 decimales antes del ``ceil`` para que el ruido
    de punto flotante (100 · 1.10 = 110.00000000000001) no sume una bicicleta.
    """
    base = np.maximum(np.asarray(prediccion, dtype=float), 0.0) * (1.0 + factor)
    return np.ceil(np.round(base, 6))


def grilla_factores(minimo: float = 0.0, maximo: float = 0.40, paso: float = 0.05) -> list[float]:
    """Factores candidatos de ``minimo`` a ``maximo`` inclusive, en pasos de ``paso``."""
    n = int(round((maximo - minimo) / paso)) + 1
    return [round(minimo + i * paso, 4) for i in range(n)]


def elegir_factor(
    real: np.ndarray,
    prediccion: np.ndarray,
    costos: dict[str, float],
    grilla: list[float] | None = None,
) -> tuple[float, dict[float, float]]:
    """Elige el factor de la grilla que minimiza el costo por hora en validación.

    Args:
        real: Demanda real del conjunto de validación.
        prediccion: Predicción del modelo para las mismas horas.
        costos: Diccionario con ``faltante`` y ``sobrante``.
        grilla: Factores candidatos; por defecto 0.00 a 0.40 en pasos de 0.05.

    Returns:
        El factor elegido y la curva completa ``{factor: costo_hora}``, útil
        para graficarla y guardarla como artefacto.
    """
    if grilla is None:
        grilla = grilla_factores()
    curva = {f: costo_hora(real, plan_con_factor(prediccion, f), costos) for f in grilla}
    # Ante empate gana el factor más pequeño, porque la grilla está ordenada.
    elegido = min(grilla, key=lambda f: curva[f])
    return elegido, curva
