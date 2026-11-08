"""Compuerta de promoción: decide si un modelo nuevo reemplaza al campeón.

Vocabulario:

- **Campeón** (*champion*): el modelo que hoy está en producción, el que lleva
  el alias ``@champion`` en el Model Registry de MLflow y el que carga
  ``predecir``.
- **Retador** (*challenger*): un modelo recién entrenado que aspira a
  reemplazarlo.
- **Compuerta** (*gate*): una lista de reglas que el retador tiene que pasar,
  en orden, antes de recibir el alias. Si falla una, se detiene ahí y se
  devuelve el motivo, para que quede registrado por qué no se promovió.

Esta función es **pura**: recibe diccionarios y umbrales, no importa mlflow ni
lee archivos, no tiene efectos secundarios. Quien la llama (un DAG de
reentrenamiento, un script) se encarga de leer las métricas de MLflow y de
mover el alias si la respuesta es sí.

Reglas, en este orden (sección Fase 6 del diseño):

1. **Features idénticas**: la misma lista, en el mismo orden. Un modelo con
   otras features no es comparable y además rompería la firma que valida
   ``predecir``.
2. **Latencia**: ``latencia_p95_ms`` del retador no supera el umbral
   (percentil 95 del tiempo de predicción de un día, en milisegundos).
3. **Tamaño**: ``tamano_modelo_mb`` del retador no supera el umbral.
4. **Mejora de costo**: el ``costo_hora_eval`` del retador es al menos un
   porcentaje mínimo más bajo que el del campeón. ``costo_hora_eval`` es el
   costo por hora (``costos.costo_hora``) calculado sobre el mismo mes de
   evaluación para los dos modelos.

Los umbrales viven en ``params.yaml`` bajo ``compuerta``.
"""

from __future__ import annotations

from pathlib import Path

RUTA_PARAMS = Path("params.yaml")

CLAVES_METRICAS = ("features", "latencia_p95_ms", "tamano_modelo_mb", "costo_hora_eval")
CLAVES_UMBRALES = ("latencia_p95_ms_max", "tamano_modelo_mb_max", "mejora_minima_costo")


def cargar_umbrales(ruta: str | Path = RUTA_PARAMS) -> dict[str, float]:
    """Lee la sección ``compuerta`` de ``params.yaml`` y valida que estén las tres claves."""
    import yaml

    datos = yaml.safe_load(Path(ruta).read_text())
    umbrales = datos.get("compuerta") if isinstance(datos, dict) else None
    if not isinstance(umbrales, dict):
        raise ValueError(f"{ruta} no tiene la sección 'compuerta'")
    faltan = [c for c in CLAVES_UMBRALES if c not in umbrales]
    if faltan:
        raise ValueError(f"A la sección 'compuerta' de {ruta} le faltan: {', '.join(faltan)}")
    return {c: float(umbrales[c]) for c in CLAVES_UMBRALES}


def _lista_features(valor) -> list[str]:
    """Acepta una lista o el texto ``"a,b,c"`` (así se guarda el tag en MLflow)."""
    if isinstance(valor, str):
        return [f.strip() for f in valor.split(",") if f.strip()]
    return [str(f) for f in valor]


def _validar(nombre: str, modelo: dict, claves=CLAVES_METRICAS) -> None:
    if not isinstance(modelo, dict):
        raise ValueError(f"{nombre} debe ser un dict con {', '.join(claves)}")
    faltan = [c for c in claves if c not in modelo]
    if faltan:
        raise ValueError(f"Al {nombre} le faltan: {', '.join(faltan)}")


def compuerta(challenger: dict, champion: dict, umbrales: dict) -> tuple[bool, str]:
    """Aplica las cuatro reglas en orden y devuelve ``(promover, motivo)``.

    ``challenger`` y ``champion`` traen ``features`` (lista o texto separado por
    comas), ``latencia_p95_ms``, ``tamano_modelo_mb`` y ``costo_hora_eval``.
    ``umbrales`` trae ``latencia_p95_ms_max``, ``tamano_modelo_mb_max`` y
    ``mejora_minima_costo`` (fracción: ``0.02`` es 2 %).

    El motivo describe la primera regla que falla, con los números que la
    hicieron fallar; si pasan todas, describe la mejora conseguida. Las
    comparaciones son inclusivas: un retador exactamente en el umbral pasa.
    """
    _validar("challenger", challenger)
    _validar("champion", champion)
    _validar("umbrales", umbrales, CLAVES_UMBRALES)

    f_ret, f_cam = _lista_features(challenger["features"]), _lista_features(champion["features"])
    if f_ret != f_cam:
        return False, f"features distintas: challenger {f_ret} vs champion {f_cam}"

    latencia = float(challenger["latencia_p95_ms"])
    if latencia > umbrales["latencia_p95_ms_max"]:
        return False, (
            f"latencia_p95_ms {latencia:.1f} supera el máximo {umbrales['latencia_p95_ms_max']:.0f}"
        )

    tamano = float(challenger["tamano_modelo_mb"])
    if tamano > umbrales["tamano_modelo_mb_max"]:
        return False, (
            f"tamano_modelo_mb {tamano:.2f} supera el máximo {umbrales['tamano_modelo_mb_max']:.0f}"
        )

    costo_ret, costo_cam = float(challenger["costo_hora_eval"]), float(champion["costo_hora_eval"])
    if costo_cam <= 0:
        raise ValueError(f"costo_hora_eval del champion debe ser positivo, llegó {costo_cam}")
    mejora = (costo_cam - costo_ret) / costo_cam
    minima = float(umbrales["mejora_minima_costo"])
    if mejora < minima:
        return False, (
            f"mejora de costo_hora_eval {mejora:+.1%} ({costo_cam:.2f} → {costo_ret:.2f}) "
            f"no alcanza el mínimo {minima:.0%}"
        )

    return True, (
        f"promover: costo_hora_eval {costo_cam:.2f} → {costo_ret:.2f} ({mejora:+.1%}), "
        f"latencia_p95_ms {latencia:.1f}, tamano_modelo_mb {tamano:.2f}"
    )
