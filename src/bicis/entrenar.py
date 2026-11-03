"""Entrena la escalera de modelos, registra cada escalón en MLflow y promueve al campeón.

La "escalera" es una secuencia de modelos de complejidad creciente que se
evalúan con el mismo criterio: ``perfil`` (promedio histórico por hora y
tipo de día), ``ridge`` (regresión lineal regularizada), ``random-forest``
y ``hgb`` (gradient boosting con búsqueda de ``learning_rate``). Cada
escalón es un run de MLflow con sus parámetros, métricas, linaje,
artefactos, datasets y el modelo con firma. La búsqueda de ``hgb`` usa
runs anidados bajo un run padre ``busqueda-hgb``.

Los datos se parten por fecha, nunca al azar: entrenamiento de
2011-01-01 a 2011-11-30, validación en diciembre de 2011. Las ventanas
viven en ``params.yaml``. 2012 no se toca aquí: queda para evaluar en
producción simulada.

El "campeón" es el escalón con menor ``costo_hora_val`` (costo promedio
por hora del plan, con el factor de ajuste elegido en validación). Se
registra como versión del modelo ``demanda-bicis`` y recibe el alias
``@champion``, que es lo único que ``predecir`` carga.

El "linaje" son los tags que permiten reconstruir con qué se entrenó:
``git_commit`` (el árbol de trabajo debe estar limpio, o el script
aborta), ``features``, ``huella_datos`` (sha256 de las filas exactas de
entrenamiento y validación) y ``dvc_md5`` (versión de ``hour.csv``).

Uso::

    MLFLOW_TRACKING_URI=http://localhost:5000 python -m bicis.entrenar
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bicis.costos import cargar_costos, elegir_factor, grilla_factores
from bicis.transformar import FEATURES

EXPERIMENTO = "demanda-bicis"
MODELO_REGISTRADO = "demanda-bicis"
ALIAS_CAMPEON = "champion"
NOMBRE_MODELO_EN_RUN = "model"

RUTA_PARAMS = Path("params.yaml")
RUTA_FUENTE = Path("fuente/bicis.db")
RUTA_DVC = Path("data/raw/hour.csv.dvc")
RUTA_COSTOS = Path("configs/costos.yaml")

TARGET = "cnt"
REPETICIONES_LATENCIA = 200
FILAS_LATENCIA = 24


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------


class PerfilEstimator:
    """Modelo base: promedio histórico de ``cnt`` por ``hr`` y ``workingday``.

    Es el escalón más simple de la escalera y la vara contra la que se leen
    los demás. Un par (hora, tipo de día) no visto en entrenamiento recibe
    el promedio global. Implementa ``fit``/``predict`` y ``get_params``
    para que MLflow lo trate como un estimador de scikit-learn.
    """

    def __init__(self) -> None:
        self.tabla_: pd.Series | None = None
        self.global_: float = 0.0

    def get_params(self, deep: bool = True) -> dict:
        return {}

    def set_params(self, **params) -> "PerfilEstimator":
        return self

    def fit(self, X: pd.DataFrame, y) -> "PerfilEstimator":
        """Calcula la tabla de promedios por (hr, workingday)."""
        y = pd.Series(np.asarray(y, dtype=float), index=X.index)
        self.tabla_ = y.groupby([X["hr"].astype(int), X["workingday"].astype(int)]).mean()
        self.global_ = float(y.mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Busca cada fila en la tabla; si el par no existe, usa el promedio global."""
        claves = list(zip(X["hr"].astype(int), X["workingday"].astype(int)))
        return np.array([self.tabla_.get(k, self.global_) for k in claves], dtype=float)


def construir_modelo(tipo: str, hiperparams: dict, semilla: int):
    """Devuelve el estimador de un escalón, listo para ``fit`` sobre ``FEATURES``.

    Args:
        tipo: ``perfil``, ``ridge``, ``random-forest`` o ``hgb``.
        hiperparams: Hiperparámetros del escalón, tomados de ``params.yaml``.
        semilla: Semilla para todo lo que tenga aleatoriedad.

    Returns:
        Un estimador con ``fit(X, y)`` y ``predict(X)`` que acepta un
        DataFrame con exactamente las columnas de ``FEATURES``.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    if tipo == "perfil":
        return PerfilEstimator()
    if tipo == "ridge":
        # hr y mnth son categóricas para una regresión lineal; el resto entra tal cual.
        columnas = ColumnTransformer(
            [("onehot", OneHotEncoder(handle_unknown="ignore"), ["hr", "mnth"])],
            remainder="passthrough",
        )
        return Pipeline([("columnas", columnas), ("ridge", Ridge(alpha=hiperparams["alpha"]))])
    if tipo == "random-forest":
        return RandomForestRegressor(
            n_estimators=hiperparams["n_estimators"],
            max_depth=hiperparams.get("max_depth"),
            n_jobs=hiperparams.get("n_jobs", -1),
            random_state=semilla,
        )
    if tipo == "hgb":
        return HistGradientBoostingRegressor(
            learning_rate=hiperparams["learning_rate"],
            max_iter=hiperparams["max_iter"],
            random_state=semilla,
        )
    raise ValueError(f"Tipo de modelo desconocido: {tipo}")


# ---------------------------------------------------------------------------
# Datos y linaje
# ---------------------------------------------------------------------------


def cargar_params(ruta: str | Path = RUTA_PARAMS) -> dict:
    """Lee ``params.yaml`` completo."""
    return yaml.safe_load(Path(ruta).read_text())


def leer_ventana(ruta_fuente: str | Path, inicio: str, fin: str) -> pd.DataFrame:
    """Filas de ``demanda_hora`` con ``inicio <= dteday <= fin``, ordenadas por fecha y hora."""
    with sqlite3.connect(ruta_fuente) as con:
        return pd.read_sql_query(
            "SELECT * FROM demanda_hora WHERE dteday BETWEEN ? AND ? ORDER BY dteday, hr",
            con,
            params=(inicio, fin),
        )


def huella_datos(*tablas: pd.DataFrame) -> str:
    """sha256 de las filas exactas de las tablas, en orden, para el tag ``huella_datos``."""
    h = hashlib.sha256()
    for df in tablas:
        h.update(pd.util.hash_pandas_object(df, index=False).values.tobytes())
    return h.hexdigest()


def dvc_md5(ruta: str | Path = RUTA_DVC) -> str:
    """md5 que DVC registró para ``hour.csv`` en su archivo ``.dvc``."""
    return yaml.safe_load(Path(ruta).read_text())["outs"][0]["md5"]


def git_commit(repo: str | Path = ".") -> str:
    """Hash del commit actual. Aborta si hay cambios sin commit.

    El tag ``git_commit`` de un run solo sirve si apunta al código que de
    verdad entrenó el modelo; con cambios pendientes no se puede garantizar.

    Raises:
        SystemExit: Si ``git status --porcelain`` no está vacío.
    """
    sucio = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()
    if sucio:
        raise SystemExit(
            "El árbol de trabajo tiene cambios sin commit. Haz commit antes de entrenar, "
            "para que el git_commit registrado en MLflow exista:\n" + sucio
        )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()


# ---------------------------------------------------------------------------
# Métricas y artefactos
# ---------------------------------------------------------------------------


def latencia_p95_ms(modelo, X: pd.DataFrame, repeticiones: int = REPETICIONES_LATENCIA) -> float:
    """Percentil 95, en milisegundos, del tiempo de predecir ``X`` (24 filas) ``repeticiones`` veces."""
    tiempos = []
    for _ in range(repeticiones):
        t0 = time.perf_counter()
        modelo.predict(X)
        tiempos.append((time.perf_counter() - t0) * 1000.0)
    return float(np.percentile(tiempos, 95))


def tamano_modelo_mb(modelo) -> float:
    """Tamaño del modelo serializado con pickle, en megabytes."""
    return len(pickle.dumps(modelo)) / 1e6


def _graficas(carpeta: Path, nombre: str, val: pd.DataFrame, pred: np.ndarray, curva: dict, modelo) -> list[Path]:
    """Escribe las gráficas del run en ``carpeta`` y devuelve sus rutas."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rutas = []
    y = val[TARGET].to_numpy(dtype=float)
    etiqueta = val["dteday"].astype(str) + " " + val["hr"].astype(int).astype(str).str.zfill(2)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(y, label="real", linewidth=0.9)
    ax.plot(pred, label="predicción", linewidth=0.9)
    ax.set_title(f"{nombre}: real contra predicción en validación (diciembre 2011)")
    ax.set_xlabel("hora de diciembre 2011 (índice)")
    ax.set_ylabel("cnt")
    ticks = np.linspace(0, len(y) - 1, 8, dtype=int)
    ax.set_xticks(ticks)
    ax.set_xticklabels(etiqueta.iloc[ticks], rotation=30, ha="right", fontsize=8)
    ax.legend()
    fig.tight_layout()
    rutas.append(carpeta / "real_vs_prediccion_val.png")
    fig.savefig(rutas[-1], dpi=110)
    plt.close(fig)

    error = pd.Series(np.abs(y - pred)).groupby(val["hr"].to_numpy()).mean()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(error.index, error.values)
    ax.set_title(f"{nombre}: MAE por hora del día (validación)")
    ax.set_xlabel("hr")
    ax.set_ylabel("MAE")
    ax.set_xticks(range(24))
    fig.tight_layout()
    rutas.append(carpeta / "mae_por_hora.png")
    fig.savefig(rutas[-1], dpi=110)
    plt.close(fig)

    factores = sorted(curva)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(factores, [curva[f] for f in factores], marker="o")
    mejor = min(factores, key=lambda f: curva[f])
    ax.axvline(mejor, linestyle="--", linewidth=0.8)
    ax.set_title(f"{nombre}: costo por hora según el factor (factor elegido = {mejor:.2f})")
    ax.set_xlabel("factor")
    ax.set_ylabel("costo_hora_val")
    fig.tight_layout()
    rutas.append(carpeta / "costo_vs_factor.png")
    fig.savefig(rutas[-1], dpi=110)
    plt.close(fig)

    importancias = getattr(modelo, "feature_importances_", None)
    if importancias is not None:
        orden = np.argsort(importancias)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(np.array(FEATURES)[orden], importancias[orden])
        ax.set_title(f"{nombre}: importancia de las features")
        fig.tight_layout()
        rutas.append(carpeta / "importancias.png")
        fig.savefig(rutas[-1], dpi=110)
        plt.close(fig)

    return rutas


# ---------------------------------------------------------------------------
# Un escalón = un run
# ---------------------------------------------------------------------------


def entrenar_escalon(
    nombre: str,
    tipo: str,
    hiperparams: dict,
    train: pd.DataFrame,
    val: pd.DataFrame,
    contexto: dict,
    nested: bool = False,
) -> dict:
    """Entrena un escalón, lo evalúa en validación y registra todo en un run.

    Args:
        nombre: Nombre del run (``perfil``, ``ridge``, ``random-forest``, ``hgb-lr-0.1``).
        tipo: Tipo de modelo para ``construir_modelo``.
        hiperparams: Hiperparámetros del escalón.
        train: Filas de entrenamiento con ``FEATURES`` y ``cnt``.
        val: Filas de validación con ``FEATURES`` y ``cnt``.
        contexto: Semilla, ventanas, costos, grilla, tags de linaje y ruta de la fuente.
        nested: ``True`` para los runs hijos de la búsqueda de ``hgb``.

    Returns:
        Diccionario con ``run_id``, ``nombre``, ``tipo``, ``factor``, ``mae_val``,
        ``costo_hora_val``, ``latencia_p95_ms`` y ``tamano_modelo_mb``.
    """
    import mlflow
    from mlflow.models import infer_signature
    from sklearn.metrics import mean_absolute_error

    X_train, y_train = train[FEATURES], train[TARGET].to_numpy(dtype=float)
    X_val, y_val = val[FEATURES], val[TARGET].to_numpy(dtype=float)

    with mlflow.start_run(run_name=nombre, nested=nested) as run:
        modelo = construir_modelo(tipo, hiperparams, contexto["semilla"])
        modelo.fit(X_train, y_train)
        pred = np.maximum(np.asarray(modelo.predict(X_val), dtype=float), 0.0)

        mae = float(mean_absolute_error(y_val, pred))
        factor, curva = elegir_factor(y_val, pred, contexto["costos"], contexto["grilla"])
        costo = curva[factor]
        latencia = latencia_p95_ms(modelo, X_val.head(FILAS_LATENCIA))
        tamano = tamano_modelo_mb(modelo)

        mlflow.log_params(
            {
                "modelo.tipo": tipo,
                **{f"modelo.{k}": v for k, v in hiperparams.items()},
                "semilla": contexto["semilla"],
                "ventana.inicio": contexto["ventana"]["inicio"],
                "ventana.fin": contexto["ventana"]["fin"],
                "validacion.mes": contexto["validacion_mes"],
                "factor": factor,
            }
        )
        mlflow.log_metrics(
            {
                "mae_val": mae,
                "costo_hora_val": costo,
                "latencia_p95_ms": latencia,
                "tamano_modelo_mb": tamano,
            }
        )
        mlflow.set_tags(
            {
                "git_commit": contexto["git_commit"],
                "features": ",".join(FEATURES),
                "huella_datos": contexto["huella_datos"],
                "dvc_md5": contexto["dvc_md5"],
                "factor": str(factor),
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            carpeta = Path(tmp)
            (carpeta / "factor.json").write_text(
                json.dumps({"factor": factor, "curva": {f"{f:.2f}": c for f, c in curva.items()}}, indent=2)
            )
            for ruta in [carpeta / "factor.json", *_graficas(carpeta, nombre, val, pred, curva, modelo)]:
                mlflow.log_artifact(str(ruta))

        fuente = str(contexto["ruta_fuente"])
        mlflow.log_input(
            mlflow.data.from_pandas(train[[*FEATURES, TARGET]], source=fuente, targets=TARGET, name="entrenamiento"),
            context="training",
        )
        mlflow.log_input(
            mlflow.data.from_pandas(val[[*FEATURES, TARGET]], source=fuente, targets=TARGET, name="validacion"),
            context="validation",
        )

        # cloudpickle en lugar del skops por defecto de MLflow 3: skops no serializa
        # clases propias como PerfilEstimator. Se carga con la misma versión de mlflow.
        mlflow.sklearn.log_model(
            modelo,
            name=NOMBRE_MODELO_EN_RUN,
            signature=infer_signature(X_val, pred),
            input_example=X_val.head(5),
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )

        return {
            "run_id": run.info.run_id,
            "nombre": nombre,
            "tipo": tipo,
            "factor": factor,
            "mae_val": mae,
            "costo_hora_val": costo,
            "latencia_p95_ms": latencia,
            "tamano_modelo_mb": tamano,
        }


# ---------------------------------------------------------------------------
# La escalera completa
# ---------------------------------------------------------------------------


def preparar_contexto(params: dict, ruta_fuente: Path, train: pd.DataFrame, val: pd.DataFrame) -> dict:
    """Reúne lo que todos los runs comparten: semilla, ventanas, costos, grilla y linaje."""
    f = params["factor"]
    return {
        "semilla": int(params["semilla"]),
        "ventana": params["ventanas"]["entrenamiento"],
        "validacion_mes": params["ventanas"]["validacion"]["inicio"][:7],
        "costos": cargar_costos(RUTA_COSTOS),
        "grilla": grilla_factores(f["minimo"], f["maximo"], f["paso"]),
        "git_commit": git_commit(),
        "huella_datos": huella_datos(train, val),
        "dvc_md5": dvc_md5(),
        "ruta_fuente": ruta_fuente,
    }


def escalera(params: dict, train: pd.DataFrame, val: pd.DataFrame, contexto: dict) -> list[dict]:
    """Corre los cuatro escalones y devuelve un resultado por run (los hijos de hgb incluidos)."""
    import mlflow

    modelos = params["modelos"]
    resultados = [
        entrenar_escalon("perfil", "perfil", modelos["perfil"], train, val, contexto),
        entrenar_escalon("ridge", "ridge", modelos["ridge"], train, val, contexto),
        entrenar_escalon("random-forest", "random-forest", modelos["random_forest"], train, val, contexto),
    ]

    hgb = modelos["hgb"]
    with mlflow.start_run(run_name="busqueda-hgb") as padre:
        hijos = []
        for lr in hgb["learning_rate"]:
            hiper = {"learning_rate": lr, "max_iter": hgb["max_iter"]}
            hijos.append(entrenar_escalon(f"hgb-lr-{lr}", "hgb", hiper, train, val, contexto, nested=True))
        mejor = min(hijos, key=lambda r: r["costo_hora_val"])
        mlflow.log_params({"modelo.tipo": "hgb", "busqueda.learning_rate": str(hgb["learning_rate"]),
                           "mejor.learning_rate": mejor["nombre"].split("-")[-1], "mejor.run_id": mejor["run_id"]})
        mlflow.log_metrics({k: mejor[k] for k in ("mae_val", "costo_hora_val", "latencia_p95_ms", "tamano_modelo_mb")})
        mlflow.set_tags({"git_commit": contexto["git_commit"], "features": ",".join(FEATURES),
                         "huella_datos": contexto["huella_datos"], "dvc_md5": contexto["dvc_md5"]})
        for h in hijos:
            h["padre_run_id"] = padre.info.run_id
    resultados.extend(hijos)
    return resultados


def promover_campeon(ganador: dict) -> str:
    """Registra el modelo del run ganador como versión de ``demanda-bicis`` y le pone ``@champion``."""
    import mlflow
    from mlflow import MlflowClient

    version = mlflow.register_model(f"runs:/{ganador['run_id']}/{NOMBRE_MODELO_EN_RUN}", MODELO_REGISTRADO)
    cliente = MlflowClient()
    cliente.set_registered_model_alias(MODELO_REGISTRADO, ALIAS_CAMPEON, version.version)
    cliente.set_model_version_tag(MODELO_REGISTRADO, version.version, "escalon", ganador["nombre"])
    cliente.set_model_version_tag(MODELO_REGISTRADO, version.version, "factor", str(ganador["factor"]))
    return str(version.version)


def main() -> None:
    """Punto de entrada de ``make entrenar``."""
    import mlflow

    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(EXPERIMENTO)

    params = cargar_params()
    v = params["ventanas"]
    train = leer_ventana(RUTA_FUENTE, v["entrenamiento"]["inicio"], v["entrenamiento"]["fin"])
    val = leer_ventana(RUTA_FUENTE, v["validacion"]["inicio"], v["validacion"]["fin"])
    contexto = preparar_contexto(params, RUTA_FUENTE, train, val)
    print(f"Entrenamiento: {len(train):,} filas ({v['entrenamiento']['inicio']} a {v['entrenamiento']['fin']})")
    print(f"Validación:    {len(val):,} filas ({v['validacion']['inicio']} a {v['validacion']['fin']})")
    print(f"git_commit={contexto['git_commit'][:12]} dvc_md5={contexto['dvc_md5'][:12]} "
          f"huella_datos={contexto['huella_datos'][:12]}")

    resultados = escalera(params, train, val, contexto)

    tabla = (
        pd.DataFrame(resultados)[["nombre", "factor", "mae_val", "costo_hora_val", "latencia_p95_ms", "tamano_modelo_mb", "run_id"]]
        .sort_values("costo_hora_val")
        .reset_index(drop=True)
    )
    pd.set_option("display.width", 160)
    print("\nEscalera ordenada por costo_hora_val:")
    print(tabla.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    ganador = tabla.iloc[0].to_dict()
    version = promover_campeon(ganador)
    print(f"\nCampeón: {ganador['nombre']} (run {ganador['run_id'][:8]}) registrado como "
          f"{MODELO_REGISTRADO} versión {version} con alias @{ALIAS_CAMPEON}.")


if __name__ == "__main__":
    sys.exit(main())
