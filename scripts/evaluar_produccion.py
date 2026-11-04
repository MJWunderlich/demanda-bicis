"""Evidencia de la deriva: cómo le fue al campeón en producción frente a validación.

Une las predicciones que el pipeline guardó en ``salidas/predicciones.db`` con
la demanda real (``cnt``) de ``fuente/bicis.db`` para un mes de producción
(por defecto enero de 2012) y calcula tres métricas:

- **MAE** (error absoluto medio): promedio de ``|prediccion - real|`` por hora.
- **Sesgo** (error medio): promedio de ``prediccion - real``. Negativo quiere
  decir que el modelo subestima la demanda.
- **Costo por hora**: el de ``costos.costo_hora``, calculado sobre el ``plan``
  (predicción con el factor de ajuste) y no sobre la predicción cruda, porque
  el plan es lo que la operación usa.

Las compara con las mismas métricas sobre la ventana de validación de
``params.yaml`` (diciembre de 2011), recalculadas aquí con el campeón actual
(``models:/demanda-bicis@champion``) para que ambas columnas salgan del mismo
modelo y el mismo factor.

**Deriva** es el cambio en la relación entre las features y la demanda con el
paso del tiempo. El modelo no ve el año, así que si la demanda de 2012 es más
alta que la de 2011 con el mismo clima y la misma hora, el modelo subestima y
el sesgo sale negativo. Eso es lo que justifica reentrenar.

Salidas:

- ``docs/capturas/deriva_enero_2012.png``: MAE por día del mes de producción,
  con el MAE de validación como referencia.
- Una tabla markdown para el README, impresa en pantalla y guardada en
  ``docs/deriva_enero_2012.md``.

Uso::

    python scripts/evaluar_produccion.py [--mes 2012-01] [--sin-validacion]

Necesita el servidor MLflow (``MLFLOW_TRACKING_URI``) salvo con
``--sin-validacion``, que omite la columna de diciembre de 2011.
"""

from __future__ import annotations

import argparse
import calendar
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from bicis import costos as mod_costos  # noqa: E402
from bicis.transformar import FEATURES  # noqa: E402

RUTA_PREDICCIONES = RAIZ / "salidas" / "predicciones.db"
RUTA_FUENTE = RAIZ / "fuente" / "bicis.db"
RUTA_PARAMS = RAIZ / "params.yaml"
RUTA_COSTOS = RAIZ / "configs" / "costos.yaml"
RUTA_FIGURA = RAIZ / "docs" / "capturas" / "deriva_enero_2012.png"
RUTA_TABLA = RAIZ / "docs" / "deriva_enero_2012.md"
MODEL_URI = "models:/demanda-bicis@champion"

COLUMNAS_EVALUACION = ["fecha", "hr", "real", "prediccion", "plan"]
MESES = [
    "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
]


def nombre_mes(fecha: str) -> str:
    """``"2012-01-15"`` → ``"enero 2012"``, sin depender del locale."""
    ts = pd.Timestamp(fecha)
    return f"{MESES[ts.month - 1]} {ts.year}"


def rango_del_mes(mes: str) -> tuple[str, str]:
    """``"2012-01"`` → ``("2012-01-01", "2012-01-31")``."""
    anio, numero = (int(x) for x in mes.split("-"))
    ultimo = calendar.monthrange(anio, numero)[1]
    return f"{anio:04d}-{numero:02d}-01", f"{anio:04d}-{numero:02d}-{ultimo:02d}"


def leer_produccion(
    ruta_predicciones: str | Path, ruta_fuente: str | Path, inicio: str, fin: str
) -> pd.DataFrame:
    """Predicciones guardadas unidas con la demanda real, una fila por (fecha, hr).

    Lee ``predicciones`` entre ``inicio`` y ``fin`` (inclusive) y le pega el
    ``cnt`` real de ``demanda_hora`` por fecha y hora. Solo conserva las horas
    que están en ambas tablas: una hora predicha sin real no se puede evaluar
    y una hora real sin predicción es un día que el pipeline no corrió.
    Devuelve las columnas ``fecha, hr, real, prediccion, plan, version_modelo,
    factor`` ordenadas por fecha y hora. Falla si no hay ninguna fila.
    """
    with sqlite3.connect(ruta_predicciones) as con:
        pred = pd.read_sql_query(
            "SELECT fecha, hr, prediccion, plan, version_modelo, factor "
            "FROM predicciones WHERE fecha BETWEEN ? AND ?",
            con,
            params=(inicio, fin),
        )
    with sqlite3.connect(ruta_fuente) as con:
        real = pd.read_sql_query(
            "SELECT dteday AS fecha, hr, cnt AS real FROM demanda_hora WHERE dteday BETWEEN ? AND ?",
            con,
            params=(inicio, fin),
        )
    unido = pred.merge(real, on=["fecha", "hr"], how="inner").sort_values(["fecha", "hr"])
    if unido.empty:
        raise ValueError(
            f"No hay predicciones con demanda real entre {inicio} y {fin} "
            f"en {ruta_predicciones}. ¿Corrió el backfill de ese mes?"
        )
    columnas = COLUMNAS_EVALUACION + ["version_modelo", "factor"]
    return unido[columnas].reset_index(drop=True)


def metricas(df: pd.DataFrame, costos: dict[str, float]) -> dict[str, float]:
    """MAE, sesgo, costo por hora, demanda media y número de filas de ``df``.

    ``df`` trae ``real``, ``prediccion`` y ``plan`` por hora. El sesgo es
    ``mean(prediccion - real)``; el costo usa ``plan`` contra ``real``.
    """
    real = df["real"].to_numpy(dtype=float)
    pred = df["prediccion"].to_numpy(dtype=float)
    plan = df["plan"].to_numpy(dtype=float)
    return {
        "filas": int(len(df)),
        "demanda_media": float(real.mean()),
        "mae": float(np.mean(np.abs(pred - real))),
        "sesgo": float(np.mean(pred - real)),
        "costo_hora": mod_costos.costo_hora(real, plan, costos),
    }


def mae_por_dia(df: pd.DataFrame) -> pd.Series:
    """MAE de cada fecha de ``df``, indexado por fecha y ordenado."""
    error = (df["prediccion"] - df["real"]).abs()
    return error.groupby(df["fecha"]).mean().sort_index()


def evaluar_validacion(
    ruta_fuente: str | Path, inicio: str, fin: str, model_uri: str = MODEL_URI
) -> pd.DataFrame:
    """Predice la ventana de validación con el campeón y arma el mismo DataFrame.

    Carga el modelo por alias (``cargar_champion``), predice las horas de la
    ventana, recorta las predicciones negativas a cero y calcula el ``plan``
    con el factor que quedó registrado con el campeón. Así la columna de
    validación es comparable con la de producción: mismo modelo, mismo factor.
    """
    from bicis.entrenar import leer_ventana
    from bicis.predecir import calcular_plan, cargar_champion

    ventana = leer_ventana(ruta_fuente, inicio, fin)
    cargado = cargar_champion(model_uri)
    prediccion = np.clip(np.asarray(cargado.modelo.predict(ventana[FEATURES]), dtype=float).ravel(), 0, None)
    return pd.DataFrame(
        {
            "fecha": ventana["dteday"].astype(str),
            "hr": ventana["hr"],
            "real": ventana["cnt"],
            "prediccion": prediccion,
            "plan": calcular_plan(prediccion, cargado.factor),
            "version_modelo": cargado.version_modelo,
            "factor": cargado.factor,
        }
    )


def tabla_markdown(columnas: dict[str, dict[str, float]]) -> str:
    """Tabla markdown con una columna por ventana y una fila por métrica.

    ``columnas`` mapea el título de cada columna (por ejemplo
    ``"diciembre 2011 (validación)"``) al dict que devuelve ``metricas``.
    Si hay exactamente dos columnas, agrega una tercera con el cambio de la
    segunda respecto a la primera.
    """
    filas = [
        ("filas (horas)", "filas", "{:d}"),
        ("demanda media por hora", "demanda_media", "{:.1f}"),
        ("MAE", "mae", "{:.2f}"),
        ("sesgo (predicción − real)", "sesgo", "{:+.2f}"),
        ("costo por hora", "costo_hora", "{:.2f}"),
    ]
    titulos = list(columnas)
    con_cambio = len(titulos) == 2
    cabecera = ["métrica", *titulos] + (["cambio"] if con_cambio else [])
    lineas = ["| " + " | ".join(cabecera) + " |", "|" + "---|" * len(cabecera)]
    for etiqueta, clave, formato in filas:
        celdas = [formato.format(columnas[t][clave]) for t in titulos]
        if con_cambio:
            a, b = columnas[titulos[0]][clave], columnas[titulos[1]][clave]
            celdas.append(f"{b - a:+.2f}" if clave != "filas" else f"{b - a:+d}")
            if clave in ("demanda_media", "mae", "costo_hora") and a:
                celdas[-1] += f" ({(b - a) / a:+.0%})"
        lineas.append(f"| {etiqueta} | " + " | ".join(celdas) + " |")
    return "\n".join(lineas) + "\n"


def graficar(mae_dia: pd.Series, mae_referencia: float | None, titulo: str, ruta: str | Path) -> str:
    """Línea del MAE por día con la referencia de validación; guarda PNG y devuelve la ruta."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    fechas = pd.to_datetime(mae_dia.index)
    fig, ax = plt.subplots(figsize=(11, 4.2), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.plot(fechas, mae_dia.to_numpy(), color="#2a78d6", linewidth=2, marker="o", markersize=5)
    if mae_referencia is not None:
        ax.axhline(mae_referencia, color="#52514e", linewidth=1.2, linestyle=(0, (4, 3)))
        ax.annotate(
            f"MAE en validación (dic 2011): {mae_referencia:.1f}",
            xy=(fechas[-1], mae_referencia),
            xytext=(0, -13),
            textcoords="offset points",
            ha="right",
            color="#52514e",
            fontsize=9,
        )
    peor = mae_dia.idxmax()
    ax.annotate(
        f"{mae_dia.max():.0f}",
        xy=(pd.Timestamp(peor), mae_dia.max()),
        xytext=(0, 7),
        textcoords="offset points",
        ha="center",
        color="#0b0b0b",
        fontsize=9,
    )
    ax.set_title(titulo, loc="left", color="#0b0b0b", fontsize=12)
    ax.set_ylabel("MAE (bicicletas por hora)", color="#52514e")
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.set_xlabel(f"día de {nombre_mes(str(fechas[0].date()))}", color="#52514e")
    ax.grid(axis="y", color="#e6e5e1", linewidth=0.8)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e")
    fig.tight_layout()
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=130)
    plt.close(fig)
    return str(ruta)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mes", default="2012-01", help="mes de producción, YYYY-MM")
    parser.add_argument("--predicciones", default=RUTA_PREDICCIONES, type=Path)
    parser.add_argument("--fuente", default=RUTA_FUENTE, type=Path)
    parser.add_argument("--params", default=RUTA_PARAMS, type=Path)
    parser.add_argument("--costos", default=RUTA_COSTOS, type=Path)
    parser.add_argument("--figura", default=RUTA_FIGURA, type=Path)
    parser.add_argument("--tabla", default=RUTA_TABLA, type=Path)
    parser.add_argument("--model-uri", default=MODEL_URI)
    parser.add_argument("--sin-validacion", action="store_true", help="no recalcula diciembre de 2011")
    args = parser.parse_args(argv)

    import yaml

    costos = mod_costos.cargar_costos(args.costos)
    inicio, fin = rango_del_mes(args.mes)
    produccion = leer_produccion(args.predicciones, args.fuente, inicio, fin)
    mes_produccion = nombre_mes(inicio)
    columnas: dict[str, dict[str, float]] = {}

    mae_val = None
    if not args.sin_validacion:
        ventana = yaml.safe_load(args.params.read_text())["ventanas"]["validacion"]
        validacion = evaluar_validacion(args.fuente, ventana["inicio"], ventana["fin"], args.model_uri)
        nombre_val = nombre_mes(ventana["inicio"])
        columnas[f"{nombre_val} (validación)"] = metricas(validacion, costos)
        mae_val = columnas[f"{nombre_val} (validación)"]["mae"]
    columnas[f"{mes_produccion} (producción)"] = metricas(produccion, costos)

    versiones = sorted(produccion["version_modelo"].unique())
    tabla = tabla_markdown(columnas)
    encabezado = (
        f"Campeón: versión {', '.join(versiones)} de demanda-bicis, factor "
        f"{produccion['factor'].iloc[0]:.2f}. Costos: faltante {costos['faltante']:g}, "
        f"sobrante {costos['sobrante']:g}.\n\n"
    )
    args.tabla.parent.mkdir(parents=True, exist_ok=True)
    args.tabla.write_text(encabezado + tabla)
    print(encabezado + tabla)

    figura = graficar(
        mae_por_dia(produccion),
        mae_val,
        f"Deriva: MAE por día del campeón en {mes_produccion}",
        args.figura,
    )
    print(f"figura: {figura}\ntabla: {args.tabla}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
