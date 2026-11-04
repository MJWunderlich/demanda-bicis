"""DAG `prediccion_diaria`: un día de demanda, de la fuente al plan guardado.

Un DAG (grafo acíclico dirigido) es, en Airflow, una receta con tareas y el
orden entre ellas. Este DAG corre una vez por día y encadena las cinco tareas
del pipeline: extraer → limpiar → transformar → predecir → almacenar. Toda la
lógica vive en `src/bicis/`; aquí solo se fijan las rutas y se conectan las
funciones.

Conceptos que usa este archivo:

- **Fecha lógica** (`logical_date`): el día que la ejecución representa, que
  no es el día en que corre. Un backfill de enero de 2012 corre hoy pero cada
  ejecución procesa su propio día de 2012. Las tareas solo usan la fecha
  lógica; nunca la fecha de hoy.
- **XCom**: el mecanismo con el que una tarea entrega un valor pequeño a la
  siguiente. Aquí viajan solo rutas de archivo y un conteo; los datos van a
  disco en Parquet bajo `trabajo/<fecha>/`.
- **Idempotencia**: ejecutar el mismo día dos veces deja el mismo resultado.
  `almacenar` borra las filas del día antes de insertar, así un *Clear* en la
  UI no duplica filas.

Rutas dentro del contenedor (montajes del docker-compose.yaml):

- fuente de solo lectura: `/opt/bicis/fuente/bicis.db`
- archivos intermedios: `/opt/bicis/proyecto/trabajo/<fecha>/`
- salida: `/opt/bicis/proyecto/salidas/predicciones.db`
"""

from __future__ import annotations

import logging
from datetime import timedelta

import pendulum
from airflow.sdk import dag, task

RUTA_FUENTE = "/opt/bicis/fuente/bicis.db"
DIR_TRABAJO = "/opt/bicis/proyecto/trabajo"
RUTA_SALIDA = "/opt/bicis/proyecto/salidas/predicciones.db"

log = logging.getLogger(__name__)


def fecha_de_ejecucion(logical_date, data_interval_start) -> str:
    """Devuelve el día a procesar como texto `YYYY-MM-DD`.

    Usa la fecha lógica. En Airflow 3 una ejecución manual puede crearse sin
    fecha lógica; en ese caso cae al inicio del intervalo de datos
    (`data_interval_start`), que para un DAG diario es el mismo día. Si tampoco
    existe, falla con un mensaje claro en vez de usar la fecha de hoy, que
    rompería la regla del pipeline y no tendría datos en la fuente.
    """
    fecha = logical_date or data_interval_start
    if fecha is None:
        raise ValueError(
            "La ejecución no tiene fecha lógica ni intervalo de datos. "
            "Dispárala desde la UI con la fecha lógica explícita."
        )
    return fecha.strftime("%Y-%m-%d")


@dag(
    schedule="@daily",
    start_date=pendulum.datetime(2012, 1, 1, tz="UTC"),
    end_date=pendulum.datetime(2012, 12, 31, tz="UTC"),
    catchup=False,
    max_active_runs=4,
    default_args={"retries": 2, "retry_delay": timedelta(minutes=1)},
    tags=["lab1", "bicis"],
    doc_md=__doc__,
)
def prediccion_diaria():
    @task
    def extraer(logical_date=None, data_interval_start=None) -> str:
        """Lee de la fuente las filas del día lógico y las deja en Parquet."""
        from bicis.extraer import extraer as _extraer

        fecha = fecha_de_ejecucion(logical_date, data_interval_start)
        return _extraer(fecha, RUTA_FUENTE, DIR_TRABAJO)

    @task
    def limpiar(ruta: str) -> str:
        """Valida el día extraído (filas, columnas, rangos) y escribe `limpio.parquet`."""
        from bicis.limpiar import limpiar as _limpiar

        return _limpiar(ruta)

    @task
    def transformar(ruta: str) -> str:
        """Deja solo las features del modelo y escribe `features.parquet`."""
        from bicis.transformar import transformar as _transformar

        return _transformar(ruta)

    @task
    def predecir(ruta: str) -> str:
        """Carga `models:/demanda-bicis@champion` y escribe `predicciones.parquet`."""
        import pandas as pd
        from bicis.predecir import predecir as _predecir

        salida = _predecir(ruta)
        resumen = pd.read_parquet(salida, columns=["fecha", "version_modelo", "factor", "plan"])
        log.info(
            "fecha=%s version_modelo=%s factor=%s filas=%d plan_total=%d",
            resumen["fecha"].iloc[0],
            resumen["version_modelo"].iloc[0],
            resumen["factor"].iloc[0],
            len(resumen),
            int(resumen["plan"].sum()),
        )
        return salida

    @task
    def almacenar(ruta: str) -> int:
        """Guarda el plan del día en `predicciones.db` y devuelve cuántas filas quedaron."""
        from bicis.almacenar import almacenar as _almacenar

        filas = _almacenar(ruta, RUTA_SALIDA)
        log.info("filas guardadas en %s: %d", RUTA_SALIDA, filas)
        return filas

    almacenar(predecir(transformar(limpiar(extraer()))))


prediccion_diaria()
