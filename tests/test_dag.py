"""Pruebas del DAG `prediccion_diaria`.

Necesitan Airflow instalado, que no forma parte de requirements.txt (vive en la
imagen de Docker). Si Airflow no está disponible se omiten; para ejecutarlas
en la imagen: `make test-dag` (docker compose run --rm airflow-cli ...).

Un DagBag es el cargador de Airflow: lee una carpeta, importa cada archivo de
DAG y guarda los errores de importación en vez de lanzarlos.
"""

from pathlib import Path

import pytest

# El módulo concreto: la carpeta airflow/ (logs) se vería como un paquete vacío.
pytest.importorskip("airflow.dag_processing.dagbag")

from airflow.dag_processing.dagbag import DagBag  # noqa: E402

DIR_DAGS = Path(__file__).resolve().parents[1] / "dags"
ORDEN = ["extraer", "limpiar", "transformar", "predecir", "almacenar"]


@pytest.fixture(scope="module")
def dagbag():
    return DagBag(dag_folder=str(DIR_DAGS))


@pytest.fixture(scope="module")
def dag(dagbag):
    return dagbag.dags["prediccion_diaria"]


def test_importa_sin_errores(dagbag):
    assert dagbag.import_errors == {}
    assert "prediccion_diaria" in dagbag.dag_ids


def test_cinco_tareas_en_orden(dag):
    assert [t.task_id for t in dag.topological_sort()] == ORDEN
    for anterior, siguiente in zip(ORDEN, ORDEN[1:]):
        assert dag.get_task(anterior).downstream_task_ids == {siguiente}
    assert dag.get_task("extraer").upstream_task_ids == set()
    assert dag.get_task("almacenar").downstream_task_ids == set()


def test_configuracion_del_dag(dag):
    assert dag.timetable.expression == "0 0 * * *"  # @daily
    assert dag.start_date.year == 2012 and dag.end_date.year == 2012
    assert dag.catchup is False
    assert dag.max_active_runs == 4
    assert set(dag.tags) == {"lab1", "bicis"}
    tarea = dag.get_task("predecir")
    assert tarea.retries == 2


def test_fecha_de_ejecucion_usa_la_fecha_logica():
    import pendulum

    from prediccion_diaria import fecha_de_ejecucion

    fecha = pendulum.datetime(2012, 1, 2, tz="UTC")
    assert fecha_de_ejecucion(fecha, None) == "2012-01-02"
    assert fecha_de_ejecucion(None, fecha) == "2012-01-02"
    with pytest.raises(ValueError, match="fecha lógica"):
        fecha_de_ejecucion(None, None)
