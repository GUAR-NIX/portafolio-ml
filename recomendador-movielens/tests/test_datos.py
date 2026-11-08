"""Partición temporal con un dataset sintético de resultado conocido: nada del futuro entra al entrenamiento."""
import numpy as np
import pandas as pd
import pytest

from recomendador import config as C
from recomendador import datos

CORTE = pd.Timestamp("2018-01-01")
FIN = pd.Timestamp("2018-07-01")


def ts(fecha: str) -> int:
    return int(pd.Timestamp(fecha).timestamp())


@pytest.fixture
def calificaciones(monkeypatch):
    monkeypatch.setattr(C, "MIN_POSITIVOS_ITEM", 1)
    monkeypatch.setattr(C, "N_ONBOARDING", 2)
    filas = [
        # usuario 1: historia antes del corte y vuelve después
        (1, 10, 5.0, "2017-01-01"), (1, 11, 2.0, "2017-02-01"), (1, 12, 4.0, "2018-02-01"), (1, 13, 3.0, "2018-03-01"),
        # usuario 2: solo antes del corte
        (2, 10, 4.5, "2017-05-01"), (2, 12, 4.0, "2017-06-01"), (2, 13, 4.0, "2017-07-01"),
        # usuario 3: nuevo (empieza después del corte): onboarding = 2 primeras, se evalúa con el resto
        (3, 10, 5.0, "2018-01-05"), (3, 11, 1.0, "2018-01-06"), (3, 12, 4.0, "2018-01-07"), (3, 13, 5.0, "2018-01-08"),
        # usuario 4: califica la película 14 después del FIN → fuera de todo
        (4, 10, 4.0, "2017-03-01"), (4, 14, 5.0, "2019-01-01"),
        # usuario 5: después del corte vuelve a la película 10 que no está... (película 99 fuera de catálogo)
        (5, 10, 4.0, "2017-03-01"), (5, 99, 5.0, "2018-02-01"),
    ]
    df = pd.DataFrame(filas, columns=["userId", "movieId", "rating", "fecha"])
    df["timestamp"] = df["fecha"].map(ts).astype(np.int32)
    return df.drop(columns="fecha")


def test_entrenamiento_solo_pasado(calificaciones):
    p = datos.particionar(calificaciones, CORTE, FIN)
    assert p.resumen["calificaciones_entrenamiento"] == 7
    assert 3 not in p.usuarios                      # el usuario nuevo no está en entrenamiento
    assert 14 not in p.items and 99 not in p.items  # películas que solo aparecen después del corte
    assert set(p.items) == {10, 12, 13}             # 11 no tiene positivos en entrenamiento


def test_matrices_positivos_y_vistos(calificaciones):
    p = datos.particionar(calificaciones, CORTE, FIN)
    i = p.idx_item()
    u1 = int(np.flatnonzero(p.usuarios == 1)[0])
    assert set(p.items[p.positivos[u1].indices]) == {10}
    assert p.positivos.sum() == 6  # (1,10) (2,10) (2,12) (2,13) (4,10) (5,10)
    assert set(p.items[p.vistos[u1].indices]) == {10}  # 11 no está en catálogo
    assert i[10] == 0


def test_evaluacion_existentes_y_nuevos(calificaciones):
    p = datos.particionar(calificaciones, CORTE, FIN)
    i = p.idx_item()
    u1 = int(np.flatnonzero(p.usuarios == 1)[0])
    assert list(p.eval_existentes) == [u1]                     # el 5 no tiene relevantes dentro del catálogo
    assert p.items[p.eval_existentes[u1]].tolist() == [12]     # la nota 3 de la película 13 no es relevante
    nuevo = p.eval_nuevos[3]
    assert p.items[nuevo["onboarding"]].tolist() == [10]       # 11 (nota 1) no es positiva y no está en catálogo
    assert sorted(p.items[nuevo["relevantes"]].tolist()) == [12, 13]
    assert p.resumen["positivos_ventana_fuera_de_catalogo_pct"] > 0
    assert i[12] in p.eval_existentes[u1]


def test_muestreo_reproducible():
    a = datos.muestrear_usuarios(list(range(1000)), 10)
    assert a == datos.muestrear_usuarios(list(range(1000)), 10) and len(set(a)) == 10
    assert datos.muestrear_usuarios([3, 1], 10) == [1, 3]
