"""Métricas calculadas a mano en casos pequeños."""
import math

import numpy as np
import pytest

from recomendador import metricas as M


def test_lista_perfecta():
    m = M.metricas_usuario(np.arange(10), np.arange(10), k=10)
    assert m["precision"] == 1 and m["recall"] == 1 and m["ndcg"] == pytest.approx(1) and m["ap"] == pytest.approx(1)


def test_sin_aciertos():
    m = M.metricas_usuario(np.arange(10), np.array([50, 60]), k=10)
    assert m == {"precision": 0, "recall": 0, "ndcg": 0, "ap": 0, "acierto": 0}


def test_caso_a_mano():
    # relevantes {1, 3}; recomendados [1, 2, 3] con k=3 → aciertos en posiciones 1 y 3
    m = M.metricas_usuario(np.array([1, 2, 3]), np.array([1, 3]), k=3)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(1)
    dcg = 1 + 1 / math.log2(4)
    idcg = 1 + 1 / math.log2(3)
    assert m["ndcg"] == pytest.approx(dcg / idcg)
    assert m["ap"] == pytest.approx((1 / 1 + 2 / 3) / 2)


def test_ap_normaliza_por_min_relevantes_k():
    # 20 relevantes, k=2, ambos aciertos → AP = 1 (no 2/20)
    m = M.metricas_usuario(np.array([0, 1]), np.arange(20), k=2)
    assert m["ap"] == pytest.approx(1) and m["recall"] == pytest.approx(0.1)


def test_orden_importa_en_ndcg():
    arriba = M.metricas_usuario(np.array([7, 0, 1]), np.array([7]), k=3)["ndcg"]
    abajo = M.metricas_usuario(np.array([0, 1, 7]), np.array([7]), k=3)["ndcg"]
    assert arriba == 1 and abajo == pytest.approx(0.5)


def test_relevantes_vacio_falla():
    with pytest.raises(ValueError):
        M.metricas_usuario(np.arange(3), np.array([]), k=3)


def test_cobertura_y_novedad():
    listas = np.array([[0, 1], [0, 2]])
    assert M.cobertura(listas, 10) == pytest.approx(0.3)
    conteo = np.array([50, 50, 50] + [1] * 7)
    # p = 50/100 → 1 bit para todos
    assert M.novedad(listas, conteo, 100) == pytest.approx(1.0)


def test_diversidad_intra_lista():
    v = np.eye(3)
    assert M.diversidad_intra_lista(np.array([[0, 1, 2]]), v) == pytest.approx(1)  # ortogonales
    assert M.diversidad_intra_lista(np.array([[0, 0, 0]]), v) == pytest.approx(0)  # idénticos


def test_sesgo_popularidad():
    conteo = np.arange(1, 101)  # el item 99 es el más popular
    populares = M.sesgo_popularidad(np.array([[99, 98]]), conteo)
    raros = M.sesgo_popularidad(np.array([[0, 1]]), conteo, historiales=[np.array([50])])
    assert populares["percentil_pop_medio"] > 95 and populares["cola_larga_pct"] == 0
    assert raros["cola_larga_pct"] == 100 and raros["delta_pop_vs_historial"] < 0


def test_bootstrap_contiene_la_media():
    x = np.random.default_rng(0).normal(0.3, 0.1, 500)
    bajo, alto = M.intervalo_bootstrap(x)
    assert bajo < x.mean() < alto and alto - bajo < 0.05
