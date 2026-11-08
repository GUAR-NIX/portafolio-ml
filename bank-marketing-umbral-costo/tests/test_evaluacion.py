"""Métricas de negocio con resultados calculables a mano."""
import numpy as np
import pytest

from campana import evaluacion as ev


def test_ganancia_a_mano():
    y = np.array([1, 0, 1, 0, 0])
    llamar = np.array([1, 1, 0, 0, 1], bool)          # 3 llamadas, 1 éxito
    assert ev.ganancia(y, llamar, costo=8, beneficio=80) == 80 - 24


def test_umbral_teorico():
    assert ev.umbral_teorico(8, 80) == pytest.approx(0.10)
    assert ev.umbral_teorico(10, 40) == pytest.approx(0.25)


def test_umbral_optimo_coincide_con_el_teorico_si_p_es_real():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 0.5, 200_000)
    y = (rng.random(p.size) < p).astype(int)           # p es la probabilidad verdadera
    assert ev.umbral_optimo(y, p, costo=8, beneficio=80) == pytest.approx(0.10, abs=0.02)


def test_ganancia_acumulada_modelo_perfecto_y_azar():
    y = np.r_[np.ones(100), np.zeros(900)].astype(int)
    perfecto = ev.ganancia_acumulada(y, y.astype(float))
    fila10 = perfecto.iloc[(perfecto["pct_llamados"] - 10).abs().argmin()]
    assert fila10["pct_capturado"] == pytest.approx(100, abs=1)
    assert fila10["lift"] == pytest.approx(10, rel=0.05)
    azar = ev.ganancia_acumulada(y, np.random.default_rng(0).random(y.size))
    assert azar["pct_capturado"].iloc[-1] == pytest.approx(100)


def test_brier_skill_de_la_tasa_base_es_cero():
    y = np.array([0, 0, 0, 1])
    assert ev.brier_skill(y, np.full(4, 0.25)) == pytest.approx(0)
    assert ev.brier_skill(y, y.astype(float)) == pytest.approx(1)


def test_politica_resumen():
    y = np.array([1, 1, 0, 0])
    r = ev.resumen_politica(y, np.array([1, 0, 1, 0], bool), costo=1, beneficio=10)
    assert r["llamadas"] == 2 and r["pct_exitos_capturados"] == 50 and r["conversion_pct"] == 50
    assert r["ganancia"] == 8


def test_bootstrap_metricas_estructura_y_contenido_del_ic():
    rng = np.random.default_rng(3)
    y = (rng.random(1500) < 0.2).astype(int)
    bueno = np.clip(0.2 + 0.5 * (y - 0.2) + rng.normal(0, 0.25, y.size), 0.01, 0.99)
    malo = rng.random(y.size)
    t = ev.bootstrap_metricas(y, {"bueno": bueno, "malo": malo}, [("bueno", "malo")], n_remuestras=300)
    assert set(t["metrica"]) == {"auc", "average_precision", "brier"}
    assert set(t["comparacion"]) == {"bueno", "malo", "bueno − malo"}
    assert (t["ic_bajo"] <= t["estimado"]).all() and (t["estimado"] <= t["ic_alto"]).all()
    d = t[(t["comparacion"] == "bueno − malo") & (t["metrica"] == "auc")].iloc[0]
    assert d["ic_bajo"] > 0                                    # el bueno gana de forma clara
    b = t[(t["comparacion"] == "bueno − malo") & (t["metrica"] == "brier")].iloc[0]
    assert b["ic_alto"] < 0                                    # y su Brier es menor


def test_bootstrap_modelo_contra_si_mismo_da_diferencia_cero():
    rng = np.random.default_rng(4)
    y = (rng.random(800) < 0.3).astype(int)
    p = rng.random(y.size)
    t = ev.bootstrap_metricas(y, {"a": p, "b": p.copy()}, [("a", "b")], n_remuestras=100)
    d = t[t["comparacion"] == "a − b"]
    assert (d["estimado"] == 0).all() and (d["ic_bajo"] == 0).all() and (d["ic_alto"] == 0).all()


def test_bootstrap_es_reproducible_y_ic_estrecho_con_mas_datos():
    rng = np.random.default_rng(5)
    def ancho(n):
        y = (rng.random(n) < 0.2).astype(int)
        p = np.clip(0.2 + 0.4 * (y - 0.2) + rng.normal(0, 0.3, n), 0.01, 0.99)
        t = ev.bootstrap_metricas(y, {"m": p}, [], n_remuestras=200)
        fila = t[t["metrica"] == "auc"].iloc[0]
        return fila["ic_alto"] - fila["ic_bajo"]
    assert ancho(6000) < ancho(600)
    y = (rng.random(500) < 0.2).astype(int)
    p = rng.random(500)
    a = ev.bootstrap_metricas(y, {"m": p}, [], n_remuestras=50)
    b = ev.bootstrap_metricas(y, {"m": p}, [], n_remuestras=50)
    assert a.equals(b)
