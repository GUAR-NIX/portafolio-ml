"""Pruebas matemáticas de BG/NBD, Gamma-Gamma, cuadratura vs 2F1 y propiedades de Bayes."""
from __future__ import annotations

import numpy as np
import pytest

from clv.modelos import (
    ModeloBGNBD,
    ModeloGammaGamma,
    ModeloSupervisadoCLV,
    ReglaQuintilesRFM,
    ReglaRitmoHistorico,
    calcular_clv,
)


def test_bgnbd_probabilidad_vivo_para_x_cero_es_uno():
    m = ModeloBGNBD(r=0.7, alpha=5.0, a=0.3, b=1.8)
    pv = m.probabilidad_vivo(np.array([0, 0, 0]), np.array([0.0, 0.0, 0.0]), np.array([5.0, 20.0, 60.0]))
    np.testing.assert_allclose(pv, 1.0, rtol=1e-12)


def test_bgnbd_probabilidad_recompra_decrece_con_edad_para_x_cero():
    m = ModeloBGNBD(r=0.7, alpha=5.0, a=0.3, b=1.8)
    pr = m.probabilidad_recompra(26.0, np.array([0, 0, 0]), np.array([0.0, 0.0, 0.0]), np.array([2.0, 20.0, 70.0]))
    assert pr[0] > pr[1] > pr[2]
    assert np.all((pr > 0.0) & (pr < 1.0))


def test_bgnbd_paradoja_alta_frecuencia_en_silencio():
    """A igual recencia baja (t_x = 15 sem) y antigüedad alta (T = 75 sem), mayor frecuencia x implica MENOR P(vivo)."""
    m = ModeloBGNBD(r=0.8, alpha=6.0, a=0.25, b=1.6)
    pv_frec_baja = m.probabilidad_vivo(np.array([2]), np.array([15.0]), np.array([75.0]))[0]
    pv_frec_alta = m.probabilidad_vivo(np.array([12]), np.array([15.0]), np.array([75.0]))[0]
    assert pv_frec_alta < pv_frec_baja


def test_bgnbd_monotonia_respecto_a_inactividad():
    """Para x y t_x fijos, cuanto mayor es T (más tiempo sin comprar), menor es P(vivo)."""
    m = ModeloBGNBD(r=0.8, alpha=6.0, a=0.25, b=1.6)
    pv = m.probabilidad_vivo(np.array([4, 4, 4]), np.array([20.0, 20.0, 20.0]), np.array([20.0, 40.0, 75.0]))
    assert pv[0] > pv[1] > pv[2]


def test_bgnbd_formula_2f1_coincide_con_cuadratura_gauss():
    """Verifica que la solución cerrada con 2F1 (Fader & Hardie 2005, Ec. 10) coincide con la integral 1D exacta."""
    m = ModeloBGNBD(r=0.85, alpha=5.5, a=1.6, b=2.4)
    x = np.array([0, 1, 3, 8], dtype=float)
    tx = np.array([0.0, 10.0, 28.0, 50.0], dtype=float)
    T = np.array([30.0, 30.0, 45.0, 60.0], dtype=float)

    cerrada = m.compras_esperadas(26.0, x, tx, T, usar_cuadratura=False)
    cuadratura = m.compras_esperadas(26.0, x, tx, T, usar_cuadratura=True)
    np.testing.assert_allclose(cerrada, cuadratura, rtol=1e-5)


def test_bgnbd_distribucion_frecuencia_suma_uno():
    m = ModeloBGNBD(r=0.7, alpha=5.0, a=0.3, b=1.8)
    T = np.linspace(5.0, 75.0, 100)
    dist = m.distribucion_frecuencia_teorica(T, max_x=7)
    assert len(dist) == 8
    assert np.all(dist >= 0.0)
    assert float(np.sum(dist)) == pytest.approx(1.0, abs=1e-6)


def test_gamma_gamma_shrinkage_entre_media_poblacional_y_observada():
    gg = ModeloGammaGamma(p=4.0, q=3.5, gamma=200.0)
    mu_0 = gg.ticket_medio_poblacional  # (4 * 200) / 2.5 = 320 GBP
    assert mu_0 == pytest.approx(320.0)

    # Cliente con ticket alto (1000 GBP): su esperanza debe quedar entre 320 y 1000, y acercarse a 1000 al subir x
    esp = gg.ticket_esperado(np.array([1, 3, 10, 50]), np.full(4, 1000.0))
    assert np.all((esp > mu_0) & (esp < 1000.0))
    assert esp[0] < esp[1] < esp[2] < esp[3]


def test_ajuste_mle_sobre_cartera_sintetica(cartera_sintetica_btyd):
    bgnbd = ModeloBGNBD().ajustar(
        cartera_sintetica_btyd["frecuencia"].to_numpy(),
        cartera_sintetica_btyd["recencia_semanas"].to_numpy(),
        cartera_sintetica_btyd["edad_semanas"].to_numpy(),
    )
    gg = ModeloGammaGamma().ajustar(
        cartera_sintetica_btyd["frecuencia"].to_numpy(),
        cartera_sintetica_btyd["ticket_medio_acotado"].to_numpy(),
    )
    assert bgnbd.r > 0 and bgnbd.alpha > 0 and bgnbd.a > 0 and bgnbd.b > 0
    assert np.isfinite(bgnbd.log_verosimilitud_)
    assert gg.p > 0 and gg.q > 1.0 and gg.gamma > 0

    clv_6m = calcular_clv(
        bgnbd, gg,
        cartera_sintetica_btyd["frecuencia"].to_numpy(),
        cartera_sintetica_btyd["recencia_semanas"].to_numpy(),
        cartera_sintetica_btyd["edad_semanas"].to_numpy(),
        cartera_sintetica_btyd["ticket_medio_acotado"].to_numpy(),
        horizonte_meses=6,
    )
    clv_12m = calcular_clv(
        bgnbd, gg,
        cartera_sintetica_btyd["frecuencia"].to_numpy(),
        cartera_sintetica_btyd["recencia_semanas"].to_numpy(),
        cartera_sintetica_btyd["edad_semanas"].to_numpy(),
        cartera_sintetica_btyd["ticket_medio_acotado"].to_numpy(),
        horizonte_meses=12,
    )
    assert np.all(clv_12m["clv_neto"] >= clv_6m["clv_neto"])


def test_clv_sensibilidad_margen_y_descuento():
    bgnbd = ModeloBGNBD(r=0.7, alpha=9.5, a=0.15, b=3.3)
    gg = ModeloGammaGamma(p=2.3, q=3.8, gamma=485.0)
    x = np.array([2.0, 5.0])
    tx = np.array([20.0, 40.0])
    T = np.array([30.0, 45.0])
    m = np.array([300.0, 500.0])

    clv_m30 = calcular_clv(bgnbd, gg, x, tx, T, m, margen_bruto=0.30, tasa_descuento_mensual=0.01)
    clv_m50 = calcular_clv(bgnbd, gg, x, tx, T, m, margen_bruto=0.50, tasa_descuento_mensual=0.01)
    clv_d05 = calcular_clv(bgnbd, gg, x, tx, T, m, margen_bruto=0.30, tasa_descuento_mensual=0.05)

    assert np.all(clv_m50["clv_neto"] > clv_m30["clv_neto"])
    assert np.all(clv_d05["clv_neto"] < clv_m30["clv_neto"])


def test_gamma_gamma_peso_credibilidad_monotono():
    gg = ModeloGammaGamma(p=2.3, q=3.8, gamma=485.0)
    pesos = gg.peso_credibilidad(np.array([0, 1, 2, 5, 20]))
    assert pesos[0] == 0.0
    assert pesos[1] < pesos[2] < pesos[3] < pesos[4] < 1.0


def test_modelos_base_y_supervisado(cartera_sintetica_btyd):
    ritmo = ReglaRitmoHistorico().predecir(cartera_sintetica_btyd)
    rfm = ReglaQuintilesRFM().ajustar(cartera_sintetica_btyd).predecir(cartera_sintetica_btyd)
    sup = ModeloSupervisadoCLV().ajustar(cartera_sintetica_btyd).predecir(cartera_sintetica_btyd)
    for pred in [ritmo, rfm, sup]:
        assert len(pred["ingreso_esperado"]) == len(cartera_sintetica_btyd)
        assert np.all(pred["ingreso_esperado"] >= 0.0)
