"""Costo por error, elección del umbral en validación y regla de pérdida esperada."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraude import costo as C
from fraude.config import COSTO_REVISION, DIR_REPORTES, FRACCION_PERDIDA


def test_costo_total_a_mano() -> None:
    y = np.array([1, 1, 0, 0, 1])
    monto = np.array([100.0, 10.0, 50.0, 5.0, 1.0])
    alerta = np.array([True, False, True, False, False])
    # 2 alertas × 5 EUR + fraudes perdidos (10 + 1)
    assert C.costo_total(y, monto, alerta, 5.0) == pytest.approx(21.0)
    assert C.costo_total(y, monto, alerta, 5.0, fraccion_perdida=0.5) == pytest.approx(15.5)
    d = C.desglose(y, monto, alerta, 5.0)
    assert d["fraudes_detectados"] == 1 and d["fraudes_perdidos"] == 2 and d["falsas_alertas"] == 1
    assert d["costo_sin_modelo_eur"] == pytest.approx(111.0)
    assert d["costo_alertas_eur"] + d["perdida_no_detectada_eur"] == pytest.approx(d["costo_total_eur"])


def test_curva_costo_coincide_con_fuerza_bruta() -> None:
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.05).astype(int)
    monto = rng.exponential(80, 400)
    s = np.round(rng.normal(size=400) + y, 1)  # con empates
    curva = C.curva_costo(y, monto, s, 5.0)
    assert curva.iloc[0]["alertas"] == 0
    assert curva.iloc[0]["costo_eur"] == pytest.approx(monto[y == 1].sum())
    for _, r in curva.iloc[1:].iterrows():
        assert C.costo_total(y, monto, s >= r["umbral"], 5.0) == pytest.approx(r["costo_eur"])
    u, _ = C.elegir_umbral(y, monto, s, 5.0)
    fuerza_bruta = min(C.costo_total(y, monto, s >= t, 5.0) for t in np.r_[np.unique(s), np.inf])
    assert C.costo_total(y, monto, s >= u, 5.0) == pytest.approx(fuerza_bruta)


def test_regla_perdida_esperada_usa_el_monto() -> None:
    p = np.array([0.01, 0.5, 0.5])
    monto = np.array([2000.0, 2.0, 20.0])
    assert C.alerta_por_perdida_esperada(p, monto, 5.0).tolist() == [True, False, True]


def test_umbral_del_reporte_se_eligio_en_validacion(pred_validacion, pred_prueba, resultados) -> None:
    """Re-elige cada umbral con validación y comprueba el costo que se reporta en prueba."""
    costo = {f["modelo"]: f for f in resultados["costo_prueba"]}
    for m in ("logistica", "lof", "autoencoder", resultados["modelo_servido"]):
        u, _ = C.elegir_umbral(pred_validacion["Class"], pred_validacion["Amount"], pred_validacion[f"s_{m}"],
                               COSTO_REVISION, FRACCION_PERDIDA)
        assert u == pytest.approx(costo[m]["umbral"])
        alerta = pred_prueba[f"s_{m}"] >= u
        assert C.costo_total(pred_prueba["Class"], pred_prueba["Amount"], alerta, COSTO_REVISION) == pytest.approx(costo[m]["costo_total_eur"])
        assert int(alerta.sum()) == costo[m]["alertas"]
        assert costo[m]["costo_ic_inf"] <= costo[m]["costo_total_eur"] <= costo[m]["costo_ic_sup"]


def test_sensibilidad_reelige_el_umbral(pred_validacion, pred_prueba) -> None:
    s = pd.read_csv(DIR_REPORTES / "sensibilidad_costo.csv")
    fila = s[(s.modelo == "logistica") & (s.costo_revision_eur == 20.0) & (s.fraccion_perdida == 0.5)].iloc[0]
    u, _ = C.elegir_umbral(pred_validacion["Class"], pred_validacion["Amount"], pred_validacion["s_logistica"], 20.0, 0.5)
    alerta = pred_prueba["s_logistica"] >= u
    assert C.costo_total(pred_prueba["Class"], pred_prueba["Amount"], alerta, 20.0, 0.5) == pytest.approx(fila["costo_total_eur"])
