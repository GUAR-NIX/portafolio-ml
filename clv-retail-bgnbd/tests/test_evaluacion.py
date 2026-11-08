"""Pruebas de evaluación en holdout, curvas de captura, segmentación y políticas de retención."""
from __future__ import annotations

import numpy as np

from clv import evaluacion, segmentacion
from clv.modelos import ModeloBGNBD, ModeloGammaGamma, ReglaRitmoHistorico, calcular_clv


def test_supuesto_gamma_gamma_devuelve_tres_variantes(cartera_sintetica_btyd):
    tabla = evaluacion.verificar_supuesto_gamma_gamma(cartera_sintetica_btyd)
    assert len(tabla) == 3
    assert {"pearson_r", "spearman_rho", "ticket_medio_gbp"}.issubset(tabla.columns)


def test_evaluacion_y_segmentacion_completas(cartera_sintetica_btyd):
    bgnbd = ModeloBGNBD().ajustar(
        cartera_sintetica_btyd["frecuencia"].to_numpy(),
        cartera_sintetica_btyd["recencia_semanas"].to_numpy(),
        cartera_sintetica_btyd["edad_semanas"].to_numpy(),
    )
    gg = ModeloGammaGamma().ajustar(
        cartera_sintetica_btyd["frecuencia"].to_numpy(),
        cartera_sintetica_btyd["ticket_medio_acotado"].to_numpy(),
    )
    pred_clv = calcular_clv(
        bgnbd, gg,
        cartera_sintetica_btyd["frecuencia"].to_numpy(),
        cartera_sintetica_btyd["recencia_semanas"].to_numpy(),
        cartera_sintetica_btyd["edad_semanas"].to_numpy(),
        cartera_sintetica_btyd["ticket_medio_acotado"].to_numpy(),
    )
    pred_ritmo = ReglaRitmoHistorico().predecir(cartera_sintetica_btyd)

    metricas = evaluacion.evaluar_modelos_holdout(
        cartera_sintetica_btyd,
        {"Ritmo": pred_ritmo, "BG/NBD + GG": pred_clv},
    )
    assert len(metricas) == 2
    assert (metricas["auc_recompra"] > 0.5).all()

    deciles = evaluacion.calibracion_por_deciles(cartera_sintetica_btyd, pred_clv)
    assert len(deciles) == 10
    assert np.isclose(deciles["pct_ingreso_real"].sum(), 100.0)

    cartera, resumen_seg = segmentacion.construir_tabla_segmentos(cartera_sintetica_btyd, pred_clv)
    assert len(cartera) == len(cartera_sintetica_btyd)
    assert int(resumen_seg["n_clientes"].sum()) == len(cartera_sintetica_btyd)

    escenario = segmentacion.simular_escenario_politicas(cartera)
    assert len(escenario) == 4
    # En el escenario (calculado con el propio modelo) la política 4 no puede perder frente a contactar a todos
    assert float(escenario.iloc[3]["ganancia_neta_gbp"]) >= float(escenario.iloc[0]["ganancia_neta_gbp"])
