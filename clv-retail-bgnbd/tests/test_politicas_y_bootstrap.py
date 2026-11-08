"""Pruebas de la validación de políticas con el holdout, el escenario con supuestos y el bootstrap pareado."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from clv import config, evaluacion, segmentacion
from clv.modelos import ModeloBGNBD, ModeloGammaGamma, ReglaRitmoHistorico, calcular_clv

POLITICA_4 = "4. BG/NBD + Gamma-Gamma (zona de riesgo y beneficio esperado > costo)"


@pytest.fixture
def cartera_y_pred(cartera_sintetica_btyd):
    df = cartera_sintetica_btyd
    bgnbd = ModeloBGNBD().ajustar(df["frecuencia"].to_numpy(), df["recencia_semanas"].to_numpy(), df["edad_semanas"].to_numpy())
    gg = ModeloGammaGamma().ajustar(df["frecuencia"].to_numpy(), df["ticket_medio_acotado"].to_numpy())
    pred = calcular_clv(
        bgnbd, gg,
        df["frecuencia"].to_numpy(), df["recencia_semanas"].to_numpy(),
        df["edad_semanas"].to_numpy(), df["ticket_medio_acotado"].to_numpy(),
    )
    cartera, _ = segmentacion.construir_tabla_segmentos(df, pred)
    return cartera, pred


def test_politica_4_es_subconjunto_del_segmento_en_riesgo(cartera_y_pred):
    cartera, _ = cartera_y_pred
    mascara = segmentacion.definir_politicas(cartera)[POLITICA_4]
    assert (cartera.loc[mascara, "segmento"] == "En riesgo de alto valor").all()
    zona = segmentacion.en_zona_riesgo(cartera["prob_vivo"], cartera["frecuencia"])
    assert np.array_equal(zona, (cartera["segmento"] == "En riesgo de alto valor").to_numpy())


def test_validacion_holdout_usa_el_gasto_real(cartera_y_pred):
    cartera, _ = cartera_y_pred
    val = segmentacion.validar_politicas_holdout(cartera, n_bootstrap=200)
    assert len(val) == 4
    politicas = segmentacion.definir_politicas(cartera)
    for (_, fila), mascara in zip(val.iterrows(), politicas.values()):
        assert fila["clientes_contactados"] == int(mascara.sum())
        if not mascara.any():
            continue
        # El margen validado es el gasto real del periodo de prueba × margen, no una predicción del modelo
        esperado = cartera.loc[mascara, "gasto_holdout_acotado"].sum() * config.MARGEN_BRUTO
        assert fila["margen_real_holdout_gbp"] == pytest.approx(esperado)
        assert fila["tasa_recompra_real_pct"] == pytest.approx(cartera.loc[mascara, "vivo_holdout"].mean() * 100)
        if fila["margen_historico_perdido_gbp"] > 0:
            assert fila["tasa_rescate_equilibrio_pct"] == pytest.approx(
                fila["costo_campana_gbp"] / fila["margen_historico_perdido_gbp"] * 100
            )
            assert (
                fila["tasa_rescate_equilibrio_ic95_inf"]
                <= fila["tasa_rescate_equilibrio_pct"]
                <= fila["tasa_rescate_equilibrio_ic95_sup"]
            )


def test_validacion_no_depende_de_las_predicciones_en_politicas_1_a_3(cartera_y_pred):
    cartera, _ = cartera_y_pred
    val = segmentacion.validar_politicas_holdout(cartera, n_bootstrap=0)
    otra = cartera.assign(compras_esperadas=cartera["compras_esperadas"] * 3)
    val2 = segmentacion.validar_politicas_holdout(otra, n_bootstrap=0)
    pd.testing.assert_series_equal(val["margen_real_holdout_gbp"].iloc[:3], val2["margen_real_holdout_gbp"].iloc[:3])


def test_sensibilidad_del_escenario(cartera_y_pred):
    cartera, _ = cartera_y_pred
    sens = segmentacion.sensibilidad_escenario(cartera, efectividades=(0.05, 0.25, 0.40), efectos_fuera=(0.0, 0.04))
    assert len(sens) == 3 * 2 * 4
    for _, grupo in sens.groupby(["politica", "efecto_fuera_zona"]):
        g = grupo.sort_values("efectividad_rescate")["ganancia_neta_gbp"].to_numpy()
        assert np.all(np.diff(g) >= -1e-9)  # más efectividad supuesta nunca reduce la ganancia


def test_bootstrap_modelo_contra_si_mismo_no_es_significativo(cartera_sintetica_btyd, cartera_y_pred):
    _, pred = cartera_y_pred
    tabla = evaluacion.bootstrap_comparacion(cartera_sintetica_btyd, pred, pred, n_bootstrap=50)
    assert set(tabla["metrica"]) == {"auc_recompra", "brier_recompra", "spearman_compras", "spearman_ingreso", "captura_top20_pct"}
    assert np.allclose(tabla["diferencia_a_menos_b"], 0.0)
    assert not tabla["significativa_95"].any()


def test_bootstrap_detecta_un_modelo_peor(cartera_sintetica_btyd, cartera_y_pred):
    _, pred = cartera_y_pred
    rng = np.random.default_rng(0)
    n = len(cartera_sintetica_btyd)
    azar = {
        "prob_vivo": rng.uniform(size=n),
        "compras_esperadas": rng.uniform(size=n),
        "ingreso_esperado": rng.uniform(size=n),
    }
    tabla = evaluacion.bootstrap_comparacion(cartera_sintetica_btyd, pred, azar, n_bootstrap=200).set_index("metrica")
    assert tabla.loc["auc_recompra", "significativa_95"]
    assert tabla.loc["auc_recompra", "ic95_inf"] > 0
    # También funciona con un modelo que no expone prob_recompra (regla de ritmo histórico)
    ritmo = ReglaRitmoHistorico().predecir(cartera_sintetica_btyd)
    assert len(evaluacion.bootstrap_comparacion(cartera_sintetica_btyd, pred, ritmo, n_bootstrap=20)) == 5
