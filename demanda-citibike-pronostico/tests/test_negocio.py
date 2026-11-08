"""Pruebas del análisis de negocio (rebalanceo)."""
from __future__ import annotations

import pandas as pd

from demanda_citibike.negocio import analizar_rebalanceo, calcular_umbrales, costo, razon_equilibrio


def _df_test() -> pd.DataFrame:
    return pd.DataFrame({
        "estacion": ["S1"] * 10,
        "salidas": [5, 20, 6, 22, 4, 18, 7, 21, 5, 19],
        "pred": [4, 21, 5, 20, 6, 17, 6, 22, 4, 20],
        "base_estacional": [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
    })


def test_umbrales_por_estacion():
    train = pd.DataFrame({"estacion": ["S1"] * 10, "salidas": list(range(10))})
    umbrales = calcular_umbrales(train)
    assert "S1" in umbrales
    assert umbrales["S1"] == 8.1  # percentil 90 de 0..9


def test_rebalanceo_aciertos_y_picos_perdidos():
    stats = analizar_rebalanceo(_df_test(), {"S1": 15.0})
    modelo = stats["estrategias"]["modelo_lightgbm"]
    base = stats["estrategias"]["linea_base_estacional"]
    # El modelo supera 15 justo en las 5 horas pico reales; la base (10) nunca alerta.
    assert modelo["aciertos"] == 5 and modelo["falsas_alarmas"] == 0 and modelo["picos_perdidos"] == 0
    assert base["alertas"] == 0 and base["picos_perdidos"] == 5
    comp = stats["frente_a_referencia"]["modelo_lightgbm"]
    assert comp["picos_perdidos_evitados"] == 5
    assert comp["domina_a_la_referencia"]


def test_costo_y_razon_de_equilibrio():
    conservadora = {"falsas_alarmas": 10, "picos_perdidos": 50}
    agresiva = {"falsas_alarmas": 40, "picos_perdidos": 30}
    assert costo(conservadora, 2) == 110
    # 10 + 50 r = 40 + 30 r  ->  r = 1,5: por encima, la estrategia agresiva cuesta menos.
    assert razon_equilibrio(conservadora, agresiva) == 1.5
    # Si una estrategia domina (menos falsas alarmas y menos picos perdidos) no hay cruce.
    assert razon_equilibrio(agresiva, {"falsas_alarmas": 5, "picos_perdidos": 5}) is None


def test_rebalanceo_claves_presentes():
    stats = analizar_rebalanceo(_df_test(), {"S1": 15.0})
    for k in ["umbral_percentil", "estrategias", "frente_a_referencia", "costo_por_razon", "estrategia_mas_barata_por_razon"]:
        assert k in stats
