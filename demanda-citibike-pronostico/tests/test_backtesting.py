"""Pruebas de backtesting: partición temporal estricta (sin fuga)."""
from __future__ import annotations

import pandas as pd

from demanda_citibike.features import construir_caracteristicas
from demanda_citibike.backtesting import ejecutar_backtesting


def test_particion_temporal_sin_fuga(demanda_sintetica, clima_sintetico):
    df = construir_caracteristicas(demanda_sintetica, clima_sintetico)
    res = ejecutar_backtesting(df, n_cortes=2, dias_test_final=7, dias_validacion=3)

    cv = res["cv"]
    test = res["test"]
    ini_test_final = res["ini_test_final"]

    # El test final es el periodo más reciente y nunca se superpone con el CV.
    assert cv["fecha_hora"].max() < ini_test_final
    assert test["fecha_hora"].min() >= ini_test_final

    # Las predicciones del CV quedan dentro del rango de backtesting (antes del test final).
    assert cv["fecha_hora"].max() < test["fecha_hora"].min()


def test_predicciones_columnas(demanda_sintetica, clima_sintetico):
    df = construir_caracteristicas(demanda_sintetica, clima_sintetico)
    res = ejecutar_backtesting(df, n_cortes=2, dias_test_final=7, dias_validacion=3)
    for col in ["pred", "base_estacional", "base_media"]:
        assert col in res["cv"].columns
        assert col in res["test"].columns
    assert "modelo_final" in res


def test_test_final_no_visto_en_entrenamiento(demanda_sintetica, clima_sintetico):
    df = construir_caracteristicas(demanda_sintetica, clima_sintetico)
    res = ejecutar_backtesting(df, n_cortes=2, dias_test_final=7, dias_validacion=3)
    # train_final se detiene antes del inicio del test final (menos la validación)
    assert res["train_final"]["fecha_hora"].max() < res["ini_test_final"]
