"""Pruebas de modelos: líneas base y LightGBM."""
from __future__ import annotations

import numpy as np
import pandas as pd

from demanda_citibike.features import COLS_CATEGORICAS, FEATURES, construir_caracteristicas
from demanda_citibike.modelos import entrenar_lgbm, predecir_linea_base_estacional, predecir_linea_base_media


def test_lineas_base(demanda_sintetica, clima_sintetico):
    df = construir_caracteristicas(demanda_sintetica, clima_sintetico)
    base_est = predecir_linea_base_estacional(df)
    base_media = predecir_linea_base_media(df)
    assert len(base_est) == len(df)
    assert len(base_media) == len(df)
    assert np.isfinite(base_est).all()
    assert np.isfinite(base_media).all()


def test_entrenar_y_predecir_lgbm(demanda_sintetica, clima_sintetico):
    df = construir_caracteristicas(demanda_sintetica, clima_sintetico)
    X = df[FEATURES]
    y = df["salidas"]
    modelo = entrenar_lgbm(X, y, COLS_CATEGORICAS, params={"n_estimators": 20, "num_leaves": 7})
    pred = modelo.predict(X)
    assert len(pred) == len(df)
    assert np.isfinite(pred).all()
    assert pred.min() >= 0  # la demanda nunca debe ser negativa


def test_modelo_unico_para_todas_las_estaciones(demanda_sintetica, clima_sintetico):
    """El modelo global usa la estación como característica (un solo modelo)."""
    df = construir_caracteristicas(demanda_sintetica, clima_sintetico)
    X = df[FEATURES]
    y = df["salidas"]
    modelo = entrenar_lgbm(X, y, COLS_CATEGORICAS, params={"n_estimators": 10, "num_leaves": 7})
    # Predecir para todas las estaciones a la vez con el mismo modelo
    pred = modelo.predict(X)
    assert pred.shape[0] == len(df)
