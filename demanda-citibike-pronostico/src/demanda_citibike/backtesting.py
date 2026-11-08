"""Backtesting con partición temporal estricta (origen móvil).

- 4 cortes semanales de backtesting (origen móvil, ventana de validación para
  *early stopping*): entrenar solo con el pasado y evaluar la semana siguiente.
- Test final: el último mes (28 días) nunca visto por ningún entrenamiento.

Los mismos cortes se usan para todos los conjuntos de predictores (día siguiente,
día siguiente sin clima y 1 hora), así las comparaciones son directas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from demanda_citibike.features import COLS_CATEGORICAS, FEATURES, OBJETIVO
from demanda_citibike.modelos import (
    entrenar_lgbm,
    predecir_linea_base_ayer,
    predecir_linea_base_estacional,
    predecir_linea_base_media,
)


def _entrenar(train: pd.DataFrame, valid: pd.DataFrame, features: list[str], params: dict | None = None):
    cats = [c for c in COLS_CATEGORICAS if c in features]
    return entrenar_lgbm(
        train[features], train[OBJETIVO], cats,
        valid=(valid[features], valid[OBJETIVO]), params=params,
    )


def _predecir(df: pd.DataFrame, modelo, features: list[str]) -> pd.DataFrame:
    # La demanda nunca es negativa: se recortan a 0 las predicciones del regresor.
    pred = np.clip(modelo.predict(df[features]), 0, None)
    return df.assign(
        pred=pred,
        base_estacional=predecir_linea_base_estacional(df),
        base_ayer=predecir_linea_base_ayer(df),
        base_media=predecir_linea_base_media(df),
    )


def fechas_de_corte(df: pd.DataFrame, n_cortes: int = 4, dias_test_final: int = 28) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Devuelve (inicio del backtesting, inicio del test final)."""
    fin = df["fecha_hora"].max()
    ini_test_final = fin - pd.Timedelta(days=dias_test_final)
    ini_backtest = ini_test_final - pd.Timedelta(days=n_cortes * 7)
    return ini_backtest, ini_test_final


def ejecutar_backtesting(
    df: pd.DataFrame,
    n_cortes: int = 4,
    dias_test_final: int = 28,
    dias_validacion: int = 14,
    features: list[str] = FEATURES,
    params: dict | None = None,
) -> dict:
    """Ejecuta backtesting CV y test final, devolviendo predicciones y modelos."""
    df = df.sort_values(["fecha_hora", "estacion"], kind="stable").reset_index(drop=True)
    ini_backtest, ini_test_final = fechas_de_corte(df, n_cortes, dias_test_final)

    # --- Backtesting CV (origen móvil) ---
    partes_cv = []
    for k in range(n_cortes):
        corte = ini_backtest + pd.Timedelta(days=k * 7)
        siguiente = corte + pd.Timedelta(days=7)
        valid = df[(df["fecha_hora"] >= corte - pd.Timedelta(days=dias_validacion)) & (df["fecha_hora"] < corte)]
        train = df[df["fecha_hora"] < corte - pd.Timedelta(days=dias_validacion)]
        modelo = _entrenar(train, valid, features, params)
        test = df[(df["fecha_hora"] >= corte) & (df["fecha_hora"] < siguiente)]
        partes_cv.append(_predecir(test, modelo, features).assign(fold=k + 1))
        print(f"    fold {k + 1}/{n_cortes}: entrenado hasta {corte.date()}, evaluada semana siguiente")

    df_cv = pd.concat(partes_cv, ignore_index=True)

    # --- Modelo final + test final (último mes nunca visto) ---
    valid_final = df[(df["fecha_hora"] >= ini_test_final - pd.Timedelta(days=dias_validacion)) & (df["fecha_hora"] < ini_test_final)]
    train_final = df[df["fecha_hora"] < ini_test_final - pd.Timedelta(days=dias_validacion)]
    modelo_final = _entrenar(train_final, valid_final, features, params)
    test_final = df[df["fecha_hora"] >= ini_test_final]
    df_test = _predecir(test_final, modelo_final, features)

    return {
        "cv": df_cv,
        "test": df_test,
        "modelo_final": modelo_final,
        "train_final": train_final,
        "ini_test_final": ini_test_final,
        "features": list(features),
    }
