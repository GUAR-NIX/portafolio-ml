"""Modelos de pronóstico: líneas base y LightGBM global (media y cuantiles)."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

from demanda_citibike.config import CUANTILES, MODELOS

PARAMS_LGBM = {
    "objective": "regression",
    "metric": "mae",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 30,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_estimators": 600,
    "n_jobs": -1,
    "verbosity": -1,
    "random_state": 42,
}


def predecir_linea_base_estacional(df: pd.DataFrame) -> np.ndarray:
    """Línea base ingenua estacional: misma hora de la semana anterior (lag_168)."""
    return df["lag_168"].fillna(0).to_numpy()


def predecir_linea_base_ayer(df: pd.DataFrame) -> np.ndarray:
    """Línea base ingenua diaria: misma hora del día anterior (lag_24)."""
    return df["lag_24"].fillna(0).to_numpy()


def predecir_linea_base_media(df: pd.DataFrame) -> np.ndarray:
    """Línea base media móvil de 3 horas (sobre el pasado; solo válida a 1 hora vista)."""
    return df["media_3"].fillna(0).to_numpy()


def entrenar_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cat_features: list[str],
    valid: tuple[pd.DataFrame, pd.Series] | None = None,
    params: dict | None = None,
) -> lgb.LGBMRegressor:
    """Entrena un LightGBM global (un solo modelo para todas las estaciones)."""
    p = dict(PARAMS_LGBM)
    if params:
        p.update(params)
    modelo = lgb.LGBMRegressor(**p)
    if valid is not None:
        modelo.fit(
            X_train, y_train,
            categorical_feature=cat_features,
            eval_set=[valid],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
    else:
        modelo.fit(X_train, y_train, categorical_feature=cat_features)
    return modelo


def entrenar_cuantiles(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cat_features: list[str],
    cuantiles: list[float] = CUANTILES,
) -> dict[float, lgb.LGBMRegressor]:
    """Entrena modelos LightGBM por cuantil para intervalos de predicción."""
    modelos: dict[float, lgb.LGBMRegressor] = {}
    for q in cuantiles:
        p = dict(PARAMS_LGBM)
        p["objective"] = "quantile"
        p["alpha"] = q
        p["n_estimators"] = 400
        m = lgb.LGBMRegressor(**p)
        m.fit(X_train, y_train, categorical_feature=cat_features)
        modelos[q] = m
    return modelos


def guardar_modelo(modelo: lgb.LGBMRegressor, ruta: Path) -> None:
    """Guarda el estimador completo (incluye la configuración de categóricas) con joblib."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, ruta)


def cargar_modelo(ruta: Path) -> lgb.LGBMRegressor:
    """Carga un estimador LightGBM guardado con joblib."""
    return joblib.load(ruta)
