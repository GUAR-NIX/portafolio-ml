"""Métricas de evaluación del pronóstico de demanda."""
from __future__ import annotations

import numpy as np
import pandas as pd


def mae(real: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(real - pred)))


def rmse(real: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((real - pred) ** 2)))


def wape(real: np.ndarray, pred: np.ndarray) -> float:
    """Weighted Absolute Percentage Error: robusto ante ceros."""
    den = np.sum(real)
    return float(np.sum(np.abs(real - pred)) / den * 100) if den else 0.0


def sesgo(real: np.ndarray, pred: np.ndarray) -> float:
    den = np.sum(real)
    return float(np.sum(pred - real) / den * 100) if den else 0.0


def metricas_completas(real: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": round(mae(real, pred), 2),
        "RMSE": round(rmse(real, pred), 2),
        "WAPE_pct": round(wape(real, pred), 2),
        "Sesgo_pct": round(sesgo(real, pred), 2),
    }


def cobertura(real: np.ndarray, inferior: np.ndarray, superior: np.ndarray) -> float:
    """Fracción de observaciones dentro del intervalo [inferior, superior]."""
    return float(np.mean((real >= inferior) & (real <= superior)) * 100)


def metricas_por_segmento(df: pd.DataFrame, col_segmento: str, real_col: str = "real", pred_col: str = "pred") -> pd.DataFrame:
    """Calcula MAE y WAPE agrupando por una columna de segmento (hora, estación, lluvia...)."""
    filas = []
    for valor, g in df.groupby(col_segmento, sort=False):
        r = g[real_col].to_numpy()
        p = g[pred_col].to_numpy()
        filas.append({
            col_segmento: valor,
            "n": int(len(g)),
            "MAE": round(mae(r, p), 2),
            "WAPE_pct": round(wape(r, p), 2),
        })
    return pd.DataFrame(filas)
