"""Pruebas de las métricas de evaluación."""
from __future__ import annotations

import numpy as np

from demanda_citibike.evaluacion import cobertura, mae, metricas_completas, rmse, sesgo, wape


def test_mae_rmse():
    real = np.array([1.0, 2.0, 3.0])
    pred = np.array([1.0, 3.0, 2.0])
    assert mae(real, pred) == 2.0 / 3.0
    assert abs(rmse(real, pred) - np.sqrt(2.0 / 3.0)) < 1e-9


def test_wape_y_sesgo():
    real = np.array([10.0, 20.0])
    pred = np.array([8.0, 22.0])
    # |10-8| + |20-22| = 4; /30 = 0.1333 -> 13.33%
    assert abs(wape(real, pred) - 13.3333) < 0.01
    # (8-10)+(22-20) = 0 -> sesgo 0
    assert abs(sesgo(real, pred)) < 1e-9


def test_wape_robusto_a_ceros():
    real = np.array([0.0, 10.0])
    pred = np.array([2.0, 10.0])
    assert wape(real, pred) == 20.0  # 2/10 = 20%


def test_metricas_completas():
    real = np.array([5.0, 5.0])
    pred = np.array([4.0, 6.0])
    m = metricas_completas(real, pred)
    assert m["MAE"] == 1.0
    assert m["WAPE_pct"] == 20.0
    assert m["Sesgo_pct"] == 0.0


def test_cobertura():
    real = np.array([1.0, 2.0, 3.0, 4.0])
    inferior = np.array([0.0, 1.0, 2.0, 3.0])
    superior = np.array([2.0, 3.0, 4.0, 5.0])
    assert cobertura(real, inferior, superior) == 100.0
    inferior2 = np.array([2.0, 2.0, 2.0, 2.0])
    superior2 = np.array([2.5, 2.5, 2.5, 2.5])
    assert cobertura(real, inferior2, superior2) == 25.0
