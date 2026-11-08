"""Las cifras clave del README y la ficha deben salir de reports/ (no escritas a mano).

1. Las métricas de reports/resultados.json se recalculan desde las predicciones guardadas.
2. Las cifras principales aparecen en README.md y MODEL_CARD.md con el formato del texto.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from demanda_citibike.evaluacion import mae, sesgo, wape

RAIZ = Path(__file__).resolve().parents[1]
REPORTES = RAIZ / "reports"

pytestmark = pytest.mark.skipif(
    not (REPORTES / "resultados.json").exists() or not (REPORTES / "predicciones_test.parquet").exists(),
    reason="Ejecuta primero el pipeline",
)


def _es(x: float, d: int = 1) -> str:
    """Formato del README: coma decimal."""
    return f"{x:.{d}f}".replace(".", ",")


def _miles(n: int) -> str:
    return f"{n:,}".replace(",", " ")


@pytest.fixture(scope="module")
def res() -> dict:
    return json.loads((REPORTES / "resultados.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def pred() -> pd.DataFrame:
    return pd.read_parquet(REPORTES / "predicciones_test.parquet")


@pytest.fixture(scope="module")
def readme() -> str:
    return (RAIZ / "README.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("col", ["pred_dia", "pred_dia_sin_clima", "pred_1h", "base_estacional", "base_ayer", "base_media"])
def test_metricas_json_se_recalculan_desde_predicciones(res, pred, col):
    r = pred["salidas"].to_numpy()
    p = pred[col].to_numpy()
    m = res["test_final"][col]
    assert abs(mae(r, p) - m["MAE"]) < 0.006
    assert abs(wape(r, p) - m["WAPE_pct"]) < 0.006
    assert abs(sesgo(r, p) - m["Sesgo_pct"]) < 0.006


def test_predicciones_no_negativas(pred):
    for c in ["pred_dia", "pred_dia_sin_clima", "pred_1h", "pred_q10", "pred_q90"]:
        assert (pred[c] >= 0).all(), c


def test_cobertura_se_recalcula(res, pred):
    cob = np.mean((pred["salidas"] >= pred["pred_q10"]) & (pred["salidas"] <= pred["pred_q90"])) * 100
    assert abs(cob - res["intervalos"]["cobertura_real_pct"]) < 0.06


def test_readme_contiene_cifras_clave(res, readme):
    t = res["test_final"]
    cv = res["backtesting_cv"]
    esperadas = [
        _es(t["pred_dia"]["WAPE_pct"]) + " %",
        _es(t["pred_dia_sin_clima"]["WAPE_pct"]) + " %",
        _es(t["pred_1h"]["WAPE_pct"]) + " %",
        _es(t["base_estacional"]["WAPE_pct"]) + " %",
        _es(t["base_estacional"]["MAE"], 2),
        _es(t["pred_dia"]["MAE"], 2),
        _es(cv["pred_dia"]["WAPE_pct"]) + " %",
        _es(res["intervalos"]["cobertura_real_pct"]) + " %",
        _es(res["mejoras_wape_pct"]["dia_vs_base_estacional_test"]) + " %",
        _miles(res["datos"]["filas_demanda_horaria"]),
        _miles(res["datos"]["horas_por_estacion"]),
    ]
    neg = res["negocio"]["estrategias"]
    for nombre in ["dia_siguiente_media", "dia_siguiente_p90", "linea_base_estacional"]:
        esperadas += [_miles(neg[nombre]["falsas_alarmas"]), _miles(neg[nombre]["picos_perdidos"])]
    faltan = [e for e in esperadas if e not in readme]
    assert not faltan, f"Cifras de reports/ que no aparecen en el README: {faltan}"


def test_readme_sin_cifras_antiguas(readme):
    for vieja in ["8 760", "18%20M", "2 344 viajes", "2,95"]:
        assert vieja not in readme, vieja


def test_model_card_completa(res):
    card = (RAIZ / "MODEL_CARD.md").read_text(encoding="utf-8")
    assert "Se completan" not in card
    assert _es(res["test_final"]["pred_dia"]["WAPE_pct"]) + " %" in card
    assert _es(res["intervalos"]["cobertura_real_pct"]) + " %" in card


def test_seleccion_estaciones_coincide_con_readme(readme):
    sel = json.loads((REPORTES / "seleccion_estaciones.json").read_text(encoding="utf-8"))
    assert _miles(sel["viajes_2024"]) in readme
    assert _es(sel["cuota_salidas_modeladas_pct"]) + " %" in readme
