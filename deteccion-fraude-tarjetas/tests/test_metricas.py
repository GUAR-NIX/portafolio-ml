"""Las métricas ponderadas coinciden con scikit-learn y el bootstrap no produce NaN."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from fraude import metricas as M


@pytest.fixture
def datos():
    rng = np.random.default_rng(3)
    y = (rng.random(5000) < 0.02).astype(int)
    s = rng.normal(size=5000) + 1.5 * y
    s[:200] = np.round(s[:200], 1)  # empates
    return y, s


def test_auprc_y_roc_igual_que_sklearn(datos) -> None:
    y, s = datos
    ev = M.evaluar(y, s)
    assert ev["auprc"] == pytest.approx(average_precision_score(y, s), abs=1e-12)
    assert ev["roc_auc"] == pytest.approx(roc_auc_score(y, s), abs=1e-12)


def test_pesos_enteros_equivalen_a_repetir_filas(datos) -> None:
    y, s = datos
    w = np.random.default_rng(5).integers(0, 3, len(y))
    yy, ss = np.repeat(y, w), np.repeat(s, w)
    ev = M.metricas_ponderadas(y, M.OrdenPuntajes.desde(s), w)
    assert ev["auprc"] == pytest.approx(average_precision_score(yy, ss), abs=1e-12)
    assert ev["roc_auc"] == pytest.approx(roc_auc_score(yy, ss), abs=1e-12)


def test_precision_at_k_y_recall_a_precision(datos) -> None:
    y, s = datos
    ev = M.evaluar(y, s, k=50, precisiones=(0.5,))
    top = np.argsort(-s, kind="mergesort")[:50]
    assert ev["precision_at_50"] == pytest.approx(y[top].mean())
    p, r, _ = precision_recall_curve(y, s)
    assert ev["recall_a_precision_50"] == pytest.approx(r[p >= 0.5].max())


def test_clasificador_perfecto_y_aleatorio() -> None:
    y = np.r_[np.zeros(990), np.ones(10)].astype(int)
    assert M.evaluar(y, y.astype(float))["auprc"] == pytest.approx(1.0)
    azar = np.mean([M.evaluar(y, np.random.default_rng(i).random(1000))["auprc"] for i in range(30)])
    assert azar < 0.05  # la AUPRC al azar ronda la tasa de positivos (1 %), no 0,5


def test_bootstrap_pareado_sin_nan_e_ic_contiene_punto(datos) -> None:
    y, s = datos
    ruido = s + np.random.default_rng(1).normal(scale=2, size=len(s))
    boot = M.bootstrap(y, {"a": s, "b": ruido}, k=50, n_rep=200)
    assert not boot.drop(columns=["replica", "modelo"]).isna().any().any()
    ic = M.intervalos(boot).set_index(["modelo", "metrica"])
    punto = M.evaluar(y, s, k=50)["auprc"]
    assert ic.loc[("a", "auprc"), "ic_inf"] <= punto <= ic.loc[("a", "auprc"), "ic_sup"]
    dif = M.diferencias_pareadas(boot, [("a", "b")])
    assert bool(dif["significativa"].iloc[0]) and dif["diferencia_media"].iloc[0] > 0


def test_metricas_del_reporte_se_recalculan(pred_prueba, resultados) -> None:
    k = resultados["capacidad"]["k_prueba"]
    y = pred_prueba["Class"].to_numpy()
    for fila in resultados["metricas_prueba"]:
        s = pred_prueba[f"s_{fila['modelo']}"].to_numpy()
        assert average_precision_score(y, s) == pytest.approx(fila["auprc"], abs=1e-9)
        assert roc_auc_score(y, s) == pytest.approx(fila["roc_auc"], abs=1e-9)
        assert fila["auprc_ic_inf"] <= fila["auprc"] <= fila["auprc_ic_sup"]
        assert M.evaluar(y, s, k=k)[f"precision_at_{k}"] == pytest.approx(fila[f"precision_at_{k}"])
