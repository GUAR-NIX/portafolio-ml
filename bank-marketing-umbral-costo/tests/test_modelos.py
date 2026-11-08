"""Modelos, desbalance y calibración sobre datos sintéticos de resultado conocido."""
import numpy as np
import pytest
from sklearn.metrics import brier_score_loss, roc_auc_score

from campana import config, modelos
from campana import evaluacion as ev


def _Xy(df):
    return df[config.VARIABLES], df[config.OBJETIVO].to_numpy()


def test_regla_segmento_devuelve_tasas_del_segmento(sinteticos):
    X, y = _Xy(sinteticos)
    p = modelos.ReglaSegmento().fit(X, y).predict_proba(X)[:, 1]
    exito = (X["resultado_previo"] == "success").to_numpy()
    assert p[exito].mean() > 3 * p[~exito].mean()


@pytest.mark.parametrize("estrategia", ["submuestreo", "smote"])
def test_remuestreo_balancea_y_respeta_el_esquema(sinteticos, estrategia):
    X, y = _Xy(sinteticos)
    Xr, yr = modelos.remuestrear(X, y, estrategia)
    assert yr.mean() == pytest.approx(0.5)
    assert list(Xr.columns) == list(X.columns)
    # "días desde el contacto" solo existe si hubo contacto previo
    assert Xr.loc[Xr["contactado_antes"] == 0, "dias_desde_contacto"].isna().all()
    assert (Xr["dias_desde_contacto"].dropna() >= 0).all()


@pytest.mark.parametrize("nombre", ["logistica", "lightgbm"])
def test_modelos_aprenden_la_senal(sinteticos, nombre):
    X, y = _Xy(sinteticos)
    oof = ev.fuera_de_pliegue(nombre, "ninguna", X, y, pliegues=3)
    # el techo es el AUC de la probabilidad verdadera con la que se generaron los datos
    techo = roc_auc_score(y, 2.5 * (X["resultado_previo"] == "success") - 0.5 * X["euribor_3m"])
    assert roc_auc_score(y, oof) > techo - 0.04


def test_pesos_inflan_la_probabilidad_y_la_calibracion_lo_corrige(sinteticos):
    X, y = _Xy(sinteticos)
    oof = ev.fuera_de_pliegue("logistica", "pesos", X, y, pliegues=3)
    assert oof.mean() > 2 * y.mean()                     # promete mucho más de lo que pasa
    cal = ev.calibrar_fuera_de_pliegue(oof, y, "isotonica", pliegues=3)
    assert abs(cal.mean() - y.mean()) < 0.02
    assert brier_score_loss(y, cal) < brier_score_loss(y, oof)


@pytest.mark.parametrize("metodo", ["platt", "isotonica"])
def test_calibrador_arregla_probabilidades_deformadas(metodo):
    rng = np.random.default_rng(3)
    p_real = rng.uniform(0, 0.6, 20_000)
    y = (rng.random(p_real.size) < p_real).astype(int)
    deformada = np.sqrt(p_real)                           # sobreestima siempre
    cal = modelos.Calibrador(metodo).fit(deformada, y)
    assert brier_score_loss(y, cal.transform(deformada)) < brier_score_loss(y, deformada) - 0.01


def test_calibrador_intercepto_iguala_la_tasa_y_conserva_el_orden():
    rng = np.random.default_rng(4)
    p = rng.uniform(0.01, 0.3, 5000)
    y = (rng.random(p.size) < 3 * p).astype(int)          # la tasa real es ~3× la predicha
    q = modelos.Calibrador("intercepto").fit(p, y).transform(p)
    assert q.mean() == pytest.approx(y.mean(), abs=1e-4)
    assert (np.argsort(q) == np.argsort(p)).all()


def test_categoria_nueva_no_rompe_lightgbm(sinteticos):
    X, y = _Xy(sinteticos)
    m = modelos.lightgbm(n_estimators=20).fit(X, y)
    nuevo = X.head(3).assign(ocupacion="astronauta")
    p = modelos.probabilidad(m, nuevo)
    assert np.all((p > 0) & (p < 1))
