"""Piezas de los modelos: autoencoder en NumPy, submuestreo, variables del híbrido y artefactos versionados."""

from __future__ import annotations

import numpy as np
import pytest

from fraude import modelos as MO
from fraude.config import DIR_MODELOS, V_COLS


def test_autoencoder_guardar_y_cargar(tmp_path) -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 29))
    ae = MO.entrenar_autoencoder(X, capas=(8, 4, 8), epocas=2, verbose=False)
    ae.guardar(tmp_path / "ae.npz")
    ae2 = MO.Autoencoder.cargar(tmp_path / "ae.npz")
    assert np.allclose(ae.puntaje(X[:50]), ae2.puntaje(X[:50]))
    assert (ae.puntaje(X) >= 0).all()


def test_autoencoder_numpy_reproduce_torch() -> None:
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, 29))
    ae = MO.entrenar_autoencoder(X, capas=(8, 4, 8), epocas=1, verbose=False)
    capas = []
    for i, (W, b) in enumerate(ae.pesos):
        lin = torch.nn.Linear(W.shape[1], W.shape[0]).double()
        lin.weight.data, lin.bias.data = torch.from_numpy(W), torch.from_numpy(b)
        capas += [lin] + ([torch.nn.Tanh()] if i < len(ae.pesos) - 1 else [])
    Z = torch.from_numpy((X - ae.media) / ae.desv)
    err_torch = ((torch.nn.Sequential(*capas)(Z) - Z) ** 2).mean(dim=1).detach().numpy()
    assert np.allclose(err_torch, ae.puntaje(X))


def test_autoencoder_versionado_es_ligero_y_puntua_anomalias_mas_alto(muestra) -> None:
    ruta = DIR_MODELOS / "autoencoder.npz"
    assert ruta.stat().st_size < 100_000
    ae = MO.Autoencoder.cargar(ruta)
    X = np.column_stack([muestra[V_COLS].to_numpy(), np.log1p(muestra["Amount"].to_numpy())])
    err = ae.puntaje(X)
    assert np.median(err[muestra["Class"] == 1]) > np.median(err[muestra["Class"] == 0])


def test_submuestreo_conserva_fraudes_y_razon() -> None:
    y = np.r_[np.zeros(10_000), np.ones(20)].astype(int)
    X = np.arange(len(y)).reshape(-1, 1).astype(float)
    Xs, ys = MO.submuestrear(X, y, 20)
    assert ys.sum() == 20 and (ys == 0).sum() == 400
    assert set(Xs[ys == 1, 0]) == set(X[y == 1, 0])


def test_con_anomalias_agrega_dos_columnas() -> None:
    X = np.ones((5, 29))
    H = MO.con_anomalias(X, np.zeros(5), np.ones(5))
    assert H.shape == (5, 31) and np.allclose(H[:, -1], np.log(1 + 1e-9))


def test_oof_no_usa_fraudes_ni_filas_propias() -> None:
    """Las normales se puntúan con detectores que no las vieron: su error no es menor que el de normales nuevas."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(3000, 29))
    y = np.zeros(3000, int)
    y[:15] = 1
    X[:15] += 4
    oof_if, oof_ae = MO.puntajes_anomalia_oof(X, y, n_pliegues=3, epocas_ae=2)
    assert oof_ae[:15].mean() > oof_ae[15:].mean() and oof_if[:15].mean() > oof_if[15:].mean()
    X_nuevo = rng.normal(size=(3000, 29))
    ae_todo = MO.entrenar_autoencoder(X[y == 0], epocas=2, verbose=False)
    ratio = np.median(oof_ae[15:]) / np.median(ae_todo.puntaje(X_nuevo))
    assert ratio > 0.9  # fuera de pliegue ≈ error en datos nuevos (sin sesgo optimista)


def test_metadata_y_booster_versionados() -> None:
    import json

    import lightgbm as lgb

    meta = json.loads((DIR_MODELOS / "metadata.json").read_text(encoding="utf-8"))
    assert {"umbral_costo_validacion", "platt", "variables", "modelo_servido"} <= set(meta)
    b = lgb.Booster(model_file=str(DIR_MODELOS / "lgbm_servido.txt"))
    assert b.num_feature() == len(meta["variables"])
    total = sum(p.stat().st_size for p in DIR_MODELOS.iterdir())
    assert total < 5_000_000
