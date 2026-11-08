"""API FastAPI con TestClient: solo necesita models/ y la muestra versionada."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app
from fraude.config import V_COLS


@pytest.fixture(scope="module")
def cliente() -> TestClient:
    return TestClient(app)


def _cuerpo(df: pd.DataFrame, **extra) -> dict:
    return {"transacciones": df[V_COLS + ["Amount"]].to_dict("records"), **extra}


def test_salud(cliente) -> None:
    r = cliente.get("/salud")
    assert r.status_code == 200 and r.json()["estado"] == "ok"


def test_puntuar_devuelve_un_resultado_por_transaccion(cliente, muestra) -> None:
    r = cliente.post("/puntuar", json=_cuerpo(muestra.head(7)))
    assert r.status_code == 200
    res = r.json()["resultados"]
    assert len(res) == 7
    for x in res:
        assert 0 <= x["puntaje_modelo"] <= 1 and 0 <= x["probabilidad_fraude_calibrada"] <= 1
        assert 0 <= x["percentil_anomalia_ae"] <= 100


def test_puntajes_iguales_a_las_predicciones_de_prueba(cliente, muestra, pred_prueba, resultados) -> None:
    """La API reproduce exactamente los puntajes del modelo servido guardados en reports/."""
    m = muestra.head(50)
    r = cliente.post("/puntuar", json=_cuerpo(m)).json()["resultados"]
    ref = pred_prueba.set_index(["Time", "Amount"])[f"s_{resultados['modelo_servido']}"]
    esperado = [ref.loc[(t, a)] for t, a in zip(m["Time"], m["Amount"])]
    esperado = [e.iloc[0] if isinstance(e, pd.Series) else e for e in esperado]
    assert np.allclose([x["puntaje_modelo"] for x in r], esperado, atol=1e-9)


def test_api_separa_fraudes_de_normales(cliente, muestra) -> None:
    r = cliente.post("/puntuar", json=_cuerpo(muestra)).json()["resultados"]
    p = np.array([x["puntaje_modelo"] for x in r])
    y = muestra["Class"].to_numpy()
    assert np.median(p[y == 1]) > 100 * np.median(p[y == 0])
    alerta = np.array([x["alerta_umbral_costo"] for x in r])
    assert alerta[y == 1].mean() > 0.6 and alerta[y == 0].mean() < 0.05


def test_costo_de_revision_cambia_la_regla_de_perdida_esperada(cliente, muestra) -> None:
    m = muestra[muestra.Class == 1]
    barato = cliente.post("/puntuar", json=_cuerpo(m, costo_revision_eur=0.01)).json()["resultados"]
    caro = cliente.post("/puntuar", json=_cuerpo(m, costo_revision_eur=500.0)).json()["resultados"]
    assert sum(x["alerta_perdida_esperada"] for x in barato) > sum(x["alerta_perdida_esperada"] for x in caro)
    # el umbral por puntaje no depende del costo que se pase en la solicitud
    assert [x["alerta_umbral_costo"] for x in barato] == [x["alerta_umbral_costo"] for x in caro]


@pytest.mark.parametrize(
    "cuerpo",
    [
        {"transacciones": []},
        {"transacciones": [{"V1": 0.0, "Amount": 10.0}]},
        {"transacciones": [{**{c: 0.0 for c in V_COLS}, "Amount": -1.0}]},
        {"transacciones": [{**{c: 0.0 for c in V_COLS}, "Amount": 1.0}], "fraccion_perdida": 2.0},
    ],
)
def test_validacion_de_entrada(cliente, cuerpo) -> None:
    assert cliente.post("/puntuar", json=cuerpo).status_code == 422
