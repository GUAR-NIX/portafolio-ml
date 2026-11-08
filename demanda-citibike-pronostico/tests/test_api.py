"""API con TestClient: usa el modelo de models/ y los datos versionados de data/procesados y data/raw (clima)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import api.main as api
from demanda_citibike.config import MODELOS, PROCESADOS, RAW

ARCHIVOS = [MODELOS / "modelo_lgbm.joblib", PROCESADOS / "demanda_horaria.parquet",
            PROCESADOS / "estaciones.csv", RAW / "clima_nyc_horario.csv"]

pytestmark = pytest.mark.skipif(not all(a.exists() for a in ARCHIVOS),
                                reason="faltan el modelo o los datos versionados (ver README)")

FECHA = "2024-12-20"


@pytest.fixture(scope="module")
def cliente() -> TestClient:
    return TestClient(api.app)


@pytest.fixture(scope="module")
def estacion(cliente) -> str:
    """W 30 St & 10 Ave: una estación con mucha demanda el 20-12-2024 (la primera de la lista estuvo cerrada)."""
    estaciones = {e["estacion"]: e["nombre"] for e in cliente.get("/estaciones").json()}
    assert estaciones["6459.07"] == "W 30 St & 10 Ave"
    return "6459.07"


def test_salud_ok(cliente):
    r = cliente.get("/salud")
    assert r.status_code == 200
    assert r.json() == {"estado": "ok", "estaciones": 250}


def test_estaciones_trae_nombre_y_coordenadas(cliente):
    e = cliente.get("/estaciones").json()
    assert len(e) == 250
    assert {"estacion", "nombre", "latitud", "longitud"} <= set(e[0])
    assert all(isinstance(x["estacion"], str) for x in e)  # "6459.10" no debe perder el cero


def test_pronostico_devuelve_24_horas_no_negativas(cliente, estacion):
    r = cliente.get(f"/pronostico/{estacion}", params={"fecha": FECHA})
    assert r.status_code == 200
    j = r.json()
    assert j["estacion"] == estacion and j["fecha"] == FECHA
    assert "día siguiente" in j["horizonte"]
    assert len(j["horas"]) == 24
    assert [h["fecha_hora"][11:13] for h in j["horas"]] == [f"{h:02d}" for h in range(24)]
    assert all(h["pronostico_salidas"] >= 0 for h in j["horas"])
    assert j["total_real"] == sum(h["salidas_reales"] for h in j["horas"])
    assert j["total_real"] > 0 and j["total_pronosticado"] > 0
    assert j["total_pronosticado"] == pytest.approx(sum(h["pronostico_salidas"] for h in j["horas"]), abs=12)


def test_pronostico_fecha_con_formato_invalido(cliente, estacion):
    r = cliente.get(f"/pronostico/{estacion}", params={"fecha": "20-12-2024-x"})
    assert r.status_code == 400


def test_pronostico_sin_fecha_es_422(cliente, estacion):
    assert cliente.get(f"/pronostico/{estacion}").status_code == 422


def test_pronostico_estacion_o_fecha_inexistente(cliente, estacion):
    assert cliente.get("/pronostico/no-existe", params={"fecha": FECHA}).status_code == 404
    assert cliente.get(f"/pronostico/{estacion}", params={"fecha": "2030-01-01"}).status_code == 404


def test_sin_archivos_responde_503_claro(monkeypatch, tmp_path):
    """Si faltan el modelo o los datos, /salud lo dice y los demás endpoints devuelven 503 con la receta."""
    monkeypatch.setattr(api, "MODELOS", tmp_path)
    monkeypatch.setitem(api._ESTADO, "cargado", False)
    c = TestClient(api.app)
    assert c.get("/salud").json()["estado"] == "no_listo"
    r = c.get("/estaciones")
    assert r.status_code == 503 and "descargar_datos.py" in r.json()["detail"]
