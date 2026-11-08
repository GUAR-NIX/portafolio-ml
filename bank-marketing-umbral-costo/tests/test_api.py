"""Pruebas de la API (requieren el modelo entrenado en models/)."""
from pathlib import Path

import pytest

MODELO = Path(__file__).resolve().parents[1] / "models" / "modelo_campana.joblib"
pytestmark = pytest.mark.skipif(not MODELO.exists(), reason="falta models/ (python -m campana.entrenar)")

CLIENTE = {"edad": 35, "ocupacion": "admin.", "estado_civil": "single", "educacion": "university.degree",
           "mora": "no", "hipoteca": "yes", "prestamo": "no", "tipo_contacto": "cellular", "mes": "may",
           "dia_semana": "thu", "dias_desde_contacto": None, "contactos_previos": 0,
           "resultado_previo": "nonexistent", "var_empleo": -1.8, "ipc": 92.893, "confianza_consumidor": -46.2,
           "euribor_3m": 1.299, "num_empleados": 5099.1}
EXITO_PREVIO = {**CLIENTE, "dias_desde_contacto": 6, "contactos_previos": 2, "resultado_previo": "success"}
AUGE = {**CLIENTE, "var_empleo": 1.4, "ipc": 93.918, "confianza_consumidor": -42.7, "euribor_3m": 4.96,
        "num_empleados": 5228.1, "tipo_contacto": "telephone"}


@pytest.fixture(scope="module")
def cliente():
    from fastapi.testclient import TestClient
    from api.main import app
    return TestClient(app)


def test_salud_y_modelo(cliente):
    assert cliente.get("/salud").json()["estado"] == "ok"
    m = cliente.get("/modelo").json()
    assert "duracion_llamada" in m["excluidas_por_fuga"]
    assert 0.5 < m["metricas_prueba"]["auc"] < 1


def test_puntuar(cliente):
    r = cliente.post("/puntuar", json=CLIENTE)
    assert r.status_code == 200
    d = r.json()
    assert 0 <= d["probabilidad"] <= 1
    assert d["llamar"] == (d["probabilidad"] >= d["umbral"])
    assert d["valor_esperado_usd"] == pytest.approx(d["probabilidad"] * 80 - 8, abs=0.01)
    assert len(d["razones"]) == 3


def test_el_modelo_tiene_sentido_de_negocio(cliente):
    base = cliente.post("/puntuar", json=CLIENTE).json()["probabilidad"]
    exito = cliente.post("/puntuar", json=EXITO_PREVIO).json()
    auge = cliente.post("/puntuar", json=AUGE).json()
    assert exito["probabilidad"] > base and exito["llamar"]       # quien ya aceptó antes, acepta más
    assert auge["probabilidad"] < base and not auge["llamar"]     # euríbor alto (2008): se contrataba menos


def test_validacion(cliente):
    assert cliente.post("/puntuar", json={**CLIENTE, "ocupacion": "astronauta"}).status_code == 422
    assert cliente.post("/puntuar", json={**CLIENTE, "edad": 7}).status_code == 422
    assert cliente.post("/priorizar?costo_llamada=5", json=[CLIENTE]).status_code == 422


def test_priorizar_ordena_y_respeta_capacidad(cliente):
    lote = [CLIENTE, EXITO_PREVIO, AUGE, {**EXITO_PREVIO, "edad": 70, "ocupacion": "retired"}]
    d = cliente.post("/priorizar?capacidad=1", json=lote).json()
    probs = [x["probabilidad"] for x in d]
    assert probs == sorted(probs, reverse=True)
    assert [x["posicion"] for x in d] == [1, 2, 3, 4]
    assert sum(x["llamar"] for x in d) == 1 and d[0]["llamar"]
    assert d[0]["indice"] in (1, 3)


def test_costos_propios_cambian_el_umbral(cliente):
    d = cliente.post("/priorizar?costo_llamada=20&beneficio_exito=80", json=[CLIENTE]).json()[0]
    assert d["umbral"] == pytest.approx(0.25)
    assert d["valor_esperado_usd"] == pytest.approx(d["probabilidad"] * 80 - 20, abs=0.01)


def test_api_y_puntuador_coinciden(cliente):
    from campana.predecir import puntuador
    assert cliente.post("/puntuar", json=CLIENTE).json()["probabilidad"] == \
        puntuador().puntuar([CLIENTE])[0]["probabilidad"]
