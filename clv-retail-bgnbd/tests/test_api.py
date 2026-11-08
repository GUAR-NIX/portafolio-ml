"""Pruebas de los endpoints FastAPI (/salud, /modelo, /valorar y /priorizar)."""
from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app

cliente = TestClient(app)


def test_endpoint_salud():
    r = cliente.get("/salud")
    assert r.status_code == 200
    assert r.json()["estado"] == "ok"


def test_endpoint_modelo():
    r = cliente.get("/modelo")
    assert r.status_code == 200
    datos = r.json()
    assert "parametros_bgnbd" in datos
    assert "parametros_gamma_gamma" in datos
    assert "metricas_holdout" in datos


def test_endpoint_valorar_cliente_activo_y_en_riesgo():
    # Cliente campeón activo (compró hace 10 días con 8 repeticiones)
    r_activo = cliente.post(
        "/valorar",
        json={
            "id_cliente": "12347",
            "frecuencia": 8,
            "recencia_dias": 390,
            "edad_dias": 400,
            "ticket_medio": 450.0,
        },
    )
    assert r_activo.status_code == 200
    d_act = r_activo.json()
    assert d_act["prob_vivo"] > 0.85
    assert d_act["clv_neto_gbp"] > 50.0
    assert d_act["segmento"] == "Campeones activos"

    # Cliente frecuente que lleva mucho tiempo en silencio (en riesgo / perdido)
    r_silencio = cliente.post(
        "/valorar",
        json={
            "id_cliente": "12348",
            "frecuencia": 8,
            "recencia_dias": 120,
            "edad_dias": 400,
            "ticket_medio": 450.0,
        },
    )
    assert r_silencio.status_code == 200
    d_sil = r_silencio.json()
    assert d_sil["prob_vivo"] < d_act["prob_vivo"]


def test_endpoint_valorar_rechaza_incoherencias():
    # recencia_dias > edad_dias -> 422
    r1 = cliente.post(
        "/valorar",
        json={"frecuencia": 2, "recencia_dias": 500, "edad_dias": 300, "ticket_medio": 200.0},
    )
    assert r1.status_code == 422

    # frecuencia == 0 con recencia_dias > 0 -> 422
    r2 = cliente.post(
        "/valorar",
        json={"frecuencia": 0, "recencia_dias": 50, "edad_dias": 300, "ticket_medio": 200.0},
    )
    assert r2.status_code == 422


def test_endpoint_priorizar_cartera_y_capacidad():
    cartera = {
        "clientes": [
            {"id_cliente": "A", "frecuencia": 1, "recencia_dias": 50, "edad_dias": 350, "ticket_medio": 120.0},
            {"id_cliente": "B", "frecuencia": 10, "recencia_dias": 340, "edad_dias": 350, "ticket_medio": 600.0},
            {"id_cliente": "C", "frecuencia": 4, "recencia_dias": 220, "edad_dias": 350, "ticket_medio": 500.0},
        ]
    }
    r = cliente.post("/priorizar?modo=clv&capacidad=2", json=cartera)
    assert r.status_code == 200
    body = r.json()
    assert body["total_evaluados"] == 3
    assert body["total_seleccionados"] == 2
    assert body["clientes"][0]["id_cliente"] == "B"
    assert body["clientes"][0]["seleccionado"] is True
    assert body["clientes"][2]["seleccionado"] is False
