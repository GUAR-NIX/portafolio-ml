"""Pruebas de integración sobre los endpoints FastAPI de clasificación satelital."""

from __future__ import annotations

import base64
from pathlib import Path

from fastapi.testclient import TestClient

from api.main import app

cliente = TestClient(app)


def test_endpoint_health() -> None:
    r = cliente.get("/health")
    assert r.status_code == 200
    datos = r.json()
    assert datos["estado"] == "ok"
    assert datos["clases_soportadas"] == 10


def test_endpoint_clases() -> None:
    r = cliente.get("/clases")
    assert r.status_code == 200
    clases = r.json()
    assert len(clases) == 10
    assert clases[0]["clase"] == "AnnualCrop"


def test_endpoint_clasificar_por_archivo_muestra() -> None:
    r = cliente.post(
        "/clasificar",
        json={"id_parcela": "TEST_BOSQUE_01", "archivo_muestra": "01_Forest_01.png"},
    )
    assert r.status_code == 200
    d = r.json()
    assert d["id_parcela"] == "TEST_BOSQUE_01"
    assert d["clase_predicha"] == "Forest"
    assert d["decision_operativa"] in {"CLASIFICACION_AUTOMATICA", "DERIVAR_FOTOINTERPRETE_SIG"}


def test_endpoint_clasificar_por_base64(dir_muestra_png: Path) -> None:
    raw = (dir_muestra_png / "09_SeaLake_01.png").read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    r = cliente.post(
        "/clasificar",
        json={"id_parcela": "TEST_MAR_01", "imagen_base64": b64},
    )
    assert r.status_code == 200
    d = r.json()
    assert d["clase_predicha"] == "SeaLake"
    assert d["confianza"] > 0.70


def test_endpoint_clasificar_lote() -> None:
    r = cliente.post(
        "/clasificar_lote",
        json={
            "parcelas": [
                {"id_parcela": "P1", "archivo_muestra": "01_Forest_01.png"},
                {"id_parcela": "P2", "archivo_muestra": "07_Residential_01.png"},
                {"id_parcela": "P3", "archivo_muestra": "09_SeaLake_01.png"},
            ]
        },
    )
    assert r.status_code == 200
    d = r.json()
    assert d["total_parcelas"] == 3
    assert len(d["resultados"]) == 3


def test_endpoint_clasificar_validacion_error() -> None:
    r = cliente.post("/clasificar", json={"id_parcela": "VACIA"})
    assert r.status_code == 422


def test_endpoint_imagen_invalida_da_400() -> None:
    r = cliente.post("/clasificar", json={"id_parcela": "MALA", "imagen_base64": "esto-no-es-base64!!"})
    assert r.status_code == 400


def test_health_informa_motor_onnx_y_tau_de_validacion(resultados_json: dict) -> None:
    d = cliente.get("/health").json()
    assert d["motor_inferencia"].startswith("onnxruntime")
    assert d["umbral_abstencion_validacion"] == resultados_json["abstencion"]["tau_validacion_m5"]
