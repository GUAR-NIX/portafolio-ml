"""Pruebas de la app Streamlit: se importa sin ejecutarse, busca clientes y coincide con los resultados guardados."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

RUTA_APP = Path(__file__).resolve().parents[1] / "app" / "app.py"


@pytest.fixture(scope="module")
def app_mod():
    spec = importlib.util.spec_from_file_location("app_clv", RUTA_APP)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)  # no llama a main(): solo define funciones
    return modulo


def test_app_importable_y_busca_clientes(app_mod):
    cartera = app_mod.cargar_cartera()
    assert len(cartera) == 4944
    assert app_mod.buscar_cliente(cartera, "no-existe") is None
    fila = app_mod.buscar_cliente(cartera, " 13093 ")
    assert fila is not None and fila["customer_id"] == "13093"


def test_ficha_coincide_con_la_cartera_guardada(app_mod):
    cartera = app_mod.cargar_cartera()
    for id_cliente in cartera.groupby("segmento")["customer_id"].first():
        fila = app_mod.buscar_cliente(cartera, id_cliente)
        v = app_mod.ficha_cliente(fila)
        assert v["segmento"] == fila["segmento"]
        assert v["clv_neto_gbp"] == pytest.approx(fila["clv_neto"], abs=0.01)
        assert v["prob_vivo"] == pytest.approx(fila["prob_vivo"], abs=1e-3)
        assert v["conviene_contactar_rescate"] == bool(fila["conviene_contactar_rescate"])


def test_app_se_ejecuta_sin_errores():
    testing = pytest.importorskip("streamlit.testing.v1")
    at = testing.AppTest.from_file(str(RUTA_APP), default_timeout=60).run()
    assert not at.exception
    assert any("Acción recomendada" in m.value for m in at.info)
    at.text_input[0].set_value("12346").run()
    assert not at.exception
    assert any("12346" in s.value for s in at.subheader)
