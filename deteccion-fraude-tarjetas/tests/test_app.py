"""La app Streamlit arranca sin excepciones y muestra las dos políticas."""

from __future__ import annotations

from fraude.config import RAIZ_PROYECTO


def test_app_streamlit_arranca() -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(RAIZ_PROYECTO / "app" / "app.py"), default_timeout=120).run()
    assert not at.exception
    etiquetas = [m.label for m in at.metric]
    assert "Umbral por costo" in etiquetas and "Pérdida esperada" in etiquetas
    at.sidebar.slider[0].set_value(50.0).run()
    assert not at.exception
