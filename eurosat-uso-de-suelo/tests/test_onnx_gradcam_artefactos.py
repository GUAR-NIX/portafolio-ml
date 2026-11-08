"""ONNX vs PyTorch, Grad-CAM y comportamiento cuando faltan los modelos."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from PIL import Image

from eurosat_suelo import predecir
from eurosat_suelo.config import ARCHIVO_ONNX_EXTRACTOR, ARCHIVO_ONNX_FT, ARCHIVO_PT, DIR_MODELOS, DIR_MUESTRA, DIR_REPORTES
from eurosat_suelo.modelos import RedConPreproceso, cargar_red_finetuned, crear_extractor_imagenet, grad_cam


@pytest.fixture(scope="module")
def lote_muestra() -> np.ndarray:
    rutas = sorted(DIR_MUESTRA.glob("*.png"))
    return np.stack([np.asarray(Image.open(r).convert("RGB")) for r in rutas]).astype(np.uint8)


@pytest.fixture(scope="module")
def red_ft() -> torch.nn.Module:
    return cargar_red_finetuned(DIR_MODELOS / ARCHIVO_PT)


def test_modelos_versionados_y_ligeros() -> None:
    for nombre in (ARCHIVO_PT, ARCHIVO_ONNX_FT, ARCHIVO_ONNX_EXTRACTOR, "clasificador_eurosat.joblib"):
        ruta = DIR_MODELOS / nombre
        assert ruta.exists(), nombre
        assert ruta.stat().st_size < 15_000_000, nombre


@pytest.mark.parametrize("cual", ["finetuned", "extractor"])
def test_onnx_igual_a_pytorch(cual: str, lote_muestra: np.ndarray, red_ft: torch.nn.Module) -> None:
    if cual == "finetuned":
        modelo_pt, archivo = RedConPreproceso(red_ft).eval(), ARCHIVO_ONNX_FT
    else:
        modelo_pt, archivo = RedConPreproceso(crear_extractor_imagenet()).eval(), ARCHIVO_ONNX_EXTRACTOR
    x = lote_muestra.astype(np.float32)
    with torch.no_grad():
        salida_pt = modelo_pt(torch.from_numpy(x)).numpy()
    sesion = predecir.crear_sesion_onnx(DIR_MODELOS / archivo)
    salida_onnx = sesion.run(None, {sesion.get_inputs()[0].name: x})[0]
    assert salida_onnx.shape == salida_pt.shape
    np.testing.assert_allclose(salida_onnx, salida_pt, atol=1e-3, rtol=1e-3)
    if cual == "finetuned":
        assert np.array_equal(salida_onnx.argmax(1), salida_pt.argmax(1))


def test_reporte_onnx_coherente() -> None:
    onnx = json.loads((DIR_REPORTES / "resultados.json").read_text(encoding="utf-8"))["onnx"]
    assert onnx["imagenes_verificadas"] >= 500
    assert onnx["diferencia_max_abs_logits"] < 1e-3
    assert onnx["pct_misma_clase_predicha"] == 100.0
    assert {l["tamano_lote"] for l in onnx["latencias"]} == {1, 64}


def test_grad_cam_forma_rango_y_clase(lote_muestra: np.ndarray, red_ft: torch.nn.Module) -> None:
    img = lote_muestra[15]  # 03_Highway_01.png (orden alfabético: 5 por clase)
    res = grad_cam(red_ft, img)
    assert res["mapa"].shape == (64, 64)
    assert res["resolucion_capa"] == (6, 6)
    assert float(res["mapa"].min()) >= 0.0 and abs(float(res["mapa"].max()) - 1.0) < 1e-5
    assert float(res["mapa"].std()) > 0.01  # no es un mapa plano
    with torch.no_grad():
        clase_pt = int(RedConPreproceso(red_ft)(torch.from_numpy(img[None].astype(np.float32))).argmax())
    assert res["clase_objetivo"] == clase_pt


def test_grad_cam_depende_de_la_clase_objetivo(lote_muestra: np.ndarray, red_ft: torch.nn.Module) -> None:
    img = lote_muestra[40]  # 08_River_01.png
    mapa_rio = grad_cam(red_ft, img, clase_objetivo=8)["mapa"]
    mapa_bosque = grad_cam(red_ft, img, clase_objetivo=1)["mapa"]
    assert not np.allclose(mapa_rio, mapa_bosque)


def test_falta_modelo_da_error_claro(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(predecir, "DIR_MODELOS", tmp_path)
    predecir.cargar_artefactos_inferencia.cache_clear()
    try:
        with pytest.raises(predecir.ArtefactoNoEncontrado, match="eurosat_suelo.entrenar"):
            predecir.cargar_artefactos_inferencia()
    finally:
        predecir.cargar_artefactos_inferencia.cache_clear()


def test_api_devuelve_503_sin_modelos(tmp_path, monkeypatch) -> None:
    from fastapi.testclient import TestClient

    from api.main import app

    monkeypatch.setattr(predecir, "DIR_MODELOS", tmp_path)
    predecir.cargar_artefactos_inferencia.cache_clear()
    try:
        r = TestClient(app).post("/clasificar", json={"archivo_muestra": "01_Forest_01.png"})
        assert r.status_code == 503
        assert "entrenar" in r.json()["detail"]
    finally:
        predecir.cargar_artefactos_inferencia.cache_clear()


def test_inferencia_onnx_igual_al_pipeline_pytorch(lote_muestra: np.ndarray, red_ft: torch.nn.Module) -> None:
    """La API (ONNX Runtime) reproduce las probabilidades del modelo 5 calculadas con PyTorch al entrenar."""
    import joblib

    from eurosat_suelo.entrenar import logits_cinco_modelos, probs_cinco_modelos

    art = joblib.load(DIR_MODELOS / "clasificador_eurosat.joblib")
    logits, _ = logits_cinco_modelos(art, red_ft, lote_muestra)
    p_pt = probs_cinco_modelos(logits, art["temperaturas"], art["peso_ensamble_finetuning"])["m5"]
    p_onnx = predecir.probabilidades_ensamble(lote_muestra)
    np.testing.assert_allclose(p_onnx, p_pt, atol=2e-3)
    assert np.array_equal(p_onnx.argmax(1), p_pt.argmax(1))
