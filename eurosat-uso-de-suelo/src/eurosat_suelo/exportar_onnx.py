"""Exporta las dos redes a ONNX, verifica que sus salidas coinciden con PyTorch y mide la latencia en CPU.

- mobilenetv3_eurosat_finetuned.onnx: modelo 4 (fine-tuning) -> logits de 10 clases.
- mobilenetv3_imagenet_extractor.onnx: backbone ImageNet congelado -> embedding de 576-d (modelos 2 y 3).

Ambos reciben parches crudos (N, 64, 64, 3) float32 en [0, 255]: el reescalado a 96x96 y la normalización
van dentro del grafo, así la API y la app infieren con onnxruntime sin depender de torch.
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from eurosat_suelo.config import ARCHIVO_ONNX_EXTRACTOR, ARCHIVO_ONNX_FT, DIR_MODELOS
from eurosat_suelo.modelos import RedConPreproceso, crear_extractor_imagenet
from eurosat_suelo.predecir import crear_sesion_onnx

HILOS_CPU = 4


def _exportar(modelo: torch.nn.Module, ruta: Path) -> None:
    modelo.eval()
    ejemplo = torch.zeros(2, 64, 64, 3, dtype=torch.float32)
    torch.onnx.export(
        modelo,
        (ejemplo,),
        str(ruta),
        input_names=["parches_rgb_0_255"],
        output_names=["salida"],
        dynamic_axes={"parches_rgb_0_255": {0: "n"}, "salida": {0: "n"}},
        opset_version=18,
        dynamo=False,
    )


def exportar_modelos_onnx(red_ft: torch.nn.Module) -> dict[str, Path]:
    """Exporta el modelo fine-tuned y el extractor ImageNet (con preprocesamiento incluido) a models/."""
    DIR_MODELOS.mkdir(parents=True, exist_ok=True)
    rutas = {"finetuned": DIR_MODELOS / ARCHIVO_ONNX_FT, "extractor": DIR_MODELOS / ARCHIVO_ONNX_EXTRACTOR}
    _exportar(RedConPreproceso(red_ft), rutas["finetuned"])
    _exportar(RedConPreproceso(crear_extractor_imagenet()), rutas["extractor"])
    return rutas


def _mediana_ms(funcion, repeticiones: int) -> float:
    funcion()  # calentamiento
    tiempos = []
    for _ in range(repeticiones):
        t0 = time.perf_counter()
        funcion()
        tiempos.append((time.perf_counter() - t0) * 1000.0)
    return float(statistics.median(tiempos))


def verificar_y_medir_onnx(red_ft: torch.nn.Module, imagenes_uint8: np.ndarray, n_verificacion: int = 512) -> dict[str, Any]:
    """Compara salidas PyTorch vs ONNX en imágenes reales de prueba y mide latencia (mediana) en CPU."""
    torch.set_num_threads(HILOS_CPU)
    envoltorio_pt = RedConPreproceso(red_ft).eval()
    sesion = crear_sesion_onnx(DIR_MODELOS / ARCHIVO_ONNX_FT, hilos=HILOS_CPU)
    nombre_in = sesion.get_inputs()[0].name

    x = imagenes_uint8[:n_verificacion].astype(np.float32)
    with torch.no_grad():
        logits_pt = envoltorio_pt(torch.from_numpy(x)).numpy()
    logits_onnx = sesion.run(None, {nombre_in: x})[0]
    dif_max = float(np.max(np.abs(logits_pt - logits_onnx)))
    coinciden_clases = float(np.mean(logits_pt.argmax(1) == logits_onnx.argmax(1)))

    resultado: dict[str, Any] = {
        "imagenes_verificadas": int(len(x)),
        "diferencia_max_abs_logits": dif_max,
        "pct_misma_clase_predicha": round(coinciden_clases * 100.0, 2),
        "hilos_cpu": HILOS_CPU,
        "tamano_onnx_finetuned_mb": round((DIR_MODELOS / ARCHIVO_ONNX_FT).stat().st_size / 1e6, 2),
        "tamano_onnx_extractor_mb": round((DIR_MODELOS / ARCHIVO_ONNX_EXTRACTOR).stat().st_size / 1e6, 2),
        "latencias": [],
    }
    for lote, reps in ((1, 300), (64, 30)):
        xb = x[:lote]
        xb_t = torch.from_numpy(xb)

        def correr_pt() -> None:
            with torch.no_grad():
                envoltorio_pt(xb_t)

        def correr_onnx() -> None:
            sesion.run(None, {nombre_in: xb})

        ms_pt = _mediana_ms(correr_pt, reps)
        ms_onnx = _mediana_ms(correr_onnx, reps)
        resultado["latencias"].append(
            {
                "tamano_lote": lote,
                "pytorch_ms_por_lote": round(ms_pt, 3),
                "onnx_ms_por_lote": round(ms_onnx, 3),
                "pytorch_ms_por_imagen": round(ms_pt / lote, 3),
                "onnx_ms_por_imagen": round(ms_onnx / lote, 3),
                "aceleracion_onnx": round(ms_pt / ms_onnx, 2),
            }
        )
    return resultado
