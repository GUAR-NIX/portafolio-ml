"""Inferencia en producción para parches Sentinel-2 (EuroSAT) en CPU con onnxruntime.

Usa tres artefactos versionados en models/ (pesan unos 13 MB en total):
- clasificador_eurosat.joblib: GBDT espectral, sonda lineal, MLP de fusión, temperaturas, peso del ensamble y tau.
- mobilenetv3_imagenet_extractor.onnx: embeddings de 576-d para el MLP de fusión (modelo 3).
- mobilenetv3_eurosat_finetuned.onnx: logits del fine-tuning (modelo 4).
Si falta alguno, se lanza ArtefactoNoEncontrado con la instrucción para regenerarlo (nunca se cae en silencio
a un modelo peor).
"""

from __future__ import annotations

import base64
import binascii
import io
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from PIL import Image
from scipy.special import softmax

from eurosat_suelo.config import (
    ARCHIVO_JOBLIB,
    ARCHIVO_ONNX_EXTRACTOR,
    ARCHIVO_ONNX_FT,
    CLASES_EUROSAT,
    DIR_MODELOS,
    DIR_MUESTRA,
    MACRO_CATEGORIA,
    NOMBRES_ESPANOL,
)
from eurosat_suelo.datos import calcular_descriptores_espectrales_lote, extraer_vector_handcrafted_imagen

INSTRUCCION_REGENERAR = (
    "Los modelos se versionan en models/. Si los borraste, regéneralos con "
    "`python descargar_datos.py` y `python -m eurosat_suelo.entrenar` (unos 30 min en CPU)."
)


class ArtefactoNoEncontrado(FileNotFoundError):
    """Falta un archivo de modelo necesario para la inferencia."""


def _exigir(ruta: Path) -> Path:
    if not ruta.exists():
        raise ArtefactoNoEncontrado(f"No se encontró {ruta.name} en {ruta.parent.name}/. {INSTRUCCION_REGENERAR}")
    return ruta


@lru_cache(maxsize=1)
def cargar_artefactos_inferencia() -> dict[str, Any]:
    """Carga (una sola vez) el paquete joblib y las dos sesiones ONNX."""
    paquete = dict(joblib.load(_exigir(DIR_MODELOS / ARCHIVO_JOBLIB)))
    paquete["sesion_finetuned"] = crear_sesion_onnx(_exigir(DIR_MODELOS / ARCHIVO_ONNX_FT))
    paquete["sesion_extractor"] = crear_sesion_onnx(_exigir(DIR_MODELOS / ARCHIVO_ONNX_EXTRACTOR))
    return paquete


def crear_sesion_onnx(ruta: Path, hilos: int = 4):
    """Sesión de onnxruntime en CPU (mismo número de hilos que PyTorch en el entrenamiento)."""
    import onnxruntime as ort

    opciones = ort.SessionOptions()
    opciones.intra_op_num_threads = hilos
    opciones.inter_op_num_threads = 1
    return ort.InferenceSession(str(ruta), sess_options=opciones, providers=["CPUExecutionProvider"])


def _a_rgb64(im: Image.Image) -> np.ndarray:
    return np.asarray(im.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR), dtype=np.uint8)


def decodificar_entrada_a_rgb64(entrada: np.ndarray | bytes | str | Path) -> np.ndarray:
    """Convierte arreglo, ruta, bytes, nombre de archivo de muestra o cadena base64 en uint8 (64, 64, 3)."""
    if isinstance(entrada, np.ndarray):
        if entrada.shape == (64, 64, 3) and entrada.dtype == np.uint8:
            return entrada
        return _a_rgb64(Image.fromarray(entrada.astype(np.uint8)))

    if isinstance(entrada, Path):
        with Image.open(entrada) as im:
            return _a_rgb64(im)

    if isinstance(entrada, str):
        posible_ruta = DIR_MUESTRA / entrada
        if posible_ruta.exists():
            with Image.open(posible_ruta) as im:
                return _a_rgb64(im)
        try:
            raw_bytes = base64.b64decode(entrada, validate=True)
        except binascii.Error as exc:
            raise ValueError(f"'{entrada[:40]}' no es un archivo de data/muestra_eurosat ni una imagen en base64.") from exc
        with Image.open(io.BytesIO(raw_bytes)) as im:
            return _a_rgb64(im)

    if isinstance(entrada, (bytes, bytearray)):
        with Image.open(io.BytesIO(entrada)) as im:
            return _a_rgb64(im)

    raise TypeError(f"Tipo de entrada no soportado: {type(entrada)}")


def probabilidades_ensamble(lote_uint8: np.ndarray) -> np.ndarray:
    """Probabilidades calibradas del modelo 5 (ensamble M4 + M3) para un lote (N, 64, 64, 3)."""
    art = cargar_artefactos_inferencia()
    temps = art["temperaturas"]
    w = float(art["peso_ensamble_finetuning"])
    x = lote_uint8.astype(np.float32)

    # Modelo 4: fine-tuning con TTA horizontal (promedio de logits original + espejo), igual que en el entrenamiento
    sesion_ft = art["sesion_finetuned"]
    nombre_ft = sesion_ft.get_inputs()[0].name
    logits_ft = sesion_ft.run(None, {nombre_ft: np.concatenate([x, x[:, :, ::-1, :].copy()])})[0]
    n = len(x)
    logits_ft = 0.5 * (logits_ft[:n] + logits_ft[n:])
    p4 = softmax(logits_ft.astype(np.float64) / temps["m4"], axis=1)

    # Modelo 3: fusión de embeddings ImageNet (ONNX) + 54 índices espectrales con MLP
    sesion_ex = art["sesion_extractor"]
    emb = sesion_ex.run(None, {sesion_ex.get_inputs()[0].name: x})[0]
    x_spec = extraer_vector_handcrafted_imagen(lote_uint8)
    p3_raw = art["modelo_fusion_hibrido"].predict_proba(np.hstack([emb, x_spec]))
    p3 = softmax(np.log(np.clip(p3_raw, 1e-7, 1.0)) / temps["m3"], axis=1)

    mezcla = w * p4 + (1.0 - w) * p3
    return softmax(np.log(np.clip(mezcla, 1e-7, 1.0)) / temps["m5"], axis=1).astype(np.float32)


def predecir_parche_satelital(
    entrada: np.ndarray | bytes | str | Path,
    umbral_confianza: float | None = None,
) -> dict[str, Any]:
    """Clasifica un parche Sentinel-2 de 64x64, calcula sus índices espectrales y aplica la regla de abstención."""
    img_rgb = decodificar_entrada_a_rgb64(entrada)
    lote = img_rgb[None, ...]

    artefactos = cargar_artefactos_inferencia()
    tau = float(umbral_confianza if umbral_confianza is not None else artefactos["umbral_abstencion_optimo"])

    desc = calcular_descriptores_espectrales_lote(lote)
    probs = probabilidades_ensamble(lote)[0]

    orden = np.argsort(probs)[::-1]
    idx_top1 = int(orden[0])
    clase_top1 = CLASES_EUROSAT[idx_top1]
    conf_top1 = float(probs[idx_top1])
    alerta_velo = bool(desc["velo_atmosferico"][0])

    # Entropía normalizada de Shannon en [0, 1]
    entropia = float(-np.sum(probs * np.log(np.clip(probs, 1e-9, 1.0))) / np.log(len(CLASES_EUROSAT)))

    requiere_triaje = (conf_top1 < tau) or (alerta_velo and conf_top1 < min(0.88, tau + 0.15))
    motivo_triaje = None
    if requiere_triaje:
        if alerta_velo and conf_top1 < tau:
            motivo_triaje = "Baja confianza y posible velo / bajo contraste"
        elif alerta_velo:
            motivo_triaje = "Posible velo o bruma detectada en bandas visibles"
        else:
            motivo_triaje = f"Confianza máxima ({conf_top1:.3f}) inferior al umbral ({tau:.2f})".replace(".", ",")

    return {
        "clase_predicha": clase_top1,
        "nombre_es": NOMBRES_ESPANOL[clase_top1],
        "macro_categoria": MACRO_CATEGORIA[clase_top1],
        "confianza": round(conf_top1, 4),
        "entropia_normalizada": round(entropia, 4),
        "umbral_aplicado": round(tau, 2),
        "decision_operativa": "DERIVAR_FOTOINTERPRETE_SIG" if requiere_triaje else "CLASIFICACION_AUTOMATICA",
        "motivo_triaje": motivo_triaje,
        "probabilidades": {c: round(float(probs[i]), 4) for i, c in enumerate(CLASES_EUROSAT)},
        "top_3": [
            {
                "clase": CLASES_EUROSAT[int(i)],
                "nombre_es": NOMBRES_ESPANOL[CLASES_EUROSAT[int(i)]],
                "macro_categoria": MACRO_CATEGORIA[CLASES_EUROSAT[int(i)]],
                "probabilidad": round(float(probs[int(i)]), 4),
            }
            for i in orden[:3]
        ],
        "telemetria_espectral": {
            "exg_vegetacion": round(float(desc["exg"][0]), 4),
            "vari_vegetacion": round(float(desc["vari"][0]), 4),
            "indice_agua": round(float(desc["indice_agua"][0]), 4),
            "brillo": round(float(desc["brillo"][0]), 4),
            "contraste": round(float(desc["contraste"][0]), 4),
            "rugosidad_espacial": round(float(desc["rugosidad"][0]), 4),
            "alerta_velo_atmosferico": alerta_velo,
        },
    }
