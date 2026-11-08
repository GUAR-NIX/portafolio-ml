"""Arquitecturas de clasificación para EuroSAT en CPU:
1. Baseline de Teledetección Clásica (54 índices espectrales + textura + HistGradientBoosting / LogReg)
2. Transfer Learning Congelado (MobileNetV3-Small ImageNet-1K -> 576-d Linear Probe)
3. Transfer Learning Híbrido (Embeddings CNN + Índices Espectrales Sentinel-2 + MLP)
4. Transfer Learning con Fine-Tuning Parcial en CPU + Invarianza Diédrica + Escalamiento por Temperatura

Incluye además Grad-CAM, mapas de oclusión y los envoltorios usados para exportar a ONNX.
"""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torchvision import models, transforms

from eurosat_suelo.config import CLASES_EUROSAT, SEMILLA

torch.set_num_threads(4)

MEDIA_IMAGENET = (0.485, 0.456, 0.406)
STD_IMAGENET = (0.229, 0.224, 0.225)


def construir_transformacion_inferencia(tamano: int = 96) -> transforms.Compose:
    """Transformación determinística para pasar parches Sentinel-2 de 64x64 al backbone preentrenado."""
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((tamano, tamano), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEDIA_IMAGENET, std=STD_IMAGENET),
        ]
    )


def preprocesar_lote_tensor(imagenes_uint8: np.ndarray, tamano: int = 96) -> torch.Tensor:
    """Convierte un arreglo (N, 64, 64, 3) uint8 en un tensor normalizado (N, 3, tamano, tamano) en CPU."""
    arr_copia = np.array(imagenes_uint8, dtype=np.uint8, copy=True)
    x = torch.from_numpy(arr_copia).permute(0, 3, 1, 2).float() / 255.0
    if tamano != 64:
        x = torch.nn.functional.interpolate(x, size=(tamano, tamano), mode="bilinear", align_corners=False)
    media = torch.tensor(MEDIA_IMAGENET, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(STD_IMAGENET, dtype=torch.float32).view(1, 3, 1, 1)
    return (x - media) / std


def crear_backbone_mobilenet_v3(num_clases: int = len(CLASES_EUROSAT), preentrenado: bool = True) -> nn.Module:
    """MobileNetV3-Small con cabezal de 10 clases.

    Con preentrenado=True parte de los pesos de ImageNet-1K (para entrenar); con False crea solo la
    arquitectura, que es lo necesario para cargar el .pt ya entrenado sin descargar nada.
    """
    pesos = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if preentrenado else None
    red = models.mobilenet_v3_small(weights=pesos)
    in_features = red.classifier[0].in_features  # 576
    red.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.Hardswish(inplace=True),
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(256, num_clases),
    )
    return red


@lru_cache(maxsize=1)
def _extractor_en_cache() -> nn.Module:
    return crear_extractor_imagenet()


@torch.no_grad()
def extraer_embeddings_cnn(
    imagenes_uint8: np.ndarray,
    batch_size: int = 128,
    tamano: int = 96,
) -> tuple[np.ndarray, float]:
    """Extrae vectores de 576 dimensiones del backbone MobileNetV3-Small congelado en CPU."""
    extractor = _extractor_en_cache()

    n = len(imagenes_uint8)
    salidas: list[np.ndarray] = []
    t0 = time.perf_counter()
    for ini in range(0, n, batch_size):
        lote = imagenes_uint8[ini : ini + batch_size]
        tensor = preprocesar_lote_tensor(lote, tamano=tamano)
        emb = extractor(tensor).cpu().numpy().astype(np.float32)
        salidas.append(emb)
    duracion_s = time.perf_counter() - t0
    return np.vstack(salidas), duracion_s


def entrenar_modelo_espectral_clasico(x_spec_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    """Modelo 1: Clasificador de gradiente impulsado sobre los 54 descriptores espectrales y de textura."""
    pipe = Pipeline(
        [
            ("escalador", StandardScaler()),
            (
                "clasificador",
                HistGradientBoostingClassifier(
                    max_iter=220,
                    learning_rate=0.07,
                    max_leaf_nodes=31,
                    min_samples_leaf=15,
                    l2_regularization=1.0,
                    random_state=SEMILLA,
                ),
            ),
        ]
    )
    pipe.fit(x_spec_train, y_train)
    return pipe


def entrenar_linear_probe_cnn(x_emb_train: np.ndarray, y_train: np.ndarray, c_reg: float = 0.5) -> Pipeline:
    """Modelo 2: Sonda lineal (Logistic Regression L2) sobre embeddings de 576-d de MobileNetV3-Small."""
    pipe = Pipeline(
        [
            ("escalador", StandardScaler()),
            (
                "clasificador",
                LogisticRegression(
                    C=c_reg,
                    max_iter=600,
                    random_state=SEMILLA,
                ),
            ),
        ]
    )
    pipe.fit(x_emb_train, y_train)
    return pipe


def entrenar_fusion_cnn_espectral(
    x_emb_train: np.ndarray,
    x_spec_train: np.ndarray,
    y_train: np.ndarray,
) -> Pipeline:
    """Modelo 3: Fusión de embeddings profundos (576-d) + índices espectrales/textura Sentinel-2 (54-d) con cabezal MLP."""
    x_fusion = np.hstack([x_emb_train, x_spec_train])
    pipe = Pipeline(
        [
            ("escalador", StandardScaler()),
            (
                "clasificador",
                MLPClassifier(
                    hidden_layer_sizes=(256, 128),
                    activation="relu",
                    alpha=1e-2,
                    batch_size=128,
                    learning_rate_init=1e-3,
                    max_iter=60,
                    early_stopping=True,
                    n_iter_no_change=8,
                    random_state=SEMILLA,
                ),
            ),
        ]
    )
    pipe.fit(x_fusion, y_train)
    return pipe


def _aumentar_lote_diedrico(tensor_bchw: torch.Tensor, generador: torch.Generator) -> torch.Tensor:
    """Aplica aumentación geométrica invariante al nadir (espejos H/V y rotaciones de 90/180/270 grados)."""
    out = tensor_bchw.clone()
    if bool(torch.randint(0, 2, (1,), generator=generador).item()):
        out = torch.flip(out, dims=[3])
    if bool(torch.randint(0, 2, (1,), generator=generador).item()):
        out = torch.flip(out, dims=[2])
    k_rot = int(torch.randint(0, 4, (1,), generator=generador).item())
    if k_rot > 0:
        out = torch.rot90(out, k=k_rot, dims=[2, 3])
    return out


def entrenar_finetuning_parcial_cpu(
    imagenes_train: np.ndarray,
    y_train: np.ndarray,
    imagenes_val: np.ndarray,
    y_val: np.ndarray,
    epocas: int = 8,
    batch_size: int = 128,
    tamano: int = 96,
) -> tuple[nn.Module, list[dict[str, float]]]:
    """Modelo 4: Fine-tuning parcial en CPU descongelando los últimos bloques invertidos de MobileNetV3-Small."""
    torch.manual_seed(SEMILLA)
    np.random.seed(SEMILLA)
    gen = torch.Generator().manual_seed(SEMILLA)

    red = crear_backbone_mobilenet_v3(num_clases=len(CLASES_EUROSAT))

    # Congelar las primeras capas genéricas de bordes y descongelar los últimos 4 bloques semánticos + clasificador
    for param in red.features.parameters():
        param.requires_grad = False
    for bloque in red.features[-4:]:
        for param in bloque.parameters():
            param.requires_grad = True
    for param in red.classifier.parameters():
        param.requires_grad = True

    criterio = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizador = torch.optim.AdamW(
        [
            {"params": [p for p in red.features[-4:].parameters() if p.requires_grad], "lr": 4e-4, "weight_decay": 1e-4},
            {"params": red.classifier.parameters(), "lr": 1.8e-3, "weight_decay": 1e-3},
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizador, T_max=epocas, eta_min=5e-5)

    x_train_t = preprocesar_lote_tensor(imagenes_train, tamano=tamano)
    y_train_t = torch.from_numpy(y_train).long()
    x_val_t = preprocesar_lote_tensor(imagenes_val, tamano=tamano)
    y_val_t = torch.from_numpy(y_val).long()

    n_train = len(x_train_t)
    historial: list[dict[str, float]] = []
    mejor_acc_val = -1.0
    mejor_estado: dict[str, torch.Tensor] | None = None

    for epoca in range(1, epocas + 1):
        t0 = time.perf_counter()
        red.train()
        perm = torch.randperm(n_train, generator=gen)
        perdida_acum = 0.0
        aciertos_train = 0

        for ini in range(0, n_train, batch_size):
            idx = perm[ini : ini + batch_size]
            xb = _aumentar_lote_diedrico(x_train_t[idx], gen)
            yb = y_train_t[idx]

            optimizador.zero_grad()
            logits = red(xb)
            loss = criterio(logits, yb)
            loss.backward()
            optimizador.step()

            perdida_acum += float(loss.item()) * len(yb)
            aciertos_train += int((logits.argmax(dim=1) == yb).sum().item())

        scheduler.step()

        # Evaluación en validación
        red.eval()
        with torch.no_grad():
            logits_val_list: list[torch.Tensor] = []
            for ini in range(0, len(x_val_t), batch_size):
                logits_val_list.append(red(x_val_t[ini : ini + batch_size]))
            logits_val = torch.cat(logits_val_list, dim=0)
            loss_val = float(criterio(logits_val, y_val_t).item())
            acc_val = float((logits_val.argmax(dim=1) == y_val_t).float().mean().item())

        if acc_val > mejor_acc_val:
            mejor_acc_val = acc_val
            mejor_estado = {k: v.cpu().clone() for k, v in red.state_dict().items()}

        historial.append(
            {
                "epoca": epoca,
                "loss_train": round(perdida_acum / n_train, 4),
                "acc_train": round(aciertos_train / n_train, 4),
                "loss_val": round(loss_val, 4),
                "acc_val": round(acc_val, 4),
                "tiempo_s": round(time.perf_counter() - t0, 2),
            }
        )

    if mejor_estado is not None:
        red.load_state_dict(mejor_estado)
    red.eval()
    return red, historial


@torch.no_grad()
def predecir_logits_finetuning(
    red: nn.Module,
    imagenes_uint8: np.ndarray,
    batch_size: int = 128,
    tamano: int = 96,
    tta_diedrico: bool = True,
) -> tuple[np.ndarray, float]:
    """Obtiene los logits del modelo fine-tuned en CPU (con opción de Test-Time Augmentation horizontal)."""
    red.eval()
    n = len(imagenes_uint8)
    salidas: list[np.ndarray] = []
    t0 = time.perf_counter()
    for ini in range(0, n, batch_size):
        xb = preprocesar_lote_tensor(imagenes_uint8[ini : ini + batch_size], tamano=tamano)
        logits = red(xb)
        if tta_diedrico:
            logits_flip = red(torch.flip(xb, dims=[3]))
            logits = 0.5 * (logits + logits_flip)
        salidas.append(logits.cpu().numpy().astype(np.float32))
    duracion_s = time.perf_counter() - t0
    return np.vstack(salidas), duracion_s


def ajustar_temperatura_calibracion(logits_val: np.ndarray, y_val: np.ndarray) -> float:
    """Ajusta la temperatura T > 0 minimizando la log-verosimilitud negativa en validación.

    Se trabaja en float64 y con búsqueda acotada sobre log(T): BFGS sobre logits float32 no sirve:
    el gradiente numérico se anula y devuelve siempre T = 1,0.
    """
    logits = np.asarray(logits_val, dtype=np.float64)
    filas = np.arange(len(y_val))

    def nll(log_t: float) -> float:
        log_probs = log_softmax(logits / np.exp(log_t), axis=1)
        return float(-np.mean(log_probs[filas, y_val]))

    res = minimize_scalar(nll, bounds=(np.log(0.05), np.log(20.0)), method="bounded", options={"xatol": 1e-5})
    return float(np.exp(res.x))


def aplicar_temperatura(logits: np.ndarray, temperatura: float) -> np.ndarray:
    """Convierte logits en probabilidades calibradas mediante Softmax(logits / T)."""
    return softmax(np.asarray(logits, dtype=np.float64) / float(temperatura), axis=1).astype(np.float32)


def probs_a_logits(probabilidades: np.ndarray) -> np.ndarray:
    """Log-probabilidades de un clasificador sklearn, usadas como logits para el escalado por temperatura."""
    return np.log(np.clip(np.asarray(probabilidades, dtype=np.float64), 1e-7, 1.0))


def cargar_red_finetuned(ruta_pt) -> nn.Module:
    """Reconstruye la arquitectura (sin descargar ImageNet) y carga los pesos fine-tuned guardados."""
    red = crear_backbone_mobilenet_v3(num_clases=len(CLASES_EUROSAT), preentrenado=False)
    red.load_state_dict(torch.load(ruta_pt, map_location="cpu", weights_only=True))
    red.eval()
    return red


class RedConPreproceso(nn.Module):
    """Envoltorio para ONNX: recibe parches (N, 64, 64, 3) en [0, 255] y hace dentro el reescalado a 96x96
    y la normalización de ImageNet, así la API no necesita torch para preprocesar."""

    def __init__(self, red: nn.Module, tamano: int = 96) -> None:
        super().__init__()
        self.red = red
        self.tamano = tamano
        self.register_buffer("media", torch.tensor(MEDIA_IMAGENET, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD_IMAGENET, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, x_nhwc: torch.Tensor) -> torch.Tensor:
        x = x_nhwc.permute(0, 3, 1, 2) / 255.0
        x = torch.nn.functional.interpolate(x, size=(self.tamano, self.tamano), mode="bilinear", align_corners=False)
        return self.red((x - self.media) / self.std)


def crear_extractor_imagenet() -> nn.Module:
    """Backbone MobileNetV3-Small de ImageNet-1K congelado hasta el vector de 576-d (modelos 2 y 3)."""
    red = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    extractor = nn.Sequential(red.features, red.avgpool, nn.Flatten())
    extractor.eval()
    return extractor


def grad_cam(
    red: nn.Module,
    imagen_uint8: np.ndarray,
    clase_objetivo: int | None = None,
    indice_capa: int = 8,
    tamano: int = 96,
) -> dict[str, Any]:
    """Grad-CAM (Selvaraju et al., 2017) sobre un bloque de MobileNetV3-Small.

    Por defecto usa features[8]: con entrada de 96x96 su mapa es de 6x6 celdas; el último bloque
    (features[12]) solo tiene 3x3 y produce mapas demasiado gruesos para parches de 64x64.
    Devuelve el mapa reescalado a 64x64 y normalizado a [0, 1].
    """
    red.eval()
    capa = red.features[indice_capa]
    guardado: dict[str, torch.Tensor] = {}

    def gancho(_modulo, _entrada, salida):
        salida.retain_grad()
        guardado["activacion"] = salida

    manija = capa.register_forward_hook(gancho)
    try:
        with torch.enable_grad():
            x = preprocesar_lote_tensor(imagen_uint8[None, ...], tamano=tamano).requires_grad_(True)
            logits = red(x)
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
            if clase_objetivo is None:
                clase_objetivo = int(np.argmax(probs))
            red.zero_grad()
            logits[0, clase_objetivo].backward()
    finally:
        manija.remove()

    act = guardado["activacion"].detach()[0]  # (C, h, w)
    grad = guardado["activacion"].grad.detach()[0]
    pesos = grad.mean(dim=(1, 2))  # importancia media de cada canal
    cam = torch.relu((pesos[:, None, None] * act).sum(dim=0))
    cam = torch.nn.functional.interpolate(cam[None, None], size=(64, 64), mode="bilinear", align_corners=False)[0, 0]
    cam_np = cam.cpu().numpy().astype(np.float32)
    if cam_np.max() > 1e-8:
        cam_np = cam_np / cam_np.max()
    return {
        "clase_objetivo": int(clase_objetivo),
        "nombre_clase": CLASES_EUROSAT[int(clase_objetivo)],
        "probabilidad": float(probs[int(clase_objetivo)]),
        "probabilidades": probs,
        "mapa": cam_np,
        "resolucion_capa": tuple(act.shape[1:]),
    }


def mapa_sensibilidad_oclusion(
    red: nn.Module,
    imagen_uint8: np.ndarray,
    clase_objetivo: int | None = None,
    parche: int = 12,
    paso: int = 4,
    tamano: int = 96,
) -> dict[str, Any]:
    """Calcula el mapa espacial de sensibilidad por oclusión (64x64) midiendo la caída de probabilidad al ocluir."""
    red.eval()
    with torch.no_grad():
        x_base = preprocesar_lote_tensor(imagen_uint8[None, ...], tamano=tamano)
        probs_base = torch.softmax(red(x_base), dim=1)[0].cpu().numpy()

    if clase_objetivo is None:
        clase_objetivo = int(np.argmax(probs_base))
    prob_original = float(probs_base[clase_objetivo])

    color_medio = imagen_uint8.mean(axis=(0, 1), keepdims=True).astype(np.uint8)
    lote_ocluido: list[np.ndarray] = []
    coordenadas: list[tuple[int, int]] = []

    for y0 in range(0, 64 - parche + 1, paso):
        for x0 in range(0, 64 - parche + 1, paso):
            copia = imagen_uint8.copy()
            copia[y0 : y0 + parche, x0 : x0 + parche] = color_medio
            lote_ocluido.append(copia)
            coordenadas.append((y0, x0))

    arr_ocluido = np.stack(lote_ocluido, axis=0)
    with torch.no_grad():
        x_ocl = preprocesar_lote_tensor(arr_ocluido, tamano=tamano)
        probs_ocl = torch.softmax(red(x_ocl), dim=1)[:, clase_objetivo].cpu().numpy()

    mapa = np.zeros((64, 64), dtype=np.float32)
    conteo = np.zeros((64, 64), dtype=np.float32)
    for (y0, x0), p_ocl in zip(coordenadas, probs_ocl):
        caida = max(0.0, prob_original - float(p_ocl))
        mapa[y0 : y0 + parche, x0 : x0 + parche] += caida
        conteo[y0 : y0 + parche, x0 : x0 + parche] += 1.0

    mapa = mapa / np.maximum(conteo, 1.0)
    if mapa.max() > 1e-6:
        mapa = mapa / mapa.max()

    return {
        "clase_objetivo": int(clase_objetivo),
        "nombre_clase": CLASES_EUROSAT[int(clase_objetivo)],
        "probabilidad_base": round(prob_original, 4),
        "mapa_importancia": mapa,
    }
