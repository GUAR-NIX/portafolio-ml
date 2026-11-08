"""Evaluación multiclase, calibración ECE/Brier, costo con abstención, bootstrap y perturbaciones de robustez."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)

from eurosat_suelo.config import (
    CLASES_EUROSAT,
    COSTO_AUDITORIA_CAMPO_USD,
    COSTO_ERROR_CRITICO_INTER_MACRO_USD,
    COSTO_ERROR_MISMA_MACROCATEGORIA_USD,
    MACRO_CATEGORIA,
    NOMBRES_ESPANOL,
)

# Macro-categoría de cada clase como entero, para contar errores de forma vectorizada
_MACROS = sorted(set(MACRO_CATEGORIA.values()))
MACRO_ID = np.array([_MACROS.index(MACRO_CATEGORIA[c]) for c in CLASES_EUROSAT])


def calcular_ece(probabilidades: np.ndarray, y_real: np.ndarray, n_bins: int = 12) -> float:
    """Expected Calibration Error (ECE) sobre la probabilidad máxima predicha."""
    confianzas = np.max(probabilidades, axis=1)
    predicciones = np.argmax(probabilidades, axis=1)
    aciertos = (predicciones == y_real).astype(np.float32)

    bordes = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_real)
    for b_ini, b_fin in zip(bordes[:-1], bordes[1:]):
        mask = (confianzas > b_ini) & (confianzas <= b_fin)
        if np.any(mask):
            acc_bin = float(np.mean(aciertos[mask]))
            conf_bin = float(np.mean(confianzas[mask]))
            ece += (np.sum(mask) / n) * abs(acc_bin - conf_bin)
    return float(ece)


def curva_fiabilidad(probabilidades: np.ndarray, y_real: np.ndarray, n_bins: int = 12) -> pd.DataFrame:
    """Confianza media vs exactitud observada por intervalo de confianza (diagrama de fiabilidad)."""
    confianzas = np.max(probabilidades, axis=1)
    aciertos = (np.argmax(probabilidades, axis=1) == y_real).astype(np.float64)
    bordes = np.linspace(0.0, 1.0, n_bins + 1)
    filas = []
    for b_ini, b_fin in zip(bordes[:-1], bordes[1:]):
        mask = (confianzas > b_ini) & (confianzas <= b_fin)
        if np.any(mask):
            filas.append(
                {
                    "confianza_media": float(confianzas[mask].mean()),
                    "exactitud": float(aciertos[mask].mean()),
                    "n": int(mask.sum()),
                }
            )
    return pd.DataFrame(filas)


def calcular_brier_multiclase(probabilidades: np.ndarray, y_real: np.ndarray) -> float:
    """Brier Score multiclase promedio: media de sum_k (p_ik - y_ik)^2."""
    n, k = probabilidades.shape
    one_hot = np.zeros((n, k), dtype=np.float32)
    one_hot[np.arange(n), y_real] = 1.0
    return float(np.mean(np.sum((probabilidades - one_hot) ** 2, axis=1)))


def tipos_de_error(y_real: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Máscaras de error dentro de la misma macro-categoría y de error crítico entre macro-categorías."""
    y_real = np.asarray(y_real)
    y_pred = np.asarray(y_pred)
    error = y_real != y_pred
    misma = MACRO_ID[y_real] == MACRO_ID[y_pred]
    return error & misma, error & ~misma


def calcular_costo_operativo_lote(
    y_real: np.ndarray,
    y_pred: np.ndarray,
    costo_misma_macro: float = COSTO_ERROR_MISMA_MACROCATEGORIA_USD,
    costo_inter_macro: float = COSTO_ERROR_CRITICO_INTER_MACRO_USD,
) -> dict[str, float]:
    """Calcula el costo (supuesto) de los errores de clasificación: misma macro-categoría vs inter-macro-categoría."""
    n = len(y_real)
    m_misma, m_inter = tipos_de_error(y_real, y_pred)
    errores_misma_macro = int(m_misma.sum())
    errores_inter_macro = int(m_inter.sum())
    costo_total = errores_misma_macro * costo_misma_macro + errores_inter_macro * costo_inter_macro
    costo_por_1000 = costo_total / max(n, 1) * 1000.0
    return {
        "errores_misma_macro": errores_misma_macro,
        "errores_inter_macro": errores_inter_macro,
        "tasa_error_critico_pct": round(errores_inter_macro / max(n, 1) * 100.0, 2),
        "costo_por_1000_parcelas_usd": round(float(costo_por_1000), 2),
    }


def evaluar_predicciones_multiclase(
    nombre_modelo: str,
    probabilidades: np.ndarray,
    y_real: np.ndarray,
    latencia_ms_por_imagen: float,
    parametros_entrenados: int,
) -> dict[str, Any]:
    """Calcula todas las métricas de exactitud, calibración, latencia y costo económico de un modelo."""
    y_pred = np.argmax(probabilidades, axis=1)
    orden_top2 = np.argsort(probabilidades, axis=1)[:, -2:]
    acierto_top2 = np.any(orden_top2 == y_real[:, None], axis=1)

    costos = calcular_costo_operativo_lote(y_real, y_pred)
    return {
        "modelo": nombre_modelo,
        "exactitud_top1": round(float(accuracy_score(y_real, y_pred)), 4),
        "exactitud_top2": round(float(np.mean(acierto_top2)), 4),
        "f1_macro": round(float(f1_score(y_real, y_pred, average="macro")), 4),
        "f1_ponderado": round(float(f1_score(y_real, y_pred, average="weighted")), 4),
        "log_loss": round(float(log_loss(y_real, np.clip(probabilidades, 1e-7, 1.0), labels=list(range(len(CLASES_EUROSAT))))), 4),
        "brier_multiclase": round(calcular_brier_multiclase(probabilidades, y_real), 4),
        "ece_calibracion": round(calcular_ece(probabilidades, y_real), 4),
        "errores_misma_macro": costos["errores_misma_macro"],
        "errores_inter_macro": costos["errores_inter_macro"],
        "tasa_error_critico_pct": costos["tasa_error_critico_pct"],
        "costo_por_1000_usd": costos["costo_por_1000_parcelas_usd"],
        "latencia_ms_img": round(float(latencia_ms_por_imagen), 3),
        "parametros_entrenados": int(parametros_entrenados),
    }


def metricas_desglosadas_por_clase(probabilidades: np.ndarray, y_real: np.ndarray) -> pd.DataFrame:
    """Devuelve precisión, sensibilidad, F1, confianza media y principal clase de confusión por cada clase."""
    y_pred = np.argmax(probabilidades, axis=1)
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_real,
        y_pred,
        labels=list(range(len(CLASES_EUROSAT))),
        zero_division=0,
    )
    cm = confusion_matrix(y_real, y_pred, labels=list(range(len(CLASES_EUROSAT))))

    filas: list[dict[str, Any]] = []
    for idx, clase in enumerate(CLASES_EUROSAT):
        fila_cm = cm[idx].copy()
        fila_cm[idx] = -1
        idx_conf = int(np.argmax(fila_cm))
        errores_conf = int(max(0, fila_cm[idx_conf]))
        mask_c = y_real == idx
        conf_media = float(np.mean(np.max(probabilidades[mask_c], axis=1))) if np.any(mask_c) else 0.0

        filas.append(
            {
                "clase_id": idx,
                "clase": clase,
                "nombre_es": NOMBRES_ESPANOL[clase],
                "macro_categoria": MACRO_CATEGORIA[clase],
                "soporte": int(sup[idx]),
                "precision": round(float(prec[idx]), 4),
                "recall": round(float(rec[idx]), 4),
                "f1": round(float(f1[idx]), 4),
                "confianza_media": round(conf_media, 4),
                "principal_confusion": CLASES_EUROSAT[idx_conf] if errores_conf > 0 else "Ninguna",
                "principal_confusion_es": NOMBRES_ESPANOL[CLASES_EUROSAT[idx_conf]] if errores_conf > 0 else "Ninguna",
                "casos_principal_confusion": errores_conf,
            }
        )
    return pd.DataFrame(filas)


UMBRALES_ABSTENCION = np.round(np.arange(0.0, 0.96, 0.05), 2)


def costo_por_parcela_con_abstencion(
    probabilidades: np.ndarray,
    y_real: np.ndarray,
    tau: float,
    costo_revision: float = COSTO_AUDITORIA_CAMPO_USD,
    costo_misma_macro: float = COSTO_ERROR_MISMA_MACROCATEGORIA_USD,
    costo_inter_macro: float = COSTO_ERROR_CRITICO_INTER_MACRO_USD,
) -> np.ndarray:
    """Costo (supuesto) de cada parcela: revisión humana si confianza < tau; si no, el costo de su error (o 0)."""
    confianzas = np.max(probabilidades, axis=1)
    y_pred = np.argmax(probabilidades, axis=1)
    m_misma, m_inter = tipos_de_error(y_real, y_pred)
    costo = m_misma * costo_misma_macro + m_inter * costo_inter_macro
    return np.where(confianzas < tau, costo_revision, costo).astype(np.float64)


def elegir_umbral_minimo_costo(probabilidades: np.ndarray, y_real: np.ndarray, **costos: float) -> float:
    """Umbral tau de la rejilla que minimiza el costo medio. Se llama con datos de VALIDACIÓN."""
    medias = [costo_por_parcela_con_abstencion(probabilidades, y_real, t, **costos).mean() for t in UMBRALES_ABSTENCION]
    return float(UMBRALES_ABSTENCION[int(np.argmin(medias))])


def resumen_politica_abstencion(probabilidades: np.ndarray, y_real: np.ndarray, tau: float, **costos: float) -> dict[str, Any]:
    """Cobertura, exactitud en automáticas, errores restantes y costo por 1.000 parcelas para un tau dado."""
    confianzas = np.max(probabilidades, axis=1)
    y_pred = np.argmax(probabilidades, axis=1)
    auto = confianzas >= tau
    m_misma, m_inter = tipos_de_error(y_real, y_pred)
    costo = costo_por_parcela_con_abstencion(probabilidades, y_real, tau, **costos)
    return {
        "umbral_confianza": float(tau),
        "cobertura_automatica_pct": round(float(auto.mean()) * 100.0, 2),
        "derivadas_fotointeprete_pct": round(float((~auto).mean()) * 100.0, 2),
        "exactitud_en_automaticas_pct": round(float((y_pred[auto] == y_real[auto]).mean()) * 100.0, 2) if auto.any() else 100.0,
        "errores_misma_macro_restantes": int((m_misma & auto).sum()),
        "errores_criticos_restantes": int((m_inter & auto).sum()),
        "costo_total_por_1000_usd": round(float(costo.mean()) * 1000.0, 2),
    }


def evaluar_curva_abstencion(probabilidades: np.ndarray, y_real: np.ndarray, **costos: float) -> pd.DataFrame:
    """Evalúa cómo varía la cobertura automática, la exactitud y el costo por 1.000 parcelas según el umbral tau."""
    return pd.DataFrame(
        [resumen_politica_abstencion(probabilidades, y_real, float(t), **costos) for t in UMBRALES_ABSTENCION]
    )


def _f1_macro_rapido(y_real: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    cm = np.bincount(y_real * k + y_pred, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(np.float64)
    denom = cm.sum(axis=0) + cm.sum(axis=1)
    return float(np.mean(np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)))


def bootstrap_metricas_pareado(
    y_real: np.ndarray,
    predicciones: dict[str, np.ndarray],
    costos_por_parcela: dict[str, np.ndarray],
    n_boot: int = 2000,
    semilla: int = 42,
) -> dict[str, dict[str, np.ndarray]]:
    """Bootstrap pareado: las MISMAS parcelas remuestreadas para todos los modelos en cada réplica.

    Devuelve, por modelo, las réplicas de exactitud, Macro-F1 y costo por 1.000 parcelas. Al ser pareado,
    las diferencias entre modelos se calculan réplica a réplica y su IC refleja la correlación entre ellos.
    """
    rng = np.random.default_rng(semilla)
    n = len(y_real)
    k = len(CLASES_EUROSAT)
    rep = {m: {"exactitud": np.empty(n_boot), "f1_macro": np.empty(n_boot), "costo_1000": np.empty(n_boot)} for m in predicciones}
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y_real[idx]
        for m, pred in predicciones.items():
            pb = pred[idx]
            rep[m]["exactitud"][b] = float(np.mean(pb == yb))
            rep[m]["f1_macro"][b] = _f1_macro_rapido(yb, pb, k)
            rep[m]["costo_1000"][b] = float(costos_por_parcela[m][idx].mean()) * 1000.0
    return rep


def intervalo(replicas: np.ndarray, nivel: float = 0.95) -> tuple[float, float]:
    """Intervalo percentil del bootstrap."""
    a = (1.0 - nivel) / 2.0
    return float(np.quantile(replicas, a)), float(np.quantile(replicas, 1.0 - a))


def simular_velo_atmosferico(imagenes_uint8: np.ndarray, intensidad_velo: float, semilla: int = 42) -> np.ndarray:
    """Velo / bruma SIMULADO: mezcla cada parche con una capa clara ondulada de orientación aleatoria.

    No es nubosidad real de Sentinel-2 (EuroSAT excluye escenas nubladas); es una perturbación sintética
    para comparar cómo se degradan los modelos cuando sube el brillo y baja el contraste.
    """
    if intensidad_velo <= 0.0:
        return imagenes_uint8.copy()
    rng = np.random.default_rng(semilla)
    arr = imagenes_uint8.astype(np.float32) / 255.0
    n = len(arr)

    # Gradiente espacial suave que imita un velo cirroiforme sobre el parche de 64x64
    y_grid, x_grid = np.mgrid[0:64, 0:64].astype(np.float32) / 64.0
    angulos = rng.uniform(0, 2 * np.pi, size=(n, 1, 1))
    onda = 0.65 + 0.35 * np.sin(2.5 * np.pi * (x_grid[None, :, :] * np.cos(angulos) + y_grid[None, :, :] * np.sin(angulos)))
    capa_nube = np.stack([onda * 0.96, onda * 0.98, onda * 1.00], axis=-1)

    mezcla = (1.0 - intensidad_velo) * arr + intensidad_velo * capa_nube
    return np.clip(mezcla * 255.0, 0, 255).astype(np.uint8)


def simular_ruido_gaussiano(imagenes_uint8: np.ndarray, sigma: float, semilla: int = 42) -> np.ndarray:
    """Ruido gaussiano aditivo independiente por píxel y banda (sigma en niveles digitales de 0 a 255)."""
    if sigma <= 0.0:
        return imagenes_uint8.copy()
    rng = np.random.default_rng(semilla)
    ruido = rng.normal(0.0, sigma, size=imagenes_uint8.shape).astype(np.float32)
    return np.clip(imagenes_uint8.astype(np.float32) + ruido, 0, 255).astype(np.uint8)


def simular_desenfoque(imagenes_uint8: np.ndarray, sigma_px: float, semilla: int = 42) -> np.ndarray:
    """Desenfoque gaussiano espacial (sigma en píxeles de 10 m), como un sensor peor enfocado o remuestreado."""
    if sigma_px <= 0.0:
        return imagenes_uint8.copy()
    from scipy.ndimage import gaussian_filter

    arr = gaussian_filter(imagenes_uint8.astype(np.float32), sigma=(0, sigma_px, sigma_px, 0), mode="reflect")
    return np.clip(np.round(arr), 0, 255).astype(np.uint8)


# perturbación -> (función, niveles evaluados además del 0, etiqueta del eje)
PERTURBACIONES = {
    "velo_simulado": (simular_velo_atmosferico, [0.10, 0.20, 0.30, 0.40], "Intensidad del velo simulado (fracción)"),
    "ruido_gaussiano": (simular_ruido_gaussiano, [5.0, 10.0, 20.0, 30.0], "Sigma del ruido (niveles 0–255)"),
    "desenfoque": (simular_desenfoque, [0.5, 1.0, 1.5, 2.0], "Sigma del desenfoque (píxeles)"),
}
