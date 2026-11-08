"""Métricas de ranking top-K y de "más allá de la exactitud" (cobertura, novedad, diversidad, sesgo de popularidad)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


# ---------------------------------------------------------------- por usuario
def metricas_usuario(recomendados: np.ndarray, relevantes: np.ndarray, k: int = C.K) -> dict[str, float]:
    """Precision@k, Recall@k, NDCG@k y AP@k de una lista ordenada contra el conjunto de relevantes.

    - Recall@k = aciertos / |relevantes| (definición estándar; con muchos relevantes no puede llegar a 1).
    - AP@k = Σ precision@i·rel_i / min(|relevantes|, k)  (MAP = promedio de AP@k entre usuarios).
    - NDCG@k con ganancia binaria e IDCG de min(|relevantes|, k) aciertos.
    """
    rec = np.asarray(recomendados)[:k]
    rel = set(np.asarray(relevantes).tolist())
    if not rel:
        raise ValueError("el usuario no tiene relevantes")
    aciertos = np.array([r in rel for r in rec.tolist()], dtype=float)
    n_ac = aciertos.sum()
    posiciones = np.arange(1, len(rec) + 1)
    descuento = 1.0 / np.log2(posiciones + 1)
    ideal = descuento[: min(len(rel), k)].sum()
    ap = (np.cumsum(aciertos) / posiciones * aciertos).sum() / min(len(rel), k)
    return {
        "precision": n_ac / k,
        "recall": n_ac / len(rel),
        "ndcg": float((aciertos * descuento).sum() / ideal),
        "ap": float(ap),
        "acierto": float(n_ac > 0),
    }


# ---------------------------------------------------------------- sobre todas las listas
def cobertura(listas: np.ndarray, n_items: int) -> float:
    """Fracción del catálogo que aparece en al menos una lista top-K."""
    return len(np.unique(listas)) / n_items


def novedad(listas: np.ndarray, conteo: np.ndarray, n_usuarios: int) -> float:
    """Autoinformación media −log2(p_i), p_i = fracción de usuarios a los que les gustó i. Más alto = menos obvio."""
    p = np.maximum(conteo[listas], 1) / n_usuarios
    return float(-np.log2(p).mean())


def diversidad_intra_lista(listas: np.ndarray, vectores: np.ndarray) -> float:
    """1 − similitud coseno promedio entre pares de una misma lista (vectores ya normalizados, p. ej. géneros)."""
    v = vectores[listas]                                    # usuarios × k × d
    sim = np.einsum("ukd,ujd->ukj", v, v)
    k = listas.shape[1]
    pares = (sim.sum(axis=(1, 2)) - np.trace(sim, axis1=1, axis2=2)) / (k * (k - 1))
    return float(1 - pares.mean())


def percentil_popularidad(conteo: np.ndarray) -> np.ndarray:
    """Percentil (0-100) de popularidad de cada película dentro del catálogo."""
    return pd.Series(conteo).rank(pct=True).to_numpy() * 100


def sesgo_popularidad(listas: np.ndarray, conteo: np.ndarray, historiales: list[np.ndarray] | None = None) -> dict:
    """ARP (percentil medio de popularidad de lo recomendado), % de la cola larga (fuera del 20 % más popular)
    y, si se dan los historiales, la diferencia de percentil frente a lo que el propio usuario ve (Δ GAP)."""
    pct = percentil_popularidad(conteo)
    salida = {
        "percentil_pop_medio": float(pct[listas].mean()),
        "cola_larga_pct": float((pct[listas] <= 80).mean() * 100),
    }
    if historiales is not None:
        hist = np.array([pct[h].mean() for h in historiales if len(h)])
        salida["delta_pop_vs_historial"] = float(pct[listas].mean(axis=1).mean() - hist.mean())
    return salida


def intervalo_bootstrap(valores: np.ndarray, n: int = 1000, nivel: float = 0.95, semilla: int = C.SEMILLA):
    """Intervalo percentil bootstrap de la media."""
    valores = np.asarray(valores, dtype=float)
    rng = np.random.default_rng(semilla)
    medias = valores[rng.integers(0, len(valores), size=(n, len(valores)))].mean(axis=1)
    a = (1 - nivel) / 2
    return float(np.quantile(medias, a)), float(np.quantile(medias, 1 - a))
