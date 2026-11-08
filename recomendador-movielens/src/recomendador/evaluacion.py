"""Evaluación offline: top-K para cada usuario evaluable y resumen por segmento."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import config as C
from . import metricas as M
from .datos import Particion, muestrear_usuarios
from .modelos import Modelo, top_k


def usuarios_existentes(part: Particion, maximo: int = C.MAX_USUARIOS_EVAL) -> np.ndarray:
    return np.array(muestrear_usuarios(list(part.eval_existentes), maximo), dtype=np.int64)


def usuarios_nuevos(part: Particion, maximo: int = C.MAX_USUARIOS_EVAL) -> list[int]:
    return muestrear_usuarios(list(part.eval_nuevos), maximo)


def _csr(listas: list[np.ndarray], n_items: int) -> sp.csr_matrix:
    filas = np.repeat(np.arange(len(listas)), [len(x) for x in listas])
    cols = np.concatenate(listas) if listas else np.array([], dtype=np.int64)
    m = sp.csr_matrix((np.ones(len(cols), np.float32), (filas, cols)), shape=(len(listas), n_items))
    m.sum_duplicates()
    m.data[:] = 1
    return m


def lote_nuevos(part: Particion, ids: list[int]):
    """Historial (onboarding positivo), películas a excluir (todo el onboarding) y relevantes de usuarios nuevos."""
    n = len(part.items)
    hist = _csr([part.eval_nuevos[u]["onboarding"] for u in ids], n)
    excl = _csr([part.eval_nuevos[u]["vistos"] for u in ids], n)
    return hist, excl, [part.eval_nuevos[u]["relevantes"] for u in ids]


def evaluar(modelo: Modelo, part: Particion, existentes: np.ndarray, nuevos: list[int], k: int = C.K,
            lote: int = 1000) -> tuple[pd.DataFrame, np.ndarray]:
    """Devuelve métricas por usuario y la matriz de listas top-k (misma fila = mismo usuario)."""
    filas, listas = [], []
    n_pos = part.n_positivos_usuario
    for ini in range(0, len(existentes), lote):
        u = existentes[ini:ini + lote]
        recs = top_k(modelo.puntuar(part.positivos[u], u), part.vistos[u], k)
        listas.append(recs)
        for fila, uu in enumerate(u):
            m = M.metricas_usuario(recs[fila], part.eval_existentes[int(uu)], k)
            filas.append({"usuario": int(part.usuarios[uu]), "tipo": "existente", "n_positivos": int(n_pos[uu]),
                          "segmento": C.segmento_de(int(n_pos[uu])), "n_relevantes": len(part.eval_existentes[int(uu)]), **m})
    for ini in range(0, len(nuevos), lote):
        ids = nuevos[ini:ini + lote]
        hist, excl, rel = lote_nuevos(part, ids)
        recs = top_k(modelo.puntuar(hist, None), excl, k)
        listas.append(recs)
        for fila, uid in enumerate(ids):
            m = M.metricas_usuario(recs[fila], rel[fila], k)
            filas.append({"usuario": uid, "tipo": "nuevo", "n_positivos": int(hist[fila].nnz),
                          "segmento": "nuevos (onboarding)", "n_relevantes": len(rel[fila]), **m})
    return pd.DataFrame(filas), (np.vstack(listas) if listas else np.zeros((0, k), dtype=np.int64))


def resumir(nombre: str, df: pd.DataFrame, listas: np.ndarray, part: Particion, vectores_generos: np.ndarray,
            historiales: list[np.ndarray] | None = None) -> pd.DataFrame:
    """Una fila por segmento (+ 'todos los existentes') con exactitud, cobertura, novedad, diversidad y popularidad."""
    conteo = np.asarray(part.positivos.sum(axis=0)).ravel()
    n_us = part.positivos.shape[0]
    grupos = [(s, df["segmento"] == s) for s in df["segmento"].unique()]
    grupos.append(("existentes (todos)", df["tipo"] == "existente"))
    salida = []
    for seg, mascara in grupos:
        idx = np.flatnonzero(mascara.to_numpy())
        if len(idx) == 0:
            continue
        sub = df.iloc[idx]
        bajo, alto = M.intervalo_bootstrap(sub["ndcg"].to_numpy())
        hist = [historiales[i] for i in idx] if historiales is not None else None
        salida.append({
            "modelo": nombre, "segmento": seg, "usuarios": len(idx),
            "precision@10": sub["precision"].mean(), "recall@10": sub["recall"].mean(),
            "ndcg@10": sub["ndcg"].mean(), "ndcg_ic95_bajo": bajo, "ndcg_ic95_alto": alto,
            "map@10": sub["ap"].mean(), "hit_rate@10": sub["acierto"].mean(),
            "cobertura_pct": 100 * M.cobertura(listas[idx], len(part.items)),
            "novedad_bits": M.novedad(listas[idx], conteo, n_us),
            "diversidad_generos": M.diversidad_intra_lista(listas[idx], vectores_generos),
            **M.sesgo_popularidad(listas[idx], conteo, hist),
        })
    return pd.DataFrame(salida)


def historiales_de(part: Particion, existentes: np.ndarray, nuevos: list[int]) -> list[np.ndarray]:
    """Índices de lo que le gustó a cada usuario evaluado (para medir el sesgo de popularidad relativo)."""
    p = part.positivos
    h = [p.indices[p.indptr[u]:p.indptr[u + 1]] for u in existentes]
    return h + [part.eval_nuevos[u]["onboarding"] for u in nuevos]


def cronometrar(f, *a, **kw):
    t = time.perf_counter()
    r = f(*a, **kw)
    return r, time.perf_counter() - t
