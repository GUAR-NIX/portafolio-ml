"""Búsqueda acotada de hiperparámetros sobre la partición de VALIDACIÓN (la prueba no se toca).

Cada configuración evaluada se guarda en un CSV apenas termina: si el proceso se corta, al relanzarlo continúa
donde quedó.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import modelos as Mo
from .datos import Particion
from .evaluacion import evaluar

# ALS: 12 combinaciones elegidas de una rejilla factores × regularización × alpha × bm25 (no la rejilla completa).
REJILLA_ALS = [
    {"factores": 64, "regularizacion": 0.1, "alpha": 10, "bm25": False},
    {"factores": 128, "regularizacion": 0.1, "alpha": 10, "bm25": False},
    {"factores": 128, "regularizacion": 1.0, "alpha": 10, "bm25": False},
    {"factores": 128, "regularizacion": 0.1, "alpha": 40, "bm25": False},
    {"factores": 128, "regularizacion": 0.1, "alpha": 2, "bm25": False},
    {"factores": 128, "regularizacion": 10.0, "alpha": 10, "bm25": False},
    {"factores": 256, "regularizacion": 1.0, "alpha": 10, "bm25": False},
    {"factores": 128, "regularizacion": 1.0, "alpha": 2, "bm25": True},
    {"factores": 128, "regularizacion": 1.0, "alpha": 10, "bm25": True},
    {"factores": 256, "regularizacion": 1.0, "alpha": 2, "bm25": True},
    {"factores": 64, "regularizacion": 1.0, "alpha": 2, "bm25": True},
    {"factores": 128, "regularizacion": 1.0, "alpha": 1, "bm25": True},
]
REJILLA_CONTENIDO = [{"peso_generos": g, "peso_popularidad": p} for g in (0.2, 0.5) for p in (0.0, 0.1, 0.3, 0.6)]
REJILLA_HIBRIDO = [{"k": k, "peso_reciente": b} for k in (3, 10, 30, 100) for b in (0.0, 0.25, 0.5, 1.0)] + [
    {"k": k, "peso_reciente": b} for k, b in ((30, 2.0), (100, 2.0), (300, 1.0), (300, 2.0))]  # se amplió: el óptimo caía en el borde
REJILLA_EASE = [200.0, 1000.0, 5000.0]


def _clave(cfg: dict) -> str:
    return json.dumps(cfg, sort_keys=True)


def _registrar(ruta: Path, fila: dict) -> None:
    df = pd.DataFrame([fila])
    df.to_csv(ruta, mode="a", header=not ruta.exists(), index=False)


def _hecho(ruta: Path) -> dict[str, dict]:
    if not ruta.exists():
        return {}
    return {r["config"]: r for r in pd.read_csv(ruta).to_dict("records")}


def _medir(modelo, part, existentes, nuevos) -> dict:
    df, _ = evaluar(modelo, part, existentes, nuevos)
    ex, nu = df[df["tipo"] == "existente"], df[df["tipo"] == "nuevo"]
    fila = {
        "ndcg_existentes": ex["ndcg"].mean(), "recall_existentes": ex["recall"].mean(),
        "ndcg_nuevos": nu["ndcg"].mean() if len(nu) else np.nan, "ndcg_todos": df["ndcg"].mean(),
    }
    for seg, g in ex.groupby("segmento"):
        fila[f"ndcg_{seg.split()[0]}"] = g["ndcg"].mean()
    return fila


def buscar_als(part: Particion, existentes, nuevos, ruta: Path) -> tuple[dict, pd.DataFrame]:
    hechos = _hecho(ruta)
    for cfg in REJILLA_ALS:
        if _clave(cfg) in hechos:
            continue
        t = time.perf_counter()
        modelo = Mo.ALS(**cfg).ajustar(part.positivos)
        seg_ajuste = time.perf_counter() - t
        fila = {"config": _clave(cfg), **cfg, "segundos_ajuste": round(seg_ajuste, 1), **_medir(modelo, part, existentes, nuevos)}
        print(f"  ALS {cfg} → NDCG existentes {fila['ndcg_existentes']:.4f} ({seg_ajuste:.0f} s)", flush=True)
        _registrar(ruta, fila)
    res = pd.read_csv(ruta)
    mejor = res.sort_values("ndcg_existentes", ascending=False).iloc[0]
    return {"factores": int(mejor["factores"]), "regularizacion": float(mejor["regularizacion"]),
            "alpha": float(mejor["alpha"]), "bm25": bool(mejor["bm25"])}, res


def buscar_contenido(part: Particion, generos, genome, existentes, nuevos, ruta: Path) -> tuple[dict, pd.DataFrame]:
    hechos = _hecho(ruta)
    base = Mo.Contenido().ajustar(part.positivos, generos=generos, genome=genome)  # el SVD se calcula una sola vez
    for cfg in REJILLA_CONTENIDO:
        if _clave(cfg) in hechos:
            continue
        m = Mo.Contenido(**cfg)
        m.tiene_genome = base.tiene_genome
        m.set_vectores(base.genome_red, base.generos, base.conteo)
        fila = {"config": _clave(cfg), **cfg, **_medir(m, part, existentes, nuevos)}
        print(f"  Contenido {cfg} → NDCG todos {fila['ndcg_todos']:.4f}", flush=True)
        _registrar(ruta, fila)
    res = pd.read_csv(ruta)
    mejor = res.sort_values("ndcg_todos", ascending=False).iloc[0]
    return {"peso_generos": float(mejor["peso_generos"]), "peso_popularidad": float(mejor["peso_popularidad"])}, res


def buscar_hibrido(part: Particion, als: Mo.ALS, contenido: Mo.Contenido, existentes, nuevos, ruta: Path):
    hechos = _hecho(ruta)
    for cfg in REJILLA_HIBRIDO:
        if _clave(cfg) in hechos:
            continue
        m = Mo.Hibrido(als, contenido, recientes=part.recientes, **cfg)
        fila = {"config": _clave(cfg), **cfg, **_medir(m, part, existentes, nuevos)}
        print(f"  Híbrido {cfg} → NDCG todos {fila['ndcg_todos']:.4f}", flush=True)
        _registrar(ruta, fila)
    res = pd.read_csv(ruta)
    mejor = res.sort_values("ndcg_todos", ascending=False).iloc[0]
    return {"k": float(mejor["k"]), "peso_reciente": float(mejor["peso_reciente"])}, res


def buscar_ease(part: Particion, existentes, nuevos, ruta: Path):
    hechos = _hecho(ruta)
    for lam in REJILLA_EASE:
        cfg = {"lambda": lam}
        if _clave(cfg) in hechos:
            continue
        m = Mo.EASE(lam).ajustar(part.positivos)
        fila = {"config": _clave(cfg), "lambda": lam, **_medir(m, part, existentes, nuevos)}
        print(f"  EASE λ={lam} → NDCG existentes {fila['ndcg_existentes']:.4f}", flush=True)
        _registrar(ruta, fila)
        del m
    res = pd.read_csv(ruta)
    return float(res.sort_values("ndcg_existentes", ascending=False).iloc[0]["lambda"]), res
