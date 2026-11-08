"""Costo de cada error con `Amount` real y elección del umbral en validación.

Supuestos (ver config.py y la sección de costo del README):
- Cada alerta cuesta `costo_revision` (tiempo del analista), sea fraude o no.
- Un fraude **no** alertado cuesta `fraccion_perdida × Amount` (sin contracargo, se pierde todo el monto).
- Un fraude alertado se bloquea a tiempo y no genera pérdida (supuesto optimista).
- Sin modelo, el banco pierde el monto de todos los fraudes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def costo_total(y, monto, alerta, costo_revision: float, fraccion_perdida: float = 1.0) -> float:
    y, monto, alerta = np.asarray(y), np.asarray(monto, dtype=float), np.asarray(alerta, dtype=bool)
    perdida = fraccion_perdida * monto[(y == 1) & ~alerta].sum()
    return float(costo_revision * alerta.sum() + perdida)


def desglose(y, monto, alerta, costo_revision: float, fraccion_perdida: float = 1.0) -> dict[str, float]:
    y, monto, alerta = np.asarray(y), np.asarray(monto, dtype=float), np.asarray(alerta, dtype=bool)
    fraude = y == 1
    return {
        "alertas": int(alerta.sum()),
        "fraudes_detectados": int((alerta & fraude).sum()),
        "fraudes_perdidos": int((~alerta & fraude).sum()),
        "falsas_alertas": int((alerta & ~fraude).sum()),
        "monto_fraude_total_eur": float(monto[fraude].sum()),
        "monto_fraude_detectado_eur": float(monto[alerta & fraude].sum()),
        "costo_alertas_eur": float(costo_revision * alerta.sum()),
        "perdida_no_detectada_eur": float(fraccion_perdida * monto[fraude & ~alerta].sum()),
        "costo_total_eur": costo_total(y, monto, alerta, costo_revision, fraccion_perdida),
        "costo_sin_modelo_eur": float(fraccion_perdida * monto[fraude].sum()),
    }


def curva_costo(y, monto, puntaje, costo_revision: float, fraccion_perdida: float = 1.0) -> pd.DataFrame:
    """Costo si se alertan las k transacciones con mayor puntaje, para cada umbral distinto.

    La fila k=0 (ninguna alerta) es el costo sin modelo.
    """
    y, monto, s = np.asarray(y), np.asarray(monto, dtype=float), np.asarray(puntaje, dtype=float)
    orden = np.argsort(-s, kind="mergesort")
    ss = s[orden]
    capturado = np.cumsum(fraccion_perdida * monto[orden] * (y[orden] == 1))
    fin = np.r_[np.flatnonzero(np.diff(ss) != 0), len(ss) - 1]  # solo umbrales alcanzables (empates juntos)
    total = fraccion_perdida * monto[y == 1].sum()
    k = fin + 1
    costo = costo_revision * k + (total - capturado[fin])
    return pd.DataFrame(
        {
            "umbral": np.r_[np.inf, ss[fin]],
            "alertas": np.r_[0, k],
            "costo_eur": np.r_[total, costo],
        }
    )


def elegir_umbral(y_val, monto_val, puntaje_val, costo_revision: float, fraccion_perdida: float = 1.0) -> tuple[float, pd.DataFrame]:
    """Umbral que minimiza el costo **en validación**. Se alerta si puntaje ≥ umbral."""
    curva = curva_costo(y_val, monto_val, puntaje_val, costo_revision, fraccion_perdida)
    i = int(curva["costo_eur"].idxmin())
    return float(curva.loc[i, "umbral"]), curva


def alerta_por_perdida_esperada(prob_calibrada, monto, costo_revision: float, fraccion_perdida: float = 1.0) -> np.ndarray:
    """Regla de Bayes: alertar si la pérdida esperada evitada supera el costo de revisar.

    p · fracción · Amount > costo_revision. Usa el monto de *cada* transacción, así que una compra de
    2.000 EUR con p = 1 % se revisa y una de 3 EUR con p = 50 % no.
    """
    return np.asarray(prob_calibrada) * fraccion_perdida * np.asarray(monto, dtype=float) > costo_revision


def bootstrap_costo(y, monto, alertas: dict[str, np.ndarray], costo_revision: float, fraccion_perdida: float = 1.0,
                    n_rep: int = 1000, semilla: int = 42) -> pd.DataFrame:
    """IC bootstrap pareado del costo total con umbrales ya fijados (no se re-eligen por réplica)."""
    y, monto = np.asarray(y), np.asarray(monto, dtype=float)
    fraude = y == 1
    rng = np.random.default_rng(semilla)
    filas = []
    comp = {m: (np.asarray(a, bool).astype(float), (fraude & ~np.asarray(a, bool)) * monto * fraccion_perdida)
            for m, a in alertas.items()}
    for r in range(n_rep):
        w = np.bincount(rng.integers(0, len(y), len(y)), minlength=len(y))
        for m, (al, perdida) in comp.items():
            filas.append({"replica": r, "modelo": m, "costo_eur": float(costo_revision * (w @ al) + w @ perdida)})
    return pd.DataFrame(filas)
