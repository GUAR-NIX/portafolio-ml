"""Métricas para clases muy desbalanceadas y sus intervalos bootstrap.

Nunca se usa la exactitud: con 0,17 % de fraude, «todo es normal» acierta el 99,83 %.

Las métricas se calculan con *pesos por fila*. Así, una réplica bootstrap no copia datos: se
traduce en un vector de pesos (cuántas veces salió cada fila) y el orden por puntaje se calcula
una sola vez por modelo. Con pesos = 1 los resultados coinciden con scikit-learn (lo verifica una prueba).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class OrdenPuntajes:
    """Orden descendente de los puntajes y fin de cada grupo de empates (umbrales distintos)."""

    orden: np.ndarray
    fin_grupo: np.ndarray

    @classmethod
    def desde(cls, puntaje: np.ndarray) -> "OrdenPuntajes":
        orden = np.argsort(-puntaje, kind="mergesort")
        s = puntaje[orden]
        fin = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]
        return cls(orden=orden, fin_grupo=fin)


def _acumulados(y: np.ndarray, op: OrdenPuntajes, w: np.ndarray | None):
    ys = y[op.orden].astype(np.float64)
    ws = np.ones_like(ys) if w is None else w[op.orden].astype(np.float64)
    tp = np.cumsum(ws * ys)
    n = np.cumsum(ws)
    return tp, n


def metricas_ponderadas(
    y: np.ndarray,
    op: OrdenPuntajes,
    w: np.ndarray | None = None,
    k: int = 100,
    precisiones: tuple[float, ...] = (0.80, 0.90),
) -> dict[str, float]:
    tp_all, n_all = _acumulados(y, op, w)
    tp, n = tp_all[op.fin_grupo], n_all[op.fin_grupo]
    fp = n - tp
    P, N = tp_all[-1], n_all[-1] - tp_all[-1]
    out: dict[str, float] = {}
    if P == 0 or N == 0:
        return {"auprc": np.nan, "roc_auc": np.nan, f"precision_at_{k}": np.nan, f"recall_at_{k}": np.nan,
                **{f"recall_a_precision_{int(p*100)}": np.nan for p in precisiones}}
    # En una réplica bootstrap las primeras filas pueden tener peso 0 (n = 0): su aporte es 0, no NaN.
    precision = np.divide(tp, n, out=np.zeros_like(tp), where=n > 0)
    recall = tp / P
    # AUPRC = average precision (suma escalonada, igual que sklearn; sin interpolación trapezoidal optimista)
    out["auprc"] = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
    tpr = np.r_[0.0, recall]
    fpr = np.r_[0.0, fp / N]
    out["roc_auc"] = float(np.trapezoid(tpr, fpr))
    # Precisión@k: las k alertas con mayor puntaje (en filas ponderadas)
    i = min(int(np.searchsorted(n_all, k, side="left")), len(n_all) - 1)
    out[f"precision_at_{k}"] = float(tp_all[i] / n_all[i]) if n_all[i] > 0 else 0.0
    out[f"recall_at_{k}"] = float(tp_all[i] / P)
    for p in precisiones:
        ok = precision >= p
        out[f"recall_a_precision_{int(round(p * 100))}"] = float(recall[ok].max()) if ok.any() else 0.0
    return out


def evaluar(y: np.ndarray, puntaje: np.ndarray, k: int = 100, precisiones=(0.80, 0.90)) -> dict[str, float]:
    return metricas_ponderadas(np.asarray(y), OrdenPuntajes.desde(np.asarray(puntaje, dtype=np.float64)), None, k, precisiones)


def bootstrap(
    y: np.ndarray,
    puntajes: dict[str, np.ndarray],
    k: int = 100,
    n_rep: int = 1000,
    semilla: int = 42,
    precisiones=(0.80, 0.90),
) -> pd.DataFrame:
    """Bootstrap *pareado*: en cada réplica todos los modelos se evalúan sobre las mismas filas.

    Devuelve una fila por (réplica, modelo) con todas las métricas.
    """
    y = np.asarray(y)
    ordenes = {m: OrdenPuntajes.desde(np.asarray(s, dtype=np.float64)) for m, s in puntajes.items()}
    rng = np.random.default_rng(semilla)
    filas = []
    for r in range(n_rep):
        w = np.bincount(rng.integers(0, len(y), len(y)), minlength=len(y))
        for m, op in ordenes.items():
            filas.append({"replica": r, "modelo": m, **metricas_ponderadas(y, op, w, k, precisiones)})
    return pd.DataFrame(filas)


def intervalos(boot: pd.DataFrame, alfa: float = 0.05) -> pd.DataFrame:
    """IC percentil por modelo y métrica."""
    met = [c for c in boot.columns if c not in ("replica", "modelo")]
    largo = boot.melt(id_vars=["replica", "modelo"], value_vars=met, var_name="metrica", value_name="valor")
    g = largo.groupby(["modelo", "metrica"])["valor"]
    return pd.DataFrame(
        {"ic_inf": g.quantile(alfa / 2), "ic_sup": g.quantile(1 - alfa / 2)}
    ).reset_index()


def diferencias_pareadas(boot: pd.DataFrame, pares: list[tuple[str, str]], metrica: str = "auprc") -> pd.DataFrame:
    ancho = boot.pivot(index="replica", columns="modelo", values=metrica)
    filas = []
    for a, b in pares:
        d = (ancho[a] - ancho[b]).dropna()
        lo, hi = np.quantile(d, [0.025, 0.975])
        filas.append(
            {
                "comparacion": f"{a} − {b}",
                "modelo_a": a,
                "modelo_b": b,
                "metrica": metrica,
                "diferencia_media": float(d.mean()),
                "ic_inf": float(lo),
                "ic_sup": float(hi),
                "prob_a_mejor": float((d > 0).mean()),
                "significativa": bool(lo > 0 or hi < 0),
            }
        )
    return pd.DataFrame(filas)
