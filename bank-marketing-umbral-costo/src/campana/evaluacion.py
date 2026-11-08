"""Métricas de clasificación, validación cruzada fuera de pliegue, ganancia económica, umbral y lift."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from . import config, modelos


def metricas(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """AUC y AP miden el ORDEN (a quién llamar primero); Brier y log loss miden si la probabilidad es real."""
    return {
        "auc": roc_auc_score(y, p),
        "average_precision": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)),
        "prob_media": float(np.mean(p)),
        "tasa_real": float(np.mean(y)),
    }


def brier_skill(y: np.ndarray, p: np.ndarray) -> float:
    """1 − Brier / Brier de predecir siempre la tasa base. 0 = no mejor que la tasa base; 1 = perfecto."""
    ref = brier_score_loss(y, np.full(len(y), np.mean(y)))
    return 1 - brier_score_loss(y, p) / ref


def fuera_de_pliegue(nombre: str, estrategia: str, X: pd.DataFrame, y: np.ndarray,
                     pliegues: int = config.PLIEGUES) -> np.ndarray:
    """Predicción de cada fila de entrenamiento con un modelo que no la vio (k pliegues estratificados).
    El remuestreo se aplica SOLO al pliegue de ajuste: la validación conserva la proporción real."""
    oof = np.zeros(len(y))
    skf = StratifiedKFold(pliegues, shuffle=True, random_state=config.SEMILLA)
    for ent, val in skf.split(X, y):
        m = modelos.entrenar(nombre, estrategia, X.iloc[ent], y[ent])
        oof[val] = modelos.probabilidad(m, X.iloc[val])
    return oof


# ---------------------------------------------------------------------------
# Incertidumbre: bootstrap pareado sobre la prueba
# ---------------------------------------------------------------------------
METRICAS_IC = {"auc": roc_auc_score, "average_precision": average_precision_score, "brier": brier_score_loss}


def bootstrap_metricas(y: np.ndarray, predicciones: dict[str, np.ndarray], diferencias: list[tuple[str, str]],
                       n_remuestras: int = 2000, nivel: float = 0.95,
                       semilla: int = config.SEMILLA) -> pd.DataFrame:
    """IC percentil de AUC, AP y Brier de cada modelo y de la diferencia A − B de cada par.

    Es un bootstrap PAREADO: cada remuestra de clientes de la prueba se evalúa con todos los modelos, así que la
    diferencia entre dos modelos se mide sobre los mismos clientes (su IC es más estrecho que el de cada uno).
    Para Brier, menos es mejor: una diferencia A − B negativa favorece a A."""
    y = np.asarray(y)
    n = len(y)
    rng = np.random.default_rng(semilla)
    muestras = {nombre: {m: np.empty(n_remuestras) for m in METRICAS_IC} for nombre in predicciones}
    for b in range(n_remuestras):
        idx = rng.integers(0, n, n)
        for nombre, p in predicciones.items():
            for m, f in METRICAS_IC.items():
                muestras[nombre][m][b] = f(y[idx], p[idx])
    a = (1 - nivel) / 2
    filas = []
    for m, f in METRICAS_IC.items():
        for nombre, p in predicciones.items():
            v = muestras[nombre][m]
            filas.append({"comparacion": nombre, "metrica": m, "estimado": f(y, p),
                          "ic_bajo": np.quantile(v, a), "ic_alto": np.quantile(v, 1 - a)})
        for x, z in diferencias:
            v = muestras[x][m] - muestras[z][m]
            filas.append({"comparacion": f"{x} − {z}", "metrica": m,
                          "estimado": f(y, predicciones[x]) - f(y, predicciones[z]),
                          "ic_bajo": np.quantile(v, a), "ic_alto": np.quantile(v, 1 - a)})
    tabla = pd.DataFrame(filas)
    tabla["n_remuestras"] = n_remuestras
    return tabla


# ---------------------------------------------------------------------------
# Economía de la campaña
# ---------------------------------------------------------------------------
def ganancia(y: np.ndarray, llamar: np.ndarray, costo: float = config.COSTO_LLAMADA,
             beneficio: float = config.BENEFICIO_EXITO) -> float:
    """Ganancia neta = beneficio × éxitos entre los llamados − costo × llamadas."""
    llamar = np.asarray(llamar, dtype=bool)
    return float(beneficio * np.sum(np.asarray(y)[llamar]) - costo * llamar.sum())


def umbral_teorico(costo: float = config.COSTO_LLAMADA, beneficio: float = config.BENEFICIO_EXITO) -> float:
    """Con probabilidad calibrada p, llamar vale la pena si p·beneficio − costo > 0  ⇔  p > costo/beneficio."""
    return costo / beneficio


def curva_ganancia_umbral(y: np.ndarray, p: np.ndarray, umbrales: np.ndarray | None = None,
                          costo: float = config.COSTO_LLAMADA,
                          beneficio: float = config.BENEFICIO_EXITO) -> pd.DataFrame:
    umbrales = np.linspace(0, 0.6, 121) if umbrales is None else umbrales
    filas = []
    for u in umbrales:
        llamar = p >= u
        filas.append({"umbral": u, "pct_llamados": llamar.mean() * 100,
                      "exitos": int(np.sum(y[llamar])), "llamadas": int(llamar.sum()),
                      "ganancia": ganancia(y, llamar, costo, beneficio)})
    return pd.DataFrame(filas)


def umbral_optimo(y: np.ndarray, p: np.ndarray, costo: float = config.COSTO_LLAMADA,
                  beneficio: float = config.BENEFICIO_EXITO) -> float:
    """Umbral que maximiza la ganancia (se busca sobre las predicciones fuera de pliegue, nunca en prueba)."""
    curva = curva_ganancia_umbral(y, p, np.round(np.arange(0.01, 0.61, 0.005), 3), costo, beneficio)
    return float(curva.loc[curva["ganancia"].idxmax(), "umbral"])


def resumen_politica(y: np.ndarray, llamar: np.ndarray, costo: float = config.COSTO_LLAMADA,
                     beneficio: float = config.BENEFICIO_EXITO) -> dict[str, float]:
    llamar = np.asarray(llamar, dtype=bool)
    exitos = int(np.sum(y[llamar]))
    return {
        "llamadas": int(llamar.sum()),
        "pct_llamados": llamar.mean() * 100,
        "exitos": exitos,
        "pct_exitos_capturados": exitos / max(int(np.sum(y)), 1) * 100,
        "conversion_pct": exitos / max(int(llamar.sum()), 1) * 100,
        "ganancia": ganancia(y, llamar, costo, beneficio),
    }


# ---------------------------------------------------------------------------
# Ganancia acumulada (gain chart) y lift
# ---------------------------------------------------------------------------
def ganancia_acumulada(y: np.ndarray, p: np.ndarray, puntos: int = 100) -> pd.DataFrame:
    """Para cada % de clientes llamados (de mayor a menor probabilidad): % de suscripciones capturadas y lift."""
    orden = np.argsort(-p, kind="stable")
    ys = np.asarray(y)[orden]
    acum = np.cumsum(ys)
    n, total = len(ys), ys.sum()
    pct = np.linspace(1, 100, puntos)
    k = np.maximum(1, np.round(pct / 100 * n).astype(int))
    capturado = acum[k - 1] / total * 100
    return pd.DataFrame({"pct_llamados": pct, "pct_capturado": capturado, "lift": capturado / pct})


def tabla_deciles(y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    d = pd.DataFrame({"y": y, "p": p})
    d["decil"] = pd.qcut(d["p"].rank(method="first", ascending=False), 10, labels=range(1, 11))
    t = d.groupby("decil", observed=True).agg(clientes=("y", "size"), suscripciones=("y", "sum"),
                                              prob_media=("p", "mean"), tasa_real=("y", "mean"))
    t["pct_capturado_acum"] = t["suscripciones"].cumsum() / t["suscripciones"].sum() * 100
    t["lift"] = t["tasa_real"] / d["y"].mean()
    return t


def tabla_calibracion(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    real, pred = calibration_curve(y, p, n_bins=bins, strategy="quantile")
    return pd.DataFrame({"prob_predicha": pred, "tasa_real": real})


def calibrar_fuera_de_pliegue(puntaje: np.ndarray, y: np.ndarray, metodo: str = "isotonica",
                              pliegues: int = config.PLIEGUES) -> np.ndarray:
    """Calibra predicciones fuera de pliegue con validación cruzada (el calibrador de cada fila no la vio):
    sirve para comparar métodos de calibración y para buscar el umbral sin hacerse trampa."""
    salida = np.zeros(len(y))
    skf = StratifiedKFold(pliegues, shuffle=True, random_state=config.SEMILLA + 1)
    for ent, val in skf.split(puntaje.reshape(-1, 1), y):
        cal = modelos.Calibrador(metodo).fit(puntaje[ent], y[ent])
        salida[val] = cal.transform(puntaje[val])
    return salida
