"""Interpretabilidad con SHAP. Se usa el TreeSHAP nativo de LightGBM (`pred_contrib=True`): es exacto, rápido
y maneja las categóricas nativas sin codificarlas. Los valores están en log-odds del modelo SIN calibrar
(la calibración isotónica es monótona: no cambia el orden ni el signo de los efectos)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def contribuciones(modelo, X: pd.DataFrame) -> tuple[np.ndarray, float | np.ndarray]:
    """Devuelve (matriz n × variables, valor base) en log-odds."""
    Xp = modelo.named_steps["prep"].transform(X)
    c = modelo.named_steps["modelo"].predict_proba(Xp, pred_contrib=True)
    return c[:, :-1], c[:, -1]


def importancia(modelo, X: pd.DataFrame) -> pd.DataFrame:
    valores, _ = contribuciones(modelo, X)
    imp = pd.DataFrame({"variable": list(X.columns), "shap_medio_abs": np.abs(valores).mean(axis=0)})
    imp["etiqueta"] = imp["variable"].map(config.ETIQUETAS)
    return imp.sort_values("shap_medio_abs", ascending=False).reset_index(drop=True)


def explicacion_shap(modelo, X: pd.DataFrame):
    """Objeto shap.Explanation para los gráficos de la librería (beeswarm, waterfall)."""
    import shap
    valores, base = contribuciones(modelo, X)
    datos = X.copy()
    for c in config.CATEGORICAS:          # sin escala de color para categóricas (se ven en gris)
        datos[c] = np.nan
    return shap.Explanation(values=valores, base_values=base, data=datos.astype(float).to_numpy(),
                            feature_names=[config.ETIQUETAS[c] for c in X.columns])


def _valor_legible(v) -> str:
    if isinstance(v, float) and np.isnan(v):
        return "sin dato"
    if isinstance(v, (float, np.floating)):
        return f"{v:g}"
    return str(v)


def razones(modelo, X: pd.DataFrame, k: int = 3) -> list[list[dict]]:
    """Las k variables que más empujan la probabilidad de cada cliente, en lenguaje claro."""
    valores, _ = contribuciones(modelo, X)
    salida = []
    for i in range(len(X)):
        orden = np.argsort(-np.abs(valores[i]))[:k]
        fila = []
        for j in orden:
            var = X.columns[j]
            efecto = float(valores[i, j])
            fila.append({
                "variable": var,
                "factor": config.ETIQUETAS[var],
                "valor": _valor_legible(X.iloc[i, j]),
                "efecto_log_odds": round(efecto, 4),
                "sentido": "sube la probabilidad" if efecto > 0 else "baja la probabilidad",
            })
        salida.append(fila)
    return salida
