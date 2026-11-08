"""Deriva temporal entre el día 1 y el día 2 del dataset."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

from fraude.config import COLUMNAS_MODELO, SEMILLA


def psi(referencia: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index con cortes por deciles de la referencia (día 1).

    Regla práctica: < 0,10 estable · 0,10–0,25 moderada · > 0,25 deriva fuerte.
    """
    cortes = np.unique(np.quantile(referencia, np.linspace(0, 1, n_bins + 1)))
    cortes[0], cortes[-1] = -np.inf, np.inf
    r = np.histogram(referencia, cortes)[0] / len(referencia)
    a = np.histogram(actual, cortes)[0] / len(actual)
    r, a = np.clip(r, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - r) * np.log(a / r)))


def deriva_por_variable(df: pd.DataFrame, columnas=COLUMNAS_MODELO) -> pd.DataFrame:
    """KS y PSI día 1 → día 2, solo en transacciones normales (el fraude es demasiado escaso)."""
    n = df[df["Class"] == 0]
    d1, d2 = n[n["dia"] == 1], n[n["dia"] == 2]
    filas = []
    for c in columnas:
        ks = ks_2samp(d1[c], d2[c])
        filas.append({"variable": c, "ks": float(ks.statistic), "ks_pvalor": float(ks.pvalue), "psi": psi(d1[c].to_numpy(), d2[c].to_numpy()),
                      "media_dia1": float(d1[c].mean()), "media_dia2": float(d2[c].mean())})
    return pd.DataFrame(filas).sort_values("psi", ascending=False).reset_index(drop=True)


def validacion_adversaria(df: pd.DataFrame, columnas=COLUMNAS_MODELO, n_max: int = 60_000) -> dict:
    """¿Puede un modelo distinguir el día 1 del día 2 solo con V1–V28 y el monto?

    ROC-AUC ≈ 0,5 → los días son intercambiables; cerca de 1 → hay deriva aprovechable.
    """
    m = df.sample(n=min(n_max, len(df)), random_state=SEMILLA)
    X, y = m[columnas].to_numpy(), (m["dia"] == 2).astype(int).to_numpy()
    clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, verbose=-1, random_state=SEMILLA)
    p = cross_val_predict(clf, X, y, cv=StratifiedKFold(3, shuffle=True, random_state=SEMILLA), method="predict_proba")[:, 1]
    clf.fit(X, y)
    imp = pd.Series(clf.booster_.feature_importance("gain"), index=columnas).sort_values(ascending=False)
    return {"roc_auc": float(roc_auc_score(y, p)), "n": int(len(m)), "top_variables": imp.head(5).round(1).to_dict()}


def fraude_por_dia(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("dia")
    return pd.DataFrame({
        "transacciones": g.size(),
        "fraudes": g["Class"].sum(),
        "tasa_fraude_pct": g["Class"].mean() * 100,
        "monto_mediano_normal": df[df.Class == 0].groupby("dia")["Amount"].median(),
        "monto_mediano_fraude": df[df.Class == 1].groupby("dia")["Amount"].median(),
    }).reset_index()
