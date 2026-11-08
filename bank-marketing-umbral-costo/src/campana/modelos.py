"""Modelos: regla de negocio (línea base), regresión logística y LightGBM, más el manejo del desbalance
(pesos de clase vs. remuestreo) y la calibración de probabilidades."""
from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config

ESTRATEGIAS = ["ninguna", "pesos", "submuestreo", "smote"]

PARAMS_LGBM = dict(n_estimators=200, learning_rate=0.03, num_leaves=15, min_child_samples=100,
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                   verbose=-1, n_jobs=4)


# ---------------------------------------------------------------------------
# Línea base: la "regla del gerente"
# ---------------------------------------------------------------------------
class ReglaSegmento(BaseEstimator, ClassifierMixin):
    """Puntúa a cada cliente con la tasa histórica de éxito de su segmento (resultado de la campaña
    anterior × tipo de teléfono). Es lo que haría un analista con una tabla dinámica."""

    def __init__(self, segmento: tuple[str, ...] = ("resultado_previo", "tipo_contacto")):
        self.segmento = segmento

    def fit(self, X: pd.DataFrame, y):
        d = X[list(self.segmento)].assign(_y=np.asarray(y))
        self.tasas_ = d.groupby(list(self.segmento))["_y"].mean()
        self.global_ = float(np.mean(y))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        idx = pd.MultiIndex.from_frame(X[list(self.segmento)])
        p = self.tasas_.reindex(idx).fillna(self.global_).to_numpy()
        return np.column_stack([1 - p, p])


# ---------------------------------------------------------------------------
# Preprocesamiento
# ---------------------------------------------------------------------------
class ACategorias(BaseEstimator, TransformerMixin):
    """Convierte las columnas de texto a `category` con las categorías vistas en entrenamiento
    (LightGBM las usa de forma nativa; una categoría nueva queda como NaN)."""

    def __init__(self, columnas: list[str]):
        self.columnas = columnas

    def fit(self, X: pd.DataFrame, y=None):
        self.categorias_ = {c: sorted(X[c].astype(str).unique()) for c in self.columnas}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c, cats in self.categorias_.items():
            X[c] = pd.Categorical(X[c].astype(str), categories=cats)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features)


def preprocesador_lineal() -> ColumnTransformer:
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), config.CATEGORICAS),
        ("num", Pipeline([("imputar", SimpleImputer(strategy="median")), ("escalar", StandardScaler())]),
         config.NUMERICAS),
    ])


def regresion_logistica(estrategia: str = "ninguna") -> Pipeline:
    return Pipeline([
        ("prep", preprocesador_lineal()),
        ("modelo", LogisticRegression(C=0.5, max_iter=3000,
                                      class_weight="balanced" if estrategia == "pesos" else None)),
    ])


def lightgbm(estrategia: str = "ninguna", **params) -> Pipeline:
    p = {**PARAMS_LGBM, "random_state": config.SEMILLA, **params}
    if estrategia == "pesos":
        p["class_weight"] = "balanced"
    return Pipeline([("prep", ACategorias(config.CATEGORICAS)), ("modelo", LGBMClassifier(**p))])


def construir(nombre: str, estrategia: str = "ninguna"):
    if nombre == "regla":
        return ReglaSegmento()
    if nombre == "logistica":
        return regresion_logistica(estrategia)
    if nombre == "lightgbm":
        return lightgbm(estrategia)
    raise ValueError(f"modelo desconocido: {nombre}")


# ---------------------------------------------------------------------------
# Desbalance: remuestreo (solo dentro del pliegue de entrenamiento, nunca en validación ni prueba)
# ---------------------------------------------------------------------------
def remuestrear(X: pd.DataFrame, y: np.ndarray, estrategia: str,
                semilla: int = config.SEMILLA) -> tuple[pd.DataFrame, np.ndarray]:
    if estrategia in ("ninguna", "pesos"):
        return X, y
    if estrategia == "submuestreo":
        return RandomUnderSampler(random_state=semilla).fit_resample(X, y)
    if estrategia == "smote":
        # SMOTENC: interpola las numéricas y toma la moda de los vecinos en las categóricas.
        # No admite NaN: "días desde el último contacto" se rellena con -1 y se restaura después.
        cats = config.CATEGORICAS + ["contactado_antes"]
        Xr = X.assign(dias_desde_contacto=X["dias_desde_contacto"].fillna(-1),
                      contactado_antes=X["contactado_antes"].astype(str))
        Xs, ys = SMOTENC(categorical_features=cats, random_state=semilla, k_neighbors=5).fit_resample(Xr, y)
        Xs["contactado_antes"] = Xs["contactado_antes"].astype(int)
        Xs["dias_desde_contacto"] = Xs["dias_desde_contacto"].clip(lower=0).where(Xs["contactado_antes"] == 1)
        return Xs, ys
    raise ValueError(f"estrategia desconocida: {estrategia}")


def entrenar(nombre: str, estrategia: str, X: pd.DataFrame, y: np.ndarray):
    Xr, yr = remuestrear(X, y, estrategia)
    return construir(nombre, estrategia).fit(Xr, yr)


def probabilidad(modelo, X: pd.DataFrame) -> np.ndarray:
    return modelo.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# Calibración
# ---------------------------------------------------------------------------
class Calibrador:
    """Mapea el puntaje del modelo a una probabilidad real. Se ajusta con predicciones FUERA DE PLIEGUE del
    entrenamiento (como CalibratedClassifierCV con ensemble=False): ninguna fila calibra su propia predicción.

    metodo = "isotonica" (no paramétrica, monótona), "platt" (logística sobre el logit del puntaje) o
             "intercepto" (solo desplaza el logit para que el promedio coincida con la tasa observada: corrige
             un cambio de tasa base sin tocar el orden; se estima bien incluso con pocos cientos de filas)."""

    def __init__(self, metodo: str = "isotonica"):
        self.metodo = metodo

    @staticmethod
    def _logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def fit(self, puntaje: np.ndarray, y: np.ndarray) -> "Calibrador":
        if self.metodo == "isotonica":
            self.modelo_ = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(puntaje, y)
        elif self.metodo == "intercepto":
            from scipy.optimize import brentq
            z, tasa = self._logit(puntaje), float(np.mean(y))
            self.modelo_ = brentq(lambda b: np.mean(1 / (1 + np.exp(-(z + b)))) - tasa, -15, 15)
        elif self.metodo == "platt":
            self.modelo_ = LogisticRegression(C=1e6).fit(self._logit(puntaje).reshape(-1, 1), y)
        else:
            raise ValueError(self.metodo)
        return self

    def transform(self, puntaje: np.ndarray) -> np.ndarray:
        if self.metodo == "isotonica":
            return self.modelo_.predict(puntaje)
        if self.metodo == "intercepto":
            return 1 / (1 + np.exp(-(self._logit(puntaje) + self.modelo_)))
        return self.modelo_.predict_proba(self._logit(puntaje).reshape(-1, 1))[:, 1]
