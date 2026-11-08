"""Puntuador usado por la API: probabilidad calibrada, decisión de llamar, valor esperado y razones (SHAP)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config, explicar
from . import evaluacion as ev


class Puntuador:
    def __init__(self, ruta: Path | None = None):
        paquete = joblib.load(ruta or config.MODELOS / "modelo_campana.joblib")
        self.modelo = paquete["modelo"]
        self.calibrador = paquete["calibrador"]
        self.umbral = float(paquete["umbral"])
        self.metadata = paquete["metadata"]

    @staticmethod
    def a_tabla(clientes: list[dict]) -> pd.DataFrame:
        """Del formato de la API (sin `contactado_antes`; `dias_desde_contacto` = None si nunca se contactó)
        a la tabla de variables del modelo."""
        df = pd.DataFrame(clientes)
        dias = pd.to_numeric(df.get("dias_desde_contacto"), errors="coerce") if "dias_desde_contacto" in df else \
            pd.Series(np.nan, index=df.index)
        df["dias_desde_contacto"] = dias.astype(float)
        df["contactado_antes"] = dias.notna().astype(int)
        for c in config.CATEGORICAS:
            df[c] = df[c].astype(str)
        return df[config.VARIABLES]

    def probabilidad(self, X: pd.DataFrame) -> np.ndarray:
        return self.calibrador.transform(self.modelo.predict_proba(X)[:, 1])

    def puntuar(self, clientes: list[dict], explicar_cuales: str = "todas", costo: float | None = None,
                beneficio: float | None = None) -> list[dict]:
        """explicar_cuales: "todas", "llamar" (solo a quienes se recomienda llamar) o "ninguna".
        Si se pasan costo y beneficio, el umbral pasa a ser costo/beneficio (válido porque p está calibrada)."""
        costo = config.COSTO_LLAMADA if costo is None else costo
        beneficio = config.BENEFICIO_EXITO if beneficio is None else beneficio
        personalizado = (costo, beneficio) != (config.COSTO_LLAMADA, config.BENEFICIO_EXITO)
        umbral = ev.umbral_teorico(costo, beneficio) if personalizado else self.umbral

        X = self.a_tabla(clientes)
        p = self.probabilidad(X)
        llamar = p >= umbral
        quienes = {"todas": np.ones(len(X), bool), "llamar": llamar, "ninguna": np.zeros(len(X), bool)}[explicar_cuales]
        razones = [[] for _ in range(len(X))]
        if quienes.any():
            idx = np.flatnonzero(quienes)
            for i, r in zip(idx, explicar.razones(self.modelo, X.iloc[idx])):
                razones[i] = r
        return [{
            "probabilidad": round(float(p[i]), 4),
            "llamar": bool(llamar[i]),
            "umbral": round(float(umbral), 4),
            "valor_esperado_usd": round(float(p[i] * beneficio - costo), 2),
            "razones": razones[i],
        } for i in range(len(X))]


@lru_cache(maxsize=1)
def puntuador() -> Puntuador:
    return Puntuador()
