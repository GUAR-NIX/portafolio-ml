"""Carga, variables y partición temporal (sin barajar) del dataset ULB Credit Card Fraud."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fraude.config import (
    ARCHIVO_DATOS,
    COLUMNAS_MODELO,
    CORTE_ENTRENAMIENTO,
    CORTE_VALIDACION,
    SEGUNDOS_DIA,
)


def cargar(ruta=ARCHIVO_DATOS) -> pd.DataFrame:
    if not ruta.exists():
        raise FileNotFoundError(f"No existe {ruta.name}. Ejecuta primero: python descargar_datos.py")
    return preparar(pd.read_parquet(ruta))


def preparar(df: pd.DataFrame) -> pd.DataFrame:
    """Añade variables derivadas. `log_monto` estabiliza la cola larga de Amount (máx. ~25.700 EUR)."""
    df = df.copy()
    df["log_monto"] = np.log1p(df["Amount"].clip(lower=0))
    df["dia"] = (df["Time"] // SEGUNDOS_DIA).astype(int) + 1
    df["hora"] = (df["Time"] % SEGUNDOS_DIA) / 3600.0
    return df


@dataclass(frozen=True)
class Particion:
    entrenamiento: pd.DataFrame
    validacion: pd.DataFrame
    prueba: pd.DataFrame
    corte_1_s: float
    corte_2_s: float

    def resumen(self) -> pd.DataFrame:
        filas = []
        for nombre, d in [("entrenamiento", self.entrenamiento), ("validacion", self.validacion), ("prueba", self.prueba)]:
            filas.append(
                {
                    "particion": nombre,
                    "desde_h": round(d["Time"].min() / 3600, 2),
                    "hasta_h": round(d["Time"].max() / 3600, 2),
                    "horas": round((d["Time"].max() - d["Time"].min()) / 3600, 2),
                    "transacciones": len(d),
                    "fraudes": int(d["Class"].sum()),
                    "tasa_fraude_pct": round(float(d["Class"].mean() * 100), 4),
                    "monto_fraude_eur": round(float(d.loc[d["Class"] == 1, "Amount"].sum()), 2),
                }
            )
        return pd.DataFrame(filas)


def particion_temporal(
    df: pd.DataFrame, corte_1: float = CORTE_ENTRENAMIENTO, corte_2: float = CORTE_VALIDACION
) -> Particion:
    """Corta por *valores* de Time (no por posición): todas las filas con el mismo segundo caen juntas.

    Entrenamiento: Time < t1 · Validación: t1 ≤ Time < t2 · Prueba: Time ≥ t2. No se baraja nada:
    el modelo solo ve el pasado, como en producción.
    """
    t1 = float(df["Time"].quantile(corte_1))
    t2 = float(df["Time"].quantile(corte_2))
    orden = df.sort_values("Time", kind="stable")
    return Particion(
        entrenamiento=orden[orden["Time"] < t1].copy(),
        validacion=orden[(orden["Time"] >= t1) & (orden["Time"] < t2)].copy(),
        prueba=orden[orden["Time"] >= t2].copy(),
        corte_1_s=t1,
        corte_2_s=t2,
    )


def matriz(d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return d[COLUMNAS_MODELO].to_numpy(dtype=np.float64), d["Class"].to_numpy(dtype=int)
