"""Lectura del CSV de UCI, limpieza, variables derivadas y particiones (estratificada y temporal)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def reconstruir_anio(meses: pd.Series) -> np.ndarray:
    """El CSV no trae año, pero las filas están en orden cronológico (mayo 2008 → noviembre 2010).
    Cada vez que el mes retrocede (dic → mar) empieza un año nuevo."""
    num = meses.map(config.MESES).to_numpy()
    saltos = np.concatenate([[0], np.cumsum(np.diff(num) < 0)])
    return 2008 + saltos


def preparar(df: pd.DataFrame) -> pd.DataFrame:
    """Columnas en español, objetivo 0/1, año y mes del contacto y variables derivadas de `pdays`."""
    df = df.rename(columns=config.RENOMBRAR).copy()
    df["anio"] = reconstruir_anio(df["mes"])
    df["mes_num"] = df["mes"].map(config.MESES)
    df[config.OBJETIVO] = (df[config.OBJETIVO] == "yes").astype(int)
    # pdays = 999 significa "nunca contactado antes": se separa en una bandera y los días reales
    df["contactado_antes"] = (df["dias_desde_contacto"] != 999).astype(int)
    df["dias_desde_contacto"] = df["dias_desde_contacto"].where(df["contactado_antes"] == 1).astype(float)
    for c in config.CATEGORICAS:
        df[c] = df[c].astype(str)
    return df


def cargar(ruta: Path | None = None) -> pd.DataFrame:
    ruta = Path(ruta or config.DATOS_CRUDOS)
    if not ruta.exists():
        raise FileNotFoundError(f"No existe {ruta}. Ejecuta primero:  python descargar_datos.py")
    df = preparar(pd.read_csv(ruta, sep=";"))
    # 12 filas idénticas en todas las columnas: se eliminan para que no caigan a la vez en entrenamiento y prueba
    return df.drop_duplicates().reset_index(drop=True)


def particion_estratificada(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return train_test_split(df, test_size=config.PROPORCION_PRUEBA, stratify=df[config.OBJETIVO],
                            random_state=config.SEMILLA)


def particion_temporal(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    periodo = df["anio"] * 100 + df["mes_num"]
    corte = config.CORTE_TEMPORAL[0] * 100 + config.CORTE_TEMPORAL[1]
    return df[periodo < corte], df[periodo >= corte]


def X_y(df: pd.DataFrame, variables: list[str] | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    return df[variables or config.VARIABLES], df[config.OBJETIVO].to_numpy()
