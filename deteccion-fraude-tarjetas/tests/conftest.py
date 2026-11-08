"""Fixtures compartidas. Ninguna prueba necesita data/raw (las que lo usan se omiten si falta)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from fraude.config import ARCHIVO_DATOS, DIR_REPORTES, RAIZ_PROYECTO


@pytest.fixture(scope="session")
def resultados() -> dict:
    return json.loads((DIR_REPORTES / "resultados.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def pred_prueba() -> pd.DataFrame:
    return pd.read_parquet(DIR_REPORTES / "predicciones_prueba.parquet")


@pytest.fixture(scope="session")
def pred_validacion() -> pd.DataFrame:
    return pd.read_parquet(DIR_REPORTES / "predicciones_validacion.parquet")


@pytest.fixture(scope="session")
def muestra() -> pd.DataFrame:
    return pd.read_csv(RAIZ_PROYECTO / "data" / "muestra" / "transacciones_prueba_muestra.csv")


@pytest.fixture(scope="session")
def datos_completos() -> pd.DataFrame:
    if not ARCHIVO_DATOS.exists():
        pytest.skip("Falta data/raw/creditcard.parquet (python descargar_datos.py)")
    from fraude.datos import cargar

    return cargar()
