"""Fixtures compartidos para la suite de pruebas de ML_06_satelite_eurosat."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from eurosat_suelo.config import DIR_DATOS, DIR_MODELOS, DIR_MUESTRA, DIR_REPORTES


@pytest.fixture(scope="session")
def resultados_json() -> dict:
    ruta = DIR_REPORTES / "resultados.json"
    assert ruta.exists(), f"Falta {ruta}"
    return json.loads(ruta.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def resumen_clases_27k() -> pd.DataFrame:
    ruta = DIR_REPORTES / "resumen_clases_27000.csv"
    assert ruta.exists(), f"Falta {ruta}"
    return pd.read_csv(ruta)


@pytest.fixture(scope="session")
def muestra_catalogo() -> pd.DataFrame:
    ruta = DIR_DATOS / "catalogo_muestra.csv"
    assert ruta.exists(), f"Falta {ruta}"
    return pd.read_csv(ruta)


@pytest.fixture(scope="session")
def ruta_modelo_joblib() -> Path:
    ruta = DIR_MODELOS / "clasificador_eurosat.joblib"
    assert ruta.exists(), f"Falta {ruta}"
    return ruta


@pytest.fixture(scope="session")
def dir_muestra_png() -> Path:
    assert DIR_MUESTRA.exists()
    return DIR_MUESTRA
