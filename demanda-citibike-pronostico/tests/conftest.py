"""Fixtures sintéticos para las pruebas (sin internet ni datos reales)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def demanda_sintetica() -> pd.DataFrame:
    """3 estaciones × 60 días de salidas/llegadas horarias sintéticas."""
    np.random.seed(42)
    estaciones = ["S1", "S2", "S3"]
    horas = pd.date_range("2024-06-01", periods=24 * 60, freq="h")
    partes = []
    for e in estaciones:
        base = np.random.poisson(6, len(horas))
        patron = 3 + 3 * np.sin((horas.hour - 8) / 24 * 2 * np.pi)
        salidas = np.maximum(0, np.round(base + patron)).astype(int)
        llegadas = np.maximum(0, salidas + np.random.randint(-3, 4, len(horas))).astype(int)
        partes.append(pd.DataFrame({
            "estacion": e,
            "fecha_hora": horas,
            "salidas": salidas,
            "llegadas": llegadas,
        }))
    return pd.concat(partes, ignore_index=True)


@pytest.fixture
def clima_sintetico() -> pd.DataFrame:
    """Clima horario sintético alineado con la demanda."""
    horas = pd.date_range("2024-06-01", periods=24 * 60, freq="h")
    return pd.DataFrame({
        "temperatura_c": np.linspace(10, 25, len(horas)),
        "precipitacion_mm": np.zeros(len(horas)),
        "lluvia_mm": np.zeros(len(horas)),
        "viento_kmh": np.full(len(horas), 10.0),
    }, index=horas).asfreq("h")
