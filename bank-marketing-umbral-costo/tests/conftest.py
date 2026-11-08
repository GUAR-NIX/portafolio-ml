import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ))

from campana import config  # noqa: E402


@pytest.fixture(scope="session")
def sinteticos() -> pd.DataFrame:
    """Clientes sintéticos con el mismo esquema que el real y un resultado conocido:
    la probabilidad sube con «éxito en la campaña anterior» y con euríbor bajo (~11 % de positivos)."""
    rng = np.random.default_rng(0)
    n = 3000
    df = pd.DataFrame({
        "edad": rng.integers(18, 90, n),
        "ocupacion": rng.choice(["admin.", "blue-collar", "student", "retired"], n),
        "estado_civil": rng.choice(["married", "single", "divorced"], n),
        "educacion": rng.choice(["university.degree", "high.school", "basic.9y"], n),
        "mora": rng.choice(["no", "unknown"], n),
        "hipoteca": rng.choice(["yes", "no"], n),
        "prestamo": rng.choice(["yes", "no"], n),
        "tipo_contacto": rng.choice(["cellular", "telephone"], n),
        "mes": rng.choice(["may", "jun", "oct"], n),
        "dia_semana": rng.choice(["mon", "tue", "wed", "thu", "fri"], n),
        "resultado_previo": rng.choice(["nonexistent", "failure", "success"], n, p=[0.85, 0.1, 0.05]),
        "contactos_previos": rng.integers(0, 3, n),
        "var_empleo": rng.uniform(-3, 1.4, n),
        "ipc": rng.uniform(92, 95, n),
        "confianza_consumidor": rng.uniform(-50, -27, n),
        "euribor_3m": rng.uniform(0.6, 5, n),
        "num_empleados": rng.uniform(4960, 5230, n),
    })
    df["contactado_antes"] = (df["resultado_previo"] != "nonexistent").astype(int)
    df["dias_desde_contacto"] = np.where(df["contactado_antes"] == 1, rng.integers(0, 27, n), np.nan)
    logit = -2.6 + 2.5 * (df["resultado_previo"] == "success") - 0.5 * (df["euribor_3m"] - 3)
    df[config.OBJETIVO] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df
