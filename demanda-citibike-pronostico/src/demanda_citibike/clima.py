"""Carga del clima horario de Nueva York y de los feriados (calendario)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from demanda_citibike.config import RAW

MAPA_CLIMA = {
    "temperature_2m": "temperatura_c",
    "precipitation": "precipitacion_mm",
    "rain": "lluvia_mm",
    "wind_speed_10m": "viento_kmh",
}


def cargar_clima(ruta: Path = RAW / "clima_nyc_horario.csv") -> pd.DataFrame:
    """Carga el clima horario de Open-Meteo con nombres en español y frecuencia horaria."""
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el clima en {ruta}. Ejecuta descargar_datos.py")
    df = pd.read_csv(ruta, parse_dates=["fecha"])
    df = df.rename(columns=MAPA_CLIMA)
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.set_index("fecha").sort_index()
    df = df.asfreq("h")
    return df


def es_feriado(fechas: pd.Series, anio: int = 2024) -> pd.Series:
    """Marca 1 si la fecha (hora local de Nueva York) cae en feriado de EE. UU./NY."""
    try:
        import holidays  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Instala la librería 'holidays' (está en requirements.txt).") from exc

    feriados = holidays.country_holidays("US", subdiv="NY", years=anio)
    dias = pd.to_datetime(fechas).dt.normalize()
    return dias.isin(pd.to_datetime(list(feriados))).astype(int)
