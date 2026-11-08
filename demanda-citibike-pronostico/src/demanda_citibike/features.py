"""Ingeniería de características sin fuga de información.

Construye, para cada estación × hora, el objetivo (salidas) y los predictores:
rezagos (lags) y medias móviles calculados solo con horas anteriores, calendario
(hora, día de semana, fin de semana, mes, feriado) y clima horario.

Hay dos conjuntos de predictores:

- ``FEATURES_DIA`` (modelo principal, horizonte de negocio): el pronóstico se
  emite a medianoche del día D para las 24 horas de D+1, por lo que solo se usan
  rezagos de 24 h o más y medias móviles sobre la serie desplazada 24 h.
- ``FEATURES`` / ``FEATURES_1H`` (comparación): usa también la hora anterior
  (``lag_1``…``lag_3``); sirve para pronosticar la próxima hora, no mañana.
"""
from __future__ import annotations

import pandas as pd

from demanda_citibike.clima import es_feriado
from demanda_citibike.config import HORIZONTE_MIN_DIA, LAGS, LAGS_DIA, VENTANAS_MEDIA, VENTANAS_MEDIA_DIA

# "mes" es numérico: el test cae en diciembre, que no aparece en el entrenamiento;
# como categórica sería un nivel nunca visto, como numérica se comporta como noviembre.
COLS_CATEGORICAS = ["estacion_cat", "hora", "dia_semana"]
COLS_CLIMA = ["temperatura_c", "precipitacion_mm", "lluvia_mm", "viento_kmh"]
COLS_CALENDARIO = ["es_fin_de_semana", "es_feriado", "mes"]

COLS_NUMERICAS = (
    [f"lag_{l}" for l in LAGS]
    + [f"media_{v}" for v in VENTANAS_MEDIA]
    + COLS_CLIMA
    + COLS_CALENDARIO
)
FEATURES_1H = COLS_NUMERICAS + COLS_CATEGORICAS
FEATURES = FEATURES_1H  # nombre histórico (modelo a 1 hora)

COLS_NUMERICAS_DIA = (
    [f"lag_{l}" for l in LAGS_DIA]
    + [f"media_{v}_d1" for v in VENTANAS_MEDIA_DIA]
    + COLS_CLIMA
    + COLS_CALENDARIO
)
FEATURES_DIA = COLS_NUMERICAS_DIA + COLS_CATEGORICAS
FEATURES_DIA_SIN_CLIMA = [c for c in FEATURES_DIA if c not in COLS_CLIMA]

# Conjuntos que entrena el pipeline (nombre → predictores).
CONJUNTOS = {
    "dia_siguiente": FEATURES_DIA,
    "dia_siguiente_sin_clima": FEATURES_DIA_SIN_CLIMA,
    "una_hora": FEATURES_1H,
}
OBJETIVO = "salidas"


def rezago_minimo(feature: str) -> int:
    """Rezago mínimo (horas) de la información que usa una característica derivada de la demanda.

    Devuelve 0 para calendario, clima y estación (no dependen de la demanda pasada).
    """
    if feature.startswith("lag_"):
        return int(feature.split("_")[1])
    if feature.startswith("media_"):
        return HORIZONTE_MIN_DIA if feature.endswith("_d1") else 1
    return 0


def construir_caracteristicas(demanda: pd.DataFrame, clima: pd.DataFrame) -> pd.DataFrame:
    """Añade calendario, clima, rezagos y medias móviles a la demanda horaria."""
    df = demanda.copy()
    df["fecha_hora"] = pd.to_datetime(df["fecha_hora"])
    df = df.sort_values(["estacion", "fecha_hora"]).reset_index(drop=True)

    # Calendario
    df["hora"] = df["fecha_hora"].dt.hour
    df["dia_semana"] = df["fecha_hora"].dt.day_of_week  # 0 = lunes
    df["es_fin_de_semana"] = (df["dia_semana"] >= 5).astype("int8")
    df["mes"] = df["fecha_hora"].dt.month.astype("int8")
    df["es_feriado"] = es_feriado(df["fecha_hora"]).to_numpy().astype("int8")

    # Clima horario (hora local de Nueva York)
    clima_df = clima.reset_index()
    clima_df = clima_df.rename(columns={clima_df.columns[0]: "fecha_hora"})
    clima_df["fecha_hora"] = pd.to_datetime(clima_df["fecha_hora"]).astype(df["fecha_hora"].dtype)
    df = df.merge(clima_df, on="fecha_hora", how="left")
    for c in COLS_CLIMA:
        df[c] = df[c].ffill().fillna(0.0).astype("float32")

    # Rezagos y medias móviles por estación (shift => solo pasado, sin fuga).
    # Re-ordenar tras el merge para garantizar el orden temporal por estación.
    df = df.sort_values(["estacion", "fecha_hora"]).reset_index(drop=True)
    g = df.groupby("estacion", sort=False)["salidas"]
    for lag in sorted(set(LAGS) | set(LAGS_DIA)):
        df[f"lag_{lag}"] = g.shift(lag).astype("float32")
    for v in VENTANAS_MEDIA:
        df[f"media_{v}"] = (
            g.shift(1).groupby(df["estacion"], sort=False).rolling(v, min_periods=1).mean()
            .reset_index(level=0, drop=True).astype("float32")
        )
    for v in VENTANAS_MEDIA_DIA:
        df[f"media_{v}_d1"] = (
            g.shift(HORIZONTE_MIN_DIA).groupby(df["estacion"], sort=False).rolling(v, min_periods=1).mean()
            .reset_index(level=0, drop=True).astype("float32")
        )

    # Codificación de estación para LightGBM (categórica)
    df["estacion_cat"] = df["estacion"].astype("category").cat.codes.astype("int32")

    # Se descartan las filas sin suficiente historia (antes de completar el rezago máximo),
    # así todos los modelos se entrenan y evalúan sobre exactamente las mismas filas.
    df = df.dropna(subset=[f"lag_{max(set(LAGS) | set(LAGS_DIA))}"]).reset_index(drop=True)
    return df
