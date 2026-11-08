"""Pruebas de la ingeniería de características (sin fuga de información)."""
from __future__ import annotations

import pandas as pd

from demanda_citibike.features import COLS_CATEGORICAS, FEATURES, OBJETIVO, construir_caracteristicas


def test_construccion_sin_fuga(demanda_sintetica, clima_sintetico):
    df = construir_caracteristicas(demanda_sintetica, clima_sintetico)
    assert OBJETIVO in df.columns
    for c in FEATURES:
        assert c in df.columns

    # Para una estación, lag_1 en la hora t debe igualar salidas en la hora t-1.
    s1 = df[df["estacion"] == "S1"].sort_values("fecha_hora").reset_index(drop=True)
    for j in range(1, len(s1)):
        assert s1["lag_1"].iloc[j] == s1["salidas"].iloc[j - 1]


def test_lag_168_es_misma_hora_semana_anterior(demanda_sintetica, clima_sintetico):
    df = construir_caracteristicas(demanda_sintetica, clima_sintetico)
    s1 = df[df["estacion"] == "S1"].sort_values("fecha_hora").reset_index(drop=True)
    # lag_168 en la posición j == salidas en la posición j-168 (una semana antes).
    for j in range(168, len(s1)):
        assert s1["lag_168"].iloc[j] == s1["salidas"].iloc[j - 168]


def test_no_hay_fechas_nulas(demanda_sintetica, clima_sintetico):
    df = construir_caracteristicas(demanda_sintetica, clima_sintetico)
    assert df["fecha_hora"].notna().all()
    # Se eliminan las primeras horas sin historia (lag máximo = 168)
    assert df["lag_168"].notna().all()


def test_columnas_categoricas_presentes(demanda_sintetica, clima_sintetico):
    df = construir_caracteristicas(demanda_sintetica, clima_sintetico)
    for c in COLS_CATEGORICAS:
        assert c in df.columns
    assert df["estacion_cat"].nunique() == 3


# --- Modelo de horizonte de negocio (día siguiente): sin fuga de las últimas 24 h ---

def test_features_dia_solo_usan_rezagos_de_24h_o_mas():
    from demanda_citibike.features import FEATURES_DIA, rezago_minimo

    for c in FEATURES_DIA:
        r = rezago_minimo(c)
        assert r == 0 or r >= 24, f"{c} usa información de hace {r} h"
    assert not any(c in FEATURES_DIA for c in ["lag_1", "lag_2", "lag_3", "media_3", "media_24", "media_168"])


def test_features_dia_no_cambian_si_cambian_las_ultimas_24h(demanda_sintetica, clima_sintetico):
    """Si se alteran las salidas de las 24 h previas a t (desconocidas a medianoche),
    las variables del modelo día siguiente en t deben quedar idénticas."""
    from demanda_citibike.features import FEATURES_DIA

    t = pd.Timestamp("2024-07-20 18:00")
    base = construir_caracteristicas(demanda_sintetica, clima_sintetico)

    alterada = demanda_sintetica.copy()
    ventana = (alterada["estacion"] == "S1") & (alterada["fecha_hora"] > t - pd.Timedelta(hours=24)) & (alterada["fecha_hora"] <= t)
    alterada.loc[ventana, "salidas"] += 100
    nueva = construir_caracteristicas(alterada, clima_sintetico)

    fila = lambda d: d[(d["estacion"] == "S1") & (d["fecha_hora"] == t)].iloc[0]  # noqa: E731
    a, b = fila(base), fila(nueva)
    for c in FEATURES_DIA:
        assert a[c] == b[c], f"{c} cambió: depende de las últimas 24 h"
    assert a["lag_1"] != b["lag_1"]  # control: el modelo a 1 hora sí lo ve


def test_media_24_d1_es_media_de_t48_a_t24(demanda_sintetica, clima_sintetico):
    df = construir_caracteristicas(demanda_sintetica, clima_sintetico)
    s1 = demanda_sintetica[demanda_sintetica["estacion"] == "S1"].set_index("fecha_hora")["salidas"]
    t = pd.Timestamp("2024-07-10 08:00")
    esperado = s1[t - pd.Timedelta(hours=47): t - pd.Timedelta(hours=24)].mean()
    fila = df[(df["estacion"] == "S1") & (df["fecha_hora"] == t)].iloc[0]
    assert abs(fila["media_24_d1"] - esperado) < 1e-4
