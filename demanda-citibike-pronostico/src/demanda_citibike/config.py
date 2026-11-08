"""Configuración central del proyecto: rutas, periodo, clima y constantes."""
from __future__ import annotations

from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
DATOS = RAIZ / "data"
RAW = DATOS / "raw"
PROCESADOS = DATOS / "procesados"
MUESTRA = DATOS / "muestra"
REPORTES = RAIZ / "reports"
FIGURAS = REPORTES / "figures"
MODELOS = RAIZ / "models"

# 12 meses de Citi Bike (2024 completo), según la ficha "usar al menos 12 meses".
MESES = [f"2024{m:02d}" for m in range(1, 13)]

# Clima de Nueva York (Midtown), coordenadas indicadas en la ficha.
LAT = 40.71
LON = -74.01
VARS_CLIMA = ["temperature_2m", "precipitation", "rain", "wind_speed_10m"]

# Número de estaciones con más actividad que se modelan (la ficha pide ~200-300).
N_ESTACIONES = 250

# Periodo modelado: año calendario 2024 (los CSV de enero traen algunos viajes
# iniciados el 31-12-2023 que se descartan).
INICIO_PERIODO = "2024-01-01 00:00:00"
FIN_PERIODO = "2024-12-31 23:00:00"

# Modelo de comparación "a 1 hora": rezagos y medias móviles desde la hora anterior.
LAGS = [1, 2, 3, 24, 168]
VENTANAS_MEDIA = [3, 24, 168]

# Modelo de horizonte de negocio ("día siguiente"): el pronóstico se emite a
# medianoche del día D para las 24 horas de D+1, así que solo se usan rezagos
# de 24 h o más (todo lo usado ocurrió como muy tarde a las 23:00 del día D).
LAGS_DIA = [24, 48, 168, 336]
VENTANAS_MEDIA_DIA = [24, 168]  # medias móviles calculadas sobre la serie desplazada 24 h
HORIZONTE_MIN_DIA = 24

# Cuantiles para los intervalos de predicción.
CUANTILES = [0.1, 0.9]

for carpeta in (RAW, PROCESADOS, REPORTES, FIGURAS, MUESTRA, MODELOS):
    carpeta.mkdir(parents=True, exist_ok=True)
