"""Rutas, fechas de corte y supuestos del proyecto."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
DATOS_CRUDOS = RAIZ / "data" / "raw" / "ml-25m"
DATOS_PROCESADOS = RAIZ / "data" / "processed"
MODELOS = RAIZ / "models"
REPORTES = RAIZ / "reports"
FIGURAS = REPORTES / "figures"

SEMILLA = 42

# Una calificación >= 4 estrellas cuenta como "le gustó" (señal implícita positiva y relevancia en la evaluación).
UMBRAL_POSITIVO = 4.0
K = 10

# Partición temporal GLOBAL (mismo reloj para todos los usuarios): nada posterior al corte entra al entrenamiento.
#   validación : entrenar con todo < 2017-07-01, evaluar 2017-07-01 → 2018-01-01  (búsqueda de hiperparámetros)
#   prueba     : entrenar con todo < 2018-01-01, evaluar 2018-01-01 → fin de los datos (2019-11-21), una sola vez
# La ventana de prueba es larga a propósito: con 6 meses solo vuelven 271 usuarios con pocos positivos.
CORTE_VALIDACION = pd.Timestamp("2017-07-01")
CORTE_PRUEBA = pd.Timestamp("2018-01-01")
FIN_PRUEBA = pd.Timestamp("2019-12-01")

# Popularidad "reciente": positivos de los últimos 180 días antes del corte.
DIAS_RECIENTES = 180

# Catálogo recomendable: películas con al menos este número de positivos en entrenamiento.
MIN_POSITIVOS_ITEM = 5

# Segmentos de usuario según sus positivos en entrenamiento.
SEGMENTOS = [(1, 20, "pocos (1-20)"), (21, 100, "medios (21-100)"), (101, 10**9, "muchos (>100)")]

# Usuarios nuevos (sin historia antes del corte): sus primeras N calificaciones del periodo de prueba hacen de
# "onboarding" (como el cuestionario inicial de MovieLens) y se evalúa con el resto.
N_ONBOARDING = 5

# Para no saturar la PC se evalúa una muestra fija de usuarios por segmento (cifras reportadas con ella).
MAX_USUARIOS_EVAL = 20_000

# Hilos para implicit/BLAS: hay otras sesiones usando la PC.
HILOS = 6


def segmento_de(n_positivos: int) -> str:
    for minimo, maximo, nombre in SEGMENTOS:
        if minimo <= n_positivos <= maximo:
            return nombre
    return "nuevos (0)"
