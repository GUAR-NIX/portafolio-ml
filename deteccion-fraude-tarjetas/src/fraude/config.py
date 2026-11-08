"""Rutas, semillas y supuestos del proyecto (todo lo que se puede cambiar está aquí)."""

from __future__ import annotations

from pathlib import Path

RAIZ_PROYECTO = Path(__file__).resolve().parents[2]
DIR_DATOS = RAIZ_PROYECTO / "data"
DIR_RAW = DIR_DATOS / "raw"
DIR_MUESTRA = DIR_DATOS / "muestra"
ARCHIVO_DATOS = DIR_RAW / "creditcard.parquet"
DIR_MODELOS = RAIZ_PROYECTO / "models"
DIR_REPORTES = RAIZ_PROYECTO / "reports"
DIR_FIGURAS = DIR_REPORTES / "figures"

SEMILLA = 42

# --- Partición temporal (por cuantiles de Time, sin barajar) ---
CORTE_ENTRENAMIENTO = 0.60  # primer 60 % de las transacciones en el tiempo
CORTE_VALIDACION = 0.80  # siguiente 20 %; el último 20 % es la prueba

V_COLS = [f"V{i}" for i in range(1, 29)]
COLUMNAS_MODELO = V_COLS + ["log_monto"]
SEGUNDOS_DIA = 86_400

# --- Supuestos de costo (en EUR, la moneda de los titulares europeos del dataset) ---
COSTO_REVISION = 5.0  # revisar una alerta: ~6 min de un analista a ~50 EUR/h
FRACCION_PERDIDA = 1.0  # fraude no detectado: se pierde el monto completo (sin contracargo)
COSTO_REVISION_SENSIBILIDAD = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
FRACCION_PERDIDA_SENSIBILIDAD = [0.25, 0.5, 1.0]

# --- Capacidad de revisión: alertas que un equipo puede revisar ---
ALERTAS_POR_DIA = 250  # supuesto: 2 analistas × ~125 alertas por día
PRECISION_OBJETIVO = [0.80, 0.90]  # recall a precisión fija

# --- Bootstrap ---
N_BOOTSTRAP = 1000

# --- Pocas etiquetas ---
N_FRAUDES_POCAS_ETIQUETAS = [5, 10, 20, 35, 50, 100, 200, 360]
REPETICIONES_POCAS_ETIQUETAS = 10

# --- Catálogo de modelos (clave interna → nombre en reportes) ---
ETIQUETAS = {
    "regla_monto": "Regla por monto",
    "logistica": "Regresión logística",
    "if_sin_etiquetas": "Isolation Forest (sin etiquetas)",
    "if_normales": "Isolation Forest (solo normales)",
    "lof": "LOF novelty (solo normales)",
    "autoencoder": "Autoencoder (solo normales)",
    "lgbm_sin_ajuste": "LightGBM sin ajuste",
    "lgbm_pesos": "LightGBM con pesos",
    "lgbm_submuestreo": "LightGBM con submuestreo 1:20",
    "hibrido": "Híbrido: LightGBM + anomalías",
}
FAMILIA = {
    "regla_monto": "Línea base", "logistica": "Línea base",
    "if_sin_etiquetas": "No supervisado", "if_normales": "No supervisado", "lof": "No supervisado", "autoencoder": "No supervisado",
    "lgbm_sin_ajuste": "Supervisado", "lgbm_pesos": "Supervisado", "lgbm_submuestreo": "Supervisado", "hibrido": "Híbrido",
}
NO_SUPERVISADOS = ["if_sin_etiquetas", "if_normales", "lof", "autoencoder"]
CANDIDATOS_SERVIDO = ["lgbm_sin_ajuste", "lgbm_pesos", "lgbm_submuestreo", "hibrido"]
RAZON_SUBMUESTREO = 20
NOMBRES_HIBRIDO = COLUMNAS_MODELO + ["puntaje_if", "log_error_ae"]
