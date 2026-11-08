"""Configuración: rutas, nombres de columnas, variables del modelo y supuestos de negocio."""
from __future__ import annotations

from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
DATOS_CRUDOS = RAIZ / "data" / "raw" / "bank-additional-full.csv"
MODELOS = RAIZ / "models"
REPORTES = RAIZ / "reports"
FIGURAS = REPORTES / "figures"

# ---------------------------------------------------------------------------
# Columnas originales (UCI, en inglés) → nombres en español
# ---------------------------------------------------------------------------
RENOMBRAR = {
    "age": "edad", "job": "ocupacion", "marital": "estado_civil", "education": "educacion",
    "default": "mora", "housing": "hipoteca", "loan": "prestamo", "contact": "tipo_contacto",
    "month": "mes", "day_of_week": "dia_semana", "duration": "duracion_llamada",
    "campaign": "contactos_campana", "pdays": "dias_desde_contacto", "previous": "contactos_previos",
    "poutcome": "resultado_previo", "emp.var.rate": "var_empleo", "cons.price.idx": "ipc",
    "cons.conf.idx": "confianza_consumidor", "euribor3m": "euribor_3m", "nr.employed": "num_empleados",
    "y": "suscribio",
}
OBJETIVO = "suscribio"
MESES = {"mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}

# ---------------------------------------------------------------------------
# Fuga de información: variables que NO se conocen al decidir a quién llamar
#   duracion_llamada  : se conoce cuando la llamada TERMINÓ (duración 0 ⇒ "no" seguro). Es la fuga clásica.
#   contactos_campana : nº de llamadas de ESTA campaña incluida la última; se deja de llamar cuando el
#                       cliente acepta, así que depende del resultado. Tampoco se conoce antes de empezar.
# ---------------------------------------------------------------------------
FUGA = ["duracion_llamada", "contactos_campana"]

# ---------------------------------------------------------------------------
# Variables del modelo (todas conocidas antes de marcar el teléfono)
# ---------------------------------------------------------------------------
CATEGORICAS = ["ocupacion", "estado_civil", "educacion", "mora", "hipoteca", "prestamo",
               "tipo_contacto", "mes", "dia_semana", "resultado_previo"]
NUMERICAS = ["edad", "contactado_antes", "dias_desde_contacto", "contactos_previos",
             "var_empleo", "ipc", "confianza_consumidor", "euribor_3m", "num_empleados"]
VARIABLES = CATEGORICAS + NUMERICAS

ETIQUETAS = {
    "edad": "Edad", "ocupacion": "Ocupación", "estado_civil": "Estado civil", "educacion": "Educación",
    "mora": "Crédito en mora", "hipoteca": "Tiene hipoteca", "prestamo": "Tiene préstamo personal",
    "tipo_contacto": "Tipo de teléfono", "mes": "Mes del contacto", "dia_semana": "Día de la semana",
    "resultado_previo": "Resultado campaña anterior", "contactado_antes": "Contactado antes",
    "dias_desde_contacto": "Días desde el último contacto", "contactos_previos": "Contactos previos",
    "var_empleo": "Variación del empleo", "ipc": "Índice de precios (IPC)",
    "confianza_consumidor": "Confianza del consumidor", "euribor_3m": "Euríbor 3 meses",
    "num_empleados": "Nº de empleados (economía)",
}

# ---------------------------------------------------------------------------
# Validación
#   Principal : 80 % entrenamiento / 20 % prueba, estratificado (semilla fija);
#               5 pliegues estratificados dentro del entrenamiento para elegir modelo, calibrar y fijar umbral.
#   Temporal  : prueba de estrés — entrenar hasta 2009-04 y evaluar 2009-05 → 2010-11.
# ---------------------------------------------------------------------------
PROPORCION_PRUEBA = 0.20
PLIEGUES = 5
CORTE_TEMPORAL = (2009, 5)          # (año, mes): desde aquí es "futuro" en la prueba de estrés
SEMILLA = 42

# ---------------------------------------------------------------------------
# Supuestos de negocio (USD por cliente) — explícitos y fáciles de cambiar
#   COSTO_LLAMADA   : costo de gestionar a un cliente en la campaña. El dataset promedia 2,6 llamadas
#                     por cliente; a ~3 USD por llamada (≈ 5 min de agente con cargas + telefonía) ≈ 8 USD.
#   BENEFICIO_EXITO : margen de un depósito a plazo. Depósito típico de 5.000 USD a 1 año con un margen
#                     neto de intereses de 1,6 % ≈ 80 USD.
#   ⇒ Con probabilidades bien calibradas, conviene llamar si p × 80 > 8, es decir si p > 10 %.
# ---------------------------------------------------------------------------
COSTO_LLAMADA = 8.0
BENEFICIO_EXITO = 80.0

COLORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
