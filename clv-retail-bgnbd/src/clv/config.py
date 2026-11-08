"""Configuración central: rutas, cortes temporales, reglas de limpieza y supuestos económicos."""
from __future__ import annotations

from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
DATOS_CRUDOS = RAIZ / "data" / "raw" / "online_retail_ii.csv"
MODELOS = RAIZ / "models"
REPORTES = RAIZ / "reports"
FIGURAS = REPORTES / "figures"

# ---------------------------------------------------------------------------
# Ventanas temporales (UCI Online Retail II: 2009-12-01 a 2011-12-09)
#   Calibración : 2009-12-01 → 2011-06-09 (555 días ≈ 79,3 semanas)
#   Prueba      : 2011-06-10 → 2011-12-09 (183 días = 6 meses exactos de holdout)
#   Supervisado : para entrenar LightGBM sin fuga de información, se construye
#                 una ventana histórica previa con corte en 2010-12-09 (373 días)
#                 y objetivo en los 182 días siguientes (2010-12-10 → 2011-06-09).
#   Estacional  : prueba de robustez de LightGBM con la misma estación que la prueba
#                 (corte 2010-06-09, objetivo 2010-06-10 → 2010-12-09), a costa de
#                 tener solo 190 días de historia para construir variables.
# ---------------------------------------------------------------------------
FECHA_INICIO = "2009-12-01"
FECHA_CORTE_SUPERVISADO = "2010-12-09"
FECHA_CORTE_ESTACIONAL = "2010-06-09"
FECHA_FIN_ESTACIONAL = "2010-12-09"
FECHA_CORTE_CAL = "2011-06-09"
FECHA_FIN = "2011-12-09"

DIAS_POR_SEMANA = 7.0
HORIZONTE_DIAS = 183
HORIZONTE_SEMANAS = HORIZONTE_DIAS / DIAS_POR_SEMANA
HORIZONTE_MESES = 6

# Códigos contables, envíos, comisiones bancarias y pruebas que no son compras de producto
CODIGOS_NO_PRODUCTO = {
    "POST", "D", "M", "BANK CHARGES", "PADS", "DOT", "CRUK",
    "ADJUST", "ADJUST2", "TEST001", "TEST002", "C2", "B", "S", "AMAZONFEE",
}

# Tope de ticket diario para separar distribuidores mayoristas extremos (p99 = ~2.500 GBP)
TOPE_TICKET_MAYORISTA = 3000.0

# ---------------------------------------------------------------------------
# Supuestos económicos y de negocio (en libras esterlinas, GBP)
#   MARGEN_BRUTO             : 30 % sobre el ingreso neto de ventas minoristas/regalos.
#   TASA_DESCUENTO_MENSUAL   : 1,0 % mensual (≈ 12,7 % anual, WACC típico retail).
#   COSTO_CONTACTO_RETENCION : 6,0 GBP por cliente contactado (incentivo + gestión).
#   EFECTIVIDAD_RESCATE      : 25 % de incremento relativo en la probabilidad de retención
#                              para clientes en zona de riesgo. Es un SUPUESTO (no hay
#                              experimento A/B en los datos): solo se usa para elegir a quién
#                              contactar y para el escenario económico, nunca para validar.
#   Zona de riesgo           : 0,20 <= P(vivo) < 0,70 y frecuencia >= 2 (= segmento
#                              «En riesgo de alto valor»). La misma definición la usan la
#                              segmentación, la política 4 de rescate y la API.
# ---------------------------------------------------------------------------
MARGEN_BRUTO = 0.30
TASA_DESCUENTO_MENSUAL = 0.01
COSTO_CONTACTO_RETENCION = 6.0
EFECTIVIDAD_RESCATE = 0.25
RANGO_RIESGO_VIVO = (0.20, 0.70)
FRECUENCIA_MIN_RIESGO = 2

# Escenario económico: sensibilidad al supuesto de efectividad (dentro de la zona de riesgo)
# y al efecto residual fuera de ella
GRILLA_EFECTIVIDAD = (0.05, 0.10, 0.25, 0.40)
GRILLA_EFECTO_FUERA = (0.0, 0.04)
EFECTO_FUERA_ZONA = 0.04
N_BOOTSTRAP = 1000

SEMILLA = 42

COLORES = [
    "#2a78d6",  # Azul principal
    "#eb6834",  # Naranja
    "#1baf7a",  # Verde
    "#eda100",  # Amarillo
    "#e87ba4",  # Rosa
    "#008300",  # Verde oscuro
    "#6250d6",  # Violeta
    "#e34948",  # Rojo
]
