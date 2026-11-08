"""Configuración global, taxonomía oficial de las 10 clases de EuroSAT y rutas del proyecto."""

from __future__ import annotations

from pathlib import Path

RAIZ_PROYECTO = Path(__file__).resolve().parents[2]
DIR_DATOS = RAIZ_PROYECTO / "data"
DIR_RAW = DIR_DATOS / "raw"
DIR_MUESTRA = DIR_DATOS / "muestra_eurosat"
DIR_MODELOS = RAIZ_PROYECTO / "models"
DIR_REPORTES = RAIZ_PROYECTO / "reports"
DIR_FIGURAS = DIR_REPORTES / "figures"

SEMILLA = 42

# Las 10 clases oficiales de EuroSAT (Sentinel-2 MSI RGB, 64x64 píxeles, 10 m/píxel)
CLASES_EUROSAT: list[str] = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

NOMBRES_ESPANOL: dict[str, str] = {
    "AnnualCrop": "Cultivo anual",
    "Forest": "Bosque denso",
    "HerbaceousVegetation": "Vegetación herbácea",
    "Highway": "Autopista / vialidad",
    "Industrial": "Zona industrial",
    "Pasture": "Pastizal ganadero",
    "PermanentCrop": "Cultivo permanente",
    "Residential": "Zona residencial",
    "River": "Río / cauce fluvial",
    "SeaLake": "Mar / cuerpo lacustre",
}

MACRO_CATEGORIA: dict[str, str] = {
    "AnnualCrop": "Agrícola",
    "PermanentCrop": "Agrícola",
    "Pasture": "Agrícola",
    "Forest": "Ecosistema natural",
    "HerbaceousVegetation": "Ecosistema natural",
    "Highway": "Infraestructura antrópica",
    "Industrial": "Infraestructura antrópica",
    "Residential": "Infraestructura antrópica",
    "River": "Cuerpo de agua",
    "SeaLake": "Cuerpo de agua",
}

# Costos SUPUESTOS (no medidos) de un servicio de monitoreo ambiental / catastro, en USD por parcela de 40 ha.
# No provienen de ninguna tarifa real: sirven para ilustrar que confundir infraestructura con bosque o agua
# es más grave que confundir dos cultivos. reports/sensibilidad_costos.csv muestra cómo cambian las
# conclusiones si se multiplican por 0,5 o por 2.
COSTO_AUDITORIA_CAMPO_USD = 18.0
COSTO_ERROR_MISMA_MACROCATEGORIA_USD = 35.0
COSTO_ERROR_CRITICO_INTER_MACRO_USD = 140.0

# Nombres cortos de los 5 modelos comparados (mismo orden en todos los reportes)
MODELOS = {
    "m1": "1 · Espectral: 54 índices + textura + GBDT",
    "m2": "2 · MobileNetV3 congelada + sonda lineal",
    "m3": "3 · Fusión: embeddings CNN + 54 índices + MLP",
    "m4": "4 · Fine-tuning parcial MobileNetV3 (D4 + TTA)",
    "m5": "5 · Ensamble M3 + M4 (calibrado con temperatura)",
}

# Archivos de modelo versionados en Git (pesan pocos MB) para que la API y la app funcionen en un clon fresco
ARCHIVO_JOBLIB = "clasificador_eurosat.joblib"
ARCHIVO_PT = "mobilenetv3_eurosat_finetuned.pt"
ARCHIVO_ONNX_FT = "mobilenetv3_eurosat_finetuned.onnx"
ARCHIVO_ONNX_EXTRACTOR = "mobilenetv3_imagenet_extractor.onnx"
