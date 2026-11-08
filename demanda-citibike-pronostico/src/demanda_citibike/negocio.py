"""Uso de negocio: alertas de rebalanceo para el día siguiente y su costo.

Supuestos explícitos:
- Una estación-hora es un **pico** si sus salidas reales superan el percentil 90
  de sus propias salidas en el entrenamiento (hora que vaciaría la estación sin
  intervención). El umbral se calcula solo con datos de entrenamiento.
- Cada noche se emite una **alerta** para cada estación-hora de mañana cuyo
  pronóstico supera ese umbral; cada alerta dispara un viaje de rebalanceo.
- Errores posibles: **falsa alarma** (alerta sin pico: un viaje de camión en vano)
  y **pico perdido** (pico sin alerta: estación vacía, viajes no servidos).
- Costo = falsas alarmas × 1 + picos perdidos × r, en unidades de "un viaje en
  vano". La razón r (cuánto más cuesta un pico perdido que una falsa alarma) no
  se conoce con estos datos, por eso se reporta una sensibilidad para varios r y
  la razón de equilibrio entre estrategias.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PERCENTIL_PICO = 90
RAZONES_COSTO = [1, 2, 5, 10]


def calcular_umbrales(df_train: pd.DataFrame) -> dict:
    """Percentil 90 de salidas por estación (calculado solo con datos de entrenamiento)."""
    return df_train.groupby("estacion")["salidas"].quantile(PERCENTIL_PICO / 100).to_dict()


def _stats(alerta: np.ndarray, necesario: np.ndarray) -> dict:
    tp = int(np.sum(alerta & necesario))
    fp = int(np.sum(alerta & ~necesario))
    fn = int(np.sum(~alerta & necesario))
    return {
        "alertas": int(np.sum(alerta)),
        "picos_reales": int(np.sum(necesario)),
        "aciertos": tp,
        "falsas_alarmas": fp,
        "picos_perdidos": fn,
        "precision_pct": round(tp / (tp + fp) * 100, 1) if (tp + fp) else 0.0,
        "recall_pct": round(tp / (tp + fn) * 100, 1) if (tp + fn) else 0.0,
    }


def costo(stats: dict, razon: float) -> float:
    """Costo en 'viajes en vano': falsas alarmas + razón × picos perdidos."""
    return stats["falsas_alarmas"] + razon * stats["picos_perdidos"]


def razon_equilibrio(a: dict, b: dict) -> float | None:
    """Razón r a partir de la cual la estrategia `b` cuesta menos que `a`.

    Devuelve None si una estrategia domina a la otra (no hay cruce con r > 0).
    """
    d_fp = b["falsas_alarmas"] - a["falsas_alarmas"]
    d_fn = a["picos_perdidos"] - b["picos_perdidos"]
    if d_fn <= 0 or d_fp <= 0:
        return None
    return round(d_fp / d_fn, 2)


def analizar_rebalanceo(
    df_test: pd.DataFrame,
    umbrales: dict,
    estrategias: dict[str, str] | None = None,
    referencia: str = "linea_base_estacional",
) -> dict:
    """Compara estrategias de alerta (columna de pronóstico) en la detección de picos.

    `estrategias` mapea nombre → columna de `df_test`; por defecto la línea base
    estacional y el modelo (`pred`).
    """
    if estrategias is None:
        estrategias = {"linea_base_estacional": "base_estacional", "modelo_lightgbm": "pred"}
    umbral = df_test["estacion"].map(umbrales).to_numpy()
    necesario = df_test["salidas"].to_numpy() > umbral

    stats = {nombre: _stats(df_test[col].to_numpy() > umbral, necesario) for nombre, col in estrategias.items()}
    costos = {
        str(r): {nombre: costo(s, r) for nombre, s in stats.items()} for r in RAZONES_COSTO
    }
    mejor_por_razon = {r: min(c, key=c.get) for r, c in costos.items()}

    ref = stats[referencia]
    comparacion = {}
    for nombre, s in stats.items():
        if nombre == referencia:
            continue
        comparacion[nombre] = {
            "falsas_alarmas_evitadas": ref["falsas_alarmas"] - s["falsas_alarmas"],
            "picos_perdidos_evitados": ref["picos_perdidos"] - s["picos_perdidos"],
            "domina_a_la_referencia": bool(
                s["falsas_alarmas"] <= ref["falsas_alarmas"] and s["picos_perdidos"] <= ref["picos_perdidos"]
            ),
        }

    return {
        "umbral_percentil": PERCENTIL_PICO,
        "horas_estacion_evaluadas": int(len(df_test)),
        "referencia": referencia,
        "estrategias": stats,
        "frente_a_referencia": comparacion,
        "costo_por_razon": costos,
        "estrategia_mas_barata_por_razon": mejor_por_razon,
    }
