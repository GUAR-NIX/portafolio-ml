"""Segmentación accionable combinando P(vivo), frecuencia y CLV, validación de políticas de retención
con el gasto real del periodo de prueba y escenario económico (con supuestos explícitos)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def en_zona_riesgo(prob_vivo: float | np.ndarray, frecuencia: int | np.ndarray) -> np.ndarray:
    """
    Zona de riesgo única del proyecto: 0,20 <= P(vivo) < 0,70 y frecuencia >= 2.
    Coincide con el segmento «En riesgo de alto valor» y la usan la política 4 y la API.
    """
    pv = np.asarray(prob_vivo, dtype=float)
    fr = np.asarray(frecuencia, dtype=int)
    lo, hi = config.RANGO_RIESGO_VIVO
    return (pv >= lo) & (pv < hi) & (fr >= config.FRECUENCIA_MIN_RIESGO)


def asignar_segmento_cliente(
    prob_vivo: float | np.ndarray,
    frecuencia: int | np.ndarray,
    clv_neto: float | np.ndarray,
    umbral_alto_valor: float = 120.0,
) -> np.ndarray:
    """
    Clasifica a cada cliente en 5 segmentos accionables cruzando su probabilidad de seguir activo,
    su historial de repetición y su valor esperado:
      1. Campeones activos       : P(vivo) >= 0,70 y CLV >= p75 (o frecuencia >= 4)
      2. En riesgo de alto valor : 0,20 <= P(vivo) < 0,70 y frecuencia >= 2 (zona de riesgo)
      3. Nuevos / prometedores   : P(vivo) >= 0,70 pero frecuencia baja (0 o 1 repetición)
      4. Regulares / ocasionales : el resto (actividad intermedia de bajo valor)
      5. Inactivos / perdidos    : P(vivo) < 0,20
    """
    pv = np.asarray(prob_vivo, dtype=float)
    fr = np.asarray(frecuencia, dtype=int)
    clv = np.asarray(clv_neto, dtype=float)
    lo, hi = config.RANGO_RIESGO_VIVO

    condiciones = [
        (pv < lo),
        en_zona_riesgo(pv, fr),
        (pv >= hi) & ((clv >= umbral_alto_valor) | (fr >= 4)),
        (pv >= hi) & (fr <= 1),
    ]
    elecciones = [
        "Inactivos / perdidos",
        "En riesgo de alto valor",
        "Campeones activos",
        "Nuevos / prometedores",
    ]
    return np.select(condiciones, elecciones, default="Regulares / ocasionales")


ACCIONES_SEGMENTO = {
    "Campeones activos": "Fidelización VIP y acceso anticipado (sin descuento agresivo)",
    "En riesgo de alto valor": "Campaña prioritaria de reactivación personalizada (rescate)",
    "Nuevos / prometedores": "Secuencia de segunda compra (cross-sell a 30 días)",
    "Regulares / ocasionales": "Newsletter automatizada de bajo costo",
    "Inactivos / perdidos": "No invertir presupuesto directo (solo retargeting pasivo)",
}

ORDEN_SEGMENTOS = list(ACCIONES_SEGMENTO)


def margen_si_sigue_vivo(
    compras_esperadas: np.ndarray,
    prob_vivo: np.ndarray,
    ticket_esperado: np.ndarray,
    margen_bruto: float = config.MARGEN_BRUTO,
) -> np.ndarray:
    """Margen a 6 meses que el modelo espera si el cliente siguiera activo con certeza."""
    tasa_si_vivo = np.asarray(compras_esperadas, dtype=float) / np.maximum(np.asarray(prob_vivo, dtype=float), 0.05)
    return tasa_si_vivo * np.asarray(ticket_esperado, dtype=float) * margen_bruto


def construir_tabla_segmentos(df_clientes: pd.DataFrame, pred_clv: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Enriquece la tabla de clientes con las predicciones CLV y resume métricas por segmento."""
    cartera = df_clientes.copy()
    cartera["prob_vivo"] = pred_clv["prob_vivo"]
    cartera["prob_recompra"] = pred_clv.get("prob_recompra", pred_clv["prob_vivo"])
    cartera["compras_esperadas"] = pred_clv["compras_esperadas"]
    cartera["ticket_esperado"] = pred_clv["ticket_esperado"]
    cartera["ingreso_esperado"] = pred_clv["ingreso_esperado"]
    cartera["clv_neto"] = pred_clv["clv_neto"]

    # Margen que se perdería por abandono (solo en la zona de riesgo) y beneficio esperado de contactar,
    # bajo el SUPUESTO de efectividad de config.EFECTIVIDAD_RESCATE
    margen_vivo = margen_si_sigue_vivo(cartera["compras_esperadas"], cartera["prob_vivo"], cartera["ticket_esperado"])
    zona = en_zona_riesgo(cartera["prob_vivo"], cartera["frecuencia"])
    cartera["margen_en_riesgo_gbp"] = np.where(zona, (1.0 - cartera["prob_vivo"]) * margen_vivo, 0.0)
    cartera["beneficio_neto_rescate_gbp"] = (
        cartera["margen_en_riesgo_gbp"] * config.EFECTIVIDAD_RESCATE - config.COSTO_CONTACTO_RETENCION
    )
    cartera["conviene_contactar_rescate"] = zona & (cartera["beneficio_neto_rescate_gbp"] > 0.0)

    umbral_alto = float(np.percentile(cartera["clv_neto"], 75))
    cartera["segmento"] = asignar_segmento_cliente(
        cartera["prob_vivo"],
        cartera["frecuencia"],
        cartera["clv_neto"],
        umbral_alto_valor=umbral_alto,
    )
    cartera["accion_recomendada"] = cartera["segmento"].map(ACCIONES_SEGMENTO)

    resumen = (
        cartera.groupby("segmento", as_index=False)
        .agg(
            n_clientes=("customer_id", "count"),
            frecuencia_media=("frecuencia", "mean"),
            inactividad_media_dias=("inactividad_dias", "mean"),
            prob_vivo_media=("prob_vivo", "mean"),
            prob_recompra_media=("prob_recompra", "mean"),
            tasa_recompra_real=("vivo_holdout", "mean"),
            clv_neto_medio_gbp=("clv_neto", "mean"),
            ingreso_esperado_total_gbp=("ingreso_esperado", "sum"),
            ingreso_real_holdout_gbp=("gasto_holdout_acotado", "sum"),
        )
    )
    resumen["orden"] = resumen["segmento"].map({s: i for i, s in enumerate(ORDEN_SEGMENTOS)})
    resumen = resumen.sort_values("orden").drop(columns=["orden"]).reset_index(drop=True)
    resumen["pct_clientes"] = resumen["n_clientes"] / max(len(cartera), 1) * 100.0
    resumen["pct_ingreso_real"] = (
        resumen["ingreso_real_holdout_gbp"]
        / max(float(resumen["ingreso_real_holdout_gbp"].sum()), 1.0)
        * 100.0
    )
    resumen["ratio_ingreso_real_vs_esperado"] = (
        resumen["ingreso_real_holdout_gbp"] / resumen["ingreso_esperado_total_gbp"].clip(lower=1.0)
    )
    resumen["accion_recomendada"] = resumen["segmento"].map(ACCIONES_SEGMENTO)
    return cartera, resumen


def definir_politicas(cartera: pd.DataFrame) -> dict[str, np.ndarray]:
    """
    Las 4 políticas de campaña de reactivación que se comparan (todas usan solo información hasta el corte):
      1. Contactar a todos los clientes con > 60 días sin comprar.
      2. Regla RFM clásica: frecuencia >= 2 e inactividad entre 60 y 240 días.
      3. Top 20 % de gasto histórico (percentil 80) con > 60 días de inactividad.
      4. BG/NBD + Gamma-Gamma: clientes de la zona de riesgo (= segmento «En riesgo de alto valor»)
         cuyo beneficio esperado de contactar supera el costo (usa el supuesto de efectividad).
    """
    fr = cartera["frecuencia"].to_numpy()
    inact = cartera["inactividad_dias"].to_numpy()
    gasto = cartera["gasto_total_cal_acotado"].to_numpy()
    umbral_gasto_top = float(np.percentile(gasto, 80))
    return {
        "1. Contactar a todos con > 60 días sin comprar": inact > 60,
        "2. Regla RFM (frecuencia >= 2 e inactividad 60–240 d)": (fr >= 2) & (inact >= 60) & (inact <= 240),
        "3. Top 20 % gasto histórico con > 60 d inactividad": (gasto >= umbral_gasto_top) & (inact > 60),
        "4. BG/NBD + Gamma-Gamma (zona de riesgo y beneficio esperado > costo)": cartera["conviene_contactar_rescate"].to_numpy(dtype=bool),
    }


def margen_historico_semestral(cartera: pd.DataFrame, margen_bruto: float = config.MARGEN_BRUTO) -> np.ndarray:
    """Margen que el cliente generaba por semestre ANTES del corte (gasto observado, no predicho por el modelo)."""
    dias = np.maximum(cartera["edad_dias"].to_numpy(dtype=float), config.HORIZONTE_DIAS)
    return cartera["gasto_total_cal_acotado"].to_numpy(dtype=float) / dias * config.HORIZONTE_DIAS * margen_bruto


def validar_politicas_holdout(
    cartera: pd.DataFrame,
    costo_contacto: float = config.COSTO_CONTACTO_RETENCION,
    margen_bruto: float = config.MARGEN_BRUTO,
    n_bootstrap: int = config.N_BOOTSTRAP,
    semilla: int = config.SEMILLA,
) -> pd.DataFrame:
    """
    Valida cada política con lo que REALMENTE pasó en los 6 meses de prueba (no hubo campaña en los datos):
      - tasa de recompra real de los contactados (si es alta, la campaña subsidia a quien iba a volver igual);
      - margen real generado por los contactados y uplift de equilibrio = costo / margen real;
      - margen histórico semestral de quienes NO volvieron (lo que había en juego) y
        tasa de rescate de equilibrio = costo / ese margen, con IC 95 % bootstrap sobre los contactados.
    Sin un experimento A/B no se puede medir el efecto causal de la campaña: estas cifras dicen cuánto
    tendría que funcionar cada política para pagarse, no cuánto funciona.
    """
    gasto_real = cartera["gasto_holdout_acotado"].to_numpy(dtype=float) * margen_bruto
    volvio = cartera["vivo_holdout"].to_numpy(dtype=int) == 1
    margen_perdido = np.where(volvio, 0.0, margen_historico_semestral(cartera, margen_bruto))
    rng = np.random.default_rng(semilla)

    filas = []
    for nombre, mascara in definir_politicas(cartera).items():
        idx = np.flatnonzero(mascara)
        n = len(idx)
        costo = n * costo_contacto
        margen_real = float(gasto_real[idx].sum())
        perdido = float(margen_perdido[idx].sum())
        if n > 0 and n_bootstrap > 0:
            muestras = rng.integers(0, n, size=(n_bootstrap, n))
            perdido_boot = margen_perdido[idx][muestras].sum(axis=1)
            tasa_boot = costo / np.maximum(perdido_boot, 1e-9) * 100.0
            ic_inf, ic_sup = np.percentile(tasa_boot, [2.5, 97.5])
        else:
            ic_inf = ic_sup = np.nan
        filas.append({
            "politica": nombre,
            "clientes_contactados": n,
            "pct_cartera": n / max(len(cartera), 1) * 100.0,
            "costo_campana_gbp": costo,
            "tasa_recompra_real_pct": float(volvio[idx].mean() * 100.0) if n else np.nan,
            "margen_real_holdout_gbp": margen_real,
            "margen_real_por_contacto_gbp": margen_real / n if n else np.nan,
            "uplift_equilibrio_pct": costo / margen_real * 100.0 if margen_real > 0 else np.nan,
            "clientes_sin_recompra": int((~volvio[idx]).sum()),
            "margen_historico_perdido_gbp": perdido,
            "tasa_rescate_equilibrio_pct": costo / perdido * 100.0 if perdido > 0 else np.nan,
            "tasa_rescate_equilibrio_ic95_inf": float(ic_inf),
            "tasa_rescate_equilibrio_ic95_sup": float(ic_sup),
        })
    return pd.DataFrame(filas)


def simular_escenario_politicas(
    cartera: pd.DataFrame,
    efectividad_rescate: float = config.EFECTIVIDAD_RESCATE,
    efecto_fuera_zona: float = config.EFECTO_FUERA_ZONA,
    costo_contacto: float = config.COSTO_CONTACTO_RETENCION,
    margen_bruto: float = config.MARGEN_BRUTO,
) -> pd.DataFrame:
    """
    ESCENARIO (no validado): margen recuperado = (1 − P(vivo)) × margen esperado si vivo × efectividad.
    Todo sale del propio modelo y de dos supuestos (efectividad dentro de 0,20 <= P(vivo) < 0,70 y efecto
    residual fuera de esa franja), por eso favorece por construcción a la política 4. Sirve para ver qué
    supuestos harían rentable cada política, no para afirmar un ROI.
    """
    pv = cartera["prob_vivo"].to_numpy(dtype=float)
    lo, hi = config.RANGO_RIESGO_VIVO
    margen_vivo = margen_si_sigue_vivo(cartera["compras_esperadas"], pv, cartera["ticket_esperado"], margen_bruto)
    sensibilidad = np.where((pv >= lo) & (pv < hi), efectividad_rescate, efecto_fuera_zona)
    margen_rescatable = (1.0 - pv) * margen_vivo * sensibilidad

    filas = []
    for nombre, mascara in definir_politicas(cartera).items():
        n = int(np.sum(mascara))
        costo = n * costo_contacto
        recuperado = float(np.sum(margen_rescatable[mascara]))
        ganancia = recuperado - costo
        filas.append({
            "politica": nombre,
            "efectividad_rescate": efectividad_rescate,
            "efecto_fuera_zona": efecto_fuera_zona,
            "clientes_contactados": n,
            "costo_campana_gbp": costo,
            "margen_recuperado_gbp": recuperado,
            "ganancia_neta_gbp": ganancia,
            "roi_pct": ganancia / costo * 100.0 if costo > 0 else 0.0,
        })
    return pd.DataFrame(filas)


def sensibilidad_escenario(
    cartera: pd.DataFrame,
    efectividades: tuple[float, ...] = config.GRILLA_EFECTIVIDAD,
    efectos_fuera: tuple[float, ...] = config.GRILLA_EFECTO_FUERA,
) -> pd.DataFrame:
    """Repite el escenario para cada combinación de supuestos (efectividad × efecto fuera de zona)."""
    return pd.concat(
        [simular_escenario_politicas(cartera, e, f) for e in efectividades for f in efectos_fuera],
        ignore_index=True,
    )
