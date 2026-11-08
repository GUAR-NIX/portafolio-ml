"""Motor de inferencia para valorar clientes individuales o priorizar carteras desde la API."""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from . import config
from .modelos import ModeloBGNBD, ModeloGammaGamma, calcular_clv
from .segmentacion import ACCIONES_SEGMENTO, asignar_segmento_cliente, en_zona_riesgo, margen_si_sigue_vivo


@lru_cache(maxsize=1)
def cargar_artefacto(ruta: Path = config.MODELOS / "modelo_clv.joblib") -> dict[str, Any]:
    if not Path(ruta).exists():
        raise FileNotFoundError(
            f"No existe el artefacto {Path(ruta).name}. Ejecuta primero: python -m clv.entrenar"
        )
    return joblib.load(ruta)


def cargar_metadata(ruta: Path = config.MODELOS / "metadata.json") -> dict[str, Any]:
    if not Path(ruta).exists():
        art = cargar_artefacto()
        return art.get("metadata", {})
    return json.loads(Path(ruta).read_text(encoding="utf-8"))


def valorar_cliente(
    frecuencia: int,
    recencia_dias: float,
    edad_dias: float,
    ticket_medio: float,
    horizonte_meses: int = config.HORIZONTE_MESES,
    margen_bruto: float = config.MARGEN_BRUTO,
    tasa_descuento_mensual: float = config.TASA_DESCUENTO_MENSUAL,
    costo_contacto: float = config.COSTO_CONTACTO_RETENCION,
    efectividad_rescate: float = config.EFECTIVIDAD_RESCATE,
) -> dict[str, Any]:
    """
    Calcula para un cliente individual su probabilidad de seguir vivo, probabilidad de recomprar
    en el horizonte, compras esperadas, ticket esperado (con shrinkage), CLV neto descontado,
    segmento estratégico y si conviene incluirlo en campaña de rescate.
    """
    if recencia_dias > edad_dias:
        raise ValueError("recencia_dias no puede ser mayor que edad_dias (T >= t_x).")
    if frecuencia == 0 and recencia_dias > 0:
        raise ValueError("Si frecuencia == 0 (sin compras repetidas), recencia_dias debe ser 0.")

    art = cargar_artefacto()
    bgnbd: ModeloBGNBD = art["bgnbd"]
    gg: ModeloGammaGamma = art["gg"]
    umbral_alto: float = float(art.get("umbral_alto_valor", 120.0))

    x_arr = np.array([frecuencia], dtype=float)
    tx_arr = np.array([recencia_dias / config.DIAS_POR_SEMANA], dtype=float)
    T_arr = np.array([edad_dias / config.DIAS_POR_SEMANA], dtype=float)
    m_acotado = min(float(ticket_medio), config.TOPE_TICKET_MAYORISTA)
    m_arr = np.array([m_acotado], dtype=float)

    res = calcular_clv(
        bgnbd=bgnbd,
        gg=gg,
        x=x_arr,
        t_x=tx_arr,
        T=T_arr,
        m_x=m_arr,
        horizonte_meses=horizonte_meses,
        margen_bruto=margen_bruto,
        tasa_descuento_mensual=tasa_descuento_mensual,
    )

    prob_vivo = float(res["prob_vivo"][0])
    prob_recompra = float(res["prob_recompra"][0])
    compras_esp = float(res["compras_esperadas"][0])
    ticket_esp = float(res["ticket_esperado"][0])
    ingreso_esp = float(res["ingreso_esperado"][0])
    clv_neto = float(res["clv_neto"][0])

    seg = str(
        asignar_segmento_cliente(
            np.array([prob_vivo]),
            np.array([frecuencia]),
            np.array([clv_neto]),
            umbral_alto_valor=umbral_alto,
        )[0]
    )
    margen_vivo = float(margen_si_sigue_vivo(compras_esp, prob_vivo, ticket_esp, margen_bruto))
    zona = bool(en_zona_riesgo(prob_vivo, frecuencia))
    margen_en_riesgo = ((1.0 - prob_vivo) * margen_vivo) if zona else 0.0
    beneficio_neto_rescate = margen_en_riesgo * efectividad_rescate - costo_contacto
    contactar_rescate = bool(zona and (beneficio_neto_rescate > 0.0))

    return {
        "prob_vivo": round(prob_vivo, 4),
        "prob_recompra": round(prob_recompra, 4),
        "compras_esperadas": round(compras_esp, 3),
        "ticket_esperado_gbp": round(ticket_esp, 2),
        "ingreso_esperado_gbp": round(ingreso_esp, 2),
        "clv_neto_gbp": round(clv_neto, 2),
        "inactividad_dias": round(float(edad_dias - recencia_dias), 1),
        "segmento": seg,
        "accion_recomendada": ACCIONES_SEGMENTO[seg],
        "conviene_contactar_rescate": contactar_rescate,
        "beneficio_neto_rescate_gbp": round(float(beneficio_neto_rescate), 2),
    }


def priorizar_cartera(
    clientes: list[dict[str, Any]],
    horizonte_meses: int = config.HORIZONTE_MESES,
    margen_bruto: float = config.MARGEN_BRUTO,
    tasa_descuento_mensual: float = config.TASA_DESCUENTO_MENSUAL,
    costo_contacto: float = config.COSTO_CONTACTO_RETENCION,
    modo: str = "clv",
    capacidad: int | None = None,
) -> dict[str, Any]:
    """
    Valora una lista de clientes y los ordena por:
      - modo='clv'     : mayor CLV neto descontado
      - modo='rescate' : mayor beneficio neto esperado en campaña de reactivación
    """
    evaluados = []
    for idx, c in enumerate(clientes):
        val = valorar_cliente(
            frecuencia=int(c["frecuencia"]),
            recencia_dias=float(c["recencia_dias"]),
            edad_dias=float(c["edad_dias"]),
            ticket_medio=float(c["ticket_medio"]),
            horizonte_meses=horizonte_meses,
            margen_bruto=margen_bruto,
            tasa_descuento_mensual=tasa_descuento_mensual,
            costo_contacto=costo_contacto,
        )
        val["id_cliente"] = str(c.get("id_cliente", f"cli_{idx + 1}"))
        evaluados.append(val)

    clave_orden = "beneficio_neto_rescate_gbp" if modo == "rescate" else "clv_neto_gbp"
    evaluados.sort(key=lambda r: r[clave_orden], reverse=True)

    for pos, item in enumerate(evaluados, start=1):
        item["ranking"] = pos
        if capacidad is not None and pos > capacidad:
            item["seleccionado"] = False
        elif modo == "rescate":
            item["seleccionado"] = bool(item["conviene_contactar_rescate"])
        else:
            item["seleccionado"] = True

    seleccionados = [e for e in evaluados if e["seleccionado"]]
    return {
        "modo": modo,
        "horizonte_meses": horizonte_meses,
        "total_evaluados": len(evaluados),
        "total_seleccionados": len(seleccionados),
        "clv_neto_total_gbp": round(sum(e["clv_neto_gbp"] for e in evaluados), 2),
        "beneficio_rescate_seleccionados_gbp": round(
            sum(e["beneficio_neto_rescate_gbp"] for e in seleccionados), 2
        ),
        "clientes": evaluados,
    }
