"""Carga, auditoría de limpieza, agregación diaria y construcción de matrices RFM/BTYD."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def cargar_transacciones_crudas(ruta: Path = config.DATOS_CRUDOS) -> pd.DataFrame:
    """Lee el CSV crudo de UCI Online Retail II y tipa fechas, cantidades y precios."""
    if not Path(ruta).exists():
        raise FileNotFoundError(
            f"No existe {Path(ruta).name}. Ejecuta primero: python descargar_datos.py"
        )
    df = pd.read_csv(
        ruta,
        dtype={
            "hoja": str,
            "invoice": str,
            "stock_code": str,
            "description": str,
            "customer_id": str,
            "country": str,
        },
        keep_default_na=False,
    )
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0).astype(int)
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0).astype(float)
    df["invoice_date"] = pd.to_datetime(df["invoice_date"], format="%Y-%m-%d %H:%M:%S")
    df["importe"] = df["quantity"] * df["price"]
    return df


def limpiar_transacciones(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aplica las reglas contables y de negocio para modelos BTYD y devuelve:
    (df_limpio, tabla_auditoria_limpieza).
    """
    n_inicial = len(df)
    imp_inicial = float(df["importe"].sum())

    etapas = [("0. Filas crudas totales", n_inicial, imp_inicial)]

    # 1. Sin Customer ID (compras de invitado)
    m_sin_cliente = df["customer_id"].str.strip() == ""
    etapas.append(("1. Excluidas: sin Customer ID (invitado)", int(m_sin_cliente.sum()), float(df.loc[m_sin_cliente, "importe"].sum())))
    actual = df.loc[~m_sin_cliente].copy()

    # 2. Facturas canceladas (empiezan por C)
    m_cancelada = actual["invoice"].str.upper().str.startswith("C")
    etapas.append(("2. Excluidas: facturas de cancelación (C*)", int(m_cancelada.sum()), float(actual.loc[m_cancelada, "importe"].sum())))
    actual = actual.loc[~m_cancelada].copy()

    # 3. Códigos contables, portes y ajustes no producto
    cod_upper = actual["stock_code"].str.upper().str.strip()
    m_no_prod = cod_upper.isin(config.CODIGOS_NO_PRODUCTO) | cod_upper.str.startswith("TEST")
    etapas.append(("3. Excluidas: códigos contables/envío (POST, M, D...)", int(m_no_prod.sum()), float(actual.loc[m_no_prod, "importe"].sum())))
    actual = actual.loc[~m_no_prod].copy()

    # 4. Cantidad o precio no positivo
    m_invalido = (actual["quantity"] <= 0) | (actual["price"] <= 0.001)
    etapas.append(("4. Excluidas: cantidad <= 0 o precio <= 0", int(m_invalido.sum()), float(actual.loc[m_invalido, "importe"].sum())))
    actual = actual.loc[~m_invalido].copy()

    # 5. Duplicados exactos de línea de factura
    cols_dup = ["invoice", "stock_code", "description", "quantity", "invoice_date", "price", "customer_id"]
    m_dup = actual.duplicated(subset=cols_dup, keep="first")
    etapas.append(("5. Excluidas: líneas duplicadas exactas", int(m_dup.sum()), float(actual.loc[m_dup, "importe"].sum())))
    actual = actual.loc[~m_dup].copy()

    actual["fecha_dia"] = actual["invoice_date"].dt.floor("D")
    etapas.append(("6. Líneas limpias retenidas", len(actual), float(actual["importe"].sum())))

    auditoria = pd.DataFrame(etapas, columns=["etapa", "filas", "importe_gbp"])
    auditoria["pct_filas"] = auditoria["filas"] / max(n_inicial, 1) * 100.0
    return actual.reset_index(drop=True), auditoria


def agregar_compras_diarias(df_limpio: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega las transacciones por (customer_id, fecha_dia).
    En BG/NBD la unidad de conteo es el día con compra activa.
    """
    diario = (
        df_limpio.groupby(["customer_id", "fecha_dia"], as_index=False)
        .agg(
            importe_dia=("importe", "sum"),
            unidades_dia=("quantity", "sum"),
            n_facturas_dia=("invoice", "nunique"),
            n_items_dia=("stock_code", "nunique"),
            pais=("country", "first"),
        )
        .sort_values(["customer_id", "fecha_dia"])
        .reset_index(drop=True)
    )
    return diario


def construir_matriz_btyd(
    compras_diarias: pd.DataFrame,
    fecha_corte: str = config.FECHA_CORTE_CAL,
    fecha_fin_holdout: str | None = config.FECHA_FIN,
    tope_ticket: float = config.TOPE_TICKET_MAYORISTA,
) -> pd.DataFrame:
    """
    Construye la tabla de clientes en el corte temporal con las variables suficientes
    de Fader & Hardie (frecuencia x, recencia_semanas t_x, edad_semanas T, ticket_medio m_x),
    variables auxiliares para ML supervisado y los resultados reales en el periodo holdout.
    """
    corte = pd.Timestamp(fecha_corte)
    cal = compras_diarias[compras_diarias["fecha_dia"] <= corte].copy()
    if cal.empty:
        raise ValueError("No hay compras antes de la fecha de corte.")

    cal = cal.sort_values(["customer_id", "fecha_dia"])
    cal["importe_dia_acotado"] = cal["importe_dia"].clip(upper=tope_ticket)
    cal["orden_compra"] = cal.groupby("customer_id").cumcount()

    # Compras repetidas (orden >= 1) para estimar el ticket medio de Gamma-Gamma según Fader & Hardie (2005)
    repetidas = cal[cal["orden_compra"] >= 1].groupby("customer_id").agg(
        ticket_medio_rep=("importe_dia", "mean"),
        ticket_medio_rep_acotado=("importe_dia_acotado", "mean"),
    )

    # Ventana reciente (últimos 90 días antes del corte) para variables del modelo supervisado
    corte_90d = corte - pd.Timedelta(days=90)
    rec90 = cal[cal["fecha_dia"] > corte_90d].groupby("customer_id").agg(
        compras_ultimos_90d=("fecha_dia", "count"),
        gasto_ultimos_90d=("importe_dia_acotado", "sum"),
    )

    base = cal.groupby("customer_id").agg(
        primera_compra=("fecha_dia", "min"),
        ultima_compra=("fecha_dia", "max"),
        n_compras=("fecha_dia", "count"),
        gasto_total_cal=("importe_dia", "sum"),
        gasto_total_cal_acotado=("importe_dia_acotado", "sum"),
        ticket_medio_total=("importe_dia", "mean"),
        ticket_medio_total_acotado=("importe_dia_acotado", "mean"),
        desv_ticket=("importe_dia_acotado", "std"),
        unidades_medias=("unidades_dia", "mean"),
        items_medios=("n_items_dia", "mean"),
        pais=("pais", "first"),
    )

    base["frecuencia"] = (base["n_compras"] - 1).astype(int)
    base["recencia_dias"] = (base["ultima_compra"] - base["primera_compra"]).dt.days.astype(float)
    base["edad_dias"] = (corte - base["primera_compra"]).dt.days.astype(float)
    base["inactividad_dias"] = (corte - base["ultima_compra"]).dt.days.astype(float)

    base["recencia_semanas"] = base["recencia_dias"] / config.DIAS_POR_SEMANA
    base["edad_semanas"] = base["edad_dias"] / config.DIAS_POR_SEMANA

    base = base.join(repetidas, how="left").join(rec90, how="left")
    base["compras_ultimos_90d"] = base["compras_ultimos_90d"].fillna(0).astype(int)
    base["gasto_ultimos_90d"] = base["gasto_ultimos_90d"].fillna(0.0)
    base["desv_ticket"] = base["desv_ticket"].fillna(0.0)
    base["es_reino_unido"] = (base["pais"] == "United Kingdom").astype(int)

    # Para clientes con x >= 1 usamos el promedio de compras repetidas; para x == 0 usamos su compra inicial
    base["ticket_medio"] = np.where(
        base["frecuencia"] >= 1,
        base["ticket_medio_rep"],
        base["ticket_medio_total"],
    )
    base["ticket_medio_acotado"] = np.where(
        base["frecuencia"] >= 1,
        base["ticket_medio_rep_acotado"],
        base["ticket_medio_total_acotado"],
    )
    base["es_mayorista_extremo"] = (base["ticket_medio"] > tope_ticket).astype(int)

    if fecha_fin_holdout is not None:
        fin = pd.Timestamp(fecha_fin_holdout)
        hold = compras_diarias[
            (compras_diarias["fecha_dia"] > corte) & (compras_diarias["fecha_dia"] <= fin)
        ].copy()
        hold["importe_dia_acotado"] = hold["importe_dia"].clip(upper=tope_ticket)
        res_hold = hold.groupby("customer_id").agg(
            compras_holdout=("fecha_dia", "count"),
            gasto_holdout=("importe_dia", "sum"),
            gasto_holdout_acotado=("importe_dia_acotado", "sum"),
            ticket_holdout=("importe_dia", "mean"),
            ticket_holdout_acotado=("importe_dia_acotado", "mean"),
        )
        base = base.join(res_hold, how="left")
        base["compras_holdout"] = base["compras_holdout"].fillna(0).astype(int)
        base["gasto_holdout"] = base["gasto_holdout"].fillna(0.0)
        base["gasto_holdout_acotado"] = base["gasto_holdout_acotado"].fillna(0.0)
        base["ticket_holdout"] = base["ticket_holdout"].fillna(0.0)
        base["ticket_holdout_acotado"] = base["ticket_holdout_acotado"].fillna(0.0)
        base["vivo_holdout"] = (base["compras_holdout"] > 0).astype(int)

    return base.reset_index()
