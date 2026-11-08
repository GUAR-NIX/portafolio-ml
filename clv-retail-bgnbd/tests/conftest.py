"""Fixtures compartidos para las pruebas automáticas del paquete clv."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def transacciones_sinteticas() -> pd.DataFrame:
    """DataFrame sintético con casos válidos y todos los motivos de exclusión contable."""
    filas = [
        # Cliente C1: 2 facturas el mismo día (deben consolidarse en 1 compra diaria) + 1 compra posterior + 1 en holdout
        ("2009-2010", 2, "500001", "85123A", "PORTAVELAS BLANCO", 6, "2010-03-10 10:00:00", 10.0, "C1", "United Kingdom"),
        ("2009-2010", 3, "500002", "22423", "SOPORTE PASTEL", 4, "2010-03-10 14:30:00", 15.0, "C1", "United Kingdom"),
        ("2010-2011", 4, "510001", "85123A", "PORTAVELAS BLANCO", 10, "2011-02-15 11:00:00", 12.0, "C1", "United Kingdom"),
        ("2010-2011", 5, "520001", "85123A", "PORTAVELAS BLANCO", 8, "2011-09-20 09:30:00", 12.5, "C1", "United Kingdom"),
        # Cliente C2: 1 sola compra en calibración (frecuencia x = 0, recencia = 0)
        ("2010-2011", 6, "510010", "20725", "BOLSA ALMUERZO", 10, "2011-01-05 16:00:00", 8.0, "C2", "United Kingdom"),
        # Exclusión 1: sin customer_id
        ("2010-2011", 7, "510011", "20725", "BOLSA ALMUERZO", 5, "2011-01-06 12:00:00", 8.0, "", "United Kingdom"),
        # Exclusión 2: factura cancelada (C*)
        ("2010-2011", 8, "C510012", "20725", "BOLSA ALMUERZO", -5, "2011-01-07 12:00:00", 8.0, "C1", "United Kingdom"),
        # Exclusión 3: código contable POST
        ("2010-2011", 9, "510013", "POST", "POSTAGE", 1, "2011-01-08 12:00:00", 18.0, "C1", "United Kingdom"),
        # Exclusión 4: precio 0
        ("2010-2011", 10, "510014", "20725", "AJUSTE", 3, "2011-01-09 12:00:00", 0.0, "C1", "United Kingdom"),
        # Exclusión 5: duplicado exacto de la primera línea
        ("2009-2010", 11, "500001", "85123A", "PORTAVELAS BLANCO", 6, "2010-03-10 10:00:00", 10.0, "C1", "United Kingdom"),
    ]
    df = pd.DataFrame(
        filas,
        columns=[
            "hoja", "fila_origen", "invoice", "stock_code", "description",
            "quantity", "invoice_date", "price", "customer_id", "country",
        ],
    )
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df["importe"] = df["quantity"] * df["price"]
    return df


@pytest.fixture
def cartera_sintetica_btyd() -> pd.DataFrame:
    """Genera 300 clientes sintéticos siguiendo el proceso generativo BG/NBD + Gamma-Gamma."""
    rng = np.random.default_rng(42)
    n = 300
    T = rng.uniform(15.0, 78.0, size=n)
    frecuencia = rng.poisson(lam=2.2, size=n)
    frac_rec = np.where(frecuencia > 0, rng.beta(2.5, 1.5, size=n), 0.0)
    t_x = frac_rec * T
    ticket_medio = rng.gamma(shape=4.0, scale=80.0, size=n)
    compras_hold = np.where(T - t_x < 20.0, rng.poisson(lam=1.5, size=n), rng.poisson(lam=0.2, size=n))
    gasto_hold = compras_hold * ticket_medio * rng.uniform(0.8, 1.2, size=n)

    return pd.DataFrame({
        "customer_id": [f"S{i:04d}" for i in range(n)],
        "n_compras": frecuencia + 1,
        "frecuencia": frecuencia,
        "recencia_semanas": t_x,
        "edad_semanas": T,
        "recencia_dias": t_x * 7.0,
        "edad_dias": T * 7.0,
        "inactividad_dias": (T - t_x) * 7.0,
        "ticket_medio": ticket_medio,
        "ticket_medio_acotado": np.minimum(ticket_medio, 3000.0),
        "gasto_total_cal": (frecuencia + 1) * ticket_medio,
        "gasto_total_cal_acotado": (frecuencia + 1) * np.minimum(ticket_medio, 3000.0),
        "desv_ticket": ticket_medio * 0.15,
        "unidades_medias": rng.uniform(20, 150, size=n),
        "items_medios": rng.uniform(5, 25, size=n),
        "compras_ultimos_90d": np.where(T - t_x < 12.0, 1, 0),
        "gasto_ultimos_90d": np.where(T - t_x < 12.0, ticket_medio, 0.0),
        "es_reino_unido": rng.integers(0, 2, size=n),
        "es_mayorista_extremo": (ticket_medio > 3000.0).astype(int),
        "compras_holdout": compras_hold,
        "gasto_holdout": gasto_hold,
        "gasto_holdout_acotado": gasto_hold,
        "ticket_holdout_acotado": np.where(compras_hold > 0, ticket_medio, 0.0),
        "vivo_holdout": (compras_hold > 0).astype(int),
    })
