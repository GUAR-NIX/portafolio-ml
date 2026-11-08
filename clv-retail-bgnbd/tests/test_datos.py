"""Pruebas de limpieza contable, consolidación diaria y construcción de variables BTYD."""
from __future__ import annotations

import pytest

from clv import config, datos


def test_limpieza_excluye_todos_los_casos_invalidos(transacciones_sinteticas):
    limpio, auditoria = datos.limpiar_transacciones(transacciones_sinteticas)
    # De las 10 filas, 5 son inválidas (sin cliente, C*, POST, precio 0, duplicada) -> quedan 5
    assert len(limpio) == 5
    assert int(auditoria.iloc[0]["filas"]) == 10
    assert int(auditoria.iloc[-1]["filas"]) == 5
    assert not limpio["invoice"].str.startswith("C").any()
    assert not (limpio["customer_id"] == "").any()
    assert not limpio["stock_code"].isin(config.CODIGOS_NO_PRODUCTO).any()


def test_agregacion_diaria_consolida_mismo_dia(transacciones_sinteticas):
    limpio, _ = datos.limpiar_transacciones(transacciones_sinteticas)
    diario = datos.agregar_compras_diarias(limpio)
    c1_marzo = diario[(diario["customer_id"] == "C1") & (diario["fecha_dia"] == "2010-03-10")]
    assert len(c1_marzo) == 1
    # 6*10 + 4*15 = 120 GBP en 2 facturas el mismo día
    assert float(c1_marzo.iloc[0]["importe_dia"]) == pytest.approx(120.0)
    assert int(c1_marzo.iloc[0]["n_facturas_dia"]) == 2


def test_matriz_btyd_calcula_frecuencia_recencia_y_holdout(transacciones_sinteticas):
    limpio, _ = datos.limpiar_transacciones(transacciones_sinteticas)
    diario = datos.agregar_compras_diarias(limpio)
    btyd = datos.construir_matriz_btyd(diario, fecha_corte="2011-06-09", fecha_fin_holdout="2011-12-09")

    c1 = btyd[btyd["customer_id"] == "C1"].iloc[0]
    # C1 compró el 2010-03-10 y el 2011-02-15 en calibración -> n_compras=2, frecuencia=1
    assert int(c1["frecuencia"]) == 1
    assert float(c1["recencia_dias"]) == (
        (datos.pd.Timestamp("2011-02-15") - datos.pd.Timestamp("2010-03-10")).days
    )
    assert float(c1["edad_dias"]) >= float(c1["recencia_dias"])
    # En holdout compró el 2011-09-20 (8 * 12.5 = 100 GBP)
    assert int(c1["compras_holdout"]) == 1
    assert float(c1["gasto_holdout"]) == pytest.approx(100.0)
    assert int(c1["vivo_holdout"]) == 1

    c2 = btyd[btyd["customer_id"] == "C2"].iloc[0]
    # C2 compró una sola vez -> frecuencia=0, recencia_dias=0, vivo_holdout=0
    assert int(c2["frecuencia"]) == 0
    assert float(c2["recencia_dias"]) == 0.0
    assert int(c2["compras_holdout"]) == 0
    assert int(c2["vivo_holdout"]) == 0


@pytest.mark.skipif(not config.DATOS_CRUDOS.exists(), reason="CSV crudo no descargado")
def test_integridad_datos_reales():
    df = datos.cargar_transacciones_crudas()
    limpio, _ = datos.limpiar_transacciones(df)
    diario = datos.agregar_compras_diarias(limpio)
    btyd = datos.construir_matriz_btyd(diario)
    assert len(btyd) > 4000
    assert (btyd["edad_semanas"] >= btyd["recencia_semanas"]).all()
    assert (btyd.loc[btyd["frecuencia"] == 0, "recencia_semanas"] == 0).all()
