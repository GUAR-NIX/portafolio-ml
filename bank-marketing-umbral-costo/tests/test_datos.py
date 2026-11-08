"""Preparación de datos y control de fuga de información."""
import numpy as np
import pandas as pd
import pytest

from campana import config, datos


def test_reconstruir_anio_por_orden_de_meses():
    meses = pd.Series(["may", "may", "dec", "mar", "nov", "mar"])
    assert datos.reconstruir_anio(meses).tolist() == [2008, 2008, 2008, 2009, 2009, 2010]


def test_preparar_separa_pdays_999_y_objetivo():
    crudo = pd.DataFrame({k: ["x", "x"] for k in config.RENOMBRAR})
    crudo["month"] = ["may", "may"]
    crudo["pdays"] = [999, 6]
    crudo["y"] = ["yes", "no"]
    d = datos.preparar(crudo)
    assert d["contactado_antes"].tolist() == [0, 1]
    assert np.isnan(d.loc[0, "dias_desde_contacto"]) and d.loc[1, "dias_desde_contacto"] == 6
    assert d[config.OBJETIVO].tolist() == [1, 0]


def test_variables_con_fuga_no_entran_al_modelo():
    assert "duracion_llamada" in config.FUGA
    assert not set(config.FUGA) & set(config.VARIABLES)
    assert set(config.VARIABLES) == set(config.CATEGORICAS) | set(config.NUMERICAS)


@pytest.mark.skipif(not config.DATOS_CRUDOS.exists(), reason="falta data/raw (python descargar_datos.py)")
def test_datos_reales_y_particiones():
    df = datos.cargar()
    assert len(df) == 41_176                                   # 41.188 − 12 duplicados exactos
    assert df["anio"].min() == 2008 and df["anio"].max() == 2010
    ent, pru = datos.particion_estratificada(df)
    assert abs(ent[config.OBJETIVO].mean() - pru[config.OBJETIVO].mean()) < 0.002
    assert not set(ent.index) & set(pru.index)
    pasado, futuro = datos.particion_temporal(df)
    assert pasado.index.max() < futuro.index.min()              # el pasado va antes, sin mezclar
