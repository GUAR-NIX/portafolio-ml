"""Pruebas sobre la descarga, auditoría radiométrica de las 27.000 imágenes y descriptores espectrales."""

from __future__ import annotations

import numpy as np
import pandas as pd

from eurosat_suelo.config import CLASES_EUROSAT, RAIZ_PROYECTO
from eurosat_suelo.datos import (
    calcular_descriptores_espectrales_lote,
    cargar_subconjunto_eurosat,
    extraer_vector_handcrafted_imagen,
)


def test_auditoria_27000_imagenes_oficiales(resumen_clases_27k: pd.DataFrame, resultados_json: dict) -> None:
    assert len(resumen_clases_27k) == 10
    assert int(resumen_clases_27k["imagenes_totales"].sum()) == 27_000
    assert set(resumen_clases_27k["clase"]) == set(CLASES_EUROSAT)
    aud = resultados_json["auditoria_datos"]
    assert aud["total_imagenes_oficial"] == 27_000
    # Se modela con las particiones oficiales completas
    assert aud["total_subconjunto_modelado"] == 27_000
    assert (aud["n_train"], aud["n_val"], aud["n_test"]) == (16_200, 5_400, 5_400)
    assert resultados_json["particiones"] == {"entrenamiento": 16_200, "validacion": 5_400, "prueba": 5_400}


def test_particiones_oficiales_completas_y_disjuntas() -> None:
    datos = cargar_subconjunto_eurosat()
    imgs, lbls, part, fnames = datos["imagenes"], datos["etiquetas"], datos["particion"], datos["filenames"]

    assert imgs.shape == (27_000, 64, 64, 3)
    assert imgs.dtype == np.uint8
    for split, esperado in [("train", 16_200), ("val", 5_400), ("test", 5_400)]:
        assert int(np.sum(part == split)) == esperado
        # Todas las clases presentes en cada partición
        assert np.all(np.bincount(lbls[part == split], minlength=10) > 0)
    # Ningún archivo aparece en dos particiones
    conjuntos = [set(fnames[part == s]) for s in ("train", "val", "test")]
    assert not (conjuntos[0] & conjuntos[1]) and not (conjuntos[0] & conjuntos[2]) and not (conjuntos[1] & conjuntos[2])


def test_descriptores_espectrales_coherencia_fisica(resumen_clases_27k: pd.DataFrame) -> None:
    df = resumen_clases_27k.set_index("clase")
    # Bosque y pastizal deben tener mayor índice de vegetación ExG que zonas industriales o autopistas
    assert float(df.loc["Forest", "exg_medio"]) > float(df.loc["Industrial", "exg_medio"])
    assert float(df.loc["Pasture", "exg_medio"]) > float(df.loc["Highway", "exg_medio"])
    # Mar/Lago debe tener el mayor índice normalizado Azul-Rojo y la menor rugosidad espacial
    assert float(df.loc["SeaLake", "indice_agua_medio"]) > float(df.loc["AnnualCrop", "indice_agua_medio"])
    assert float(df.loc["SeaLake", "rugosidad_media"]) < float(df.loc["Residential", "rugosidad_media"])


def test_extraccion_handcrafted_54_atributos() -> None:
    lote_ficticio = np.random.default_rng(7).integers(0, 256, size=(8, 64, 64, 3), dtype=np.uint8)
    vec = extraer_vector_handcrafted_imagen(lote_ficticio)
    assert vec.shape == (8, 54)
    assert np.all(np.isfinite(vec))

    desc = calcular_descriptores_espectrales_lote(lote_ficticio)
    assert desc["exg"].shape == (8,)
    assert desc["rugosidad"].shape == (8,)


def test_muestra_versionada_50_pngs(muestra_catalogo: pd.DataFrame, dir_muestra_png) -> None:
    assert len(muestra_catalogo) == 50
    for archivo in muestra_catalogo["archivo_muestra"]:
        assert (dir_muestra_png / archivo).exists()


def test_ningun_archivo_versionado_supera_50mb() -> None:
    limite = 50 * 1024 * 1024
    for ruta in RAIZ_PROYECTO.rglob("*"):
        if not ruta.is_file():
            continue
        partes = set(ruta.parts)
        if ".venv" in partes or "raw" in partes or "cache" in partes:
            continue
        assert ruta.stat().st_size < limite, f"Archivo {ruta} excede 50 MB"
