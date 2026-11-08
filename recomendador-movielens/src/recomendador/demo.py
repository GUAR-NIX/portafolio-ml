"""Paquete de demostración: los tres archivos de models/ que contienen datos de MovieLens.

La licencia de MovieLens no permite redistribuir el dataset, así que estos archivos NO se versionan:

- ``peliculas.parquet``    títulos, géneros y conteos de positivos por película (viene de movies.csv y ratings.csv)
- ``demo_positivos.npz``   qué películas le gustaron (≥ 4★) a cada usuario de demostración
- ``demo_vistos.npz``      qué películas calificó cada usuario de demostración

Se generan en la instalación, a partir de los datos descargados de GroupLens y del resto del paquete (que sí se
versiona: ``factores.npz``, ``demo_usuarios.npz`` y ``metadata.json``):

    python descargar_datos.py          # 262 MB desde GroupLens
    python generar_demo.py             # ~2 min
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import config as C
from . import datos as D

ARCHIVOS_DEMO = ("peliculas.parquet", "demo_positivos.npz", "demo_vistos.npz")
ARCHIVOS_VERSIONADOS = ("metadata.json", "factores.npz", "demo_usuarios.npz")
FIN_MODELO_FINAL = pd.Timestamp("2019-12-01")  # el modelo final usa todos los datos


class PaqueteIncompleto(FileNotFoundError):
    """Faltan archivos del paquete de models/ (por la licencia de MovieLens no se suben al repositorio)."""


def faltantes(carpeta: Path = C.MODELOS) -> list[str]:
    return [a for a in (*ARCHIVOS_VERSIONADOS, *ARCHIVOS_DEMO) if not (carpeta / a).exists()]


def mensaje_faltantes(carpeta: Path = C.MODELOS) -> str:
    falta = faltantes(carpeta)
    if not falta:
        return ""
    receta = []
    if any(a in ARCHIVOS_DEMO for a in falta):
        receta.append("Por la licencia de MovieLens no se redistribuyen los de datos: genéralos con "
                      "`python descargar_datos.py` y luego `python generar_demo.py`.")
    if any(a in ARCHIVOS_VERSIONADOS for a in falta):
        receta.append("Los demás vienen con el repositorio (o se crean con `python -m recomendador.entrenar`).")
    return f"Faltan en {carpeta.name}/: {', '.join(falta)}. {' '.join(receta)}"


def tabla_peliculas(peliculas: pd.DataFrame, part: D.Particion, tiene_genome: np.ndarray) -> pd.DataFrame:
    """Catálogo recomendable en el orden de ``factores.npz``: título, año, géneros y conteos."""
    conteo = np.asarray(part.positivos.sum(axis=0)).ravel().astype(np.int32)
    pel = peliculas.set_index("movieId").reindex(part.items).reset_index()
    return pel[["movieId", "titulo", "anio", "generos"]].assign(
        generos=lambda d: d["generos"].map(lambda g: "|".join(g)), positivos=conteo,
        tiene_genome=tiene_genome, recientes=part.recientes.astype(np.int32))


def guardar_matrices_demo(part: D.Particion, filas: np.ndarray, carpeta: Path) -> None:
    sp.save_npz(carpeta / "demo_positivos.npz", part.positivos[filas].astype(np.float32))
    sp.save_npz(carpeta / "demo_vistos.npz", part.vistos[filas].astype(np.float32))


def generar(carpeta: Path = C.MODELOS) -> None:
    """Reconstruye los tres archivos con datos de MovieLens usando los usuarios de ``demo_usuarios.npz``."""
    previos = [a for a in ARCHIVOS_VERSIONADOS if not (carpeta / a).exists()]
    if previos:
        raise PaqueteIncompleto(f"Faltan archivos que sí se versionan en {carpeta.name}/: {', '.join(previos)}")
    if not (C.DATOS_CRUDOS / "ratings.csv").exists():
        raise PaqueteIncompleto("Faltan los datos de MovieLens: ejecuta primero `python descargar_datos.py`.")

    D.convertir_a_parquet()
    cal = D.cargar_calificaciones()
    peliculas = D.cargar_peliculas()
    genome, _ = D.cargar_genome()
    part = D.particionar(cal, FIN_MODELO_FINAL, FIN_MODELO_FINAL + pd.Timedelta(days=1))
    _, gnm = D.caracteristicas_items(part.items, peliculas, genome)
    tiene_genome = ~np.isnan(gnm).any(axis=1)

    usuarios = np.load(carpeta / "demo_usuarios.npz")["usuarios"]
    fila_de = {int(u): i for i, u in enumerate(part.usuarios)}
    sin_fila = [int(u) for u in usuarios if int(u) not in fila_de]
    if sin_fila:
        raise ValueError(f"{len(sin_fila)} usuarios de demo_usuarios.npz no están en los datos descargados "
                         "(¿otra versión de MovieLens?)")
    filas = np.array([fila_de[int(u)] for u in usuarios])

    tabla_peliculas(peliculas, part, tiene_genome).to_parquet(carpeta / "peliculas.parquet", index=False)
    guardar_matrices_demo(part, filas, carpeta)
    print(f"Listo: {', '.join(ARCHIVOS_DEMO)} en {carpeta}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    try:
        generar()
    except PaqueteIncompleto as e:
        raise SystemExit(str(e)) from None


if __name__ == "__main__":
    main()
