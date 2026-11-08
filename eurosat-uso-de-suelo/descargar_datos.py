"""Punto de entrada para descargar el dataset oficial EuroSAT RGB (Sentinel-2, 27.000 imágenes)."""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ / "src"))

from eurosat_suelo.datos import preparar_datos_eurosat  # noqa: E402


def main() -> None:
    resumen = preparar_datos_eurosat(forzar=False)
    print(
        f"[ok] EuroSAT preparado: {resumen['total_imagenes_oficial']:,} imágenes auditadas en 10 clases | "
        f"Imágenes usadas para modelar: {resumen['total_subconjunto_modelado']:,} "
        f"(train={resumen['n_train']:,}, val={resumen['n_val']:,}, test={resumen['n_test']:,})."
    )


if __name__ == "__main__":
    main()
