"""
Descarga el dataset UCI Bank Marketing (Moro, Cortez y Rita, 2014) desde el repositorio oficial de UCI
y deja `bank-additional-full.csv` (41.188 filas) en data/raw/.

    python descargar_datos.py

Licencia de los datos: CC BY 4.0 (https://archive.ics.uci.edu/dataset/222/bank+marketing).
"""
from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
DESTINO = Path(__file__).resolve().parent / "data" / "raw"
ARCHIVO = "bank-additional-full.csv"
FILAS_ESPERADAS = 41_188


def descargar(url: str = URL, destino: Path = DESTINO, forzar: bool = False) -> Path:
    destino.mkdir(parents=True, exist_ok=True)
    salida = destino / ARCHIVO
    if salida.exists() and not forzar:
        print(f"Ya existe {salida} (usa forzar=True para volver a bajarlo)")
        return salida

    print(f"Descargando {url} …")
    with urllib.request.urlopen(url, timeout=120) as r:
        contenido = r.read()
    print(f"  {len(contenido) / 1e6:.1f} MB")

    # El zip de UCI trae dos zips anidados: bank.zip y bank-additional.zip
    with zipfile.ZipFile(io.BytesIO(contenido)) as externo:
        interno = next(n for n in externo.namelist() if n.endswith("bank-additional.zip"))
        with zipfile.ZipFile(io.BytesIO(externo.read(interno))) as z:
            miembro = next(n for n in z.namelist() if n.endswith(ARCHIVO) and "__MACOSX" not in n)
            datos = z.read(miembro)

    filas = datos.count(b"\n") - 1
    print(f"  {salida.name}: {filas:,} filas · sha256 {hashlib.sha256(datos).hexdigest()[:16]}…")
    if filas != FILAS_ESPERADAS:  # se valida antes de guardar: un CSV incorrecto no se da por bueno en la siguiente corrida
        raise RuntimeError(f"se esperaban {FILAS_ESPERADAS:,} filas y llegaron {filas:,}")
    salida.write_bytes(datos)
    return salida


if __name__ == "__main__":
    descargar()
