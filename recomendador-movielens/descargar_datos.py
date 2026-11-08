"""
Descarga MovieLens 25M (GroupLens, Universidad de Minnesota) desde el sitio oficial y la descomprime
en data/raw/ml-25m/ (ratings.csv, movies.csv, tags.csv, links.csv, genome-scores.csv, genome-tags.csv).

    python descargar_datos.py

El zip pesa ~250 MB y descomprimido ~1,2 GB. Se verifica el MD5 publicado por GroupLens.
Licencia de uso: ver https://files.grouplens.org/datasets/movielens/ml-25m-README.html
(uso de investigación/educativo, citar a Harper y Konstan 2015, no redistribuir).
"""
from __future__ import annotations

import hashlib
import shutil
import ssl
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
URL_MD5 = "https://files.grouplens.org/datasets/movielens/ml-25m.zip.md5"
RAIZ = Path(__file__).resolve().parent / "data" / "raw"
CARPETA = RAIZ / "ml-25m"
ARCHIVOS = ["ratings.csv", "movies.csv", "tags.csv", "links.csv", "genome-scores.csv", "genome-tags.csv"]


def _abrir(url: str, timeout: int):
    """urlopen con verificación TLS.

    Si el almacén de certificados del sistema no tiene la raíz del servidor (pasa en Windows con
    files.grouplens.org: Windows la instala bajo demanda para el navegador, pero Python no la pide),
    reintenta con los certificados raíz de Mozilla del paquete certifi. Nunca desactiva la verificación.
    """
    try:
        return urllib.request.urlopen(url, timeout=timeout)
    except urllib.error.URLError as e:
        if not isinstance(e.reason, ssl.SSLCertVerificationError):
            raise
        import certifi

        print("  (certificado no verificable con el sistema: se usa el paquete certifi)")
        return urllib.request.urlopen(url, timeout=timeout, context=ssl.create_default_context(cafile=certifi.where()))


def _md5(ruta: Path) -> str:
    h = hashlib.md5()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def descargar(forzar: bool = False) -> Path:
    RAIZ.mkdir(parents=True, exist_ok=True)
    if not forzar and all((CARPETA / a).exists() for a in ARCHIVOS):
        print(f"Ya existe {CARPETA} (usa forzar=True para volver a bajarlo)")
        return CARPETA

    zip_local = RAIZ / "ml-25m.zip"
    # is_zipfile busca el índice que va al final del ZIP: un archivo cortado no lo tiene y se vuelve a bajar.
    if forzar or not zipfile.is_zipfile(zip_local):
        print(f"Descargando {URL} (~250 MB) …")
        parcial = zip_local.with_suffix(zip_local.suffix + ".part")
        with _abrir(URL, timeout=300) as r, open(parcial, "wb") as f:
            shutil.copyfileobj(r, f, length=1 << 20)
        if not zipfile.is_zipfile(parcial):
            raise RuntimeError(f"La descarga de {zip_local.name} no es un ZIP completo; vuelve a ejecutar el script")
        parcial.replace(zip_local)
    print(f"  {zip_local.stat().st_size / 1e6:.0f} MB")

    try:
        with _abrir(URL_MD5, timeout=60) as r:
            esperado = r.read().decode().split()[0].strip().lower()
        obtenido = _md5(zip_local)
        if obtenido != esperado:
            raise RuntimeError(f"MD5 no coincide: {obtenido} vs {esperado}; borra {zip_local} y reintenta")
        print(f"  MD5 verificado: {obtenido}")
    except OSError as e:  # sin conexión al .md5: se avisa pero no se bloquea
        print(f"  (no se pudo verificar el MD5: {e})")

    with zipfile.ZipFile(zip_local) as z:
        for miembro in z.namelist():
            if Path(miembro).name in ARCHIVOS:
                destino = CARPETA / Path(miembro).name
                destino.parent.mkdir(parents=True, exist_ok=True)
                with z.open(miembro) as src, open(destino, "wb") as dst:
                    shutil.copyfileobj(src, dst, length=1 << 20)
                print(f"  {destino.name}: {destino.stat().st_size / 1e6:.0f} MB")
    zip_local.unlink()
    return CARPETA


if __name__ == "__main__":
    descargar()
