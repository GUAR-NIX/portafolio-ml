"""
descargar_datos.py
Descarga el dataset oficial UCI Online Retail II (01/12/2009 – 09/12/2011, 2 hojas de Excel)
y genera data/raw/online_retail_ii.csv con todas las transacciones normalizadas en UTF-8.
Si el archivo ya fue descargado previamente en el repositorio o en retail-rfm-cohortes-sql,
lo reutiliza automáticamente para ahorrar tiempo y ancho de banda.

Uso:
    python descargar_datos.py
"""
from __future__ import annotations

import csv
import shutil
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

URL_FUENTE = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
RAIZ = Path(__file__).resolve().parent
CARPETA_RAW = RAIZ / "data" / "raw"
ZIP_DESTINO = CARPETA_RAW / "online_retail_ii.zip"
CSV_DESTINO = CARPETA_RAW / "online_retail_ii.csv"

# Caché local en caso de que el proyecto hermano de SQL ya haya descargado la fuente oficial
RAW_HERMANO = RAIZ.parents[1] / "portafolio-sql" / "retail-rfm-cohortes-sql" / "data" / "raw"
CACHE_HERMANO_CSV = RAW_HERMANO / "online_retail_ii.csv"
CACHE_HERMANO_ZIP = RAW_HERMANO / "online_retail_ii.zip"

COLUMNAS = [
    "hoja",
    "fila_origen",
    "invoice",
    "stock_code",
    "description",
    "quantity",
    "invoice_date",
    "price",
    "customer_id",
    "country",
]


def _texto(valor) -> str:
    if pd.isna(valor):
        return ""
    return str(valor).replace("\r", " ").replace("\n", " ").strip()


def descargar_zip(url: str, destino: Path) -> None:
    # is_zipfile busca el índice que va al final del ZIP: un archivo cortado no lo tiene y se vuelve a bajar.
    if zipfile.is_zipfile(destino):
        print(f"[OMITIDO] {destino.name} ya existe ({destino.stat().st_size / 1e6:.1f} MB)")
        return
    if zipfile.is_zipfile(CACHE_HERMANO_ZIP):
        print(f"Copiando {CACHE_HERMANO_ZIP.name} desde caché local...")
        shutil.copy2(CACHE_HERMANO_ZIP, destino)
        return
    print(f"Descargando {url} ...")
    parcial = destino.with_suffix(destino.suffix + ".part")
    urllib.request.urlretrieve(url, parcial)
    if not zipfile.is_zipfile(parcial):
        raise RuntimeError(f"La descarga de {destino.name} no es un ZIP completo; vuelve a ejecutar el script")
    parcial.replace(destino)
    print(f"  Descargado: {destino.stat().st_size / 1e6:.1f} MB")


def extraer_excel(zip_path: Path) -> Path:
    with zipfile.ZipFile(zip_path) as z:
        nombre = next(n for n in z.namelist() if n.lower().endswith(".xlsx"))
        destino = CARPETA_RAW / Path(nombre).name
        if not destino.exists():
            parcial = destino.with_suffix(destino.suffix + ".part")
            with z.open(nombre) as origen, open(parcial, "wb") as salida:
                salida.write(origen.read())
            parcial.replace(destino)
    return destino


def convertir_excel_a_csv(excel: Path, csv_destino: Path) -> int:
    hojas = pd.read_excel(excel, sheet_name=None, dtype=str, engine="openpyxl")
    total_filas = 0
    with open(csv_destino, "w", encoding="utf-8", newline="") as f:
        escritor = csv.writer(f, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        escritor.writerow(COLUMNAS)
        for nombre_hoja, df in hojas.items():
            df.columns = [c.strip() for c in df.columns]
            filas_hoja = 0
            for i, r in enumerate(df.itertuples(index=False), start=2):
                cliente = _texto(r[6])
                if cliente.endswith(".0"):
                    cliente = cliente[:-2]
                fecha = pd.Timestamp(r[4]).strftime("%Y-%m-%d %H:%M:%S")
                escritor.writerow([
                    nombre_hoja,
                    i,
                    _texto(r[0]),
                    _texto(r[1]),
                    _texto(r[2]),
                    _texto(r[3]),
                    fecha,
                    _texto(r[5]),
                    cliente,
                    _texto(r[7]),
                ])
                filas_hoja += 1
            total_filas += filas_hoja
            print(f"  Hoja '{nombre_hoja}': {filas_hoja:,} filas")
    return total_filas


def main() -> None:
    CARPETA_RAW.mkdir(parents=True, exist_ok=True)
    if CSV_DESTINO.exists():
        print(f"[OK] {CSV_DESTINO.relative_to(RAIZ).as_posix()} ya existe ({CSV_DESTINO.stat().st_size / 1e6:.1f} MB)")
        return
    if CACHE_HERMANO_CSV.exists():
        print(f"Reutilizando CSV oficial desde {CACHE_HERMANO_CSV.relative_to(RAIZ.parents[1]).as_posix()} ...")
        shutil.copy2(CACHE_HERMANO_CSV, CSV_DESTINO)
        print(f"[OK] Copiado a {CSV_DESTINO.relative_to(RAIZ).as_posix()} ({CSV_DESTINO.stat().st_size / 1e6:.1f} MB)")
        return

    descargar_zip(URL_FUENTE, ZIP_DESTINO)
    excel = extraer_excel(ZIP_DESTINO)
    print(f"Convirtiendo {excel.name} a CSV...")
    total = convertir_excel_a_csv(excel, CSV_DESTINO)
    print(f"[OK] {CSV_DESTINO.relative_to(RAIZ).as_posix()} generado con {total:,} filas ({CSV_DESTINO.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
