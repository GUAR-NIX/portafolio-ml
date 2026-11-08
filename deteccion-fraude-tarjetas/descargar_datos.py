"""Descarga reanudable del dataset ULB Credit Card Fraud desde OpenML (data_id=1597).

Uso:
    python descargar_datos.py            # descarga si falta, valida y guarda data/raw/creditcard.parquet
    python descargar_datos.py --forzar   # vuelve a descargar aunque exista

Cómo funciona:
- Fuente principal: el Parquet público de OpenML (~73 MB). Se baja a un archivo `.part` con cabeceras
  HTTP `Range`, así que si la conexión se corta el siguiente intento continúa donde quedó.
- Respaldo: el ARFF público de OpenML (mismo contenido), también reanudable.
- No se usa `fetch_openml` porque OpenML marca `Time` como identificador de fila y lo descarta, y
  `Time` es imprescindible para la partición temporal.
- Si `data/raw/creditcard.parquet` ya existe y pasa la validación, no descarga nada.
- Valida la forma conocida: 284.807 filas, 492 fraudes, columnas Time, V1–V28, Amount, Class.

Fuente: Dal Pozzolo, A., Caelen, O., Johnson, R. A. y Bontempi, G. (2015). *Calibrating Probability with
Undersampling for Unbalanced Classification*. IEEE SSCI. Worldline y Machine Learning Group (ULB).
Licencia de los datos: Open Database License (ODbL) 1.0 según la publicación original del MLG-ULB;
OpenML la redistribuye como dataset 1597 («creditcard»).
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent
DIR_RAW = RAIZ / "data" / "raw"
ARCHIVO = DIR_RAW / "creditcard.parquet"
URL_PARQUET = "https://data.openml.org/datasets/0000/1597/dataset_1597.pq"
URL_ARFF = "https://openml.org/data/v1/download/1673544/creditcard.arff"
FILAS_ESPERADAS = 284_807
FRAUDES_ESPERADOS = 492
COLUMNAS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]


def validar(df: pd.DataFrame) -> None:
    faltan = [c for c in COLUMNAS if c not in df.columns]
    if faltan:
        raise ValueError(f"Faltan columnas: {faltan}")
    if len(df) != FILAS_ESPERADAS:
        raise ValueError(f"Se esperaban {FILAS_ESPERADAS} filas y hay {len(df)}")
    if int(df["Class"].sum()) != FRAUDES_ESPERADOS:
        raise ValueError(f"Se esperaban {FRAUDES_ESPERADOS} fraudes y hay {int(df['Class'].sum())}")


def descarga_reanudable(url: str, destino: Path, reintentos: int = 5) -> Path:
    """Descarga `url` a `destino` continuando un `.part` previo con HTTP Range."""
    parcial = destino.with_name(destino.name + ".part")
    for intento in range(1, reintentos + 1):
        ya = parcial.stat().st_size if parcial.exists() else 0
        req = urllib.request.Request(url, headers={"User-Agent": "ML_07_fraude_tarjetas/1.0"})
        if ya:
            req.add_header("Range", f"bytes={ya}-")
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                # 206 = el servidor respetó el Range; 200 = manda todo de nuevo
                modo = "ab" if (ya and r.status == 206) else "wb"
                total = r.headers.get("Content-Length")
                print(f"[descarga] {url} (intento {intento}, desde byte {ya if modo == 'ab' else 0}, {total} bytes por bajar)")
                with open(parcial, modo) as f:
                    while bloque := r.read(1 << 20):
                        f.write(bloque)
            parcial.replace(destino)
            return destino
        except Exception as e:  # red caída, servidor lento
            print(f"[aviso] Intento {intento} falló: {e}. Se conserva {parcial.name} para continuar.")
            time.sleep(3 * intento)
    raise RuntimeError(f"No se pudo descargar {url}")


def leer_arff(ruta: Path) -> pd.DataFrame:
    texto = ruta.read_text(encoding="utf-8", errors="replace")
    inicio = texto.lower().index("@data")
    cuerpo = texto[inicio:].split("\n", 1)[1]
    return pd.read_csv(io.StringIO(cuerpo), header=None, names=COLUMNAS, quotechar="'")


def normalizar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Class"] = df["Class"].astype(str).str.strip("'\"").astype(int)  # OpenML lo entrega como categoría
    for c in COLUMNAS[:-1]:
        df[c] = df[c].astype("float64")
    return df[COLUMNAS].reset_index(drop=True)


def obtener() -> pd.DataFrame:
    try:
        ruta = descarga_reanudable(URL_PARQUET, DIR_RAW / "dataset_1597.pq")
        return normalizar(pd.read_parquet(ruta))
    except Exception as e:
        print(f"[aviso] Parquet no disponible ({e}); se intenta el ARFF de OpenML.")
        ruta = descarga_reanudable(URL_ARFF, DIR_RAW / "creditcard.arff")
        return normalizar(leer_arff(ruta))


def main() -> int:
    parser = argparse.ArgumentParser(description="Descarga reanudable de ULB Credit Card Fraud (OpenML 1597)")
    parser.add_argument("--forzar", action="store_true", help="Descargar aunque ya exista el archivo")
    args = parser.parse_args()

    DIR_RAW.mkdir(parents=True, exist_ok=True)
    if ARCHIVO.exists() and not args.forzar:
        df = pd.read_parquet(ARCHIVO)
        validar(df)
        print(f"[ok] Ya existe y es válido: data/raw/{ARCHIVO.name} ({len(df):,} filas)")
        return 0

    df = obtener()
    validar(df)
    tmp = ARCHIVO.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(ARCHIVO)  # escritura atómica: nunca queda un parquet a medias
    resumen = {
        "fuente": "OpenML data_id=1597 (creditcard, MLG-ULB / Worldline)",
        "filas": len(df),
        "fraudes": int(df["Class"].sum()),
        "tasa_fraude_pct": round(float(df["Class"].mean() * 100), 4),
        "time_min_s": float(df["Time"].min()),
        "time_max_s": float(df["Time"].max()),
    }
    (DIR_RAW / "resumen_descarga.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ok] Guardado data/raw/{ARCHIVO.name}: {resumen}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
