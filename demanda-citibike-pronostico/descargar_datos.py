"""Descarga reanudable de viajes Citi Bike (S3) y clima horario (Open-Meteo).

- Viajes: 12 archivos mensuales `AAAA-MM-citibike-tripdata.csv.zip` del bucket
  público de Citi Bike, extraídos a data/raw/.
- Clima: Open-Meteo Historical API (sin clave) para NYC (40.71, -74.01),
  variables horarias de temperatura, precipitación, lluvia y viento en 2024.

La descarga es idempotente y reanudable: salta archivos ya presentes y escribe
con archivo `.parcial` (atómico) antes de renombrar.
"""
from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests

RAIZ = Path(__file__).resolve().parent
RAW = RAIZ / "data" / "raw"
MANIFIESTO = RAIZ / "data" / "manifiesto_fuente.json"

URL_TRIP = "https://s3.amazonaws.com/tripdata/{mes}-citibike-tripdata.zip"
URL_CLIMA = "https://archive-api.open-meteo.com/v1/archive"

MESES = [f"2024{m:02d}" for m in range(1, 13)]
LAT, LON = 40.71, -74.01
VARS_CLIMA = ["temperature_2m", "precipitation", "rain", "wind_speed_10m"]


def descargar_archivo(sesion: requests.Session, url: str, destino: Path) -> Path:
    """Descarga con escritura atómica (`.parcial` → renombrar)."""
    parcial = destino.with_suffix(destino.suffix + ".parcial")
    parcial.parent.mkdir(parents=True, exist_ok=True)
    with sesion.get(url, stream=True, timeout=(30, 900)) as r:
        r.raise_for_status()
        with parcial.open("wb") as f:
            for bloque in r.iter_content(4 * 1024 * 1024):
                if bloque:
                    f.write(bloque)
    parcial.replace(destino)
    return destino


def extraer_zip(zip_path: Path, destino_dir: Path) -> list[Path]:
    """Extrae todos los CSV de un zip (los meses de 2024 vienen partidos en varios)."""
    with zipfile.ZipFile(zip_path) as z:
        nombres = [n for n in z.namelist() if n.lower().endswith(".csv")]
        extraidos: list[Path] = []
        for n in nombres:
            z.extract(n, destino_dir)
            extraidos.append(destino_dir / n)
    return extraidos


def descargar_clima(sesion: requests.Session) -> Path:
    """Descarga el clima horario de NYC para 2024 desde Open-Meteo."""
    destino = RAW / "clima_nyc_horario.csv"
    if destino.exists() and destino.stat().st_size > 1000:
        print(f"OK clima: {destino.name} ({destino.stat().st_size:,} bytes)")
        return destino

    print("Descargando clima horario Open-Meteo (2024, hora local de Nueva York)...")
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": "2024-01-01",
        "end_date": "2024-12-31",
        "hourly": ",".join(VARS_CLIMA),
        "timezone": "America/New_York",
    }
    r = sesion.get(URL_CLIMA, params=params, timeout=180)
    r.raise_for_status()
    datos = r.json()
    horas = datos["hourly"]["time"]
    columnas = ["fecha"] + VARS_CLIMA
    filas = [horas] + [datos["hourly"][v] for v in VARS_CLIMA]

    with destino.open("w", encoding="utf-8", newline="") as f:
        f.write(",".join(columnas) + "\n")
        for i in range(len(horas)):
            f.write(",".join(str(filas[c][i]) for c in range(len(filas))) + "\n")
    print(f"  → clima guardado ({destino.stat().st_size:,} bytes)")
    return destino


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    sesion = requests.Session()
    sesion.headers.update({"User-Agent": "portafolio-ml03-citibike/1.0"})

    manifiesto = {
        "fuente_viajes": "Citi Bike System Data (Lyft)",
        "url_viajes": "https://s3.amazonaws.com/tripdata/",
        "licencia_viajes": "Citi Bike Trip Data License (uso público con atribución)",
        "fuente_clima": "Open-Meteo Historical API",
        "url_clima": URL_CLIMA,
        "licencia_clima": "CC BY 4.0",
        "consultado_utc": datetime.now(timezone.utc).isoformat(),
        "meses": MESES,
        "archivos_viajes": [],
    }

    # 1. Viajes por mes
    for mes in MESES:
        zip_path = RAW / f"{mes}-citibike-tripdata.zip"
        csvs = sorted(RAW.glob(f"{mes}-citibike-tripdata*.csv"))
        if csvs and all(c.stat().st_size > 0 for c in csvs):
            print(f"OK {mes}: {len(csvs)} CSV ya extraídos")
            manifiesto["archivos_viajes"].append({"mes": mes, "csvs": [c.name for c in csvs]})
            continue

        if not (zip_path.exists() and zip_path.stat().st_size > 0):
            url = URL_TRIP.format(mes=mes)
            print(f"DESCARGA {mes}: {url}")
            try:
                descargar_archivo(sesion, url, zip_path)
                print(f"  → zip guardado ({zip_path.stat().st_size / 1e6:.1f} MB)")
            except Exception as e:  # noqa: BLE001
                print(f"  Error descargando {mes}: {e}")
                continue

        print(f"EXTRAYENDO {mes} ...")
        try:
            extraidos = extraer_zip(zip_path, RAW)
            print(f"  → {len(extraidos)} CSV extraídos")
            manifiesto["archivos_viajes"].append({"mes": mes, "csvs": [p.name for p in extraidos]})
        except Exception as e:  # noqa: BLE001
            print(f"  Error extrayendo {mes}: {e}")

    # 2. Clima
    clima = descargar_clima(sesion)
    manifiesto["archivo_clima"] = clima.name

    MANIFIESTO.write_text(json.dumps(manifiesto, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\nDescarga finalizada.")


if __name__ == "__main__":
    main()
