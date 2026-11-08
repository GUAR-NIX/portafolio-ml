"""Ingesta de viajes Citi Bike: agregación a salidas/llegadas por estación y hora.

Lee los CSV mensuales con DuckDB (sin cargar decenas de millones de filas en
pandas), agrega salidas y llegadas por estación × hora, elige las N estaciones
con más actividad y construye la serie completa con ceros en las horas sin viajes.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from demanda_citibike.config import FIN_PERIODO, INICIO_PERIODO, MUESTRA, N_ESTACIONES, PROCESADOS, RAW, REPORTES


def recortar_periodo(demanda: pd.DataFrame) -> pd.DataFrame:
    """Deja solo las horas del año 2024 (8 784 h, bisiesto)."""
    fh = pd.to_datetime(demanda["fecha_hora"])
    return demanda[(fh >= INICIO_PERIODO) & (fh <= FIN_PERIODO)].reset_index(drop=True)


def agregar_viajes(
    carpeta_raw: Path = RAW,
    n_estaciones: int = N_ESTACIONES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve (demanda horaria, metadatos de estaciones).

    `demanda horaria`: filas `estacion × fecha_hora` con columnas `salidas` y `llegadas`.
    """
    archivos = sorted(carpeta_raw.glob("*tripdata*.csv"))
    if not archivos:
        raise FileNotFoundError("No hay CSVs de viajes en data/raw. Ejecuta descargar_datos.py")

    con = duckdb.connect()
    patron = str(carpeta_raw / "*tripdata*.csv").replace("\\", "/")
    con.execute(f"""
        CREATE OR REPLACE VIEW viajes AS
        SELECT
            TRY_CAST(started_at AS TIMESTAMP) AS inicio,
            TRY_CAST(ended_at AS TIMESTAMP) AS fin,
            TRY_CAST(start_station_id AS VARCHAR) AS estacion_inicio,
            TRY_CAST(end_station_id AS VARCHAR) AS estacion_fin,
            start_station_name AS nombre_inicio,
            TRY_CAST(start_lat AS DOUBLE) AS lat_inicio,
            TRY_CAST(start_lng AS DOUBLE) AS lng_inicio
        FROM read_csv_auto('{patron}', header=true, union_by_name=true, sample_size=300000)
    """)

    salidas = con.execute("""
        SELECT estacion_inicio AS estacion,
               date_trunc('hour', inicio) AS fecha_hora,
               count(*) AS salidas
        FROM viajes
        WHERE estacion_inicio IS NOT NULL AND inicio IS NOT NULL
        GROUP BY 1, 2
    """).df()

    llegadas = con.execute("""
        SELECT estacion_fin AS estacion,
               date_trunc('hour', fin) AS fecha_hora,
               count(*) AS llegadas
        FROM viajes
        WHERE estacion_fin IS NOT NULL AND fin IS NOT NULL
        GROUP BY 1, 2
    """).df()

    # Estaciones con más salidas (la ficha pide ~200-300 con más actividad).
    top = salidas.groupby("estacion")["salidas"].sum().sort_values(ascending=False).head(n_estaciones).index.tolist()
    salidas = salidas[salidas["estacion"].isin(top)].copy()
    llegadas = llegadas[llegadas["estacion"].isin(top)].copy()

    # Grilla completa estación × hora (con ceros) para todo 2024.
    horas = pd.date_range(INICIO_PERIODO, FIN_PERIODO, freq="h")
    idx = pd.MultiIndex.from_product([top, horas], names=["estacion", "fecha_hora"])
    s = salidas.set_index(["estacion", "fecha_hora"]).reindex(idx, fill_value=0)
    l = llegadas.set_index(["estacion", "fecha_hora"]).reindex(idx, fill_value=0)

    demanda = pd.DataFrame(
        {"salidas": s["salidas"].astype("int32").values, "llegadas": l["llegadas"].astype("int32").values},
        index=idx,
    ).reset_index()

    # Metadatos de estaciones (nombre, latitud, longitud más frecuentes).
    meta = con.execute("""
        SELECT estacion_inicio AS estacion,
               mode(nombre_inicio) AS nombre,
               mode(lat_inicio) AS latitud,
               mode(lng_inicio) AS longitud
        FROM viajes
        WHERE estacion_inicio IS NOT NULL
        GROUP BY 1
    """).df()
    meta = meta[meta["estacion"].isin(top)].reset_index(drop=True)

    return recortar_periodo(demanda), meta


def resumir_seleccion(carpeta_raw: Path = RAW, n_estaciones: int = N_ESTACIONES) -> dict:
    """Cuenta los viajes de 2024 y documenta la selección de las N estaciones más activas.

    Compara la selección por salidas de todo el año con la que se obtendría usando solo
    enero-noviembre (sin mirar el mes de test) y guarda reports/seleccion_estaciones.json.
    """
    con = duckdb.connect()
    con.execute("SET memory_limit='4GB'; SET threads=2; SET preserve_insertion_order=false")
    patron = str(carpeta_raw / "*tripdata*.csv").replace("\\", "/")
    c = con.execute(f"""
        SELECT TRY_CAST(start_station_id AS VARCHAR) AS estacion,
               month(TRY_CAST(started_at AS TIMESTAMP)) AS mes,
               count(*) AS n
        FROM read_csv_auto('{patron}', header=true, union_by_name=true, sample_size=300000)
        WHERE year(TRY_CAST(started_at AS TIMESTAMP)) = 2024
        GROUP BY ALL
    """).df()
    por_estacion = c.dropna(subset=["estacion"]).groupby("estacion")["n"].sum().sort_values(ascending=False)
    top = por_estacion.head(n_estaciones)
    ene_nov = c[c["mes"] <= 11].dropna(subset=["estacion"]).groupby("estacion")["n"].sum().sort_values(ascending=False)
    resumen = {
        "viajes_2024": int(c["n"].sum()),
        "viajes_sin_estacion_origen": int(c.loc[c["estacion"].isna(), "n"].sum()),
        "estaciones_con_salidas": int(por_estacion.size),
        "estaciones_modeladas": int(n_estaciones),
        "salidas_estaciones_modeladas": int(top.sum()),
        "cuota_salidas_modeladas_pct": round(float(top.sum() / c["n"].sum() * 100), 1),
        "salidas_anuales_estacion_250": int(top.iloc[-1]),
        "salidas_anuales_estacion_1": int(top.iloc[0]),
        "mediana_salidas_anuales_todas": float(por_estacion.median()),
        "coinciden_si_se_elige_con_ene_nov": int(len(set(top.index) & set(ene_nov.head(n_estaciones).index))),
    }
    REPORTES.mkdir(parents=True, exist_ok=True)
    (REPORTES / "seleccion_estaciones.json").write_text(
        __import__("json").dumps(resumen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return resumen


def guardar_demanda(demanda: pd.DataFrame, meta: pd.DataFrame) -> None:
    """Persiste la demanda horaria y los metadatos a data/procesados/."""
    PROCESADOS.mkdir(parents=True, exist_ok=True)
    demanda.to_parquet(PROCESADOS / "demanda_horaria.parquet", index=False)
    meta.to_csv(PROCESADOS / "estaciones.csv", index=False)
    (PROCESADOS / "resumen_ingesta.json").write_text(
        __import__("json").dumps(
            {
                "filas_demanda_horaria": int(len(demanda)),
                "estaciones": int(meta["estacion"].nunique()),
                "total_salidas": int(demanda["salidas"].sum()),
                "total_llegadas": int(demanda["llegadas"].sum()),
                "periodo_inicio": str(demanda["fecha_hora"].min()),
                "periodo_fin": str(demanda["fecha_hora"].max()),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def generar_muestra_ci(demanda: pd.DataFrame, n_estaciones: int = 6, n_horas: int = 24 * 14) -> Path:
    """Extrae una muestra pequeña (estaciones × horas) para CI y pruebas."""
    MUESTRA.mkdir(parents=True, exist_ok=True)
    estaciones = demanda["estacion"].unique()[:n_estaciones]
    sub = demanda[demanda["estacion"].isin(estaciones)].copy()
    horas_unicas = sorted(sub["fecha_hora"].unique())[:n_horas]
    sub = sub[sub["fecha_hora"].isin(horas_unicas)]
    ruta = MUESTRA / "muestra_demanda.parquet"
    sub.to_parquet(ruta, index=False)
    return ruta
