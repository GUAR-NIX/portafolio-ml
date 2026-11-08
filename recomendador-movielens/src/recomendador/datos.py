"""Lectura de MovieLens 25M, conversión a Parquet, partición temporal y matrices dispersas."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
import scipy.sparse as sp

from . import config as C

GENEROS = [
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "Film-Noir", "Horror", "IMAX", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


# ---------------------------------------------------------------- conversión y lectura
def convertir_a_parquet(forzar: bool = False) -> None:
    """CSV → Parquet en bloques (pico de RAM bajo). ratings: userId/movieId int32, rating float32, timestamp int32."""
    C.DATOS_PROCESADOS.mkdir(parents=True, exist_ok=True)
    tipos = {
        "ratings": {"userId": pa.int32(), "movieId": pa.int32(), "rating": pa.float32(), "timestamp": pa.int32()},
        "genome-scores": {"movieId": pa.int32(), "tagId": pa.int16(), "relevance": pa.float32()},
    }
    for nombre, esquema in tipos.items():
        salida = C.DATOS_PROCESADOS / f"{nombre}.parquet"
        if salida.exists() and not forzar:
            continue
        lector = pacsv.open_csv(
            C.DATOS_CRUDOS / f"{nombre}.csv",
            read_options=pacsv.ReadOptions(block_size=64 << 20),
            convert_options=pacsv.ConvertOptions(column_types=esquema),
        )
        with pq.ParquetWriter(salida, lector.schema, compression="zstd") as escritor:
            for lote in lector:
                escritor.write_batch(lote)
        print(f"  {salida.name}: {salida.stat().st_size / 1e6:.0f} MB")


def cargar_calificaciones(columnas: list[str] | None = None) -> pd.DataFrame:
    return pq.read_table(C.DATOS_PROCESADOS / "ratings.parquet", columns=columnas).to_pandas()


def cargar_peliculas() -> pd.DataFrame:
    """movieId, titulo, anio, generos (lista) y una columna 0/1 por género."""
    pel = pd.read_csv(C.DATOS_CRUDOS / "movies.csv")
    pel = pel.rename(columns={"title": "titulo"})
    pel["anio"] = pd.to_numeric(pel["titulo"].str.extract(r"\((\d{4})\)\s*$")[0], errors="coerce").astype("Int16")
    pel["generos"] = pel["genres"].replace("(no genres listed)", "").str.split("|")
    pel["generos"] = pel["generos"].apply(lambda g: [x for x in g if x])
    for g in GENEROS:
        pel[g] = pel["genres"].str.contains(g, regex=False).astype(np.int8)
    return pel.drop(columns="genres")


def cargar_genome() -> tuple[pd.DataFrame, list[str]]:
    """Matriz película × 1.128 etiquetas del Tag Genome (relevancia 0-1) y los nombres de las etiquetas."""
    g = pq.read_table(C.DATOS_PROCESADOS / "genome-scores.parquet").to_pandas()
    nombres = pd.read_csv(C.DATOS_CRUDOS / "genome-tags.csv").sort_values("tagId")["tag"].tolist()
    matriz = g.pivot(index="movieId", columns="tagId", values="relevance").astype(np.float32)
    return matriz, nombres


# ---------------------------------------------------------------- partición temporal
@dataclass
class Particion:
    """Entrenamiento (todo < corte) y evaluación (corte ≤ t < fin), con índices internos 0..n-1."""

    corte: pd.Timestamp
    fin: pd.Timestamp
    usuarios: np.ndarray            # userId por índice interno
    items: np.ndarray               # movieId por índice interno (catálogo recomendable)
    positivos: sp.csr_matrix        # usuarios × items, 1 si calificó ≥ 4 antes del corte
    vistos: sp.csr_matrix           # usuarios × items, 1 si calificó (cualquier nota) antes del corte
    recientes: np.ndarray           # positivos por item en los últimos DIAS_RECIENTES antes del corte
    eval_existentes: dict[int, np.ndarray] = field(default_factory=dict)   # u interno → items relevantes
    eval_nuevos: dict[int, dict] = field(default_factory=dict)             # userId → onboarding y relevantes
    resumen: dict = field(default_factory=dict)

    @property
    def n_positivos_usuario(self) -> np.ndarray:
        return np.diff(self.positivos.indptr)

    def idx_item(self) -> dict[int, int]:
        return {int(m): i for i, m in enumerate(self.items)}


def particionar(cal: pd.DataFrame, corte: pd.Timestamp, fin: pd.Timestamp) -> Particion:
    """Partición temporal global.

    - Entrenamiento: todas las calificaciones con fecha < corte (de todos los usuarios).
    - Evaluación de usuarios existentes: sus positivos en [corte, fin) sobre películas que no habían calificado.
    - Usuarios nuevos (primera calificación en [corte, fin)): las N_ONBOARDING primeras hacen de cuestionario
      inicial; se evalúa con sus positivos posteriores dentro de la ventana.
    """
    t_corte, t_fin = int(corte.timestamp()), int(fin.timestamp())
    ts = cal["timestamp"].to_numpy()
    ent = cal[ts < t_corte]
    ven = cal[(ts >= t_corte) & (ts < t_fin)]

    pos_ent = ent[ent["rating"] >= C.UMBRAL_POSITIVO]
    cuenta_item = pos_ent["movieId"].value_counts()
    items = np.sort(cuenta_item.index[cuenta_item >= C.MIN_POSITIVOS_ITEM].to_numpy())
    usuarios = np.sort(ent["userId"].unique())
    u_idx = pd.Series(np.arange(len(usuarios), dtype=np.int32), index=usuarios)
    i_idx = pd.Series(np.arange(len(items), dtype=np.int32), index=items)

    def matriz(df: pd.DataFrame) -> sp.csr_matrix:
        df = df[df["movieId"].isin(i_idx.index)]
        filas = u_idx.loc[df["userId"].to_numpy()].to_numpy()
        cols = i_idx.loc[df["movieId"].to_numpy()].to_numpy()
        m = sp.csr_matrix((np.ones(len(df), np.float32), (filas, cols)), shape=(len(usuarios), len(items)))
        m.sum_duplicates()
        m.data[:] = 1.0
        return m

    positivos = matriz(pos_ent)
    vistos = matriz(ent)
    t_rec = t_corte - C.DIAS_RECIENTES * 86400
    rec = pos_ent[pos_ent["timestamp"] >= t_rec]["movieId"].value_counts()
    recientes = rec.reindex(items, fill_value=0).to_numpy(np.float32)

    # --- usuarios existentes con al menos un positivo en entrenamiento
    con_hist = set(usuarios[np.diff(positivos.indptr) > 0])
    pos_ven = ven[ven["rating"] >= C.UMBRAL_POSITIVO]
    eval_existentes: dict[int, np.ndarray] = {}
    ex = pos_ven[pos_ven["userId"].isin(con_hist) & pos_ven["movieId"].isin(i_idx.index)]
    for uid, grupo in ex.groupby("userId")["movieId"]:
        u = int(u_idx[uid])
        rel = i_idx.loc[grupo.to_numpy()].to_numpy()
        ya = vistos.indices[vistos.indptr[u]:vistos.indptr[u + 1]]
        rel = np.setdiff1d(rel, ya)
        if len(rel):
            eval_existentes[u] = rel

    # --- usuarios nuevos: sin ninguna calificación antes del corte
    nuevos_ids = np.setdiff1d(ven["userId"].unique(), usuarios)
    vn = ven[ven["userId"].isin(nuevos_ids)].sort_values(["userId", "timestamp", "movieId"])
    eval_nuevos: dict[int, dict] = {}
    for uid, g in vn.groupby("userId"):
        onb, resto = g.iloc[: C.N_ONBOARDING], g.iloc[C.N_ONBOARDING:]
        onb_pos = onb[(onb["rating"] >= C.UMBRAL_POSITIVO) & onb["movieId"].isin(i_idx.index)]
        rel = resto[(resto["rating"] >= C.UMBRAL_POSITIVO) & resto["movieId"].isin(i_idx.index)]["movieId"]
        if len(onb_pos) == 0 or len(rel) == 0:
            continue
        eval_nuevos[int(uid)] = {
            "onboarding": i_idx.loc[onb_pos["movieId"].to_numpy()].to_numpy(),
            "vistos": i_idx.loc[onb[onb["movieId"].isin(i_idx.index)]["movieId"].to_numpy()].to_numpy(),
            "relevantes": np.unique(i_idx.loc[rel.to_numpy()].to_numpy()),
        }

    pos_ven_total = len(pos_ven)
    pos_item_nuevo = int((~pos_ven["movieId"].isin(i_idx.index)).sum())
    resumen = {
        "corte": str(corte.date()), "fin": str(fin.date()),
        "calificaciones_entrenamiento": int(len(ent)), "positivos_entrenamiento": int(positivos.nnz),
        "usuarios_entrenamiento": int(len(usuarios)), "items_catalogo": int(len(items)),
        "calificaciones_ventana": int(len(ven)), "positivos_ventana": int(pos_ven_total),
        "usuarios_existentes_evaluables": len(eval_existentes), "usuarios_nuevos_evaluables": len(eval_nuevos),
        "positivos_ventana_fuera_de_catalogo_pct": round(100 * pos_item_nuevo / max(pos_ven_total, 1), 2),
    }
    return Particion(corte, fin, usuarios, items, positivos, vistos, recientes, eval_existentes, eval_nuevos, resumen)


def muestrear_usuarios(usuarios: list[int], n: int, semilla: int = C.SEMILLA) -> list[int]:
    if len(usuarios) <= n:
        return sorted(usuarios)
    rng = np.random.default_rng(semilla)
    return sorted(rng.choice(np.asarray(usuarios), size=n, replace=False).tolist())


def caracteristicas_items(items: np.ndarray, peliculas: pd.DataFrame | None = None,
                          genome: pd.DataFrame | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Géneros (n_items × 19, 0/1) y Tag Genome (n_items × 1128, NaN si la película no tiene) alineados al catálogo."""
    peliculas = cargar_peliculas() if peliculas is None else peliculas
    genome = cargar_genome()[0] if genome is None else genome
    gen = peliculas.set_index("movieId").reindex(items)[GENEROS].fillna(0).to_numpy(np.float32)
    gnm = genome.reindex(items).to_numpy(np.float32)
    return gen, gnm
