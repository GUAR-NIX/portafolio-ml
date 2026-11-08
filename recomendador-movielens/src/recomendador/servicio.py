"""Recomendador listo para servir: carga el paquete liviano de models/ (sin necesitar los datos crudos).

Lo usan la API (api/main.py) y la app (app/app.py). Si faltan peliculas.parquet o demo_*.npz (llevan datos de
MovieLens y no se redistribuyen), lanza PaqueteIncompleto con la receta para generarlos (python generar_demo.py).
"""
from __future__ import annotations

import json
import unicodedata
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import config as C
from .datos import GENEROS
from .demo import PaqueteIncompleto, mensaje_faltantes
from .modelos import ALS, Contenido, Hibrido, top_k, zscore_filas

GENEROS_ES = {
    "Action": "Acción", "Adventure": "Aventura", "Animation": "Animación", "Children": "Infantil",
    "Comedy": "Comedia", "Crime": "Crimen", "Documentary": "Documental", "Drama": "Drama", "Fantasy": "Fantasía",
    "Film-Noir": "Cine negro", "Horror": "Terror", "IMAX": "IMAX", "Musical": "Musical", "Mystery": "Misterio",
    "Romance": "Romance", "Sci-Fi": "Ciencia ficción", "Thriller": "Suspenso", "War": "Bélica", "Western": "Western",
}
METODOS = ("hibrido", "als", "contenido", "popular")


def _sin_tildes(t: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", t.lower()) if unicodedata.category(c) != "Mn")


_ALIAS_GENERO = {_sin_tildes(g): g for g in GENEROS} | {_sin_tildes(es): en for en, es in GENEROS_ES.items()}


def normalizar_genero(g: str) -> str:
    """Acepta el nombre de MovieLens o el español, con o sin tildes ('ciencia ficcion' → 'Sci-Fi')."""
    clave = _sin_tildes(g.strip())
    if clave not in _ALIAS_GENERO:
        raise ValueError(f"género desconocido: {g!r}. Válidos: {', '.join(GENEROS)}")
    return _ALIAS_GENERO[clave]


class PeliculaDesconocida(KeyError):
    pass


class UsuarioDesconocido(KeyError):
    pass


class Recomendador:
    def __init__(self, carpeta: Path = C.MODELOS):
        if mensaje_faltantes(carpeta):
            raise PaqueteIncompleto(mensaje_faltantes(carpeta))
        self.metadata = json.loads((carpeta / "metadata.json").read_text(encoding="utf-8"))
        hp = self.metadata["hiperparametros"]
        self.peliculas = pd.read_parquet(carpeta / "peliculas.parquet")
        f = np.load(carpeta / "factores.npz")
        n = len(self.peliculas)

        self.als = ALS(**hp["als"])
        self.als.factores_item = f["factores_item"].astype(np.float32)
        self.als._preparar()
        self.als.idf = f["idf_bm25"].astype(np.float32)  # fold-in con los pesos BM25 del entrenamiento
        self.als.largo_medio = float(self.metadata["modelo_final"].get("largo_medio_bm25", 1.0))
        demo = np.load(carpeta / "demo_usuarios.npz")
        self.als.factores_usuario = demo["factores"].astype(np.float32)
        self.demo_usuarios = {int(u): i for i, u in enumerate(demo["usuarios"])}
        self.demo_positivos = sp.load_npz(carpeta / "demo_positivos.npz").tocsr()
        self.demo_vistos = sp.load_npz(carpeta / "demo_vistos.npz").tocsr()

        self.contenido = Contenido(**hp["contenido"])
        self.contenido.set_vectores(f["genome_reducido"].astype(np.float32), f["generos"].astype(np.float32),
                                    self.peliculas["positivos"].to_numpy())
        self.recientes = self.peliculas["recientes"].to_numpy(np.float32)
        self.hibrido = Hibrido(self.als, self.contenido, recientes=self.recientes, **hp["hibrido"])
        self.idx = {int(m): i for i, m in enumerate(self.peliculas["movieId"])}
        self.n = n
        self._titulos_norm = self.peliculas["titulo"].map(_sin_tildes)

    # ------------------------------------------------------------ utilidades
    def _indices(self, peliculas: list[int]) -> np.ndarray:
        faltan = [p for p in peliculas if int(p) not in self.idx]
        if faltan:
            raise PeliculaDesconocida(f"películas fuera del catálogo recomendable: {faltan[:10]}")
        return np.array([self.idx[int(p)] for p in peliculas], dtype=np.int64)

    def _fila(self, indices: np.ndarray) -> sp.csr_matrix:
        m = sp.csr_matrix((np.ones(len(indices), np.float32), (np.zeros(len(indices), int), indices)), shape=(1, self.n))
        m.sum_duplicates()
        m.data[:] = 1
        return m

    def _puntajes(self, historial: sp.csr_matrix, metodo: str, usuario: int | None = None) -> np.ndarray:
        u = None if usuario is None else np.array([usuario])
        if metodo == "hibrido":
            return self.hibrido.puntuar(historial, u)[0]
        if metodo == "als":
            return self.als.puntuar(historial, u)[0]
        if metodo == "contenido":
            return self.contenido.puntuar(historial)[0]
        if metodo == "popular":
            return np.log1p(self.recientes) + 1e-3 * np.log1p(self.contenido.conteo)
        raise ValueError(f"método desconocido: {metodo!r}. Válidos: {METODOS}")

    def _formatear(self, indices: np.ndarray, puntajes: np.ndarray, gustos: np.ndarray | None = None) -> list[dict]:
        salida = []
        for i in indices:
            p = self.peliculas.iloc[int(i)]
            fila = {
                "movieId": int(p["movieId"]), "titulo": p["titulo"],
                "anio": None if pd.isna(p["anio"]) else int(p["anio"]),
                "generos": [g for g in p["generos"].split("|") if g],
                "puntaje": round(float(puntajes[int(i)]), 4), "popularidad": int(p["positivos"]),
            }
            if gustos is not None and len(gustos):
                sim = self.contenido.vectores[gustos] @ self.contenido.vectores[int(i)]
                fila["porque_te_gusto"] = self.peliculas.iloc[int(gustos[int(np.argmax(sim))])]["titulo"]
            salida.append(fila)
        return salida

    # ------------------------------------------------------------ casos de uso
    def recomendar_usuario(self, usuario: int, n: int = C.K, metodo: str = "hibrido") -> dict:
        if usuario not in self.demo_usuarios:
            raise UsuarioDesconocido(usuario)
        fila = self.demo_usuarios[usuario]
        hist, vistos = self.demo_positivos[fila], self.demo_vistos[fila]
        p = self._puntajes(hist, metodo, fila)
        recs = top_k(p[None, :], vistos, n)[0]
        n_pos = int(hist.nnz)
        return {"usuario": usuario, "metodo": metodo, "positivos_en_historial": n_pos,
                "peso_als": round(float(self.hibrido.peso_als([n_pos])[0]), 3),
                "segmento": C.segmento_de(n_pos),
                "recomendaciones": self._formatear(recs, p, hist.indices)}

    def recomendar_historial(self, le_gustaron: list[int], n: int = C.K, metodo: str = "hibrido",
                             vistas: list[int] | None = None) -> dict:
        """Usuario que no estaba en el entrenamiento: ALS por fold-in + contenido, sin re-entrenar."""
        gustos = self._indices(le_gustaron)
        hist = self._fila(gustos)
        excluir = self._fila(np.concatenate([gustos, self._indices(vistas or [])]))
        p = self._puntajes(hist, metodo)
        recs = top_k(p[None, :], excluir, n)[0]
        return {"metodo": metodo, "positivos_en_historial": int(hist.nnz),
                "peso_als": round(float(self.hibrido.peso_als([hist.nnz])[0]), 3),
                "recomendaciones": self._formatear(recs, p, gustos)}

    def usuario_nuevo(self, generos: list[str], n: int = C.K, excluir: list[int] | None = None) -> dict:
        """Arranque en frío puro: solo los géneros que el usuario marca al registrarse."""
        elegidos = sorted({normalizar_genero(g) for g in generos})
        if not elegidos:
            raise ValueError("elige al menos un género")
        v = np.array([[1.0 if g in elegidos else 0.0 for g in GENEROS]], dtype=np.float32)
        p = self.contenido.puntuar_generos(v)[0]
        recs = top_k(p[None, :], self._fila(self._indices(excluir or [])), n)[0]
        return {"generos": elegidos, "generos_es": [GENEROS_ES[g] for g in elegidos],
                "recomendaciones": self._formatear(recs, p)}

    def similares(self, pelicula: int, n: int = C.K, metodo: str = "hibrido") -> dict:
        i = int(self._indices([pelicula])[0])
        cont = self.contenido.vectores @ self.contenido.vectores[i]
        y = self.als.factores_item / np.maximum(np.linalg.norm(self.als.factores_item, axis=1, keepdims=True), 1e-9)
        colab = y @ y[i]
        if metodo == "contenido":
            p = cont
        elif metodo == "als":
            p = colab
        elif metodo == "hibrido":
            p = zscore_filas(np.vstack([cont, colab])).mean(axis=0)
        else:
            raise ValueError(f"método desconocido: {metodo!r}. Válidos: hibrido, als, contenido")
        p = p.astype(np.float32)
        recs = top_k(p[None, :], self._fila(np.array([i])), n)[0]
        return {"pelicula": self._formatear(np.array([i]), p)[0] | {"puntaje": None}, "metodo": metodo,
                "similares": self._formatear(recs, p)}

    def buscar(self, texto: str, n: int = 10) -> list[dict]:
        t = _sin_tildes(texto.strip())
        if not t:
            return []
        m = self.peliculas[self._titulos_norm.str.contains(t, regex=False)]
        m = m.sort_values("positivos", ascending=False).head(n)
        return [{"movieId": int(r.movieId), "titulo": r.titulo, "popularidad": int(r.positivos)} for r in m.itertuples()]

    def usuarios_demo(self, n: int = 20) -> list[int]:
        return sorted(self.demo_usuarios)[:n]


@lru_cache(maxsize=1)
def cargar() -> Recomendador:
    return Recomendador()
