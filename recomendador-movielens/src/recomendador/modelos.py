"""Modelos de recomendación con una interfaz común.

Todos devuelven una matriz de puntajes (usuarios × catálogo) a partir del historial de positivos de cada usuario
(una fila de una matriz CSR). Así sirven igual para usuarios conocidos y para usuarios nuevos (onboarding o API).
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # implicit lo recomienda: su paralelismo es propio

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp

from . import config as C


def _filas(historial: sp.csr_matrix) -> sp.csr_matrix:
    h = sp.csr_matrix(historial, dtype=np.float32)
    h.data[:] = 1.0
    return h


def zscore_filas(m: np.ndarray) -> np.ndarray:
    """Estandariza cada fila (usuario) para poder mezclar puntajes de modelos distintos."""
    media = m.mean(axis=1, keepdims=True)
    desv = m.std(axis=1, keepdims=True)
    return (m - media) / np.where(desv > 0, desv, 1.0)


class Modelo:
    nombre = "modelo"

    def ajustar(self, positivos: sp.csr_matrix, **kw) -> "Modelo":
        return self

    def puntuar(self, historial: sp.csr_matrix, usuarios: np.ndarray | None = None) -> np.ndarray:
        raise NotImplementedError


# ---------------------------------------------------------------- líneas base
class Popularidad(Modelo):
    """Las películas con más positivos en todo el entrenamiento (igual para todos)."""

    nombre = "Popularidad global"

    def ajustar(self, positivos, **kw):
        self.conteo = np.asarray(positivos.sum(axis=0)).ravel().astype(np.float32)
        return self

    def puntuar(self, historial, usuarios=None):
        return np.broadcast_to(np.log1p(self.conteo), (historial.shape[0], len(self.conteo))).copy()


class PopularidadReciente(Modelo):
    """Las películas con más positivos en los últimos 180 días antes del corte (desempate: popularidad global)."""

    nombre = "Popularidad reciente"

    def ajustar(self, positivos, recientes=None, **kw):
        global_ = np.asarray(positivos.sum(axis=0)).ravel().astype(np.float32)
        self.puntaje = np.log1p(recientes) + 1e-3 * np.log1p(global_)
        return self

    def puntuar(self, historial, usuarios=None):
        return np.broadcast_to(self.puntaje, (historial.shape[0], len(self.puntaje))).copy()


# ---------------------------------------------------------------- filtrado colaborativo implícito
class ALS(Modelo):
    """ALS implícito (Hu, Koren y Volinsky 2008) con la librería `implicit`.

    Confianza = alpha para cada positivo. Para usuarios que no estaban en el entrenamiento (nuevos o de la API)
    los factores se recalculan con un paso de mínimos cuadrados con los factores de las películas fijos (fold-in),
    la misma fórmula que usa `implicit` con `recalculate_user=True`.
    """

    nombre = "ALS implícito"

    def __init__(self, factores=128, regularizacion=0.1, alpha=20.0, iteraciones=15, bm25=False, semilla=C.SEMILLA):
        self.factores, self.regularizacion, self.alpha = factores, regularizacion, alpha
        self.iteraciones, self.bm25, self.semilla = iteraciones, bm25, semilla

    K1, B = 100.0, 0.8

    def _pesos(self, m: sp.csr_matrix) -> sp.csr_matrix:
        """Confianza de cada positivo. Con BM25 (usuarios = documentos, películas = términos) se baja el peso de las
        películas muy populares y de los usuarios que ven de todo. El IDF y el largo medio son los del ENTRENAMIENTO,
        también al hacer fold-in de un usuario nuevo (calcularlos con el lote evaluado sería incorrecto)."""
        m = _filas(m)
        if self.bm25:
            largo = np.diff(m.indptr).astype(np.float32)
            norma = (1 - self.B) + self.B * largo / self.largo_medio
            filas = np.repeat(np.arange(m.shape[0]), np.diff(m.indptr))
            m.data = (m.data * (self.K1 + 1) / (self.K1 * norma[filas] + m.data) * self.idf[m.indices]).astype(np.float32)
        m.data *= self.alpha
        return m

    def preparar_bm25(self, positivos: sp.csr_matrix) -> None:
        x = _filas(positivos)
        df = np.bincount(x.indices, minlength=x.shape[1])
        self.idf = np.maximum(np.log(x.shape[0]) - np.log1p(df), 1e-3).astype(np.float32)
        self.largo_medio = float(np.diff(x.indptr).mean())

    def ajustar(self, positivos, **kw):
        from implicit.als import AlternatingLeastSquares

        modelo = AlternatingLeastSquares(
            factors=self.factores, regularization=self.regularizacion, alpha=1.0, iterations=self.iteraciones,
            random_state=self.semilla, num_threads=C.HILOS, calculate_training_loss=False,
        )
        if self.bm25:
            self.preparar_bm25(positivos)
        from threadpoolctl import threadpool_limits

        with threadpool_limits(1, "blas"):  # sin esto BLAS compite con los hilos de implicit (16 min vs ~1-2 min)
            modelo.fit(self._pesos(positivos), show_progress=False)
        self.factores_usuario = np.asarray(modelo.user_factors, dtype=np.float32)
        self.factores_item = np.asarray(modelo.item_factors, dtype=np.float32)
        self._preparar()
        return self

    def _preparar(self):
        y = self.factores_item
        self.yty = (y.T @ y).astype(np.float64)

    def factores_de(self, historial: sp.csr_matrix) -> np.ndarray:
        """Fold-in: x_u = (YᵀY + Yᵀ(C_u − I)Y + λI)⁻¹ YᵀC_u p_u para cada fila del historial."""
        h = self._pesos(historial)
        y = self.factores_item.astype(np.float64)
        f = y.shape[1]
        salida = np.zeros((h.shape[0], f), dtype=np.float32)
        for fila in range(h.shape[0]):
            ini, fin = h.indptr[fila], h.indptr[fila + 1]
            if ini == fin:
                continue
            idx, conf = h.indices[ini:fin], h.data[ini:fin].astype(np.float64)
            yu = y[idx]
            a = self.yty + (yu.T * (conf - 1.0)) @ yu + self.regularizacion * np.eye(f)
            b = yu.T @ conf
            salida[fila] = sla.solve(a, b, assume_a="pos")
        return salida

    def puntuar(self, historial, usuarios=None):
        x = self.factores_usuario[usuarios] if usuarios is not None else self.factores_de(historial)
        return x @ self.factores_item.T

    def similares(self, item: int, n: int = 10) -> tuple[np.ndarray, np.ndarray]:
        y = self.factores_item / np.maximum(np.linalg.norm(self.factores_item, axis=1, keepdims=True), 1e-9)
        s = y @ y[item]
        s[item] = -np.inf
        top = np.argpartition(-s, n)[:n]
        top = top[np.argsort(-s[top])]
        return top, s[top]


# ---------------------------------------------------------------- referencia fuerte: EASE (solo evaluación)
class EASE(Modelo):
    """EASE^R (Steck 2019): autoencoder lineal item-item con solución cerrada.

    B = I − P·diag(1/diag(P)), P = (XᵀX + λI)⁻¹. Muy fuerte en MovieLens, pero B es catálogo × catálogo
    (17 mil² ≈ 1,2 GB): se usa como referencia en la evaluación, no en la API.
    """

    nombre = "EASE (referencia)"

    def __init__(self, lambda_=500.0):
        self.lambda_ = lambda_

    def ajustar(self, positivos, **kw):
        x = _filas(positivos)
        g = (x.T @ x).toarray().astype(np.float32)
        g[np.diag_indices_from(g)] += self.lambda_
        p = sla.inv(g, overwrite_a=True, check_finite=False)
        del g
        b = p / (-np.diag(p))
        b[np.diag_indices_from(b)] = 0.0
        self.b = b.astype(np.float32)
        return self

    def puntuar(self, historial, usuarios=None):
        return np.asarray(_filas(historial) @ self.b)


# ---------------------------------------------------------------- contenido: géneros + Tag Genome
class Contenido(Modelo):
    """Perfil de contenido: cada película es un vector [Tag Genome reducido con SVD | géneros].

    El perfil del usuario es el promedio de los vectores de lo que le gustó y el puntaje es la similitud coseno,
    más un pequeño término de popularidad (log de positivos estandarizado) que evita recomendar rarezas.
    Funciona sin historial colaborativo: sirve para usuarios nuevos (con 1 película o solo con géneros).
    """

    nombre = "Contenido (géneros + genome)"

    def __init__(self, dimensiones=64, peso_generos=0.5, peso_popularidad=0.1):
        self.dimensiones, self.peso_generos, self.peso_popularidad = dimensiones, peso_generos, peso_popularidad

    def ajustar(self, positivos, generos=None, genome=None, **kw):
        """generos: (n_items × 19) 0/1 · genome: (n_items × 1128) con NaN donde la película no tiene genome."""
        from sklearn.decomposition import TruncatedSVD

        tiene = ~np.isnan(genome).any(axis=1)
        g = np.zeros((len(genome), self.dimensiones), dtype=np.float32)
        centrado = genome[tiene] - genome[tiene].mean(axis=0)
        svd = TruncatedSVD(self.dimensiones, random_state=C.SEMILLA)
        g[tiene] = svd.fit_transform(centrado).astype(np.float32)
        self.varianza_explicada = float(svd.explained_variance_ratio_.sum())
        self.tiene_genome = tiene
        self.set_vectores(g, generos.astype(np.float32), np.asarray(positivos.sum(axis=0)).ravel())
        return self

    def set_vectores(self, genome_red: np.ndarray, generos: np.ndarray, conteo: np.ndarray):
        def normal(m):
            return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)

        self.genome_red, self.generos, self.conteo = genome_red, generos, conteo.astype(np.float32)
        self.vectores_generos = normal(generos)
        w = self.peso_generos
        self.vectores = normal(np.hstack([np.sqrt(1 - w) * normal(genome_red), np.sqrt(w) * self.vectores_generos]))
        lp = np.log1p(self.conteo)
        self.pop_z = ((lp - lp.mean()) / lp.std()).astype(np.float32)

    def puntuar(self, historial, usuarios=None):
        h = _filas(historial)
        perfil = np.asarray(h @ self.vectores)
        perfil /= np.maximum(np.linalg.norm(perfil, axis=1, keepdims=True), 1e-9)
        return perfil @ self.vectores.T + self.peso_popularidad * self.pop_z

    def puntuar_generos(self, generos_elegidos: np.ndarray) -> np.ndarray:
        """Usuario nuevo que solo elige géneros (vector 0/1 de 19): coseno con los géneros de cada película + popularidad."""
        v = np.atleast_2d(generos_elegidos).astype(np.float32)
        v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
        return v @ self.vectores_generos.T + max(self.peso_popularidad, 0.3) * self.pop_z

    def similares(self, item: int, n: int = 10) -> tuple[np.ndarray, np.ndarray]:
        s = self.vectores @ self.vectores[item]
        s[item] = -np.inf
        top = np.argpartition(-s, n)[:n]
        top = top[np.argsort(-s[top])]
        return top, s[top]


# ---------------------------------------------------------------- híbrido
class Hibrido(Modelo):
    """Mezcla ALS, contenido y popularidad reciente con un peso que depende de cuánta historia tiene el usuario.

    puntaje = w·z(ALS) + (1 − w)·z(Contenido) + β·z(log positivos recientes),  w = n / (n + k)
    Con pocos positivos manda el contenido (arranque en frío); con muchos, el colaborativo. El término β corrige
    que ALS y contenido no saben qué está "de moda" ahora (en una partición temporal la gente ve estrenos).
    """

    nombre = "Híbrido (ALS + contenido + recientes)"

    def __init__(self, als: ALS, contenido: Contenido, k: float = 10.0, peso_reciente: float = 0.0,
                 recientes: np.ndarray | None = None):
        self.als, self.contenido, self.k, self.peso_reciente = als, contenido, k, peso_reciente
        r = np.log1p(np.zeros(contenido.vectores.shape[0]) if recientes is None else np.asarray(recientes, np.float32))
        self.reciente_z = ((r - r.mean()) / (r.std() or 1.0)).astype(np.float32)

    def peso_als(self, n_positivos: np.ndarray) -> np.ndarray:
        n = np.asarray(n_positivos, dtype=np.float32)
        return n / (n + self.k)

    def puntuar(self, historial, usuarios=None):
        n = np.diff(historial.indptr)
        w = self.peso_als(n)[:, None]
        return (w * zscore_filas(self.als.puntuar(historial, usuarios))
                + (1 - w) * zscore_filas(self.contenido.puntuar(historial, usuarios))
                + self.peso_reciente * self.reciente_z)


# ---------------------------------------------------------------- utilidades de ranking
def top_k(puntajes: np.ndarray, excluir: sp.csr_matrix | None = None, k: int = C.K) -> np.ndarray:
    """Índices de los k mayores puntajes por fila, sin las películas ya vistas."""
    p = np.array(puntajes, dtype=np.float32, copy=True)
    if excluir is not None:
        filas = np.repeat(np.arange(excluir.shape[0]), np.diff(excluir.indptr))
        p[filas, excluir.indices] = -np.inf
    parte = np.argpartition(-p, k, axis=1)[:, :k]
    orden = np.take_along_axis(p, parte, axis=1).argsort(axis=1)[:, ::-1]
    return np.take_along_axis(parte, orden, axis=1)


class GenerosElegidos(Modelo):
    """Simula al usuario nuevo de la API que solo marca géneros: toma los `n_generos` géneros más frecuentes de lo
    que le gustó en el onboarding y recomienda con `Contenido.puntuar_generos` (no usa qué películas fueron)."""

    nombre = "Contenido solo con géneros elegidos"

    def __init__(self, contenido: Contenido, n_generos: int = 3):
        self.contenido, self.n_generos = contenido, n_generos

    def generos_de(self, historial: sp.csr_matrix) -> np.ndarray:
        cuenta = np.asarray(_filas(historial) @ self.contenido.generos)
        elegidos = np.zeros_like(cuenta)
        top = np.argsort(-cuenta, axis=1)[:, : self.n_generos]
        np.put_along_axis(elegidos, top, 1.0, axis=1)
        return elegidos * (cuenta > 0)

    def puntuar(self, historial, usuarios=None):
        return self.contenido.puntuar_generos(self.generos_de(historial))
