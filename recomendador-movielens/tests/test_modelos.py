"""Modelos con datos sintéticos de estructura conocida: dos grupos de usuarios con gustos distintos."""
import numpy as np
import pytest
import scipy.sparse as sp

from recomendador import modelos as Mo

N_ITEMS = 40  # items 0-19 = grupo A (Acción), 20-39 = grupo B (Romance)


@pytest.fixture(scope="module")
def datos_sinteticos():
    rng = np.random.default_rng(0)
    filas, cols = [], []
    for u in range(200):
        grupo = 0 if u < 100 else 20
        for i in rng.choice(np.arange(grupo, grupo + 20), size=8, replace=False):
            filas.append(u)
            cols.append(i)
    # algunos items muy populares en cada grupo
    m = sp.csr_matrix((np.ones(len(filas), np.float32), (filas, cols)), shape=(200, N_ITEMS))
    generos = np.zeros((N_ITEMS, 19), np.float32)
    generos[:20, 0] = 1   # Action
    generos[20:, 14] = 1  # Romance
    genome = rng.normal(size=(N_ITEMS, 30)).astype(np.float32)
    genome[:20] += 3
    genome[35:] = np.nan  # películas sin genome: solo géneros
    return m, generos, genome


def proporcion_en_grupo(recs: np.ndarray, grupo_a: bool) -> float:
    return float(((recs < 20) == grupo_a).mean())


def test_top_k_excluye_vistos_y_ordena():
    p = np.array([[0.1, 0.9, 0.5, 0.7]])
    vistos = sp.csr_matrix(np.array([[0, 1, 0, 0]], np.float32))
    assert Mo.top_k(p, vistos, 2).tolist() == [[3, 2]]
    assert Mo.top_k(p, None, 2).tolist() == [[1, 3]]


def test_popularidad(datos_sinteticos):
    m, *_ = datos_sinteticos
    pop = Mo.Popularidad().ajustar(m)
    p = pop.puntuar(m[:3])
    assert p.shape == (3, N_ITEMS) and np.allclose(p[0], p[2])
    assert np.argmax(p[0]) == np.argmax(np.asarray(m.sum(axis=0)).ravel())
    rec = Mo.PopularidadReciente().ajustar(m, recientes=np.arange(N_ITEMS, dtype=np.float32))
    assert np.argmax(rec.puntuar(m[:1])[0]) == N_ITEMS - 1


@pytest.fixture(scope="module")
def als(datos_sinteticos):
    m, *_ = datos_sinteticos
    return Mo.ALS(factores=4, regularizacion=1.0, alpha=10, iteraciones=20).ajustar(m)


def test_als_respeta_los_grupos(datos_sinteticos, als):
    m, *_ = datos_sinteticos
    recs = Mo.top_k(als.puntuar(m[:5], np.arange(5)), m[:5], 5)
    assert proporcion_en_grupo(recs, True) == 1.0
    recs_b = Mo.top_k(als.puntuar(m[150:155], np.arange(150, 155)), m[150:155], 5)
    assert proporcion_en_grupo(recs_b, False) == 1.0


def test_fold_in_aproxima_factores_entrenados(datos_sinteticos, als):
    m, *_ = datos_sinteticos
    x = als.factores_de(m[:10])
    corr = [np.corrcoef(x[i], als.factores_usuario[i])[0, 1] for i in range(10)]
    assert np.mean(corr) > 0.95
    # un usuario nuevo con tres películas del grupo B recibe recomendaciones del grupo B
    nuevo = sp.csr_matrix(([1.0] * 3, ([0] * 3, [21, 25, 30])), shape=(1, N_ITEMS))
    assert proporcion_en_grupo(Mo.top_k(als.puntuar(nuevo), nuevo, 5), False) == 1.0
    # historial vacío → factores en cero, sin error
    vacio = sp.csr_matrix((1, N_ITEMS), dtype=np.float32)
    assert np.allclose(als.factores_de(vacio), 0)


def test_als_similares(als):
    top, sim = als.similares(3, 5)
    assert 3 not in top and (top < 20).all() and np.all(np.diff(sim) <= 1e-6)


def test_ease_respeta_los_grupos(datos_sinteticos):
    m, *_ = datos_sinteticos
    ease = Mo.EASE(lambda_=10).ajustar(m)
    assert np.allclose(np.diag(ease.b), 0)
    recs = Mo.top_k(ease.puntuar(m[:5]), m[:5], 5)
    assert proporcion_en_grupo(recs, True) == 1.0


@pytest.fixture(scope="module")
def contenido(datos_sinteticos):
    m, gen, gnm = datos_sinteticos
    return Mo.Contenido(dimensiones=4, peso_generos=0.5, peso_popularidad=0.0).ajustar(m, generos=gen, genome=gnm)


def test_contenido_sin_historia_colaborativa(datos_sinteticos, contenido):
    _, gen, _ = datos_sinteticos
    nuevo = sp.csr_matrix(([1.0], ([0], [36])), shape=(1, N_ITEMS))  # película B sin genome
    assert proporcion_en_grupo(Mo.top_k(contenido.puntuar(nuevo), nuevo, 5), False) == 1.0
    por_genero = contenido.puntuar_generos(gen[0])  # elige "Action"
    assert proporcion_en_grupo(Mo.top_k(por_genero, None, 10), True) == 1.0
    top, _ = contenido.similares(2, 5)
    assert (top < 20).all()


def test_hibrido_peso_segun_historia(als, contenido, datos_sinteticos):
    m, *_ = datos_sinteticos
    h = Mo.Hibrido(als, contenido, k=10)
    assert h.peso_als([0])[0] == 0 and h.peso_als([10])[0] == pytest.approx(0.5) and h.peso_als([990])[0] > 0.98
    p = h.puntuar(m[:3], np.arange(3))
    assert p.shape == (3, N_ITEMS) and np.isfinite(p).all()
    assert proporcion_en_grupo(Mo.top_k(p, m[:3], 5), True) == 1.0


def test_generos_elegidos(contenido, datos_sinteticos):
    m, *_ = datos_sinteticos
    g = Mo.GenerosElegidos(contenido, n_generos=1)
    elegidos = g.generos_de(m[150:152])
    assert elegidos[:, 14].tolist() == [1, 1] and elegidos.sum() == 2
    assert proporcion_en_grupo(Mo.top_k(g.puntuar(m[150:152]), m[150:152], 5), False) == 1.0


def test_zscore_filas():
    z = Mo.zscore_filas(np.array([[1.0, 2.0, 3.0], [5.0, 5.0, 5.0]]))
    assert np.allclose(z[0].mean(), 0) and np.allclose(z[0].std(), 1) and np.allclose(z[1], 0)


def test_bm25_fold_in_usa_idf_del_entrenamiento(datos_sinteticos):
    m, *_ = datos_sinteticos
    als = Mo.ALS(factores=4, regularizacion=1.0, alpha=2, iteraciones=10, bm25=True).ajustar(m)
    # el peso de un usuario no depende de con quién se evalúe en el lote
    solo = als._pesos(m[[3]]).toarray()
    en_lote = als._pesos(m[[3, 150, 151]]).toarray()[0]
    assert np.allclose(solo[0], en_lote)
    # las películas más populares pesan menos
    df = np.asarray(m.sum(axis=0)).ravel()
    assert als.idf[np.argmax(df)] < als.idf[np.argmin(df)]
    x = als.factores_de(m[:10])
    assert np.mean([np.corrcoef(x[i], als.factores_usuario[i])[0, 1] for i in range(10)]) > 0.95
