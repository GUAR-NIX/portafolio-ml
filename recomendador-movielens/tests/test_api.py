"""API y servicio con el paquete de models/ (los archivos con datos de MovieLens se generan con generar_demo.py; sin ellos se omiten)."""
import pytest
from fastapi.testclient import TestClient

from api.main import app
from recomendador import demo
from recomendador.servicio import cargar, normalizar_genero

pytestmark = pytest.mark.skipif(bool(demo.faltantes()), reason=demo.mensaje_faltantes() or "paquete completo")

TOY_STORY, MATRIX, LOTR = 1, 2571, 4993


@pytest.fixture(scope="module")
def cliente():
    return TestClient(app)


@pytest.fixture(scope="module")
def usuario_demo(cliente):
    return cliente.get("/usuarios_demo", params={"n": 50}).json()["usuarios"][10]


def test_salud_y_modelo(cliente):
    s = cliente.get("/salud").json()
    assert s["estado"] == "ok" and s["peliculas"] > 10_000 and s["usuarios_demo"] > 1000
    m = cliente.get("/modelo").json()
    assert {"als", "contenido", "hibrido"} <= set(m["hiperparametros"]) and m["metricas_prueba"]


@pytest.mark.parametrize("metodo", ["hibrido", "als", "contenido", "popular"])
def test_recomendar_usuario_no_repite_lo_visto(cliente, usuario_demo, metodo):
    r = cliente.get(f"/recomendar/{usuario_demo}", params={"n": 20, "metodo": metodo})
    assert r.status_code == 200
    recs = [x["movieId"] for x in r.json()["recomendaciones"]]
    assert len(recs) == 20 and len(set(recs)) == 20
    serv = cargar()
    fila = serv.demo_usuarios[usuario_demo]
    vistos = set(serv.peliculas["movieId"].iloc[serv.demo_vistos[fila].indices])
    assert not vistos & set(recs)


def test_puntajes_ordenados(cliente, usuario_demo):
    p = [x["puntaje"] for x in cliente.get(f"/recomendar/{usuario_demo}").json()["recomendaciones"]]
    assert p == sorted(p, reverse=True)


def test_usuario_desconocido_404(cliente):
    assert cliente.get("/recomendar/999999999").status_code == 404


def test_recomendar_por_historial(cliente):
    r = cliente.post("/recomendar", json={"le_gustaron": [TOY_STORY, 3114, 4886]}).json()  # Toy Story 1-2, Monsters Inc.
    recs = r["recomendaciones"]
    assert len(recs) == 10 and not {TOY_STORY, 3114, 4886} & {x["movieId"] for x in recs}
    # con 3 películas infantiles, la mayoría de lo recomendado es animación o infantil
    infantiles = sum(bool({"Animation", "Children"} & set(x["generos"])) for x in recs)
    assert infantiles >= 6
    assert all("porque_te_gusto" in x for x in recs)
    assert 0 < r["peso_als"] < 1


def test_historial_con_pelicula_inexistente(cliente):
    assert cliente.post("/recomendar", json={"le_gustaron": [123456789]}).status_code == 422
    assert cliente.post("/recomendar", json={"le_gustaron": []}).status_code == 422


def test_usuario_nuevo_por_generos(cliente):
    r = cliente.post("/usuario_nuevo", json={"generos": ["Horror"]})
    assert r.status_code == 200
    recs = r.json()["recomendaciones"]
    assert sum("Horror" in x["generos"] for x in recs) >= 8
    # en español y sin tildes también vale
    r2 = cliente.post("/usuario_nuevo", json={"generos": ["ciencia ficcion", "Animación"]}).json()
    assert r2["generos"] == ["Animation", "Sci-Fi"]
    assert cliente.post("/usuario_nuevo", json={"generos": ["telenovela"]}).status_code == 422


def test_similares(cliente):
    for metodo in ("hibrido", "als", "contenido"):
        r = cliente.get(f"/similares/{LOTR}", params={"metodo": metodo, "n": 10}).json()
        ids = [x["movieId"] for x in r["similares"]]
        assert LOTR not in ids and len(ids) == 10
    # las otras dos de El Señor de los Anillos aparecen entre las similares híbridas
    ids = {x["movieId"] for x in cliente.get(f"/similares/{LOTR}").json()["similares"]}
    assert {5952, 7153} <= ids
    assert cliente.get("/similares/123456789").status_code == 404


def test_buscar(cliente):
    res = cliente.get("/buscar", params={"q": "matrix"}).json()["resultados"]
    assert res[0]["movieId"] == MATRIX


def test_normalizar_genero():
    assert normalizar_genero("Suspenso") == "Thriller" and normalizar_genero("sci-fi") == "Sci-Fi"
    with pytest.raises(ValueError):
        normalizar_genero("xyz")
