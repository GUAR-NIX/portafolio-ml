"""API del recomendador.

    uvicorn api.main:app --reload        # documentación interactiva en http://localhost:8000/docs
"""
from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from recomendador import __version__
from recomendador.demo import PaqueteIncompleto
from recomendador.servicio import GENEROS_ES, PeliculaDesconocida, UsuarioDesconocido, cargar

app = FastAPI(
    title="Recomendador de películas · MovieLens 25M",
    version=__version__,
    description="ALS implícito + contenido (géneros y Tag Genome) + popularidad reciente, con arranque en frío.",
)



@app.exception_handler(PaqueteIncompleto)
def paquete_incompleto(_: Request, e: PaqueteIncompleto):
    """Faltan los archivos con datos de MovieLens (no se redistribuyen): mensaje claro en vez de un error 500."""
    return JSONResponse(status_code=503, content={"detail": str(e)})


Metodo = Literal["hibrido", "als", "contenido", "popular"]


class Historial(BaseModel):
    le_gustaron: list[int] = Field(..., min_length=1, max_length=2000, description="movieId que el usuario calificó con ≥ 4",
                                   examples=[[1, 260, 1196, 1210, 2571]])
    vistas: list[int] = Field(default_factory=list, max_length=5000, description="otras películas ya vistas (se excluyen)")


class Generos(BaseModel):
    generos: list[str] = Field(..., min_length=1, max_length=19, examples=[["Sci-Fi", "Animación"]],
                               description="nombres de MovieLens o en español")
    excluir: list[int] = Field(default_factory=list, max_length=5000)


@app.get("/salud")
def salud():
    r = cargar()
    return {"estado": "ok", "version": __version__, "peliculas": r.n, "usuarios_demo": len(r.demo_usuarios)}


@app.get("/modelo")
def modelo():
    m = cargar().metadata
    return {"version": m["version"], "hiperparametros": m["hiperparametros"], "corte_prueba": m["corte_prueba"],
            "modelo_final": m["modelo_final"], "metricas_prueba": m["metricas_prueba"]}


@app.get("/usuarios_demo")
def usuarios_demo(n: int = Query(20, ge=1, le=2000)):
    """Usuarios incluidos en el paquete liviano (los demás se atienden con POST /recomendar)."""
    return {"usuarios": cargar().usuarios_demo(n)}


@app.get("/recomendar/{usuario}")
def recomendar_usuario(usuario: int, n: int = Query(10, ge=1, le=100), metodo: Metodo = "hibrido"):
    try:
        return cargar().recomendar_usuario(usuario, n, metodo)
    except UsuarioDesconocido:
        raise HTTPException(404, f"usuario {usuario} no está en el paquete de demostración; "
                                 "usa GET /usuarios_demo o POST /recomendar con su historial")


@app.post("/recomendar")
def recomendar_historial(h: Historial, n: int = Query(10, ge=1, le=100), metodo: Metodo = "hibrido"):
    """Cualquier usuario (también uno nuevo con pocas películas): ALS por fold-in, sin re-entrenar."""
    try:
        return cargar().recomendar_historial(h.le_gustaron, n, metodo, h.vistas)
    except PeliculaDesconocida as e:
        raise HTTPException(422, str(e.args[0]))


@app.post("/usuario_nuevo")
def usuario_nuevo(g: Generos, n: int = Query(10, ge=1, le=100)):
    """Arranque en frío: el usuario solo marcó géneros al registrarse."""
    try:
        return cargar().usuario_nuevo(g.generos, n, g.excluir)
    except (ValueError, PeliculaDesconocida) as e:
        raise HTTPException(422, str(e.args[0]))


@app.get("/similares/{pelicula}")
def similares(pelicula: int, n: int = Query(10, ge=1, le=100),
              metodo: Literal["hibrido", "als", "contenido"] = "hibrido"):
    try:
        return cargar().similares(pelicula, n, metodo)
    except PeliculaDesconocida:
        raise HTTPException(404, f"la película {pelicula} no está en el catálogo recomendable (≥ 5 positivos)")


@app.get("/buscar")
def buscar(q: str = Query(..., min_length=2), n: int = Query(10, ge=1, le=50)):
    return {"resultados": cargar().buscar(q, n)}


@app.get("/generos")
def generos():
    return GENEROS_ES
