"""API de puntuación de fraude.

Arranque:  uvicorn api.main:app --port 8000
Docs:      http://127.0.0.1:8000/docs
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, create_model

from fraude.config import V_COLS
from fraude.predecir import puntuador

app = FastAPI(
    title="Detección de fraude con tarjetas (ULB)",
    version="1.0.0",
    description="Puntúa transacciones con el LightGBM servido. Devuelve dos políticas de alerta: umbral por costo "
    "(elegido en validación) y pérdida esperada (p calibrada × monto > costo de revisar).",
)

_campos = {c: (float, Field(..., description=f"Componente PCA anónima {c}")) for c in V_COLS}
Transaccion = create_model(
    "Transaccion",
    **_campos,
    Amount=(float, Field(..., ge=0, description="Monto de la transacción en EUR")),
)


class Solicitud(BaseModel):
    transacciones: list[Transaccion] = Field(..., min_length=1, max_length=1000)  # type: ignore[valid-type]
    costo_revision_eur: float | None = Field(None, gt=0, description="Opcional: reemplaza el supuesto de 5 EUR por alerta")
    fraccion_perdida: float | None = Field(None, gt=0, le=1, description="Opcional: fracción del monto que se pierde si no se detecta")


class Resultado(BaseModel):
    puntaje_modelo: float
    probabilidad_fraude_calibrada: float
    alerta_umbral_costo: bool
    perdida_esperada_eur: float
    alerta_perdida_esperada: bool
    error_reconstruccion_ae: float
    percentil_anomalia_ae: int


class Respuesta(BaseModel):
    modelo: str
    umbral_costo_validacion: float
    costo_revision_eur: float
    resultados: list[Resultado]


@app.get("/salud")
def salud() -> dict:
    try:
        meta = puntuador().meta
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Faltan artefactos en models/: {e}") from e
    return {"estado": "ok", "modelo": meta["nombre_servido"]}


@app.post("/puntuar", response_model=Respuesta)
def puntuar(sol: Solicitud) -> Respuesta:
    p = puntuador()
    filas = [t.model_dump() for t in sol.transacciones]
    res = p.puntuar(filas, sol.costo_revision_eur, sol.fraccion_perdida)
    return Respuesta(
        modelo=p.meta["nombre_servido"],
        umbral_costo_validacion=p.meta["umbral_costo_validacion"],
        costo_revision_eur=sol.costo_revision_eur or p.meta["costo_revision_eur"],
        resultados=[Resultado(**r) for r in res],
    )
