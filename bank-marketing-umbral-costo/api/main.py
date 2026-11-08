"""
API del modelo de priorización de llamadas.

    uvicorn api.main:app --reload
    Documentación interactiva: http://localhost:8000/docs
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from campana.predecir import puntuador  # noqa: E402

app = FastAPI(
    title="¿A quién llamar? · Campaña de depósitos a plazo",
    version="1.0.0",
    description=("Devuelve la probabilidad calibrada de que un cliente contrate un depósito a plazo, si conviene "
                 "llamarlo según el costo de la llamada y el margen del depósito, y las 3 razones principales "
                 "(SHAP). Modelo LightGBM entrenado con UCI Bank Marketing (41 mil clientes, Portugal 2008–2010)."),
)

Ocupacion = Literal["admin.", "blue-collar", "technician", "services", "management", "retired", "entrepreneur",
                    "self-employed", "housemaid", "unemployed", "student", "unknown"]
SiNo = Literal["yes", "no", "unknown"]


class Cliente(BaseModel):
    edad: int = Field(..., ge=17, le=100, examples=[35])
    ocupacion: Ocupacion = Field(..., examples=["admin."])
    estado_civil: Literal["married", "single", "divorced", "unknown"] = Field(..., examples=["single"])
    educacion: Literal["university.degree", "high.school", "basic.9y", "professional.course", "basic.4y",
                       "basic.6y", "illiterate", "unknown"] = Field(..., examples=["university.degree"])
    mora: SiNo = Field("no", description="¿Tiene crédito en mora?")
    hipoteca: SiNo = Field("no")
    prestamo: SiNo = Field("no", description="¿Tiene préstamo personal?")
    tipo_contacto: Literal["cellular", "telephone"] = Field("cellular", description="Celular o teléfono fijo")
    mes: Literal["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"] = Field(..., examples=["may"])
    dia_semana: Literal["mon", "tue", "wed", "thu", "fri"] = Field(..., examples=["thu"])
    dias_desde_contacto: int | None = Field(None, ge=0, le=60,
                                            description="Días desde el último contacto de una campaña anterior "
                                                        "(null si nunca fue contactado)")
    contactos_previos: int = Field(0, ge=0, le=50, description="Contactos en campañas anteriores")
    resultado_previo: Literal["nonexistent", "failure", "success"] = Field("nonexistent")
    var_empleo: float = Field(..., ge=-5, le=5, examples=[-1.8], description="Tasa de variación del empleo (trimestral)")
    ipc: float = Field(..., ge=90, le=97, examples=[92.893], description="Índice de precios al consumidor (mensual)")
    confianza_consumidor: float = Field(..., ge=-60, le=-20, examples=[-46.2])
    euribor_3m: float = Field(..., ge=0, le=6, examples=[1.299])
    num_empleados: float = Field(..., ge=4900, le=5300, examples=[5099.1], description="Miles de empleados (trimestral)")


class Razon(BaseModel):
    variable: str
    factor: str
    valor: str
    efecto_log_odds: float
    sentido: str


class Puntaje(BaseModel):
    probabilidad: float = Field(..., description="Probabilidad calibrada de contratar el depósito")
    llamar: bool
    umbral: float
    valor_esperado_usd: float = Field(..., description="probabilidad × beneficio − costo de la llamada")
    razones: list[Razon]


class ClientePriorizado(Puntaje):
    posicion: int = Field(..., description="1 = el primero a quien llamar")
    indice: int = Field(..., description="Posición del cliente en la lista enviada")


@app.get("/salud")
def salud() -> dict:
    p = puntuador()
    return {"estado": "ok", "modelo": p.metadata["version"], "entrenado": p.metadata["entrenado"]}


@app.get("/modelo")
def modelo() -> dict:
    m = puntuador().metadata
    return {k: m[k] for k in ("version", "entrenado", "modelo", "estrategia_desbalance", "calibracion", "umbral",
                              "supuestos", "excluidas_por_fuga", "filas", "metricas_prueba", "politica_prueba")}


@app.post("/puntuar", response_model=Puntaje)
def puntuar(cliente: Cliente) -> Puntaje:
    return puntuador().puntuar([cliente.model_dump()])[0]


@app.post("/priorizar", response_model=list[ClientePriorizado])
def priorizar(
    clientes: list[Cliente],
    capacidad: int | None = Query(None, ge=1, description="Máximo de llamadas que puede hacer el call center"),
    costo_llamada: float | None = Query(None, gt=0, description="USD por cliente gestionado (por defecto 8)"),
    beneficio_exito: float | None = Query(None, gt=0, description="USD de margen por depósito (por defecto 80)"),
    explicar: Literal["llamar", "todas", "ninguna"] = "llamar",
) -> list[ClientePriorizado]:
    """Ordena hasta 10.000 clientes de mayor a menor probabilidad. Se marca `llamar` a quienes superan el umbral
    de costo/beneficio y, si hay `capacidad`, solo a los primeros `capacidad` de ellos."""
    if not 1 <= len(clientes) <= 10_000:
        raise HTTPException(422, "envía entre 1 y 10.000 clientes")
    if (costo_llamada is None) != (beneficio_exito is None):
        raise HTTPException(422, "indica costo_llamada y beneficio_exito juntos")
    res = puntuador().puntuar([c.model_dump() for c in clientes], explicar_cuales=explicar,
                              costo=costo_llamada, beneficio=beneficio_exito)
    orden = sorted(range(len(res)), key=lambda i: -res[i]["probabilidad"])
    salida = []
    for pos, i in enumerate(orden, start=1):
        r = {**res[i], "posicion": pos, "indice": i}
        if capacidad is not None and pos > capacidad:
            r["llamar"] = False
        salida.append(r)
    return salida
