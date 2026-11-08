"""API FastAPI para estimar probabilidad de actividad, compras futuras, CLV neto y rescate."""
from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, model_validator

from clv import config, predecir

app = FastAPI(
    title="API de Valor de Vida del Cliente (BG/NBD + Gamma-Gamma)",
    description=(
        "Estima la probabilidad de seguir activo P(vivo), compras esperadas, ticket medio contraído "
        "(shrinkage bayesiano), CLV neto descontado y retorno esperado de campañas de retención."
    ),
    version="1.0.0",
)


class SolicitudCliente(BaseModel):
    id_cliente: str = Field(default="12347", description="Identificador del cliente")
    frecuencia: int = Field(
        default=4,
        ge=0,
        le=500,
        description="Número de días con compras repetidas tras la primera compra (x = n_compras - 1)",
    )
    recencia_dias: float = Field(
        default=280.0,
        ge=0.0,
        le=3650.0,
        description="Días entre la primera compra y la última compra del cliente (t_x)",
    )
    edad_dias: float = Field(
        default=400.0,
        ge=1.0,
        le=3650.0,
        description="Días entre la primera compra del cliente y la fecha de corte (T)",
    )
    ticket_medio: float = Field(
        default=380.0,
        gt=0.0,
        le=50000.0,
        description="Importe medio por día de compra en GBP (m_x)",
    )
    horizonte_meses: int = Field(
        default=config.HORIZONTE_MESES,
        ge=1,
        le=36,
        description="Horizonte de proyección del CLV en meses",
    )
    margen_bruto: float = Field(
        default=config.MARGEN_BRUTO,
        gt=0.0,
        le=1.0,
        description="Margen bruto sobre las ventas (ej. 0.30 = 30 %)",
    )
    tasa_descuento_mensual: float = Field(
        default=config.TASA_DESCUENTO_MENSUAL,
        ge=0.0,
        le=0.20,
        description="Tasa de descuento mensual para el flujo de caja (ej. 0.01 = 1 % mensual)",
    )
    costo_contacto: float = Field(
        default=config.COSTO_CONTACTO_RETENCION,
        ge=0.0,
        le=500.0,
        description="Costo de una acción de retención en GBP",
    )

    @model_validator(mode="after")
    def validar_coherencia_btyd(self) -> "SolicitudCliente":
        if self.recencia_dias > self.edad_dias:
            raise ValueError("recencia_dias (t_x) no puede superar a edad_dias (T).")
        if self.frecuencia == 0 and self.recencia_dias > 0:
            raise ValueError("Cuando frecuencia == 0 (sin compras repetidas), recencia_dias debe ser 0.")
        return self


class ItemCartera(BaseModel):
    id_cliente: str = Field(..., description="Identificador del cliente")
    frecuencia: int = Field(..., ge=0, le=500)
    recencia_dias: float = Field(..., ge=0.0, le=3650.0)
    edad_dias: float = Field(..., ge=1.0, le=3650.0)
    ticket_medio: float = Field(..., gt=0.0, le=50000.0)

    @model_validator(mode="after")
    def validar_item(self) -> "ItemCartera":
        if self.recencia_dias > self.edad_dias:
            raise ValueError(f"Cliente {self.id_cliente}: recencia_dias > edad_dias.")
        if self.frecuencia == 0 and self.recencia_dias > 0:
            raise ValueError(f"Cliente {self.id_cliente}: frecuencia == 0 exige recencia_dias == 0.")
        return self


class SolicitudCartera(BaseModel):
    clientes: list[ItemCartera] = Field(..., min_length=1, max_length=5000)


@app.get("/salud")
def salud():
    return {"estado": "ok", "servicio": "clv_bgnbd_gamma_gamma", "version": "1.0.0"}


@app.get("/modelo")
def info_modelo():
    return predecir.cargar_metadata()


@app.post("/valorar")
def valorar(req: SolicitudCliente):
    try:
        res = predecir.valorar_cliente(
            frecuencia=req.frecuencia,
            recencia_dias=req.recencia_dias,
            edad_dias=req.edad_dias,
            ticket_medio=req.ticket_medio,
            horizonte_meses=req.horizonte_meses,
            margen_bruto=req.margen_bruto,
            tasa_descuento_mensual=req.tasa_descuento_mensual,
            costo_contacto=req.costo_contacto,
        )
        res["id_cliente"] = req.id_cliente
        return res
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/priorizar")
def priorizar(
    req: SolicitudCartera,
    modo: Literal["clv", "rescate"] = Query(
        default="clv",
        description="Criterio de ordenamiento: 'clv' (mayor CLV neto) o 'rescate' (mayor beneficio de reactivación)",
    ),
    capacidad: int | None = Query(default=None, ge=1, le=5000, description="Máximo de clientes a seleccionar"),
    horizonte_meses: int = Query(default=config.HORIZONTE_MESES, ge=1, le=36),
    margen_bruto: float = Query(default=config.MARGEN_BRUTO, gt=0.0, le=1.0),
    tasa_descuento_mensual: float = Query(default=config.TASA_DESCUENTO_MENSUAL, ge=0.0, le=0.20),
    costo_contacto: float = Query(default=config.COSTO_CONTACTO_RETENCION, ge=0.0, le=500.0),
):
    lista = [c.model_dump() for c in req.clientes]
    return predecir.priorizar_cartera(
        clientes=lista,
        horizonte_meses=horizonte_meses,
        margen_bruto=margen_bruto,
        tasa_descuento_mensual=tasa_descuento_mensual,
        costo_contacto=costo_contacto,
        modo=modo,
        capacidad=capacidad,
    )
