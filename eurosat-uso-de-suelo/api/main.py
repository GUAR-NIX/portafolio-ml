"""Servicio FastAPI para clasificar parches satelitales Sentinel-2 (EuroSAT) en CPU."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from eurosat_suelo.config import CLASES_EUROSAT, MACRO_CATEGORIA, NOMBRES_ESPANOL
from eurosat_suelo.predecir import ArtefactoNoEncontrado, cargar_artefactos_inferencia, predecir_parche_satelital

app = FastAPI(
    title="API de Clasificación de Uso de Suelo Satelital (Sentinel-2 · EuroSAT)",
    description="Ensamble MobileNetV3-Small fine-tuned + fusión espectral, inferido con ONNX Runtime en CPU, "
    "calibrado con temperatura en validación y con abstención a fotointérprete SIG.",
    version="2.0.0",
)


class SolicitudClasificacion(BaseModel):
    id_parcela: str = Field(default="PARCELA_001", description="Identificador catastral o de escena")
    archivo_muestra: str | None = Field(
        default=None,
        description="Nombre de archivo en data/muestra_eurosat (p. ej. 01_Forest_01.png)",
    )
    imagen_base64: str | None = Field(
        default=None,
        description="Imagen PNG/JPEG codificada en base64 (alternativa a archivo_muestra)",
    )
    umbral_confianza: float | None = Field(
        default=None,
        ge=0.0,
        le=0.99,
        description="Umbral opcional de confianza para derivar a fotointérprete humano",
    )


class SolicitudLote(BaseModel):
    parcelas: list[SolicitudClasificacion] = Field(..., min_length=1, max_length=50)


def _artefactos_o_503() -> dict[str, Any]:
    try:
        return cargar_artefactos_inferencia()
    except ArtefactoNoEncontrado as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/health")
def estado_servicio() -> dict[str, Any]:
    art = _artefactos_o_503()
    return {
        "estado": "ok",
        "modelo": "M5: MobileNetV3-Small fine-tuned (ONNX) + fusión espectral, calibrado con temperatura",
        "motor_inferencia": "onnxruntime (CPU)",
        "clases_soportadas": len(CLASES_EUROSAT),
        "umbral_abstencion_validacion": art["umbral_abstencion_optimo"],
        "temperaturas_validacion": art["temperaturas"],
    }


@app.get("/clases")
def listar_clases() -> list[dict[str, Any]]:
    return [
        {
            "clase_id": i,
            "clase": c,
            "nombre_es": NOMBRES_ESPANOL[c],
            "macro_categoria": MACRO_CATEGORIA[c],
        }
        for i, c in enumerate(CLASES_EUROSAT)
    ]


@app.post("/clasificar")
def clasificar_parcela(req: SolicitudClasificacion) -> dict[str, Any]:
    fuente = req.archivo_muestra or req.imagen_base64
    if not fuente:
        raise HTTPException(status_code=422, detail="Debe proporcionar 'archivo_muestra' o 'imagen_base64'.")
    _artefactos_o_503()
    try:
        resultado = predecir_parche_satelital(fuente, umbral_confianza=req.umbral_confianza)
    except (ValueError, TypeError, OSError) as exc:
        raise HTTPException(status_code=400, detail=f"Error al procesar la imagen: {exc}") from exc

    return {"id_parcela": req.id_parcela, **resultado}


@app.post("/clasificar_lote")
def clasificar_lote(req: SolicitudLote) -> dict[str, Any]:
    resultados = [clasificar_parcela(p) for p in req.parcelas]
    n_auto = sum(1 for r in resultados if r["decision_operativa"] == "CLASIFICACION_AUTOMATICA")
    return {
        "total_parcelas": len(resultados),
        "clasificadas_automaticamente": n_auto,
        "derivadas_fotointeprete": len(resultados) - n_auto,
        "tasa_automatizacion_pct": round(n_auto / len(resultados) * 100.0, 2),
        "resultados": resultados,
    }
