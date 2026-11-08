"""API FastAPI de pronóstico de demanda de bicicletas por estación y hora.

Endpoints:
- GET /salud                     → estado del servicio.
- GET /estaciones                → estaciones disponibles.
- GET /pronostico/{estacion}?fecha=YYYY-MM-DD → pronóstico horario del día.

El modelo servido es el de horizonte de negocio ("día siguiente"): para cada hora
del día pedido solo usa información disponible a la medianoche anterior
(rezagos de 24 h o más) más calendario y clima.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from demanda_citibike.clima import cargar_clima
from demanda_citibike.config import MODELOS, PROCESADOS
from demanda_citibike.features import FEATURES_DIA, construir_caracteristicas
from demanda_citibike.modelos import cargar_modelo

app = FastAPI(title="Pronóstico de demanda Citi Bike", version="1.0.0")

_ESTADO: dict = {"modelo": None, "features": None, "meta": None, "cargado": False}


def cargar_todo() -> None:
    """Carga modelo y características una sola vez (caché en memoria)."""
    if _ESTADO["cargado"]:
        return
    ruta_modelo = MODELOS / "modelo_lgbm.joblib"
    ruta_demanda = PROCESADOS / "demanda_horaria.parquet"
    ruta_estaciones = PROCESADOS / "estaciones.csv"
    faltan = [r.name for r in (ruta_modelo, ruta_demanda, ruta_estaciones) if not r.exists()]
    if faltan:
        raise HTTPException(
            status_code=503,
            detail=f"Faltan {', '.join(faltan)}. Ejecuta `python descargar_datos.py` y "
                   "`python -m demanda_citibike.pipeline` para generarlos.")
    _ESTADO["modelo"] = cargar_modelo(ruta_modelo)
    demanda = pd.read_parquet(ruta_demanda)
    clima = cargar_clima()
    _ESTADO["features"] = construir_caracteristicas(demanda, clima)
    _ESTADO["meta"] = pd.read_csv(ruta_estaciones, dtype={"estacion": str})
    _ESTADO["cargado"] = True


@app.get("/salud")
def salud() -> dict:
    try:
        cargar_todo()
        return {"estado": "ok", "estaciones": int(_ESTADO["meta"]["estacion"].nunique())}
    except HTTPException as e:
        return {"estado": "no_listo", "detalle": e.detail}


@app.get("/estaciones")
def estaciones() -> list[dict]:
    cargar_todo()
    return _ESTADO["meta"].to_dict(orient="records")


@app.get("/pronostico/{estacion}")
def pronostico(estacion: str, fecha: str = Query(..., description="Fecha YYYY-MM-DD")) -> dict:
    cargar_todo()
    try:
        dia = pd.to_datetime(fecha).date()
    except Exception:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Formato de fecha inválido. Usa YYYY-MM-DD.")

    df = _ESTADO["features"]
    sub = df[(df["estacion"] == estacion) & (df["fecha_hora"].dt.date == dia)]
    if sub.empty:
        raise HTTPException(status_code=404, detail=f"Sin datos para la estación {estacion} en {fecha}.")

    sub = sub.sort_values("fecha_hora")
    pred = np.clip(_ESTADO["modelo"].predict(sub[FEATURES_DIA]), 0, None)
    salida = []
    for (_, fila), p in zip(sub.iterrows(), pred):
        salida.append({
            "fecha_hora": str(fila["fecha_hora"]),
            "pronostico_salidas": round(float(p), 1),
            "salidas_reales": int(fila["salidas"]),
        })
    return {
        "estacion": estacion,
        "fecha": fecha,
        "horizonte": "día siguiente (solo información hasta la medianoche anterior)",
        "horas": salida,
        "total_pronosticado": round(float(sum(pred)), 0),
        "total_real": int(sub["salidas"].sum()),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
