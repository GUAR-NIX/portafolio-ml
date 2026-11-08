"""Puntuación de transacciones nuevas con los artefactos de models/ (sin torch ni datos crudos).

- Probabilidad del modelo servido (LightGBM) y su versión calibrada con Platt (ajustada en validación).
- Política 1 · umbral por costo: alerta si el puntaje ≥ umbral elegido en validación.
- Política 2 · pérdida esperada: alerta si p_calibrada · fracción · Amount > costo de revisión.
- Señal de anomalía: error de reconstrucción del autoencoder y su percentil entre normales de validación.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import lightgbm as lgb
import numpy as np

from fraude.config import DIR_MODELOS, V_COLS
from fraude.modelos import Autoencoder


@dataclass
class Puntuador:
    booster: lgb.Booster
    autoencoder: Autoencoder
    meta: dict

    @classmethod
    def cargar(cls, dir_modelos: Path = DIR_MODELOS) -> "Puntuador":
        meta = json.loads((dir_modelos / "metadata.json").read_text(encoding="utf-8"))
        if meta["modelo_servido"] == "hibrido":
            raise NotImplementedError("La API sirve un LightGBM sobre V1–V28 + log(monto); el híbrido no está empaquetado.")
        return cls(
            booster=lgb.Booster(model_file=str(dir_modelos / "lgbm_servido.txt")),
            autoencoder=Autoencoder.cargar(dir_modelos / "autoencoder.npz"),
            meta=meta,
        )

    @staticmethod
    def matriz(transacciones: list[dict]) -> np.ndarray:
        X = np.array([[float(t[c]) for c in V_COLS] for t in transacciones], dtype=np.float64).reshape(len(transacciones), len(V_COLS))
        monto = np.array([float(t["Amount"]) for t in transacciones], dtype=np.float64)
        return np.column_stack([X, np.log1p(np.clip(monto, 0, None))])

    def puntuar(self, transacciones: list[dict], costo_revision: float | None = None, fraccion_perdida: float | None = None) -> list[dict]:
        cr = self.meta["costo_revision_eur"] if costo_revision is None else costo_revision
        fp = self.meta["fraccion_perdida"] if fraccion_perdida is None else fraccion_perdida
        X = self.matriz(transacciones)
        monto = np.array([float(t["Amount"]) for t in transacciones])
        p = self.booster.predict(X)
        z = np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))
        p_cal = 1 / (1 + np.exp(-(self.meta["platt"]["a"] * z + self.meta["platt"]["b"])))
        err = self.autoencoder.puntaje(X)
        pct = np.searchsorted(np.asarray(self.meta["percentiles_error_ae_normales_validacion"]), err, side="right") - 1
        perdida = p_cal * fp * monto
        umbral = self.meta["umbral_costo_validacion"]
        return [
            {
                "puntaje_modelo": float(p[i]),
                "probabilidad_fraude_calibrada": float(p_cal[i]),
                "alerta_umbral_costo": bool(p[i] >= umbral),
                "perdida_esperada_eur": round(float(perdida[i]), 4),
                "alerta_perdida_esperada": bool(perdida[i] > cr),
                "error_reconstruccion_ae": float(err[i]),
                "percentil_anomalia_ae": int(np.clip(pct[i], 0, 100)),
            }
            for i in range(len(X))
        ]


@lru_cache(maxsize=1)
def puntuador() -> Puntuador:
    return Puntuador.cargar()
