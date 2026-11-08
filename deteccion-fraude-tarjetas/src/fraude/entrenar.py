"""Entrenamiento: partición temporal → líneas base → no supervisados → LightGBM → híbrido.

Uso:  python -m fraude.entrenar                    (entrena ≈ 2 min y luego llama a fraude.evaluar)
      python -m fraude.entrenar --solo-entrenar

Guarda los puntajes de validación y prueba de los 10 modelos en reports/predicciones_*.parquet y el
modelo servido en models/. Todo lo que se *decide* aquí (parada temprana, esquema de desbalance, modelo
servido) usa validación; la prueba solo se puntúa para que `fraude.evaluar` la reporte.
"""

from __future__ import annotations

import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd

from fraude import metricas as M
from fraude import modelos as MO
from fraude.config import (
    CANDIDATOS_SERVIDO,
    DIR_FIGURAS,
    DIR_MODELOS,
    DIR_REPORTES,
    ETIQUETAS,
    RAZON_SUBMUESTREO,
)
from fraude.datos import cargar, matriz, particion_temporal


def main(evaluar: bool = True) -> dict:
    t_inicio = time.time()
    for d in (DIR_MODELOS, DIR_REPORTES, DIR_FIGURAS):
        d.mkdir(parents=True, exist_ok=True)

    part = particion_temporal(cargar())
    tr, va, te = part.entrenamiento, part.validacion, part.prueba
    print(part.resumen().to_string(index=False))
    X_tr, y_tr = matriz(tr)
    X_va, y_va = matriz(va)
    X_te, y_te = matriz(te)
    normales_tr = X_tr[y_tr == 0]

    s_va: dict[str, np.ndarray] = {}
    s_te: dict[str, np.ndarray] = {}
    tiempos: dict[str, dict] = {}

    def registrar(nombre, f_puntaje, t_ent):
        t0 = time.time()
        s_va[nombre], s_te[nombre] = f_puntaje(X_va), f_puntaje(X_te)
        tiempos[nombre] = {"entrenamiento_s": round(t_ent, 2),
                           "puntuar_prueba_ms_por_1000": round((time.time() - t0) / (len(X_va) + len(X_te)) * 1e6, 3)}
        ev = M.evaluar(y_va, s_va[nombre])
        print(f"  {ETIQUETAS[nombre]:<36} AUPRC val {ev['auprc']:.4f} · ROC {ev['roc_auc']:.4f}")

    print("\n[1] Líneas base")
    registrar("regla_monto", MO.puntaje_regla_monto, 0.0)
    t0 = time.time(); logit = MO.entrenar_logistica(X_tr, y_tr)
    registrar("logistica", lambda X: logit.predict_proba(X)[:, 1], time.time() - t0)

    print("\n[2] No supervisados")
    t0 = time.time(); if_todo = MO.entrenar_isolation_forest(X_tr)
    registrar("if_sin_etiquetas", lambda X: MO.puntaje_isolation_forest(if_todo, X), time.time() - t0)
    t0 = time.time(); if_norm = MO.entrenar_isolation_forest(normales_tr)
    registrar("if_normales", lambda X: MO.puntaje_isolation_forest(if_norm, X), time.time() - t0)
    t0 = time.time(); lof = MO.entrenar_lof(normales_tr)
    registrar("lof", lambda X: MO.puntaje_lof(lof, X), time.time() - t0)
    t0 = time.time(); ae = MO.entrenar_autoencoder(normales_tr)
    registrar("autoencoder", ae.puntaje, time.time() - t0)
    pd.DataFrame(ae.historial).to_csv(DIR_REPORTES / "historial_autoencoder.csv", index=False)

    print("\n[3] LightGBM: sin ajuste vs pesos vs submuestreo")
    razon = float((y_tr == 0).sum() / (y_tr == 1).sum())
    lgbms = {}
    t0 = time.time(); lgbms["lgbm_sin_ajuste"] = MO.entrenar_lgbm(X_tr, y_tr, X_va, y_va)
    registrar("lgbm_sin_ajuste", lambda X: MO.puntaje_sup(lgbms["lgbm_sin_ajuste"], X), time.time() - t0)
    t0 = time.time(); lgbms["lgbm_pesos"] = MO.entrenar_lgbm(X_tr, y_tr, X_va, y_va, peso_positivos=razon)
    registrar("lgbm_pesos", lambda X: MO.puntaje_sup(lgbms["lgbm_pesos"], X), time.time() - t0)
    t0 = time.time()
    Xs, ys = MO.submuestrear(X_tr, y_tr, RAZON_SUBMUESTREO)
    lgbms["lgbm_submuestreo"] = MO.entrenar_lgbm(Xs, ys, X_va, y_va, min_child_samples=10)
    registrar("lgbm_submuestreo", lambda X: MO.puntaje_sup(lgbms["lgbm_submuestreo"], X), time.time() - t0)
    auprc_val = {m: M.evaluar(y_va, s_va[m])["auprc"] for m in lgbms}
    mejor_esquema = max(auprc_val, key=auprc_val.get)
    print(f"  Esquema de desbalance elegido en validación: {mejor_esquema}")

    print("\n[4] Híbrido: puntajes de anomalía (fuera de pliegue) como variables")
    t0 = time.time()
    oof_if, oof_ae = MO.puntajes_anomalia_oof(X_tr, y_tr)
    Xh_tr = MO.con_anomalias(X_tr, oof_if, oof_ae)
    Xh_va = MO.con_anomalias(X_va, MO.puntaje_isolation_forest(if_norm, X_va), ae.puntaje(X_va))
    if mejor_esquema == "lgbm_submuestreo":
        Xhs, yhs = MO.submuestrear(Xh_tr, y_tr, RAZON_SUBMUESTREO)
        hibrido = MO.entrenar_lgbm(Xhs, yhs, Xh_va, y_va, min_child_samples=10)
    else:
        hibrido = MO.entrenar_lgbm(Xh_tr, y_tr, Xh_va, y_va, peso_positivos=razon if mejor_esquema == "lgbm_pesos" else 1.0)
    registrar("hibrido", lambda X: MO.puntaje_sup(hibrido, MO.con_anomalias(X, MO.puntaje_isolation_forest(if_norm, X), ae.puntaje(X))),
              time.time() - t0)

    servido = max(CANDIDATOS_SERVIDO, key=lambda m: M.evaluar(y_va, s_va[m])["auprc"])
    print(f"\nModelo servido (mejor AUPRC en validación): {ETIQUETAS[servido]}")

    # ---------------- artefactos
    for nombre, d, s in [("validacion", va, s_va), ("prueba", te, s_te)]:
        pd.DataFrame({"Time": d["Time"].to_numpy(), "Amount": d["Amount"].to_numpy(), "Class": d["Class"].to_numpy(),
                      **{f"s_{m}": s[m] for m in ETIQUETAS}}).to_parquet(DIR_REPORTES / f"predicciones_{nombre}.parquet", index=False)
    modelo_final = hibrido if servido == "hibrido" else lgbms[servido]
    modelo_final.booster_.save_model(str(DIR_MODELOS / "lgbm_servido.txt"))
    if servido == "hibrido":  # el híbrido necesita el Isolation Forest para construir sus variables
        joblib.dump(if_norm, DIR_MODELOS / "isolation_forest.joblib", compress=3)
    ae.guardar(DIR_MODELOS / "autoencoder.npz")
    entrenamiento = {
        "esquema_desbalance_elegido": mejor_esquema,
        "razon_normales_por_fraude_entrenamiento": razon,
        "razon_submuestreo": RAZON_SUBMUESTREO,
        "modelo_servido": servido,
        "autoencoder": {"epocas": len(ae.historial), "mse_normales_reservadas": float(min(h["mse_validacion_normales"] for h in ae.historial))},
        "tiempos": tiempos,
        "segundos_entrenamiento": round(time.time() - t_inicio, 1),
    }
    (DIR_REPORTES / "entrenamiento.json").write_text(json.dumps(entrenamiento, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ok] Entrenamiento en {entrenamiento['segundos_entrenamiento']} s")

    if evaluar:
        from fraude import evaluar as EV

        return EV.main()
    return entrenamiento


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo-entrenar", action="store_true")
    main(evaluar=not ap.parse_args().solo_entrenar)
