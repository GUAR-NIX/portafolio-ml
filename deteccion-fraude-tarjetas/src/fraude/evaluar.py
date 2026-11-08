"""Evaluación completa a partir de lo que dejó `fraude.entrenar` (no reentrena los modelos guardados).

Uso:  python -m fraude.evaluar                    (≈ 5 min: incluye la curva de pocas etiquetas)
      python -m fraude.evaluar --sin-pocas-etiquetas

Lee reports/predicciones_{validacion,prueba}.parquet, reports/entrenamiento.json y models/, y escribe:
métricas con IC bootstrap, diferencias pareadas, costo (umbral elegido en validación), curvas de
costo, sensibilidad, curva de pocas etiquetas, SHAP, deriva, models/metadata.json y reports/resultados.json.
"""

from __future__ import annotations

import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from fraude import costo as C
from fraude import deriva as D
from fraude import metricas as M
from fraude import modelos as MO
from fraude.config import (
    ALERTAS_POR_DIA,
    CANDIDATOS_SERVIDO,
    COLUMNAS_MODELO,
    COSTO_REVISION,
    COSTO_REVISION_SENSIBILIDAD,
    DIR_MODELOS,
    DIR_MUESTRA,
    DIR_REPORTES,
    ETIQUETAS,
    FAMILIA,
    FRACCION_PERDIDA,
    FRACCION_PERDIDA_SENSIBILIDAD,
    N_BOOTSTRAP,
    N_FRAUDES_POCAS_ETIQUETAS,
    NO_SUPERVISADOS,
    REPETICIONES_POCAS_ETIQUETAS,
    SEMILLA,
)
from fraude.datos import cargar, matriz, particion_temporal

MODELOS = list(ETIQUETAS)


# ---------------------------------------------------------------- calibración
def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def calibrar_platt(puntaje_val: np.ndarray, y_val: np.ndarray) -> tuple[float, float]:
    """Calibración de Platt sobre el logit del puntaje (2 parámetros: robusta con solo 57 fraudes en validación)."""
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(puntaje_val).reshape(-1, 1), y_val)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def aplicar_platt(puntaje: np.ndarray, a: float, b: float) -> np.ndarray:
    return 1 / (1 + np.exp(-(a * _logit(puntaje) + b)))


# ---------------------------------------------------------------- utilidades
def leer_predicciones(particion: str) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    p = pd.read_parquet(DIR_REPORTES / f"predicciones_{particion}.parquet")
    return p, {m: p[f"s_{m}"].to_numpy() for m in MODELOS}


def capacidades(resumen_part: pd.DataFrame) -> tuple[int, int]:
    """Alertas revisables en cada ventana: ALERTAS_POR_DIA × horas / 24."""
    h = resumen_part.set_index("particion")["horas"]
    return int(round(ALERTAS_POR_DIA * h["validacion"] / 24)), int(round(ALERTAS_POR_DIA * h["prueba"] / 24))


def elegir_servido(met_va: pd.DataFrame) -> str:
    return max(CANDIDATOS_SERVIDO, key=lambda m: met_va.set_index("modelo").loc[m, "auprc"])


def curva_pocas_etiquetas(X_tr, y_tr, X_te, y_te) -> pd.DataFrame:
    """Supervisados con solo n fraudes etiquetados (y todas las normales), repetidos con 10 semillas.

    Sin parada temprana: con pocas etiquetas tampoco habría una validación etiquetada confiable.
    """
    fraudes, normales = np.flatnonzero(y_tr == 1), np.flatnonzero(y_tr == 0)
    filas = []
    for n in N_FRAUDES_POCAS_ETIQUETAS:
        reps = 1 if n >= len(fraudes) else REPETICIONES_POCAS_ETIQUETAS
        for r in range(reps):
            rng = np.random.default_rng(SEMILLA + 1000 * n + r)
            sel = rng.choice(fraudes, size=min(n, len(fraudes)), replace=False)
            idx = np.sort(np.r_[normales, sel])
            y = y_tr[idx]
            peso = (y == 0).sum() / (y == 1).sum()
            ajustes = {
                "Regresión logística": lambda: MO.entrenar_logistica(X_tr[idx], y),
                "LightGBM con pesos": lambda: MO.entrenar_lgbm(X_tr[idx], y, peso_positivos=peso, n_estimadores=300, semilla=SEMILLA + r),
            }
            for nombre, ajustar in ajustes.items():
                ev = M.evaluar(y_te, ajustar().predict_proba(X_te)[:, 1])
                filas.append({"n_fraudes": int(len(sel)), "repeticion": r, "modelo": nombre,
                              "auprc_prueba": ev["auprc"], "roc_auc_prueba": ev["roc_auc"]})
        print(f"  [pocas etiquetas] n = {n} listo")
    return pd.DataFrame(filas)


def resumen_pocas_etiquetas(pe: pd.DataFrame, ref: float) -> tuple[pd.DataFrame, dict]:
    g = pe.groupby(["modelo", "n_fraudes"])["auprc_prueba"].agg(media="mean", desv="std", minimo="min").reset_index()
    cruces = {}
    for mod, sub in g.groupby("modelo"):
        sub = sub.sort_values("n_fraudes")
        supera = (sub["media"] > ref).to_numpy()
        primero = int(sub["n_fraudes"].iloc[supera.argmax()]) if supera.any() else None
        # estable: desde ese n, *todas* las cantidades mayores también superan la referencia
        estable = next((int(n) for i, n in enumerate(sub["n_fraudes"]) if supera[i:].all()), None)
        cruces[mod] = {"primer_n": primero, "n_estable": estable}
    return g, cruces


# ---------------------------------------------------------------- principal
def main(pocas_etiquetas: bool = True) -> dict:
    t0 = time.time()
    entrenamiento = json.loads((DIR_REPORTES / "entrenamiento.json").read_text(encoding="utf-8"))
    df = cargar()
    part = particion_temporal(df)
    resumen_part = part.resumen()
    resumen_part.to_csv(DIR_REPORTES / "particion_temporal.csv", index=False)
    k_val, k_te = capacidades(resumen_part)
    pred_va, s_va = leer_predicciones("validacion")
    pred_te, s_te = leer_predicciones("prueba")
    y_va, y_te = pred_va["Class"].to_numpy(), pred_te["Class"].to_numpy()
    monto_va, monto_te = pred_va["Amount"].to_numpy(), pred_te["Amount"].to_numpy()
    assert len(pred_te) == len(part.prueba) and int(y_te.sum()) == int(part.prueba["Class"].sum())

    # ---------------- métricas e IC
    print("[1] Métricas en validación y prueba (IC bootstrap pareado)")
    met_va = pd.DataFrame([{"modelo": m, **M.evaluar(y_va, s_va[m], k=k_val)} for m in MODELOS])
    met_va.to_csv(DIR_REPORTES / "metricas_validacion.csv", index=False)
    servido = elegir_servido(met_va)
    assert servido == entrenamiento["modelo_servido"], "El modelo servido no coincide con el elegido al entrenar"
    mejor_esquema = entrenamiento["esquema_desbalance_elegido"]
    mejor_no_sup = max(NO_SUPERVISADOS, key=lambda m: met_va.set_index("modelo").loc[m, "auprc"])

    met_te = pd.DataFrame([{"modelo": m, **M.evaluar(y_te, s_te[m], k=k_te)} for m in MODELOS])
    boot = M.bootstrap(y_te, s_te, k=k_te, n_rep=N_BOOTSTRAP)
    assert not boot.drop(columns=["replica", "modelo"]).isna().any().any(), "Réplicas bootstrap con NaN"
    ic = M.intervalos(boot).pivot(index="modelo", columns="metrica", values=["ic_inf", "ic_sup"])
    ic.columns = [f"{met}_{lim}" for lim, met in ic.columns]
    met_te = met_te.merge(ic.reset_index(), on="modelo")
    met_te.insert(1, "nombre", met_te["modelo"].map(ETIQUETAS))
    met_te.insert(2, "familia", met_te["modelo"].map(FAMILIA))
    met_te["auprc_validacion"] = met_te["modelo"].map(met_va.set_index("modelo")["auprc"])
    met_te.to_csv(DIR_REPORTES / "metricas_prueba.csv", index=False)

    pares = [
        ("hibrido", mejor_esquema), (servido, mejor_no_sup), (servido, "logistica"),
        ("lgbm_pesos", "lgbm_submuestreo"), ("lgbm_pesos", "lgbm_sin_ajuste"), ("lgbm_submuestreo", "lgbm_sin_ajuste"),
        ("logistica", mejor_no_sup), ("lof", "autoencoder"), ("autoencoder", "if_normales"), ("if_normales", "if_sin_etiquetas"),
    ]
    pares = list(dict.fromkeys((a, b) for a, b in pares if a != b))
    dif = pd.concat([M.diferencias_pareadas(boot, pares, "auprc"), M.diferencias_pareadas(boot, pares, f"precision_at_{k_te}")])
    dif.to_csv(DIR_REPORTES / "bootstrap_diferencias.csv", index=False)
    print(met_te[["nombre", "auprc", "auprc_ic_inf", "auprc_ic_sup", "roc_auc", f"precision_at_{k_te}"]].round(4).to_string(index=False))

    # ---------------- costo
    print("\n[2] Costo: umbral elegido en validación, reportado en prueba")
    filas, alertas_te, umbrales = [], {}, {}
    for m in MODELOS:
        u, _ = C.elegir_umbral(y_va, monto_va, s_va[m], COSTO_REVISION, FRACCION_PERDIDA)
        umbrales[m] = u
        alertas_te[m] = s_te[m] >= u
        d_va = C.desglose(y_va, monto_va, s_va[m] >= u, COSTO_REVISION, FRACCION_PERDIDA)
        filas.append({"modelo": m, "nombre": ETIQUETAS[m], "politica": "umbral por costo en validación", "umbral": u,
                      "costo_validacion_eur": d_va["costo_total_eur"],
                      **C.desglose(y_te, monto_te, alertas_te[m], COSTO_REVISION, FRACCION_PERDIDA)})
    a, b = calibrar_platt(s_va[servido], y_va)
    p_va_cal, p_te_cal = aplicar_platt(s_va[servido], a, b), aplicar_platt(s_te[servido], a, b)
    alertas_te["perdida_esperada"] = C.alerta_por_perdida_esperada(p_te_cal, monto_te, COSTO_REVISION, FRACCION_PERDIDA)
    d_va = C.desglose(y_va, monto_va, C.alerta_por_perdida_esperada(p_va_cal, monto_va, COSTO_REVISION, FRACCION_PERDIDA), COSTO_REVISION, FRACCION_PERDIDA)
    filas.append({"modelo": "perdida_esperada", "nombre": f"{ETIQUETAS[servido]} + regla p·monto > costo",
                  "politica": "pérdida esperada (Platt en validación)", "umbral": np.nan, "costo_validacion_eur": d_va["costo_total_eur"],
                  **C.desglose(y_te, monto_te, alertas_te["perdida_esperada"], COSTO_REVISION, FRACCION_PERDIDA)})
    top = np.zeros(len(y_te), bool)
    top[np.argsort(-s_te[servido], kind="mergesort")[:k_te]] = True
    alertas_te["capacidad"] = top
    filas.append({"modelo": "capacidad", "nombre": f"{ETIQUETAS[servido]} · top {k_te} (capacidad {ALERTAS_POR_DIA}/día)",
                  "politica": "capacidad fija", "umbral": np.nan, "costo_validacion_eur": np.nan,
                  **C.desglose(y_te, monto_te, top, COSTO_REVISION, FRACCION_PERDIDA)})
    costo_df = pd.DataFrame(filas)
    bc = C.bootstrap_costo(y_te, monto_te, alertas_te, COSTO_REVISION, FRACCION_PERDIDA, n_rep=N_BOOTSTRAP)
    q = bc.groupby("modelo")["costo_eur"].quantile([0.025, 0.975]).unstack()
    costo_df["costo_ic_inf"] = costo_df["modelo"].map(q[0.025])
    costo_df["costo_ic_sup"] = costo_df["modelo"].map(q[0.975])
    costo_df["ahorro_vs_sin_modelo_pct"] = 100 * (1 - costo_df["costo_total_eur"] / costo_df["costo_sin_modelo_eur"])
    costo_df.to_csv(DIR_REPORTES / "costo_prueba.csv", index=False)
    print(costo_df[["nombre", "alertas", "fraudes_detectados", "costo_total_eur", "ahorro_vs_sin_modelo_pct"]].round(1).to_string(index=False))

    curvas = []
    for m in dict.fromkeys([servido, mejor_no_sup, "logistica", "regla_monto"]):
        for nom, yy, mm, ss in [("validacion", y_va, monto_va, s_va[m]), ("prueba", y_te, monto_te, s_te[m])]:
            c = C.curva_costo(yy, mm, ss, COSTO_REVISION, FRACCION_PERDIDA)
            curvas.append(c[c["alertas"] <= 1500].assign(modelo=m, particion=nom))
    pd.concat(curvas).to_csv(DIR_REPORTES / "curvas_costo.csv", index=False)

    sens = []
    for cr in COSTO_REVISION_SENSIBILIDAD:
        for fp in FRACCION_PERDIDA_SENSIBILIDAD:
            for m in dict.fromkeys(["logistica", mejor_no_sup, servido, "hibrido"]):
                u, _ = C.elegir_umbral(y_va, monto_va, s_va[m], cr, fp)
                sens.append({"costo_revision_eur": cr, "fraccion_perdida": fp, "modelo": m, "nombre": ETIQUETAS[m], "umbral": u,
                             **C.desglose(y_te, monto_te, s_te[m] >= u, cr, fp)})
            sens.append({"costo_revision_eur": cr, "fraccion_perdida": fp, "modelo": "perdida_esperada",
                         "nombre": f"{ETIQUETAS[servido]} + regla p·monto", "umbral": np.nan,
                         **C.desglose(y_te, monto_te, C.alerta_por_perdida_esperada(p_te_cal, monto_te, cr, fp), cr, fp)})
    sens = pd.DataFrame(sens)
    sens["ahorro_vs_sin_modelo_pct"] = 100 * (1 - sens["costo_total_eur"] / sens["costo_sin_modelo_eur"])
    sens.to_csv(DIR_REPORTES / "sensibilidad_costo.csv", index=False)

    # ---------------- pocas etiquetas
    X_tr, y_tr = matriz(part.entrenamiento)
    X_te, _ = matriz(part.prueba)
    ruta_pe = DIR_REPORTES / "pocas_etiquetas.csv"
    if pocas_etiquetas or not ruta_pe.exists():
        print("\n[3] Curva de pocas etiquetas")
        curva_pocas_etiquetas(X_tr, y_tr, X_te, y_te).to_csv(ruta_pe, index=False)
    pe = pd.read_csv(ruta_pe)
    ref = float(met_te.set_index("modelo").loc[mejor_no_sup, "auprc"])
    g_pe, cruces = resumen_pocas_etiquetas(pe, ref)
    g_pe.to_csv(DIR_REPORTES / "pocas_etiquetas_resumen.csv", index=False)

    # ---------------- SHAP del modelo servido (booster guardado)
    print("\n[4] SHAP del modelo servido")
    import shap

    booster = lgb.Booster(model_file=str(DIR_MODELOS / "lgbm_servido.txt"))
    assert not entrenamiento["modelo_servido"] == "hibrido", "SHAP del híbrido requiere puntajes de anomalía"
    s_chk = booster.predict(X_te)
    assert np.allclose(s_chk, s_te[servido], atol=1e-9), "El booster guardado no reproduce las predicciones de prueba"
    rng = np.random.default_rng(SEMILLA)
    idx = np.unique(np.r_[rng.choice(np.flatnonzero(y_te == 0), 3000, replace=False), np.flatnonzero(y_te == 1)])
    sv = np.asarray(shap.TreeExplainer(booster).shap_values(X_te[idx]))
    if sv.ndim == 3:
        sv = sv[..., 1]
    imp = pd.DataFrame({"variable": COLUMNAS_MODELO, "shap_medio_abs": np.abs(sv).mean(axis=0),
                        "shap_medio_abs_fraudes": np.abs(sv[y_te[idx] == 1]).mean(axis=0)}).sort_values("shap_medio_abs", ascending=False)
    imp.to_csv(DIR_REPORTES / "shap_importancia.csv", index=False)
    np.savez_compressed(DIR_REPORTES / "shap_muestra_prueba.npz", shap=sv.astype(np.float32), X=X_te[idx].astype(np.float32),
                        y=y_te[idx], nombres=np.array(COLUMNAS_MODELO))

    # ---------------- deriva
    print("\n[5] Deriva temporal día 1 → día 2")
    der = D.deriva_por_variable(df)
    der.to_csv(DIR_REPORTES / "deriva_variables.csv", index=False)
    adv = D.validacion_adversaria(df)
    por_dia = D.fraude_por_dia(df)
    por_dia.to_csv(DIR_REPORTES / "fraude_por_dia.csv", index=False)
    por_hora = (df.assign(hora_abs=(df["Time"] // 3600).astype(int)).groupby("hora_abs")
                .agg(transacciones=("Class", "size"), fraudes=("Class", "sum")).reset_index())
    por_hora.to_csv(DIR_REPORTES / "transacciones_por_hora.csv", index=False)
    # ¿dónde está la deriva? correlación de cada V con Time dentro de las normales
    normales = df[df["Class"] == 0]
    corr_time = normales[COLUMNAS_MODELO].corrwith(normales["Time"]).abs().sort_values(ascending=False)
    print(f"  Validación adversaria ROC-AUC {adv['roc_auc']:.3f} · |corr(V, Time)| máx {corr_time.iloc[0]:.2f} ({corr_time.index[0]})")

    # ---------------- artefactos para la API y la app
    rng = np.random.default_rng(SEMILLA)
    te = part.prueba
    idx_m = np.sort(np.r_[np.flatnonzero(y_te == 1), rng.choice(np.flatnonzero(y_te == 0), 425, replace=False)])
    DIR_MUESTRA.mkdir(parents=True, exist_ok=True)
    te.iloc[idx_m][["Time"] + COLUMNAS_MODELO[:-1] + ["Amount", "Class"]].to_csv(DIR_MUESTRA / "transacciones_prueba_muestra.csv", index=False)
    ae = MO.Autoencoder.cargar(DIR_MODELOS / "autoencoder.npz")
    metadata = {
        "modelo_servido": servido,
        "nombre_servido": ETIQUETAS[servido],
        "variables": COLUMNAS_MODELO,
        "umbral_costo_validacion": umbrales[servido],
        "platt": {"a": a, "b": b},
        "costo_revision_eur": COSTO_REVISION,
        "fraccion_perdida": FRACCION_PERDIDA,
        "percentiles_error_ae_normales_validacion": np.quantile(ae.puntaje(matriz(part.validacion)[0][y_va == 0]), np.linspace(0, 1, 101)).tolist(),
        "umbral_ae_costo_validacion": umbrales["autoencoder"],
        "particion_cortes_s": [part.corte_1_s, part.corte_2_s],
    }
    (DIR_MODELOS / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---------------- resultados.json
    resultados = {
        "datos": {
            "filas": len(df), "fraudes": int(df["Class"].sum()), "tasa_fraude_pct": float(df["Class"].mean() * 100),
            "duplicados_exactos": int(df.duplicated(subset=["Time"] + COLUMNAS_MODELO[:-1] + ["Amount", "Class"]).sum()),
            "monto_fraude_total_eur": float(df.loc[df.Class == 1, "Amount"].sum()),
            "monto_fraude_mediana_eur": float(df.loc[df.Class == 1, "Amount"].median()),
            "monto_normal_mediana_eur": float(df.loc[df.Class == 0, "Amount"].median()),
        },
        "particion": resumen_part.to_dict(orient="records"),
        "capacidad": {"alertas_por_dia": ALERTAS_POR_DIA, "k_validacion": k_val, "k_prueba": k_te},
        "supuestos_costo": {"costo_revision_eur": COSTO_REVISION, "fraccion_perdida": FRACCION_PERDIDA},
        "entrenamiento": entrenamiento,
        "mejor_no_supervisado": mejor_no_sup,
        "modelo_servido": servido,
        "metricas_prueba": met_te.to_dict(orient="records"),
        "diferencias": dif.to_dict(orient="records"),
        "costo_prueba": costo_df.to_dict(orient="records"),
        "platt": {"a": a, "b": b},
        "pocas_etiquetas": {"referencia_no_supervisado": mejor_no_sup, "auprc_referencia": ref,
                            "n_fraudes_para_superar": cruces, "curva": g_pe.to_dict(orient="records")},
        "shap_top": imp.head(8).to_dict(orient="records"),
        "deriva": {"adversaria": adv, "top_psi": der.head(5)[["variable", "psi", "ks"]].to_dict(orient="records"),
                   "corr_time_top": {k: float(v) for k, v in corr_time.head(5).items()},
                   "por_dia": por_dia.to_dict(orient="records")},
        "segundos_evaluacion": round(time.time() - t0, 1),
    }
    (DIR_REPORTES / "resultados.json").write_text(json.dumps(resultados, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"\n[ok] Evaluación completa en {resultados['segundos_evaluacion']} s · cruces pocas etiquetas: {cruces}")
    return resultados


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sin-pocas-etiquetas", action="store_true", help="Reutiliza reports/pocas_etiquetas.csv si existe")
    main(pocas_etiquetas=not ap.parse_args().sin_pocas_etiquetas)
