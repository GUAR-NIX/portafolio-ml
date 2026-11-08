"""
Entrenamiento y evaluación completos (un comando, ~3 min):

    python -m campana.entrenar

1. Datos y particiones (80/20 estratificada + prueba de estrés temporal)
2. Fuga de información: cuánto «ayuda» la duración de la llamada
3. Desbalance: 2 modelos × 4 estrategias (ninguna, pesos, submuestreo, SMOTE) en 5 pliegues
4. Elección del modelo, calibración y umbral por costo/beneficio (todo con predicciones fuera de pliegue)
5. Evaluación final en prueba: métricas con IC bootstrap, políticas de llamada, ganancia acumulada, sensibilidad
6. SHAP
7. Prueba de estrés temporal con recalibración mensual
8. Guardar modelo, metadata y figuras
"""
from __future__ import annotations

import json
import sys
import time
from datetime import date

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from . import config, datos, explicar, graficos, modelos
from . import evaluacion as ev

VERSION = "1.0.0"


def _paso(i: int, texto: str):
    print(f"\n[{i}/8] {texto}", flush=True)


def _r(d: dict, dec: int = 4) -> dict:
    return {k: (round(float(v), dec) if isinstance(v, (float, np.floating)) else v) for k, v in d.items()}


def main() -> dict:
    t0 = time.time()
    config.REPORTES.mkdir(parents=True, exist_ok=True)
    config.MODELOS.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ 1
    _paso(1, "Datos y particiones")
    df = datos.cargar()
    ent, pru = datos.particion_estratificada(df)
    X, y = datos.X_y(ent)
    Xp, yp = datos.X_y(pru)
    print(f"  {len(df):,} clientes · tasa de suscripción {df[config.OBJETIVO].mean():.1%}")
    print(f"  entrenamiento {len(ent):,} · prueba {len(pru):,} (estratificada, semilla {config.SEMILLA})")

    # ------------------------------------------------------------------ 2
    _paso(2, "Fuga de información (duración de la llamada)")
    tramos = pd.cut(df["duracion_llamada"], [-1, 0, 60, 120, 180, 300, 600, 5000],
                    labels=["0", "1–60", "61–120", "121–180", "181–300", "301–600", "> 600"])
    tasa_duracion = df.groupby(tramos, observed=True)[config.OBJETIVO].agg(tasa="mean", n="size")
    fuga = {}
    for etiqueta, extra in [("Con duración (fuga)", ["duracion_llamada"]),
                            ("Con nº de llamadas de la campaña", ["contactos_campana"]),
                            ("Sin variables con fuga (modelo final)", [])]:
        vars_ = config.VARIABLES + extra
        m = modelos.lightgbm().fit(ent[vars_], y)
        p = modelos.probabilidad(m, pru[vars_])
        fuga[etiqueta] = {"auc": roc_auc_score(yp, p), "average_precision": average_precision_score(yp, p)}
    tabla_fuga = pd.DataFrame(fuga).T
    tabla_fuga.round(4).to_csv(config.REPORTES / "fuga_informacion.csv", index_label="variante")
    tasa_duracion.round(4).to_csv(config.REPORTES / "tasa_por_duracion.csv", index_label="duracion_s")
    print(tabla_fuga.round(4).to_string())

    # ------------------------------------------------------------------ 3
    _paso(3, "Desbalance: 2 modelos × 4 estrategias, 5 pliegues estratificados")
    filas, oof = [], {}
    p_regla = ev.fuera_de_pliegue("regla", "ninguna", X, y)
    filas.append({"modelo": "regla", "estrategia": "ninguna", **ev.metricas(y, p_regla)})
    oof[("regla", "ninguna")] = p_regla
    umbral_t = ev.umbral_teorico()
    for nombre in ["logistica", "lightgbm"]:
        for estrategia in modelos.ESTRATEGIAS:
            ti = time.time()
            p = ev.fuera_de_pliegue(nombre, estrategia, X, y)
            p_cal = ev.calibrar_fuera_de_pliegue(p, y, "isotonica")
            oof[(nombre, estrategia)] = p
            filas.append({"modelo": nombre, "estrategia": estrategia, **ev.metricas(y, p),
                          "brier_calibrado": brier_score_loss(y, p_cal),
                          "ganancia_umbral_teorico": ev.ganancia(y, p >= umbral_t),
                          "ganancia_calibrado": ev.ganancia(y, p_cal >= umbral_t),
                          "segundos": time.time() - ti})
            print(f"  {nombre:9s} {estrategia:11s} AP {filas[-1]['average_precision']:.4f} · "
                  f"Brier {filas[-1]['brier']:.4f} → calibrado {filas[-1]['brier_calibrado']:.4f} · "
                  f"{filas[-1]['segundos']:.0f} s")
    comp = pd.DataFrame(filas)
    comp["n"] = len(y)
    comp.round(4).to_csv(config.REPORTES / "comparacion_desbalance.csv", index=False)

    # ------------------------------------------------------------------ 4
    _paso(4, "Elección, calibración y umbral (fuera de pliegue)")
    candidatos = comp[comp["modelo"] != "regla"].sort_values("ganancia_calibrado", ascending=False)
    elegido = candidatos.iloc[0]
    nombre_f, estrategia_f = elegido["modelo"], elegido["estrategia"]
    print(f"  elegido por ganancia calibrada fuera de pliegue: {nombre_f} + {estrategia_f}")
    p_oof = oof[(nombre_f, estrategia_f)]
    calib_cmp = {"sin calibrar": brier_score_loss(y, p_oof)}
    for metodo in ["platt", "isotonica"]:
        calib_cmp[metodo] = brier_score_loss(y, ev.calibrar_fuera_de_pliegue(p_oof, y, metodo))
    metodo_cal = min(["platt", "isotonica"], key=calib_cmp.get)
    print(f"  Brier fuera de pliegue: " + " · ".join(f"{k} {v:.5f}" for k, v in calib_cmp.items())
          + f" → {metodo_cal}")
    p_oof_cal = ev.calibrar_fuera_de_pliegue(p_oof, y, metodo_cal)
    umbral = ev.umbral_optimo(y, p_oof_cal)
    print(f"  umbral teórico costo/beneficio = {umbral_t:.3f} · óptimo fuera de pliegue = {umbral:.3f}")

    modelo = modelos.entrenar(nombre_f, estrategia_f, X, y)
    calibrador = modelos.Calibrador(metodo_cal).fit(p_oof, y)

    # Logística (mejor estrategia) y regla, entrenadas igual, para comparar en prueba
    estr_log = candidatos[candidatos["modelo"] == "logistica"].iloc[0]["estrategia"]
    m_log = modelos.entrenar("logistica", estr_log, X, y)
    cal_log = modelos.Calibrador(metodo_cal).fit(oof[("logistica", estr_log)], y)
    m_regla = modelos.ReglaSegmento().fit(X, y)
    m_pesos = modelos.entrenar("lightgbm", "pesos", X, y)

    # ------------------------------------------------------------------ 5
    _paso(5, "Evaluación final en prueba")
    s_final = modelos.probabilidad(modelo, Xp)
    p_final = calibrador.transform(s_final)
    p_log = cal_log.transform(modelos.probabilidad(m_log, Xp))
    p_reg = modelos.probabilidad(m_regla, Xp)
    p_pesos = modelos.probabilidad(m_pesos, Xp)
    etq_final = "LightGBM" if nombre_f == "lightgbm" else "Regresión logística"
    preds = {
        "Regla por segmento": p_reg,
        f"Regresión logística ({estr_log}, calibrada)": p_log,
        f"{etq_final} ({estrategia_f}, sin calibrar)": s_final,
        f"{etq_final} ({estrategia_f}, calibrado) · FINAL": p_final,
        "LightGBM (pesos, sin calibrar)": p_pesos,
    }
    met_prueba = pd.DataFrame({k: {**ev.metricas(yp, v), "brier_skill": ev.brier_skill(yp, v)}
                               for k, v in preds.items()}).T
    met_prueba.round(4).to_csv(config.REPORTES / "metricas_prueba.csv", index_label="modelo")
    print(met_prueba[["auc", "average_precision", "brier", "brier_skill", "prob_media"]].round(4).to_string())

    # Incertidumbre de la prueba: IC 95 % bootstrap pareado de AUC, AP y Brier, y de LightGBM frente a la logística
    nombres_ic = {"Regla por segmento": p_reg, "Regresión logística": p_log, "LightGBM calibrado (final)": p_final}
    ic = ev.bootstrap_metricas(yp, nombres_ic, [("LightGBM calibrado (final)", "Regresión logística"),
                                                ("LightGBM calibrado (final)", "Regla por segmento")])
    ic.round(5).to_csv(config.REPORTES / "bootstrap_metricas_prueba.csv", index=False)
    print(ic[ic["metrica"] == "auc"].round(4).to_string(index=False))

    politicas = {
        "No llamar a nadie": np.zeros(len(yp), bool),
        "Llamar a todos": np.ones(len(yp), bool),
        "Regla: segmentos con tasa > 10 %": p_reg >= umbral_t,
        "Logística calibrada, p ≥ umbral": p_log >= umbral,
        f"{etq_final} calibrado, p ≥ umbral (FINAL)": p_final >= umbral,
        "Capacidad limitada: top 20 % del modelo": p_final >= np.quantile(p_final, 0.80),
        "Oráculo (solo quienes suscriben)": yp == 1,
    }
    tabla_pol = pd.DataFrame({k: ev.resumen_politica(yp, v) for k, v in politicas.items()}).T
    tabla_pol["ganancia_por_1000"] = tabla_pol["ganancia"] / len(yp) * 1000
    tabla_pol.round(2).to_csv(config.REPORTES / "politicas_prueba.csv", index_label="politica")
    print(tabla_pol[["pct_llamados", "pct_exitos_capturados", "conversion_pct", "ganancia"]].round(1).to_string())

    curva_u = ev.curva_ganancia_umbral(yp, p_final)
    curva_u.round(4).to_csv(config.REPORTES / "ganancia_por_umbral.csv", index=False)
    deciles = ev.tabla_deciles(yp, p_final)
    deciles.round(4).to_csv(config.REPORTES / "deciles_prueba.csv")
    gan = {f"{etq_final} (final)": ev.ganancia_acumulada(yp, p_final),
           "Regresión logística": ev.ganancia_acumulada(yp, p_log),
           "Regla por segmento": ev.ganancia_acumulada(yp, p_reg)}
    lift = {k: {f"capturado_top{q}": float(np.interp(q, v["pct_llamados"], v["pct_capturado"]))
                for q in (10, 20, 30, 50)} | {"lift_top10": float(np.interp(10, v["pct_llamados"], v["lift"]))}
            for k, v in gan.items()}

    sens = []
    for costo in (4.0, 8.0, 16.0):
        for beneficio in (40.0, 80.0, 160.0):
            u = ev.umbral_teorico(costo, beneficio)
            llamar = p_final >= u
            sens.append({"costo": costo, "beneficio": beneficio, "umbral": u, "pct_llamados": llamar.mean() * 100,
                         "ganancia_modelo": ev.ganancia(yp, llamar, costo, beneficio),
                         "ganancia_todos": ev.ganancia(yp, np.ones(len(yp), bool), costo, beneficio)})
    sens = pd.DataFrame(sens)
    sens["mejora_vs_todos"] = sens["ganancia_modelo"] - sens["ganancia_todos"]
    sens.round(2).to_csv(config.REPORTES / "sensibilidad_costos.csv", index=False)

    # ------------------------------------------------------------------ 6
    _paso(6, "SHAP")
    muestra = Xp.sample(min(3000, len(Xp)), random_state=config.SEMILLA)
    imp = explicar.importancia(modelo, muestra)
    imp.round(5).to_csv(config.REPORTES / "importancia_shap.csv", index=False)
    print("  " + " > ".join(imp["etiqueta"].head(6)))

    # ------------------------------------------------------------------ 7
    _paso(7, "Prueba de estrés temporal")
    pasado, futuro = datos.particion_temporal(df)
    Xa, ya = datos.X_y(pasado)
    Xf, yf = datos.X_y(futuro)
    m_t = modelos.entrenar(nombre_f, estrategia_f, Xa, ya)
    oof_a = ev.fuera_de_pliegue(nombre_f, estrategia_f, Xa, ya)
    cal_t = modelos.Calibrador(metodo_cal).fit(oof_a, ya)
    s_f = modelos.probabilidad(m_t, Xf)
    p_fijo = cal_t.transform(s_f)
    # Recalibración mensual: se desplaza el intercepto (logit) para igualar la tasa observada en los ÚLTIMOS
    # 3 MESES con etiqueta conocida. Platt completo (pendiente + intercepto) mes a mes resultó inestable:
    # las variables macro cambian cada mes y con 128–700 filas la pendiente se estima mal.
    per_a = (pasado["anio"] * 100 + pasado["mes_num"]).to_numpy()
    hist = pd.DataFrame({"periodo": per_a, "p": cal_t.transform(oof_a), "y": ya})
    fut = pd.DataFrame({"periodo": (futuro["anio"] * 100 + futuro["mes_num"]).to_numpy(), "s": s_f, "y": yf,
                        "p_fijo": p_fijo})
    fut["p_recal"] = np.nan
    conocido = hist[hist["periodo"].isin(sorted(hist["periodo"].unique())[-3:])]
    for per in sorted(fut["periodo"].unique()):
        idx = fut["periodo"] == per
        recal = modelos.Calibrador("intercepto").fit(conocido["p"].to_numpy(), conocido["y"].to_numpy())
        fut.loc[idx, "p_recal"] = recal.transform(fut.loc[idx, "p_fijo"].to_numpy())
        nuevo_mes = fut.loc[idx, ["periodo", "p_fijo", "y"]].rename(columns={"p_fijo": "p"})
        conocido = pd.concat([conocido, nuevo_mes])
        conocido = conocido[conocido["periodo"].isin(sorted(conocido["periodo"].unique())[-3:])]
    # Re-entrenar cada mes con todo lo conocido hasta el mes anterior (ventana creciente). LightGBM sin pesos
    # sale casi calibrado (paso 4), así que aquí no se recalibra aparte.
    per_todo = (df["anio"] * 100 + df["mes_num"]).to_numpy()
    fut["p_reent"] = np.nan
    for per in sorted(fut["periodo"].unique()):
        conocido_df = df[per_todo < per]
        m_mes = modelos.entrenar(nombre_f, estrategia_f, *datos.X_y(conocido_df))
        idx = (fut["periodo"] == per).to_numpy()
        fut.loc[idx, "p_reent"] = modelos.probabilidad(m_mes, Xf[idx])
    mensual = fut.groupby("periodo").agg(n=("y", "size"), tasa_real=("y", "mean"),
                                         prob_sin_recalibrar=("p_fijo", "mean"),
                                         prob_recalibrada=("p_recal", "mean"),
                                         prob_reentrenado=("p_reent", "mean")).reset_index()
    for col, nombre_auc in [("s", "auc_fijo"), ("p_reent", "auc_reentrenado")]:
        mensual[nombre_auc] = [roc_auc_score(g["y"], g[col]) for _, g in fut.groupby("periodo")]
    mensual["periodo"] = mensual["periodo"].map(lambda v: f"{v // 100}-{v % 100:02d}")
    mensual.round(4).to_csv(config.REPORTES / "prueba_temporal_mensual.csv", index=False)
    pf, pr, pe = fut["p_fijo"].to_numpy(), fut["p_recal"].to_numpy(), fut["p_reent"].to_numpy()
    temporal = {
        "filas_pasado": int(len(pasado)), "filas_futuro": int(len(futuro)),
        "tasa_pasado": float(ya.mean()), "tasa_futuro": float(yf.mean()),
        "auc": roc_auc_score(yf, s_f), "average_precision": average_precision_score(yf, s_f),
        "auc_reentrenado": roc_auc_score(yf, pe), "average_precision_reentrenado": average_precision_score(yf, pe),
        "brier_fijo": brier_score_loss(yf, pf), "brier_recalibrado": brier_score_loss(yf, pr),
        "brier_reentrenado": brier_score_loss(yf, pe),
        "prob_media_fijo": float(pf.mean()), "prob_media_recalibrado": float(pr.mean()),
        "ganancia_fijo": ev.ganancia(yf, pf >= umbral), "ganancia_recalibrado": ev.ganancia(yf, pr >= umbral),
        "ganancia_reentrenado": ev.ganancia(yf, pe >= umbral),
        "ganancia_oraculo": ev.ganancia(yf, yf == 1),
        "ganancia_todos": ev.ganancia(yf, np.ones(len(yf), bool)),
        "pct_llamados_fijo": float((pf >= umbral).mean() * 100),
        "pct_llamados_recalibrado": float((pr >= umbral).mean() * 100),
        "pct_llamados_reentrenado": float((pe >= umbral).mean() * 100),
    }
    print("  " + " · ".join(f"{k} {v:.3f}" for k, v in _r(temporal).items() if isinstance(v, float)))

    # ------------------------------------------------------------------ 8
    _paso(8, "Modelo, metadata y figuras")
    final = met_prueba.loc[f"{etq_final} ({estrategia_f}, calibrado) · FINAL"]
    pol_final = tabla_pol.loc[f"{etq_final} calibrado, p ≥ umbral (FINAL)"]
    metadata = {
        "version": VERSION,
        "entrenado": date.today().isoformat(),
        "modelo": nombre_f, "estrategia_desbalance": estrategia_f, "calibracion": metodo_cal,
        "variables": config.VARIABLES, "categoricas": config.CATEGORICAS, "excluidas_por_fuga": config.FUGA,
        "supuestos": {"costo_por_cliente_usd": config.COSTO_LLAMADA, "beneficio_por_deposito_usd": config.BENEFICIO_EXITO,
                      "umbral_teorico": umbral_t},
        "umbral": umbral,
        "filas": {"total": int(len(df)), "entrenamiento": int(len(ent)), "prueba": int(len(pru))},
        "tasa_base": float(y.mean()),
        "metricas_prueba": _r({k: final[k] for k in ["auc", "average_precision", "brier", "brier_skill",
                                                     "log_loss", "prob_media", "tasa_real"]}),
        "politica_prueba": _r(pol_final.to_dict(), 2),
        "llamar_a_todos_prueba": _r(tabla_pol.loc["Llamar a todos"].to_dict(), 2),
        "lift_prueba": {k: _r(v, 2) for k, v in lift.items()},
        "calibracion_oof_brier": _r(calib_cmp, 5),
        "fuga": {k: _r(v) for k, v in fuga.items()},
        "temporal": _r(temporal),
        "categorias": modelo.named_steps["prep"].categorias_,
    }
    joblib.dump({"modelo": modelo, "calibrador": calibrador, "umbral": umbral, "metadata": metadata},
                config.MODELOS / "modelo_campana.joblib", compress=3)
    (config.MODELOS / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    (config.REPORTES / "resultados.json").write_text(json.dumps(
        {k: metadata[k] for k in ["metricas_prueba", "politica_prueba", "llamar_a_todos_prueba", "lift_prueba",
                                  "umbral", "calibracion_oof_brier", "fuga", "temporal"]},
        indent=2, ensure_ascii=False), encoding="utf-8")

    graficos.fuga_duracion(tabla_fuga, tasa_duracion)
    graficos.comparacion_desbalance(comp)
    cal_curvas = {"LightGBM con pesos, sin calibrar": p_pesos,
                  f"{etq_final} sin calibrar": s_final,
                  f"{etq_final} calibrado (final)": p_final}
    graficos.calibracion({k: ev.tabla_calibracion(yp, v) for k, v in cal_curvas.items()},
                         {k: brier_score_loss(yp, v) for k, v in cal_curvas.items()})
    graficos.ganancia_umbral(curva_u, umbral, ev.ganancia(yp, np.ones(len(yp), bool)))
    graficos.ganancia_acumulada(gan, float(yp.mean()))
    graficos.importancia_shap(imp)
    graficos.beeswarm_shap(explicar.explicacion_shap(modelo, muestra.head(1500)))
    graficos.prueba_temporal(mensual)
    print(f"\nListo en {time.time() - t0:.0f} s")
    return metadata


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
