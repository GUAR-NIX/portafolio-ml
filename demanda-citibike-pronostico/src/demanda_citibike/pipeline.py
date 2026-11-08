"""Pipeline integral: ingesta → features → backtesting → evaluación → negocio → figuras.

Entrena con los mismos cortes temporales tres conjuntos de predictores:
- ``dia_siguiente`` (principal): horizonte de negocio, solo rezagos ≥ 24 h.
- ``dia_siguiente_sin_clima``: ablación, el mismo modelo sin variables de clima.
- ``una_hora``: comparación, usa también la hora anterior (lag_1…lag_3).
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from demanda_citibike import graficos
from demanda_citibike.backtesting import ejecutar_backtesting
from demanda_citibike.clima import cargar_clima
from demanda_citibike.config import FIGURAS, MODELOS, PROCESADOS, REPORTES
from demanda_citibike.evaluacion import cobertura, metricas_completas, metricas_por_segmento, wape
from demanda_citibike.features import COLS_CATEGORICAS, CONJUNTOS, FEATURES_DIA, OBJETIVO, construir_caracteristicas
from demanda_citibike.ingesta import agregar_viajes, generar_muestra_ci, guardar_demanda, recortar_periodo
from demanda_citibike.modelos import entrenar_cuantiles, guardar_modelo
from demanda_citibike.negocio import analizar_rebalanceo, calcular_umbrales, razon_equilibrio

# Nombre del conjunto → columna de predicción en los reportes.
COLUMNA_PRED = {
    "dia_siguiente": "pred_dia",
    "dia_siguiente_sin_clima": "pred_dia_sin_clima",
    "una_hora": "pred_1h",
}
NOMBRES_ESTRATEGIA = {
    "linea_base_estacional": "Base estacional (t−168)",
    "dia_siguiente_media": "Día siguiente, alerta con la media",
    "dia_siguiente_p90": "Día siguiente, alerta con el cuantil p90",
    "una_hora_media": "1 hora (no disponible la noche anterior)",
}


def _metricas(df: pd.DataFrame, cols: list[str]) -> dict:
    real = df["salidas"].to_numpy()
    return {c: metricas_completas(real, df[c].to_numpy()) for c in cols}


def _error_por_estacion(df_test: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for est, g in df_test.groupby("estacion", sort=False):
        r = g["salidas"].to_numpy()
        filas.append({
            "estacion": est,
            "n": int(len(g)),
            "salidas": int(r.sum()),
            "MAE": round(float(np.mean(np.abs(r - g["pred_dia"].to_numpy()))), 2),
            # Sin salidas en el test (estación cerrada o con ID cambiado) el WAPE no está definido.
            "WAPE_pct": round(wape(r, g["pred_dia"].to_numpy()), 2) if r.sum() else np.nan,
            "WAPE_base_pct": round(wape(r, g["base_estacional"].to_numpy()), 2) if r.sum() else np.nan,
            "WAPE_1h_pct": round(wape(r, g["pred_1h"].to_numpy()), 2) if r.sum() else np.nan,
            "sesgo_pct": round(float((g["pred_dia"].sum() - r.sum()) / r.sum() * 100), 2) if r.sum() else 0.0,
        })
    d = pd.DataFrame(filas)
    m = meta.assign(estacion=meta["estacion"].astype(str))[["estacion", "nombre", "latitud", "longitud"]]
    d = d.assign(estacion=d["estacion"].astype(str)).merge(m, on="estacion", how="left")
    d["nombre"] = d["nombre"].fillna("Estación " + d["estacion"])
    return d.sort_values("WAPE_pct", ascending=False).reset_index(drop=True)


def _error_por_dia(df_test: pd.DataFrame) -> pd.DataFrame:
    df = df_test.assign(fecha=df_test["fecha_hora"].dt.normalize())
    filas = []
    for fecha, g in df.groupby("fecha"):
        r = g["salidas"].to_numpy()
        # El clima es el mismo para todas las estaciones: se toma una fila por hora.
        clima_h = g.drop_duplicates("fecha_hora")
        filas.append({
            "fecha": fecha.date(),
            "dia_semana": int(fecha.day_of_week),
            "es_feriado": int(g["es_feriado"].max()),
            "lluvia_mm_dia": round(float(clima_h["lluvia_mm"].sum()), 1),
            "temperatura_media_c": round(float(clima_h["temperatura_c"].mean()), 1),
            "salidas": int(r.sum()),
            "WAPE_pred_dia": round(wape(r, g["pred_dia"].to_numpy()), 2),
            "WAPE_pred_dia_sin_clima": round(wape(r, g["pred_dia_sin_clima"].to_numpy()), 2),
            "WAPE_pred_1h": round(wape(r, g["pred_1h"].to_numpy()), 2),
            "WAPE_base_estacional": round(wape(r, g["base_estacional"].to_numpy()), 2),
            "sesgo_pred_dia_pct": round(float((g["pred_dia"].sum() - r.sum()) / r.sum() * 100), 2),
        })
    return pd.DataFrame(filas)


def ejecutar_pipeline() -> dict:
    inicio = time.time()
    print("=" * 70)
    print("INICIANDO PIPELINE: PRONÓSTICO DE DEMANDA Citi Bike (estación × hora)")
    print("=" * 70)

    # 1. Ingesta
    ruta_demanda = PROCESADOS / "demanda_horaria.parquet"
    if ruta_demanda.exists():
        print("[1/8] Cargando demanda horaria existente...")
        demanda = pd.read_parquet(ruta_demanda)
        meta = pd.read_csv(PROCESADOS / "estaciones.csv")
        n_antes = len(demanda)
        demanda = recortar_periodo(demanda)
        if len(demanda) != n_antes:  # descarta las horas del 31-12-2023 que traen los CSV de enero
            guardar_demanda(demanda, meta)
    else:
        print("[1/8] Agregando viajes a salidas/llegadas por estación y hora (DuckDB)...")
        demanda, meta = agregar_viajes()
        guardar_demanda(demanda, meta)
        generar_muestra_ci(demanda)
    print(f"  → {len(demanda):,} filas estación-hora, {demanda['estacion'].nunique()} estaciones")

    # 2. Features
    print("[2/8] Construyendo características (clima + calendario + rezagos sin fuga)...")
    clima = cargar_clima()
    df = construir_caracteristicas(demanda, clima)
    print(f"  → {len(df):,} filas de modelado")

    # 3. Backtesting de los tres conjuntos con los mismos cortes
    df_cv = df_test = None
    modelos = {}
    ini_test_final = None
    pct_negativas = {}
    for i, (nombre, feats) in enumerate(CONJUNTOS.items(), start=1):
        print(f"[3/8] Backtesting {i}/{len(CONJUNTOS)}: {nombre} ({len(feats)} variables)...")
        res = ejecutar_backtesting(df, features=feats)
        col = COLUMNA_PRED[nombre]
        modelos[nombre] = res["modelo_final"]
        ini_test_final = res["ini_test_final"]
        crudo = res["modelo_final"].predict(res["test"][feats])
        pct_negativas[col] = round(float(np.mean(crudo < 0) * 100), 2)
        cv = res["cv"].rename(columns={"pred": col})
        test = res["test"].rename(columns={"pred": col})
        if df_test is None:
            df_cv, df_test = cv, test
        else:
            df_cv[col] = cv[col].to_numpy()
            df_test[col] = test[col].to_numpy()
        del res
    df_test = df_test.reset_index(drop=True)

    cols_modelo = list(COLUMNA_PRED.values())
    cols_base = ["base_estacional", "base_ayer", "base_media"]

    # 4. Métricas
    print("[4/8] Calculando métricas (test final y backtesting)...")
    m_test = _metricas(df_test, cols_modelo + cols_base)
    m_cv = _metricas(df_cv, cols_modelo + cols_base)
    m_folds = {
        int(k): {c: round(wape(g["salidas"].to_numpy(), g[c].to_numpy()), 2) for c in cols_modelo + ["base_estacional"]}
        for k, g in df_cv.groupby("fold")
    }

    # 5. Cuantiles del modelo principal (intervalos de predicción 80 %)
    print("[5/8] Entrenando modelos por cuantil (p10/p90) del modelo día siguiente...")
    train_q = df[df["fecha_hora"] < ini_test_final]
    cats = [c for c in COLS_CATEGORICAS if c in FEATURES_DIA]
    modelos_q = entrenar_cuantiles(train_q[FEATURES_DIA], train_q[OBJETIVO], cats)
    df_test["pred_q10"] = np.clip(modelos_q[0.1].predict(df_test[FEATURES_DIA]), 0, None)
    df_test["pred_q90"] = np.clip(modelos_q[0.9].predict(df_test[FEATURES_DIA]), 0, None)
    real_test = df_test["salidas"].to_numpy()
    cobertura_pct = cobertura(real_test, df_test["pred_q10"].to_numpy(), df_test["pred_q90"].to_numpy())

    # 6. Importancia SHAP (misma muestra de horas del test para los dos modelos)
    print("[6/8] Importancia SHAP de los modelos día siguiente y 1 hora...")
    muestra = df_test.sample(min(20000, len(df_test)), random_state=42)
    imp_dia = graficos.importancia_shap(modelos["dia_siguiente"], muestra[CONJUNTOS["dia_siguiente"]])
    imp_1h = graficos.importancia_shap(modelos["una_hora"], muestra[CONJUNTOS["una_hora"]])
    imp = pd.concat([imp_dia.assign(modelo="dia_siguiente"), imp_1h.assign(modelo="una_hora")], ignore_index=True)
    imp["media_abs_shap"] = imp["media_abs_shap"].round(4)
    imp["pct"] = imp["pct"].round(2)
    imp[["modelo", "variable", "nombre", "familia", "media_abs_shap", "pct"]].to_csv(REPORTES / "importancia_shap.csv", index=False)
    familias = {
        m: g.groupby("familia")["pct"].sum().round(1).to_dict() for m, g in imp.groupby("modelo")
    }

    # 7. Negocio (rebalanceo nocturno)
    print("[7/8] Análisis de negocio: alertas de rebalanceo y costos...")
    umbrales = calcular_umbrales(train_q)
    estrategias = {
        "linea_base_estacional": "base_estacional",
        "dia_siguiente_media": "pred_dia",
        "dia_siguiente_p90": "pred_q90",
        "una_hora_media": "pred_1h",
    }
    stats_rebalanceo = analizar_rebalanceo(df_test, umbrales, estrategias)
    est = stats_rebalanceo["estrategias"]
    stats_rebalanceo["razon_equilibrio_p90_vs_media"] = razon_equilibrio(est["dia_siguiente_media"], est["dia_siguiente_p90"])
    stats_rebalanceo["razon_equilibrio_media_vs_base"] = razon_equilibrio(est["linea_base_estacional"], est["dia_siguiente_media"])
    stats_rebalanceo["razon_equilibrio_p90_vs_base"] = razon_equilibrio(est["linea_base_estacional"], est["dia_siguiente_p90"])

    # 8. Segmentos, figuras y persistencia
    print("[8/8] Reportes, figuras y modelos...")
    error_por_hora = pd.concat(
        [metricas_por_segmento(df_test, "hora", real_col="salidas", pred_col=c).assign(modelo=c)
         for c in ["pred_dia", "pred_1h", "base_estacional"]],
        ignore_index=True,
    ).sort_values(["modelo", "hora"])
    error_por_estacion = _error_por_estacion(df_test, meta)
    error_por_dia = _error_por_dia(df_test)
    lluvia = graficos.tabla_lluvia(df_test, ["base_estacional", "pred_dia_sin_clima", "pred_dia", "pred_1h"])

    for vieja in ["07_rebalanceo_ahorro.png"]:  # figura reemplazada por 07_rebalanceo_costos.png
        (FIGURAS / vieja).unlink(missing_ok=True)
    graficos.graficar_demanda_horaria(df)
    graficos.graficar_prediccion_vs_real(df_test)
    graficos.graficar_importancia(imp_dia, imp_1h)
    graficos.graficar_error_por_hora(df_test)
    graficos.graficar_intervalos(df_test, cobertura_pct)
    graficos.graficar_error_lluvia(df_test)
    graficos.graficar_rebalanceo(stats_rebalanceo, NOMBRES_ESTRATEGIA)
    graficos.graficar_error_estaciones(error_por_estacion)
    graficos.graficar_error_por_dia(error_por_dia)

    guardar_modelo(modelos["dia_siguiente"], MODELOS / "modelo_lgbm.joblib")
    guardar_modelo(modelos["una_hora"], MODELOS / "modelo_1h.joblib")

    # Predicciones en Parquet comprimido (~2 MB cada archivo).
    cols_pred = cols_modelo + cols_base
    salida_test = df_test[["fecha_hora", "estacion", "hora", "es_feriado", "lluvia_mm", "salidas",
                           *cols_pred, "pred_q10", "pred_q90"]].copy()
    salida_cv = df_cv[["fecha_hora", "estacion", "fold", "salidas", *cols_pred]].copy()
    for d in (salida_test, salida_cv):
        d["estacion"] = d["estacion"].astype(str)
        d["salidas"] = d["salidas"].astype("int16")
        for c in d.columns:
            # 3 decimales bastan para recalcular las métricas; los cuantiles se redondean hacia
            # dentro del intervalo (ceil/floor) para que la cobertura con salidas enteras no cambie.
            if c == "pred_q10":
                d[c] = (np.ceil(d[c] * 1000) / 1000).astype("float32")
            elif c == "pred_q90":
                d[c] = (np.floor(d[c] * 1000) / 1000).astype("float32")
            elif c.startswith("pred") or c.startswith("base"):
                d[c] = d[c].round(3).astype("float32")
    salida_test.to_parquet(REPORTES / "predicciones_test.parquet", index=False, compression="zstd")
    salida_cv.to_parquet(REPORTES / "predicciones_cv.parquet", index=False, compression="zstd")
    (REPORTES / "predicciones_test.csv").unlink(missing_ok=True)
    error_por_hora.to_csv(REPORTES / "error_por_hora.csv", index=False)
    error_por_estacion.to_csv(REPORTES / "error_por_estacion.csv", index=False)
    error_por_dia.to_csv(REPORTES / "error_por_dia.csv", index=False)
    lluvia.to_csv(REPORTES / "error_lluvia_ablacion.csv", index=False)

    def _mejora(col: str, ref: str, m: dict) -> float:
        return round((m[ref]["WAPE_pct"] - m[col]["WAPE_pct"]) / m[ref]["WAPE_pct"] * 100, 1)

    resultados = {
        "proyecto": "ML_03_demanda_citibike",
        "descripcion": "Pronóstico de demanda de bicicletas por estación y hora (Citi Bike + clima)",
        "tiempo_ejecucion_segundos": round(time.time() - inicio, 1),
        "datos": {
            "filas_demanda_horaria": int(len(demanda)),
            "estaciones_modeladas": int(demanda["estacion"].nunique()),
            "horas_por_estacion": int(demanda.groupby("estacion").size().iloc[0]),
            "total_salidas": int(demanda["salidas"].sum()),
            "periodo_demanda": f"{demanda['fecha_hora'].min()} → {demanda['fecha_hora'].max()}",
            "filas_modelado": int(len(df)),
            "periodo_modelado": f"{df['fecha_hora'].min()} → {df['fecha_hora'].max()}",
            "test_final_desde": str(ini_test_final),
            "horas_estacion_test": int(len(df_test)),
            "horas_estacion_cv": int(len(df_cv)),
        },
        "horizonte": {
            "principal": "dia_siguiente: emitido a medianoche de D para las 24 h de D+1; solo rezagos >= 24 h",
            "comparacion": "una_hora: usa la hora anterior (lag_1..lag_3); no disponible la noche anterior",
        },
        "test_final": m_test,
        "backtesting_cv": m_cv,
        "backtesting_wape_por_fold": m_folds,
        "mejoras_wape_pct": {
            "dia_vs_base_estacional_test": _mejora("pred_dia", "base_estacional", m_test),
            "dia_vs_base_ayer_test": _mejora("pred_dia", "base_ayer", m_test),
            "dia_vs_base_estacional_cv": _mejora("pred_dia", "base_estacional", m_cv),
            "una_hora_vs_dia_test": _mejora("pred_1h", "pred_dia", m_test),
            "con_clima_vs_sin_clima_test": _mejora("pred_dia", "pred_dia_sin_clima", m_test),
            "con_clima_vs_sin_clima_cv": _mejora("pred_dia", "pred_dia_sin_clima", m_cv),
        },
        "ablacion_clima_lluvia": lluvia.to_dict(orient="records"),
        "predicciones_negativas_antes_de_recortar_pct": pct_negativas,
        "intervalos": {
            "modelo": "dia_siguiente",
            "nominal_pct": 80,
            "cobertura_real_pct": round(cobertura_pct, 1),
        },
        "importancia_shap_por_familia_pct": familias,
        "negocio": stats_rebalanceo,
        "artefactos": {
            "figuras_png": len(list(FIGURAS.glob("*.png"))),
            "reportes_csv": len(list(REPORTES.glob("*.csv"))),
        },
    }

    (REPORTES / "resultados.json").write_text(json.dumps(resultados, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("=" * 70)
    print(f"PIPELINE COMPLETADO en {resultados['tiempo_ejecucion_segundos']}s")
    for c in cols_modelo + ["base_estacional", "base_ayer"]:
        print(f"  WAPE test {c:<20} {m_test[c]['WAPE_pct']:6.2f} %   (CV {m_cv[c]['WAPE_pct']:6.2f} %)")
    print(f"Cobertura intervalo 80%: {round(cobertura_pct, 1)}%")
    print("=" * 70)
    return resultados


if __name__ == "__main__":
    ejecutar_pipeline()
