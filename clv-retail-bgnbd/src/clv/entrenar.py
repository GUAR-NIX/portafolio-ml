"""Pipeline completo en un comando: limpieza, corte temporal, BG/NBD + Gamma-Gamma,
comparación contra heurísticas y LightGBM, segmentación, gráficos y artefactos."""
from __future__ import annotations

import json
import time

import joblib
import numpy as np

from . import config, datos, evaluacion, graficos, segmentacion
from .modelos import (
    ModeloBGNBD,
    ModeloGammaGamma,
    ModeloSupervisadoCLV,
    ReglaQuintilesRFM,
    ReglaRitmoHistorico,
    calcular_clv,
)


def main() -> None:
    t0 = time.perf_counter()
    config.MODELOS.mkdir(parents=True, exist_ok=True)
    config.REPORTES.mkdir(parents=True, exist_ok=True)
    config.FIGURAS.mkdir(parents=True, exist_ok=True)

    print("1/8 · Cargando y limpiando transacciones de UCI Online Retail II...")
    crudo = datos.cargar_transacciones_crudas()
    limpio, auditoria = datos.limpiar_transacciones(crudo)
    compras_diarias = datos.agregar_compras_diarias(limpio)
    auditoria.to_csv(config.REPORTES / "auditoria_limpieza.csv", index=False)
    print(f"      Filas crudas: {len(crudo):,} -> limpias: {len(limpio):,} -> compras diarias: {len(compras_diarias):,}")

    print("2/8 · Construyendo matrices BTYD (ventana histórica previa y corte principal)...")
    # Ventana previa (corte 2010-12-09 -> objetivo 2010-12-10..2011-06-09) para entrenar modelos supervisados sin fuga
    df_previo = datos.construir_matriz_btyd(
        compras_diarias,
        fecha_corte=config.FECHA_CORTE_SUPERVISADO,
        fecha_fin_holdout=config.FECHA_CORTE_CAL,
    )
    # Ventana principal (corte 2011-06-09 -> holdout 2011-06-10..2011-12-09, 183 días = 6 meses)
    df_principal = datos.construir_matriz_btyd(
        compras_diarias,
        fecha_corte=config.FECHA_CORTE_CAL,
        fecha_fin_holdout=config.FECHA_FIN,
    )
    # Ventana estacional (corte 2010-06-09 -> objetivo jun-dic 2010): misma estación que la prueba, para robustez
    df_estacional = datos.construir_matriz_btyd(
        compras_diarias,
        fecha_corte=config.FECHA_CORTE_ESTACIONAL,
        fecha_fin_holdout=config.FECHA_FIN_ESTACIONAL,
    )
    print(
        f"      Clientes en calibración (hasta {config.FECHA_CORTE_CAL}): {len(df_principal):,} "
        f"(tasa de recompra en 6 meses: {df_principal['vivo_holdout'].mean() * 100:.1f} %)"
    )

    print("3/8 · Verificando supuesto de independencia de Gamma-Gamma...")
    tabla_indep = evaluacion.verificar_supuesto_gamma_gamma(df_principal)
    tabla_indep.to_csv(config.REPORTES / "supuesto_gamma_gamma.csv", index=False)

    print("4/8 · Ajustando BG/NBD y Gamma-Gamma por máxima verosimilitud...")
    bgnbd = ModeloBGNBD(penalizacion=0.001).ajustar(
        df_principal["frecuencia"].to_numpy(),
        df_principal["recencia_semanas"].to_numpy(),
        df_principal["edad_semanas"].to_numpy(),
    )
    gg = ModeloGammaGamma(penalizacion=0.001).ajustar(
        df_principal["frecuencia"].to_numpy(),
        df_principal["ticket_medio_acotado"].to_numpy(),
    )
    print(
        f"      BG/NBD     : r={bgnbd.r:.4f}, alpha={bgnbd.alpha:.4f}, a={bgnbd.a:.4f}, b={bgnbd.b:.4f} "
        f"(LL={bgnbd.log_verosimilitud_:,.1f})"
    )
    print(
        f"      Gamma-Gamma: p={gg.p:.4f}, q={gg.q:.4f}, gamma={gg.gamma:.2f} "
        f"(ticket medio poblacional={gg.ticket_medio_poblacional:.1f} GBP)"
    )

    print("5/8 · Entrenando líneas base y modelo supervisado LightGBM (Hurdle)...")
    regla_ritmo = ReglaRitmoHistorico()
    regla_rfm = ReglaQuintilesRFM().ajustar(df_previo)
    modelo_lgb = ModeloSupervisadoCLV(semilla=config.SEMILLA).ajustar(df_previo)
    modelo_lgb_estacional = ModeloSupervisadoCLV(semilla=config.SEMILLA).ajustar(df_estacional)

    pred_bgnbd_gg = calcular_clv(
        bgnbd=bgnbd,
        gg=gg,
        x=df_principal["frecuencia"].to_numpy(),
        t_x=df_principal["recencia_semanas"].to_numpy(),
        T=df_principal["edad_semanas"].to_numpy(),
        m_x=df_principal["ticket_medio_acotado"].to_numpy(),
        horizonte_meses=config.HORIZONTE_MESES,
        margen_bruto=config.MARGEN_BRUTO,
        tasa_descuento_mensual=config.TASA_DESCUENTO_MENSUAL,
    )
    predicciones = {
        "Ritmo histórico (heurística)": regla_ritmo.predecir(df_principal),
        "Tabla RFM por tramos": regla_rfm.predecir(df_principal),
        "LightGBM supervisado": modelo_lgb.predecir(df_principal),
        "BG/NBD + Gamma-Gamma": pred_bgnbd_gg,
    }

    print("6/8 · Evaluando en el periodo de prueba (holdout de 183 días) y segmentando cartera...")
    metricas = evaluacion.evaluar_modelos_holdout(df_principal, predicciones)
    deciles = evaluacion.calibracion_por_deciles(df_principal, pred_bgnbd_gg)
    cal_frec = evaluacion.calibracion_por_frecuencia(df_principal, pred_bgnbd_gg)
    curvas = evaluacion.curvas_captura_ingreso(df_principal, predicciones)

    # LightGBM: ventana previa (dic-jun, la usada en la tabla principal) frente a ventana de la misma estación
    pred_lgb_estacional = modelo_lgb_estacional.predecir(df_principal)
    lgb_ventanas = evaluacion.evaluar_modelos_holdout(
        df_principal,
        {
            "LightGBM · ventana previa (corte 2010-12-09, objetivo dic–jun)": predicciones["LightGBM supervisado"],
            "LightGBM · misma estación (corte 2010-06-09, objetivo jun–dic)": pred_lgb_estacional,
            "BG/NBD + Gamma-Gamma": pred_bgnbd_gg,
        },
    )
    lgb_ventanas.insert(1, "clientes_entrenamiento", [len(df_previo), len(df_estacional), len(df_principal)])

    print("      Bootstrap pareado BG/NBD + GG frente a LightGBM (1.000 remuestreos)...")
    bootstrap = evaluacion.bootstrap_comparacion(
        df_principal,
        pred_bgnbd_gg,
        predicciones["LightGBM supervisado"],
        n_bootstrap=config.N_BOOTSTRAP,
        semilla=config.SEMILLA,
    )

    cartera, resumen_seg = segmentacion.construir_tabla_segmentos(df_principal, pred_bgnbd_gg)
    politicas = segmentacion.validar_politicas_holdout(cartera)
    escenario = segmentacion.simular_escenario_politicas(cartera)
    sensibilidad = segmentacion.sensibilidad_escenario(cartera)

    metricas.to_csv(config.REPORTES / "comparacion_modelos.csv", index=False)
    deciles.to_csv(config.REPORTES / "calibracion_deciles.csv", index=False)
    cal_frec.to_csv(config.REPORTES / "calibracion_frecuencia.csv", index=False)
    curvas.to_csv(config.REPORTES / "curvas_captura.csv", index=False)
    resumen_seg.to_csv(config.REPORTES / "resumen_segmentos.csv", index=False)
    politicas.to_csv(config.REPORTES / "politicas_retencion.csv", index=False)
    escenario.to_csv(config.REPORTES / "politicas_escenario.csv", index=False)
    sensibilidad.to_csv(config.REPORTES / "politicas_sensibilidad.csv", index=False)
    bootstrap.to_csv(config.REPORTES / "bootstrap_bgnbd_vs_lightgbm.csv", index=False)
    lgb_ventanas.to_csv(config.REPORTES / "lightgbm_ventanas.csv", index=False)

    # Cartera completa puntuada (la usa la app Streamlit para buscar clientes)
    cartera[
        [
            "customer_id", "pais", "frecuencia", "recencia_dias", "edad_dias", "inactividad_dias",
            "ticket_medio_acotado", "gasto_total_cal_acotado", "prob_vivo", "prob_recompra",
            "compras_esperadas", "ticket_esperado", "ingreso_esperado", "clv_neto", "segmento",
            "accion_recomendada", "margen_en_riesgo_gbp", "beneficio_neto_rescate_gbp",
            "conviene_contactar_rescate", "compras_holdout", "gasto_holdout_acotado",
        ]
    ].round(4).to_csv(config.REPORTES / "cartera_clientes.csv", index=False)

    top_riesgo = (
        cartera[cartera["conviene_contactar_rescate"]]
        .sort_values("beneficio_neto_rescate_gbp", ascending=False)
        .head(100)[
            [
                "customer_id",
                "frecuencia",
                "recencia_dias",
                "edad_dias",
                "inactividad_dias",
                "ticket_medio_acotado",
                "prob_vivo",
                "prob_recompra",
                "compras_esperadas",
                "ticket_esperado",
                "clv_neto",
                "margen_en_riesgo_gbp",
                "beneficio_neto_rescate_gbp",
                "segmento",
            ]
        ]
    )
    top_riesgo.to_csv(config.REPORTES / "top100_clientes_rescate.csv", index=False)

    print("7/8 · Generando las 9 figuras analíticas en reports/figures/...")
    graficos.fig_01_limpieza_y_corte(auditoria, compras_diarias)
    graficos.fig_02_independencia_frecuencia_ticket(df_principal, tabla_indep)
    graficos.fig_03_ajuste_bgnbd_frecuencia(df_principal, bgnbd, cal_frec)
    graficos.fig_04_probabilidad_vivo_matriz(bgnbd)
    graficos.fig_05_calibracion_vivo_y_compras(deciles)
    graficos.fig_06_shrinkage_gamma_gamma(df_principal, gg)
    graficos.fig_07_comparacion_modelos_clv(metricas, curvas)
    graficos.fig_08_segmentos_y_campana_retencion(resumen_seg, politicas)
    graficos.fig_09_sensibilidad_escenario(sensibilidad)

    print("8/8 · Guardando artefacto del modelo y metadatos...")
    umbral_alto_valor = float(np.percentile(cartera["clv_neto"], 75))
    metadata = {
        "nombre_modelo": "clv_bgnbd_gamma_gamma",
        "version": "1.1.0",
        "fuente_datos": "UCI Online Retail II (01/12/2009 - 09/12/2011)",
        "ventana_calibracion": f"{config.FECHA_INICIO} a {config.FECHA_CORTE_CAL} (555 días)",
        "ventana_holdout": f"2011-06-10 a {config.FECHA_FIN} (183 días = 6 meses)",
        "clientes_calibracion": int(len(df_principal)),
        "clientes_repetidores": int((df_principal["frecuencia"] >= 1).sum()),
        "tasa_recompra_holdout": round(float(df_principal["vivo_holdout"].mean()), 4),
        "parametros_bgnbd": {
            "r": round(bgnbd.r, 4),
            "alpha_semanas": round(bgnbd.alpha, 4),
            "a": round(bgnbd.a, 4),
            "b": round(bgnbd.b, 4),
            "log_verosimilitud": round(bgnbd.log_verosimilitud_, 2),
            "tasa_media_compra_semanal": round(bgnbd.r / bgnbd.alpha, 4),
            "prob_media_abandono_post_compra": round(bgnbd.a / (bgnbd.a + bgnbd.b), 4),
        },
        "parametros_gamma_gamma": {
            "p": round(gg.p, 4),
            "q": round(gg.q, 4),
            "gamma": round(gg.gamma, 2),
            "ticket_medio_poblacional_gbp": round(gg.ticket_medio_poblacional, 2),
            "log_verosimilitud": round(gg.log_verosimilitud_, 2),
        },
        "supuestos_negocio": {
            "horizonte_meses": config.HORIZONTE_MESES,
            "margen_bruto": config.MARGEN_BRUTO,
            "tasa_descuento_mensual": config.TASA_DESCUENTO_MENSUAL,
            "costo_contacto_retencion_gbp": config.COSTO_CONTACTO_RETENCION,
            "efectividad_rescate_supuesta": config.EFECTIVIDAD_RESCATE,
            "zona_riesgo": "0,20 <= P(vivo) < 0,70 y frecuencia >= 2",
            "tope_ticket_mayorista_gbp": config.TOPE_TICKET_MAYORISTA,
        },
        "metricas_holdout": metricas.to_dict(orient="records"),
    }

    artefacto = {
        "bgnbd": bgnbd,
        "gg": gg,
        "umbral_alto_valor": umbral_alto_valor,
        "metadata": metadata,
    }
    joblib.dump(artefacto, config.MODELOS / "modelo_clv.joblib")
    (config.MODELOS / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    resultados_resumen = {
        "metadata": metadata,
        "auditoria_limpieza": auditoria.to_dict(orient="records"),
        "supuesto_gamma_gamma": tabla_indep.to_dict(orient="records"),
        "comparacion_modelos": metricas.to_dict(orient="records"),
        "segmentos": resumen_seg.to_dict(orient="records"),
        "politicas_retencion_holdout": politicas.to_dict(orient="records"),
        "politicas_escenario": escenario.to_dict(orient="records"),
        "bootstrap_bgnbd_vs_lightgbm": bootstrap.to_dict(orient="records"),
        "lightgbm_ventanas": lgb_ventanas.to_dict(orient="records"),
    }
    (config.REPORTES / "resultados.json").write_text(
        json.dumps(resultados_resumen, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Listo en {time.perf_counter() - t0:.1f} s.")


if __name__ == "__main__":
    main()
