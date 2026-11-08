"""Pipeline completo: entrenamiento de 5 modelos, calibración en validación, umbral de abstención elegido en
validación, evaluación en prueba con IC bootstrap, sensibilidad de costos, Few-Shot, robustez y exportación.

Regla de oro: todo lo que se AJUSTA (temperaturas, peso del ensamble, umbral tau, época del fine-tuning)
se decide con la partición de validación (5.400 imágenes). La de prueba (5.400) solo se usa para reportar.
"""

from __future__ import annotations

import itertools
import json
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss

from eurosat_suelo.config import (
    ARCHIVO_JOBLIB,
    ARCHIVO_PT,
    CLASES_EUROSAT,
    COSTO_AUDITORIA_CAMPO_USD,
    COSTO_ERROR_CRITICO_INTER_MACRO_USD,
    COSTO_ERROR_MISMA_MACROCATEGORIA_USD,
    DIR_DATOS,
    DIR_MODELOS,
    DIR_REPORTES,
    MODELOS,
    NOMBRES_ESPANOL,
    SEMILLA,
)
from eurosat_suelo.datos import cargar_subconjunto_eurosat, extraer_vector_handcrafted_imagen
from eurosat_suelo.evaluacion import (
    PERTURBACIONES,
    bootstrap_metricas_pareado,
    calcular_brier_multiclase,
    calcular_ece,
    costo_por_parcela_con_abstencion,
    curva_fiabilidad,
    elegir_umbral_minimo_costo,
    evaluar_curva_abstencion,
    evaluar_predicciones_multiclase,
    intervalo,
    metricas_desglosadas_por_clase,
    resumen_politica_abstencion,
)
from eurosat_suelo.exportar_onnx import exportar_modelos_onnx, verificar_y_medir_onnx
from eurosat_suelo.modelos import (
    ajustar_temperatura_calibracion,
    aplicar_temperatura,
    entrenar_finetuning_parcial_cpu,
    entrenar_fusion_cnn_espectral,
    entrenar_linear_probe_cnn,
    entrenar_modelo_espectral_clasico,
    extraer_embeddings_cnn,
    predecir_logits_finetuning,
    probs_a_logits,
)

EPOCAS_FINETUNING = 10
N_BOOTSTRAP = 2000


def logits_cinco_modelos(art: dict[str, Any], red_ft: torch.nn.Module, imgs: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Logits SIN calibrar de los modelos 1 a 4 (log-probabilidades en los de sklearn) y tiempos en segundos."""
    t: dict[str, float] = {}
    t0 = time.perf_counter()
    x_spec = extraer_vector_handcrafted_imagen(imgs)
    t["espectral"] = time.perf_counter() - t0
    x_emb, t["embeddings"] = extraer_embeddings_cnn(imgs, batch_size=128)

    salida: dict[str, np.ndarray] = {}
    t0 = time.perf_counter()
    salida["m1"] = probs_a_logits(art["modelo_espectral"].predict_proba(x_spec))
    t["m1"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    salida["m2"] = probs_a_logits(art["modelo_probe_cnn"].predict_proba(x_emb))
    t["m2"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    salida["m3"] = probs_a_logits(art["modelo_fusion_hibrido"].predict_proba(np.hstack([x_emb, x_spec])))
    t["m3"] = time.perf_counter() - t0
    salida["m4"], t["m4"] = predecir_logits_finetuning(red_ft, imgs, batch_size=128, tta_diedrico=True)
    return salida, t


def probs_cinco_modelos(logits: dict[str, np.ndarray], temperaturas: dict[str, float], peso_ft: float) -> dict[str, np.ndarray]:
    """Probabilidades de los 5 modelos aplicando las temperaturas dadas (1,0 = sin calibrar).

    El modelo 5 mezcla M4 y M3 ya calibrados con peso `peso_ft` y luego recibe su propia temperatura.
    """
    p = {m: aplicar_temperatura(logits[m], temperaturas.get(m, 1.0)) for m in ("m1", "m2", "m3", "m4")}
    mezcla = peso_ft * p["m4"] + (1.0 - peso_ft) * p["m3"]
    p["m5"] = aplicar_temperatura(probs_a_logits(mezcla), temperaturas.get("m5", 1.0))
    return p


def ejecutar_curva_few_shot(
    x_spec_train: np.ndarray,
    x_emb_train: np.ndarray,
    y_train: np.ndarray,
    x_spec_test: np.ndarray,
    x_emb_test: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """Evalúa cómo escalan el clasificador espectral y el Transfer Learning de 5 a 1.000 imágenes por clase."""
    tamanos_por_clase = [5, 10, 25, 50, 100, 250, 1000]
    rng = np.random.default_rng(SEMILLA)
    filas: list[dict[str, Any]] = []

    for k in tamanos_por_clase:
        idxs: list[np.ndarray] = []
        for c in range(len(CLASES_EUROSAT)):
            pos = np.where(y_train == c)[0]
            idxs.append(rng.choice(pos, size=min(k, len(pos)), replace=False))
        sub_idx = np.concatenate(idxs)

        pred_spec = entrenar_modelo_espectral_clasico(x_spec_train[sub_idx], y_train[sub_idx]).predict(x_spec_test)
        pred_probe = entrenar_linear_probe_cnn(x_emb_train[sub_idx], y_train[sub_idx], c_reg=0.3).predict(x_emb_test)
        x_hyb_tr = np.hstack([x_emb_train[sub_idx], x_spec_train[sub_idx]])
        x_hyb_te = np.hstack([x_emb_test, x_spec_test])
        pred_hyb = entrenar_linear_probe_cnn(x_hyb_tr, y_train[sub_idx], c_reg=0.3).predict(x_hyb_te)

        filas.append(
            {
                "ejemplos_por_clase": int(k),
                "total_entrenamiento": int(len(sub_idx)),
                "acc_espectral": round(float(accuracy_score(y_test, pred_spec)), 4),
                "f1_espectral": round(float(f1_score(y_test, pred_spec, average="macro")), 4),
                "acc_cnn_probe": round(float(accuracy_score(y_test, pred_probe)), 4),
                "f1_cnn_probe": round(float(f1_score(y_test, pred_probe, average="macro")), 4),
                "acc_hibrido": round(float(accuracy_score(y_test, pred_hyb)), 4),
                "f1_hibrido": round(float(f1_score(y_test, pred_hyb, average="macro")), 4),
            }
        )
        print(f"      few-shot {k:>4}/clase -> espectral {filas[-1]['acc_espectral']:.4f} | híbrido {filas[-1]['acc_hibrido']:.4f}")
    return pd.DataFrame(filas)


def ejecutar_robustez(
    imgs_test: np.ndarray,
    y_test: np.ndarray,
    art: dict[str, Any],
    red_ft: torch.nn.Module,
) -> pd.DataFrame:
    """Exactitud de los 5 modelos (calibrados) bajo velo simulado, ruido gaussiano y desenfoque."""
    filas: list[dict[str, Any]] = []
    base: dict[str, float] | None = None
    for perturbacion, (funcion, niveles, _) in PERTURBACIONES.items():
        for nivel in [0.0, *niveles]:
            if nivel == 0.0 and base is not None:
                accs = base
            else:
                imgs_p = funcion(imgs_test, nivel, semilla=SEMILLA)
                logits, _ = logits_cinco_modelos(art, red_ft, imgs_p)
                probs = probs_cinco_modelos(logits, art["temperaturas"], art["peso_ensamble_finetuning"])
                accs = {m: float(accuracy_score(y_test, probs[m].argmax(1))) for m in MODELOS}
                if nivel == 0.0:
                    base = accs
            fila = {"perturbacion": perturbacion, "nivel": float(nivel)}
            fila.update({f"acc_{m}": round(a, 4) for m, a in accs.items()})
            filas.append(fila)
            print(f"      {perturbacion:<16} nivel {nivel:>5} -> " + " | ".join(f"{m} {a:.3f}" for m, a in accs.items()))
    return pd.DataFrame(filas)


def _parametros_sklearn(modelo_pipe: Any) -> int:
    clf = modelo_pipe.named_steps["clasificador"]
    if hasattr(clf, "coefs_"):  # MLP
        return int(sum(w.size for w in clf.coefs_) + sum(b.size for b in clf.intercepts_))
    if hasattr(clf, "coef_"):  # regresión logística
        return int(clf.coef_.size + clf.intercept_.size)
    # GBDT: número de hojas (valores aprendidos) en todos los árboles
    return int(sum(int(arbol.nodes["is_leaf"].sum()) for iteracion in clf._predictors for arbol in iteracion))


def entrenar_y_evaluar_todo() -> dict[str, Any]:
    """Ejecuta el pipeline completo sobre las particiones oficiales de EuroSAT (16.200 / 5.400 / 5.400)."""
    DIR_MODELOS.mkdir(parents=True, exist_ok=True)
    DIR_REPORTES.mkdir(parents=True, exist_ok=True)
    t_inicio = time.perf_counter()

    datos = cargar_subconjunto_eurosat()
    imgs, lbls, part = datos["imagenes"], datos["etiquetas"], datos["particion"]
    imgs_tr, y_tr = imgs[part == "train"], lbls[part == "train"]
    imgs_va, y_va = imgs[part == "val"], lbls[part == "val"]
    imgs_te, y_te = imgs[part == "test"], lbls[part == "test"]
    print(f"[1/9] Particiones: entrenamiento {len(y_tr):,} · validación {len(y_va):,} · prueba {len(y_te):,}")

    print("[2/9] Descriptores espectrales (54) y embeddings MobileNetV3 congelada (576-d)...")
    x_spec_tr = extraer_vector_handcrafted_imagen(imgs_tr)
    x_emb_tr, _ = extraer_embeddings_cnn(imgs_tr, batch_size=128)

    print("[3/9] Entrenando M1 (GBDT espectral), M2 (sonda lineal) y M3 (fusión MLP)...")
    m1 = entrenar_modelo_espectral_clasico(x_spec_tr, y_tr)
    m2 = entrenar_linear_probe_cnn(x_emb_tr, y_tr, c_reg=0.4)
    m3 = entrenar_fusion_cnn_espectral(x_emb_tr, x_spec_tr, y_tr)

    print(f"[4/9] Fine-tuning parcial de MobileNetV3-Small en CPU ({EPOCAS_FINETUNING} épocas, D4)...")
    red_ft, historial_ft = entrenar_finetuning_parcial_cpu(
        imgs_tr, y_tr, imgs_va, y_va, epocas=EPOCAS_FINETUNING, batch_size=128, tamano=96
    )
    for h in historial_ft:
        print(f"      época {h['epoca']:>2}: acc_val {h['acc_val']:.4f} ({h['tiempo_s']:.0f} s)")

    art: dict[str, Any] = {
        "modelo_espectral": m1,
        "modelo_probe_cnn": m2,
        "modelo_fusion_hibrido": m3,
        "clases": CLASES_EUROSAT,
        "nombres_es": NOMBRES_ESPANOL,
    }

    print("[5/9] Calibración en VALIDACIÓN: temperatura por modelo, peso del ensamble y umbral tau...")
    logits_va, _ = logits_cinco_modelos(art, red_ft, imgs_va)
    logits_te, tiempos_te = logits_cinco_modelos(art, red_ft, imgs_te)

    temperaturas = {m: ajustar_temperatura_calibracion(logits_va[m], y_va) for m in ("m1", "m2", "m3", "m4")}
    p_va_cal = {m: aplicar_temperatura(logits_va[m], temperaturas[m]) for m in ("m3", "m4")}
    rejilla_pesos = np.round(np.arange(0.0, 1.0001, 0.05), 2)
    perdidas = [log_loss(y_va, np.clip(w * p_va_cal["m4"] + (1 - w) * p_va_cal["m3"], 1e-7, 1), labels=list(range(10))) for w in rejilla_pesos]
    peso_ft = float(rejilla_pesos[int(np.argmin(perdidas))])
    mezcla_va = peso_ft * p_va_cal["m4"] + (1 - peso_ft) * p_va_cal["m3"]
    temperaturas["m5"] = ajustar_temperatura_calibracion(probs_a_logits(mezcla_va), y_va)
    art["temperaturas"] = {m: float(t) for m, t in temperaturas.items()}
    art["peso_ensamble_finetuning"] = peso_ft

    probs_va = probs_cinco_modelos(logits_va, temperaturas, peso_ft)
    probs_te = probs_cinco_modelos(logits_te, temperaturas, peso_ft)
    probs_te_sin_cal = probs_cinco_modelos(logits_te, {}, peso_ft)

    tau_m5 = elegir_umbral_minimo_costo(probs_va["m5"], y_va)
    tau_m1 = elegir_umbral_minimo_costo(probs_va["m1"], y_va)
    art["umbral_abstencion_optimo"] = tau_m5
    print(f"      T = {', '.join(f'{m}: {t:.3f}' for m, t in temperaturas.items())} | peso M4 = {peso_ft:.2f} | tau(M5) = {tau_m5:.2f}")

    print("[6/9] Métricas en PRUEBA, calibración antes/después y bootstrap pareado...")
    n_te = len(y_te)
    lat = {
        "m1": (tiempos_te["espectral"] + tiempos_te["m1"]) / n_te * 1000.0,
        "m2": (tiempos_te["embeddings"] + tiempos_te["m2"]) / n_te * 1000.0,
        "m3": (tiempos_te["embeddings"] + tiempos_te["espectral"] + tiempos_te["m3"]) / n_te * 1000.0,
        "m4": tiempos_te["m4"] / n_te * 1000.0,
    }
    lat["m5"] = lat["m3"] + lat["m4"]
    params_ft = int(sum(p.numel() for p in red_ft.parameters() if p.requires_grad))
    params = {"m1": _parametros_sklearn(m1), "m2": _parametros_sklearn(m2), "m3": _parametros_sklearn(m3), "m4": params_ft}
    params["m5"] = params["m3"] + params["m4"]

    comparacion = [
        evaluar_predicciones_multiclase(MODELOS[m], probs_te[m], y_te, latencia_ms_por_imagen=lat[m], parametros_entrenados=params[m])
        for m in MODELOS
    ]
    calibracion = []
    for m in MODELOS:
        calibracion.append(
            {
                "modelo": m,
                "nombre": MODELOS[m],
                "temperatura_validacion": round(float(temperaturas[m]), 4),
                "ece_antes": round(calcular_ece(probs_te_sin_cal[m], y_te), 4),
                "ece_despues": round(calcular_ece(probs_te[m], y_te), 4),
                "log_loss_antes": round(float(log_loss(y_te, np.clip(probs_te_sin_cal[m], 1e-7, 1), labels=list(range(10)))), 4),
                "log_loss_despues": round(float(log_loss(y_te, np.clip(probs_te[m], 1e-7, 1), labels=list(range(10)))), 4),
                "brier_antes": round(calcular_brier_multiclase(probs_te_sin_cal[m], y_te), 4),
                "brier_despues": round(calcular_brier_multiclase(probs_te[m], y_te), 4),
            }
        )

    # Políticas: sin abstención (tau = 0) y con el tau elegido en validación, para M1 y M5
    politicas = {
        "m1_sin_abstencion": ("m1", 0.0),
        "m1_con_abstencion": ("m1", tau_m1),
        "m5_sin_abstencion": ("m5", 0.0),
        "m5_con_abstencion": ("m5", tau_m5),
    }
    costos_parcela = {m: costo_por_parcela_con_abstencion(probs_te[m], y_te, 0.0) for m in MODELOS}
    for nombre, (m, tau) in politicas.items():
        costos_parcela[nombre] = costo_por_parcela_con_abstencion(probs_te[m], y_te, tau)
    preds_te = {m: probs_te[m].argmax(1) for m in MODELOS}
    preds_boot = {**preds_te, **{n: preds_te[m] for n, (m, _) in politicas.items()}}
    rep = bootstrap_metricas_pareado(y_te, preds_boot, costos_parcela, n_boot=N_BOOTSTRAP, semilla=SEMILLA)

    for fila, m in zip(comparacion, MODELOS):
        for metrica in ("exactitud", "f1_macro", "costo_1000"):
            lo, hi = intervalo(rep[m][metrica])
            fila[f"{metrica}_ic95_inf"] = round(lo, 4 if metrica != "costo_1000" else 2)
            fila[f"{metrica}_ic95_sup"] = round(hi, 4 if metrica != "costo_1000" else 2)

    def diferencia(a: str, b: str, metrica: str) -> dict[str, Any]:
        d = rep[a][metrica] - rep[b][metrica]
        lo, hi = intervalo(d)
        if metrica == "costo_1000":
            observada = costos_parcela[a].mean() * 1000 - costos_parcela[b].mean() * 1000
        elif metrica == "exactitud":
            observada = float(np.mean(preds_boot[a] == y_te) - np.mean(preds_boot[b] == y_te))
        else:
            observada = float(f1_score(y_te, preds_boot[a], average="macro") - f1_score(y_te, preds_boot[b], average="macro"))
        return {
            "comparacion": f"{a} − {b}",
            "metrica": metrica,
            "diferencia": round(float(observada), 4 if metrica != "costo_1000" else 2),
            "ic95_inf": round(lo, 4 if metrica != "costo_1000" else 2),
            "ic95_sup": round(hi, 4 if metrica != "costo_1000" else 2),
            "significativa": bool(lo > 0 or hi < 0),
        }

    pares = [("m2", "m1"), ("m3", "m2"), ("m4", "m3"), ("m5", "m4"), ("m5", "m3"), ("m4", "m1"), ("m5", "m1")]
    diferencias = [diferencia(a, b, met) for a, b in pares for met in ("exactitud", "f1_macro")]
    diferencias += [
        diferencia("m5_sin_abstencion", "m1_sin_abstencion", "costo_1000"),
        diferencia("m1_con_abstencion", "m1_sin_abstencion", "costo_1000"),
        diferencia("m5_con_abstencion", "m5_sin_abstencion", "costo_1000"),
        diferencia("m5_con_abstencion", "m1_con_abstencion", "costo_1000"),
    ]

    comparacion_justa = []
    for nombre, (m, tau) in politicas.items():
        r = resumen_politica_abstencion(probs_te[m], y_te, tau)
        lo, hi = intervalo(rep[nombre]["costo_1000"])
        r.update({"politica": nombre, "modelo": m, "costo_ic95_inf": round(lo, 2), "costo_ic95_sup": round(hi, 2)})
        comparacion_justa.append(r)

    print("[7/9] Sensibilidad a los costos supuestos (tau re-elegido en validación en cada escenario)...")
    sensibilidad = []
    base = {
        "costo_revision": COSTO_AUDITORIA_CAMPO_USD,
        "costo_misma_macro": COSTO_ERROR_MISMA_MACROCATEGORIA_USD,
        "costo_inter_macro": COSTO_ERROR_CRITICO_INTER_MACRO_USD,
    }
    for f_rev, f_misma, f_inter in itertools.product((0.5, 1.0, 2.0), repeat=3):
        costos = {
            "costo_revision": base["costo_revision"] * f_rev,
            "costo_misma_macro": base["costo_misma_macro"] * f_misma,
            "costo_inter_macro": base["costo_inter_macro"] * f_inter,
        }
        tau_esc = elegir_umbral_minimo_costo(probs_va["m5"], y_va, **costos)
        c_m1 = costo_por_parcela_con_abstencion(probs_te["m1"], y_te, 0.0, **costos).mean() * 1000
        c_m5 = costo_por_parcela_con_abstencion(probs_te["m5"], y_te, 0.0, **costos).mean() * 1000
        r = resumen_politica_abstencion(probs_te["m5"], y_te, tau_esc, **costos)
        sensibilidad.append(
            {
                **costos,
                "un_solo_cambio": int((f_rev != 1.0) + (f_misma != 1.0) + (f_inter != 1.0)) <= 1,
                "tau_validacion": tau_esc,
                "cobertura_automatica_pct": r["cobertura_automatica_pct"],
                "costo_m1_sin_abstencion": round(float(c_m1), 2),
                "costo_m5_sin_abstencion": round(float(c_m5), 2),
                "costo_m5_con_abstencion": r["costo_total_por_1000_usd"],
                "ahorro_abstencion_vs_m5_pct": round((1 - r["costo_total_por_1000_usd"] / c_m5) * 100, 1),
                "ahorro_m5_abst_vs_m1_pct": round((1 - r["costo_total_por_1000_usd"] / c_m1) * 100, 1),
            }
        )
    df_sens = pd.DataFrame(sensibilidad)

    print("[8/9] Curva Few-Shot y robustez (velo simulado, ruido gaussiano, desenfoque) en los 5 modelos...")
    x_spec_te = extraer_vector_handcrafted_imagen(imgs_te)
    x_emb_te, _ = extraer_embeddings_cnn(imgs_te, batch_size=128)
    df_fewshot = ejecutar_curva_few_shot(x_spec_tr, x_emb_tr, y_tr, x_spec_te, x_emb_te, y_te)
    df_robustez = ejecutar_robustez(imgs_te, y_te, art, red_ft)

    print("[9/9] Guardando modelos (.joblib, .pt, .onnx) y reportes...")
    torch.save(red_ft.state_dict(), DIR_MODELOS / ARCHIVO_PT)
    joblib.dump(art, DIR_MODELOS / ARCHIVO_JOBLIB, compress=3)
    exportar_modelos_onnx(red_ft)
    info_onnx = verificar_y_medir_onnx(red_ft, imgs_te)

    df_comp = pd.DataFrame(comparacion)
    df_comp.insert(0, "id", list(MODELOS))
    df_comp.to_csv(DIR_REPORTES / "comparacion_modelos.csv", index=False, encoding="utf-8")
    pd.DataFrame(calibracion).to_csv(DIR_REPORTES / "calibracion_temperatura.csv", index=False, encoding="utf-8")
    pd.DataFrame(diferencias).to_csv(DIR_REPORTES / "bootstrap_diferencias.csv", index=False, encoding="utf-8")
    pd.DataFrame(comparacion_justa).to_csv(DIR_REPORTES / "comparacion_abstencion.csv", index=False, encoding="utf-8")
    df_sens.to_csv(DIR_REPORTES / "sensibilidad_costos.csv", index=False, encoding="utf-8")
    pd.DataFrame(historial_ft).to_csv(DIR_REPORTES / "historial_finetuning.csv", index=False, encoding="utf-8")
    df_fewshot.to_csv(DIR_REPORTES / "curva_few_shot_eurosat.csv", index=False, encoding="utf-8")
    df_robustez.to_csv(DIR_REPORTES / "robustez_perturbaciones.csv", index=False, encoding="utf-8")

    curva_val = evaluar_curva_abstencion(probs_va["m5"], y_va).assign(particion="validacion")
    curva_test = evaluar_curva_abstencion(probs_te["m5"], y_te).assign(particion="prueba")
    df_abst = pd.concat([curva_val, curva_test], ignore_index=True)
    df_abst.to_csv(DIR_REPORTES / "curva_abstencion_fotointerpretacion.csv", index=False, encoding="utf-8")

    fiab = []
    for m in ("m1", "m4", "m5"):
        for etapa, p in (("antes", probs_te_sin_cal[m]), ("despues", probs_te[m])):
            fiab.append(curva_fiabilidad(p, y_te).assign(modelo=m, etapa=etapa))
    pd.concat(fiab, ignore_index=True).to_csv(DIR_REPORTES / "curva_fiabilidad.csv", index=False, encoding="utf-8")

    df_clases = metricas_desglosadas_por_clase(probs_te["m5"], y_te)
    df_clases_spec = metricas_desglosadas_por_clase(probs_te["m1"], y_te)
    df_clases["f1_espectral_clasico"] = df_clases_spec["f1"]
    df_clases["ganancia_f1_transfer"] = (df_clases["f1"] - df_clases_spec["f1"]).round(4)
    df_clases.to_csv(DIR_REPORTES / "metricas_por_clase.csv", index=False, encoding="utf-8")
    for nombre, m in (("matriz_confusion_mejor_modelo.csv", "m5"), ("matriz_confusion_espectral.csv", "m1")):
        pd.DataFrame(confusion_matrix(y_te, preds_te[m]), index=CLASES_EUROSAT, columns=CLASES_EUROSAT).to_csv(
            DIR_REPORTES / nombre, encoding="utf-8"
        )

    # Predicciones de prueba: permiten recalcular las cifras del README sin reentrenar (ver tests)
    pred_df = pd.DataFrame({"y_real": y_te})
    for m in MODELOS:
        pred_df[f"pred_{m}"] = preds_te[m]
        pred_df[f"conf_{m}"] = probs_te[m].max(1).astype(np.float32)
    pred_df.to_parquet(DIR_REPORTES / "predicciones_test.parquet", index=False)

    tau_oraculo = float(curva_test.loc[curva_test["costo_total_por_1000_usd"].idxmin(), "umbral_confianza"])
    auditoria_datos = json.loads((DIR_DATOS / "cifras_control_eurosat.json").read_text(encoding="utf-8"))
    resultados = {
        "proyecto": "ML_06_satelite_eurosat",
        "auditoria_datos": auditoria_datos,
        "particiones": {"entrenamiento": int(len(y_tr)), "validacion": int(len(y_va)), "prueba": int(len(y_te))},
        "costos_supuestos_usd": base,
        "comparacion_modelos": comparacion,
        "calibracion": calibracion,
        "ensamble": {"peso_finetuning": peso_ft, "peso_fusion": round(1 - peso_ft, 2)},
        "abstencion": {
            "tau_validacion_m5": tau_m5,
            "tau_validacion_m1": tau_m1,
            "tau_que_minimizaria_prueba_m5": tau_oraculo,
            "validacion_m5": resumen_politica_abstencion(probs_va["m5"], y_va, tau_m5),
            "prueba_m5": resumen_politica_abstencion(probs_te["m5"], y_te, tau_m5),
        },
        "comparacion_abstencion": comparacion_justa,
        "bootstrap": {"replicas": N_BOOTSTRAP, "diferencias": diferencias},
        "sensibilidad_costos": sensibilidad,
        "onnx": info_onnx,
        "curva_few_shot": df_fewshot.to_dict(orient="records"),
        "robustez": df_robustez.to_dict(orient="records"),
        "metricas_por_clase": df_clases.to_dict(orient="records"),
        "historial_finetuning": historial_ft,
        "minutos_pipeline": round((time.perf_counter() - t_inicio) / 60.0, 1),
    }
    (DIR_REPORTES / "resultados.json").write_text(json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8")
    (DIR_MODELOS / "metadata.json").write_text(
        json.dumps(
            {
                "version": "2.0.0",
                "backbone": "MobileNetV3-Small (ImageNet-1K) + 54 índices espectrales Sentinel-2",
                "particiones": resultados["particiones"],
                "exactitud_top1_test_m5": comparacion[-1]["exactitud_top1"],
                "f1_macro_test_m5": comparacion[-1]["f1_macro"],
                "umbral_abstencion_validacion": tau_m5,
                "temperaturas_validacion": art["temperaturas"],
                "peso_ensamble_finetuning": peso_ft,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return resultados


if __name__ == "__main__":
    res = entrenar_y_evaluar_todo()
    mejor = res["comparacion_modelos"][-1]
    print(
        f"[ok] {mejor['modelo']}: Top-1={mejor['exactitud_top1']*100:.2f}% | "
        f"Macro-F1={mejor['f1_macro']:.4f} | pipeline {res['minutos_pipeline']} min"
    )
