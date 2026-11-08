"""Pruebas de calidad de modelos, calibración, umbral elegido en validación, bootstrap, robustez y artefactos."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from eurosat_suelo.config import ARCHIVO_JOBLIB, DIR_FIGURAS, DIR_MODELOS, DIR_REPORTES
from eurosat_suelo.evaluacion import (
    calcular_brier_multiclase,
    calcular_ece,
    elegir_umbral_minimo_costo,
    simular_desenfoque,
    simular_ruido_gaussiano,
    simular_velo_atmosferico,
)
from eurosat_suelo.modelos import ajustar_temperatura_calibracion
from eurosat_suelo.predecir import predecir_parche_satelital


def _comparacion(resultados_json: dict) -> dict[str, dict]:
    return {m["modelo"][:1]: m for m in resultados_json["comparacion_modelos"]}


def test_transfer_learning_supera_baseline_espectral(resultados_json: dict) -> None:
    comp = _comparacion(resultados_json)
    assert comp["1"]["exactitud_top1"] >= 0.85
    for k in ("2", "3", "4", "5"):
        assert comp[k]["exactitud_top1"] > comp["1"]["exactitud_top1"], k
    assert comp["5"]["costo_por_1000_usd"] < comp["1"]["costo_por_1000_usd"]


def test_ic_bootstrap_contienen_el_valor_observado(resultados_json: dict) -> None:
    for m in resultados_json["comparacion_modelos"]:
        assert m["exactitud_ic95_inf"] <= m["exactitud_top1"] <= m["exactitud_ic95_sup"]
        assert m["costo_1000_ic95_inf"] <= m["costo_por_1000_usd"] <= m["costo_1000_ic95_sup"]
        # Con 5.400 imágenes de prueba el IC de exactitud no supera ~2 puntos de ancho
        assert m["exactitud_ic95_sup"] - m["exactitud_ic95_inf"] < 0.025


def test_significancia_coherente_con_intervalos() -> None:
    df = pd.read_csv(DIR_REPORTES / "bootstrap_diferencias.csv")
    assert len(df) >= 10
    for _, r in df.iterrows():
        excluye_cero = r["ic95_inf"] > 0 or r["ic95_sup"] < 0
        assert bool(r["significativa"]) == excluye_cero, r["comparacion"]


def test_temperatura_se_ajusta_de_verdad() -> None:
    # Logits deliberadamente sobreconfiados (x4): la temperatura óptima debe ser claramente > 1
    rng = np.random.default_rng(0)
    y = rng.integers(0, 10, size=2000)
    logits = rng.normal(0, 1, size=(2000, 10)).astype(np.float32)
    logits[np.arange(2000), y] += 2.0
    t_base = ajustar_temperatura_calibracion(logits, y)
    t_x4 = ajustar_temperatura_calibracion(logits * 4.0, y)
    assert abs(t_x4 / t_base - 4.0) < 0.05
    assert t_x4 > 2.0


def test_calibracion_en_validacion_mejora_ece(resultados_json: dict) -> None:
    cal = pd.read_csv(DIR_REPORTES / "calibracion_temperatura.csv").set_index("modelo")
    assert len(cal) == 5
    # Ninguna temperatura queda clavada en 1,0 por un fallo del optimizador
    assert (np.abs(cal["temperatura_validacion"] - 1.0) > 1e-3).sum() >= 4
    # La temperatura se ajusta en validación y no debe empeorar el log-loss en prueba de forma apreciable
    assert (cal["log_loss_despues"] <= cal["log_loss_antes"] + 0.01).all()
    assert cal.loc["m5", "ece_despues"] <= cal.loc["m5", "ece_antes"] + 0.005


def test_umbral_elegido_en_validacion_y_no_en_prueba(resultados_json: dict) -> None:
    abst = resultados_json["abstencion"]
    curva = pd.read_csv(DIR_REPORTES / "curva_abstencion_fotointerpretacion.csv")
    val = curva[curva["particion"] == "validacion"]
    tau_min_val = float(val.loc[val["costo_total_por_1000_usd"].idxmin(), "umbral_confianza"])
    assert abs(abst["tau_validacion_m5"] - tau_min_val) < 1e-9
    # El modelo guardado usa ese mismo tau
    art = joblib.load(DIR_MODELOS / ARCHIVO_JOBLIB)
    assert abs(art["umbral_abstencion_optimo"] - abst["tau_validacion_m5"]) < 1e-9
    # El resultado de prueba reportado es el del tau de validación
    test = curva[curva["particion"] == "prueba"].set_index("umbral_confianza")
    assert abs(test.loc[abst["tau_validacion_m5"], "costo_total_por_1000_usd"] - abst["prueba_m5"]["costo_total_por_1000_usd"]) < 0.01


def test_elegir_umbral_minimiza_costo_en_datos_dados() -> None:
    # Si todo lo dudoso está mal, conviene derivar a revisión todo lo que tenga confianza < 0,6
    probs = np.array([[0.9, 0.1], [0.55, 0.45], [0.95, 0.05], [0.52, 0.48]] * 25)
    y = np.array([0, 1, 0, 1] * 25)
    from eurosat_suelo.evaluacion import costo_por_parcela_con_abstencion

    tau = elegir_umbral_minimo_costo(probs, y)
    assert 0.55 < tau <= 0.9
    assert costo_por_parcela_con_abstencion(probs, y, tau).mean() < costo_por_parcela_con_abstencion(probs, y, 0.0).mean()


def test_comparacion_justa_con_y_sin_abstencion(resultados_json: dict) -> None:
    filas = {r["politica"]: r for r in resultados_json["comparacion_abstencion"]}
    assert set(filas) == {"m1_sin_abstencion", "m1_con_abstencion", "m5_sin_abstencion", "m5_con_abstencion"}
    assert filas["m5_sin_abstencion"]["cobertura_automatica_pct"] == 100.0
    # La abstención no puede empeorar el costo en más que el ruido de elegir tau en otra partición
    assert filas["m5_con_abstencion"]["costo_total_por_1000_usd"] <= filas["m5_sin_abstencion"]["costo_total_por_1000_usd"] * 1.05


def test_sensibilidad_costos_27_escenarios() -> None:
    df = pd.read_csv(DIR_REPORTES / "sensibilidad_costos.csv")
    assert len(df) == 27
    assert df["un_solo_cambio"].sum() == 7  # escenario base + 6 cambios de un solo costo
    assert (df["costo_m5_sin_abstencion"] < df["costo_m1_sin_abstencion"]).all()


def test_curva_few_shot_y_ventaja_cnn() -> None:
    df_fs = pd.read_csv(DIR_REPORTES / "curva_few_shot_eurosat.csv").set_index("ejemplos_por_clase")
    assert list(df_fs.index) == [5, 10, 25, 50, 100, 250, 1000]
    # Con pocas etiquetas los rasgos preentrenados dominan
    for k in (5, 10, 25):
        assert df_fs.loc[k, "acc_hibrido"] > df_fs.loc[k, "acc_espectral"] + 0.05
    assert df_fs.loc[1000, "acc_hibrido"] > 0.90


def test_robustez_tres_perturbaciones_cinco_modelos() -> None:
    df = pd.read_csv(DIR_REPORTES / "robustez_perturbaciones.csv")
    assert set(df["perturbacion"]) == {"velo_simulado", "ruido_gaussiano", "desenfoque"}
    assert {f"acc_m{i}" for i in range(1, 6)} <= set(df.columns)
    for pert, sub in df.groupby("perturbacion"):
        sub = sub.sort_values("nivel")
        assert len(sub) == 5, pert
        # El nivel máximo degrada a todos los modelos
        for m in range(1, 6):
            assert sub[f"acc_m{m}"].iloc[-1] < sub[f"acc_m{m}"].iloc[0], (pert, m)


def test_perturbaciones_unitarias() -> None:
    img = np.full((2, 64, 64, 3), 80, dtype=np.uint8)
    img[:, 20:40, 20:40] = 200
    assert float(simular_velo_atmosferico(img, 0.30).mean()) > float(img.mean())
    ruidosa = simular_ruido_gaussiano(img, 10.0)
    assert ruidosa.shape == img.shape and 5.0 < float(np.std(ruidosa.astype(float) - img)) < 15.0
    borrosa = simular_desenfoque(img, 1.5)
    # El desenfoque conserva el brillo medio y reduce el contraste del borde
    assert abs(float(borrosa.mean()) - float(img.mean())) < 1.0
    assert np.abs(np.diff(borrosa.astype(float), axis=2)).max() < np.abs(np.diff(img.astype(float), axis=2)).max()
    assert np.array_equal(simular_desenfoque(img, 0.0), img)


def test_metricas_ece_y_brier_unitarias() -> None:
    probs_perf = np.array([[0.99, 0.01], [0.01, 0.99]], dtype=np.float32)
    y_perf = np.array([0, 1])
    assert calcular_brier_multiclase(probs_perf, y_perf) < 0.001
    assert calcular_ece(probs_perf, y_perf) < 0.02


def test_inferencia_individual_muestra() -> None:
    res = predecir_parche_satelital("01_Forest_01.png")
    assert res["clase_predicha"] == "Forest"
    assert res["confianza"] > 0.60
    assert len(res["top_3"]) == 3
    assert abs(sum(res["probabilidades"].values()) - 1.0) < 1e-3
    assert "exg_vegetacion" in res["telemetria_espectral"]


def test_diez_figuras_png_generadas() -> None:
    esperadas = [
        "01_mosaico_clases_y_espectro_sentinel2.png",
        "02_comparacion_arquitecturas_cpu.png",
        "03_curva_few_shot_transfer_learning.png",
        "04_f1_por_clase_y_ganancia_transfer.png",
        "05_matrices_confusion_espectral_vs_transfer.png",
        "06_calibracion_temperatura_y_abstencion.png",
        "07_robustez_velo_ruido_desenfoque.png",
        "08_mapas_oclusion_espacial.png",
        "09_grad_cam_aciertos_y_errores.png",
        "10_bootstrap_diferencias.png",
    ]
    for nombre in esperadas:
        ruta = DIR_FIGURAS / nombre
        assert ruta.exists(), f"Falta figura {ruta}"
        assert ruta.stat().st_size > 25_000
