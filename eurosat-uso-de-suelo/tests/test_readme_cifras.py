"""Las cifras clave del README y de la ficha deben salir de reports/ (no escritas a mano).

1. Las métricas de reports/resultados.json se recalculan desde las predicciones de prueba guardadas.
2. Las cifras principales aparecen en README.md y MODEL_CARD.md con el formato del texto.
3. El README no usa bloques que GitHub no renderiza ni cifras viejas, y el análisis anual/permanente sale de las matrices de confusión.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from eurosat_suelo.config import DIR_REPORTES, RAIZ_PROYECTO
from eurosat_suelo.evaluacion import tipos_de_error


def _pct(x: float) -> str:
    """0,9698 -> '96,98 %'"""
    return f"{x * 100:.2f} %".replace(".", ",")


def _usd(x: float) -> str:
    """3350.93 -> '3.350,93'"""
    return f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


@pytest.fixture(scope="module")
def readme() -> str:
    return (RAIZ_PROYECTO / "README.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def ficha() -> str:
    return (RAIZ_PROYECTO / "MODEL_CARD.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def pred() -> pd.DataFrame:
    return pd.read_parquet(DIR_REPORTES / "predicciones_test.parquet")


def test_metricas_se_recalculan_desde_predicciones(resultados_json: dict, pred: pd.DataFrame) -> None:
    assert len(pred) == resultados_json["particiones"]["prueba"] == 5_400
    y = pred["y_real"].to_numpy()
    for i, fila in enumerate(resultados_json["comparacion_modelos"], start=1):
        p = pred[f"pred_m{i}"].to_numpy()
        assert abs(float(np.mean(p == y)) - fila["exactitud_top1"]) < 6e-5
        _, criticos = tipos_de_error(y, p)
        assert int(criticos.sum()) == fila["errores_inter_macro"]


def test_politica_abstencion_se_recalcula(resultados_json: dict, pred: pd.DataFrame) -> None:
    tau = resultados_json["abstencion"]["tau_validacion_m5"]
    auto = pred["conf_m5"].to_numpy() >= tau
    y, p = pred["y_real"].to_numpy(), pred["pred_m5"].to_numpy()
    esperado = resultados_json["abstencion"]["prueba_m5"]
    assert abs(auto.mean() * 100 - esperado["cobertura_automatica_pct"]) < 0.01
    assert abs(np.mean(p[auto] == y[auto]) * 100 - esperado["exactitud_en_automaticas_pct"]) < 0.01


def test_readme_contiene_cifras_clave(resultados_json: dict, readme: str) -> None:
    esperadas = []
    for m in resultados_json["comparacion_modelos"]:
        esperadas += [_pct(m["exactitud_top1"]), _usd(m["costo_por_1000_usd"]), str(m["errores_inter_macro"])]
        esperadas.append(f"[{_pct(m['exactitud_ic95_inf'])[:-2]}; {_pct(m['exactitud_ic95_sup'])[:-2]}]")
    abst = resultados_json["abstencion"]
    esperadas += [
        f"τ = {abst['tau_validacion_m5']:.2f}".replace(".", ","),
        _pct(abst["prueba_m5"]["exactitud_en_automaticas_pct"] / 100),
        _pct(abst["prueba_m5"]["cobertura_automatica_pct"] / 100),
        _usd(abst["prueba_m5"]["costo_total_por_1000_usd"]),
    ]
    for c in resultados_json["calibracion"]:
        esperadas.append(f"{c['ece_antes']:.4f} → **{c['ece_despues']:.4f}**".replace(".", ","))
    onnx = resultados_json["onnx"]
    lote1 = next(l for l in onnx["latencias"] if l["tamano_lote"] == 1)
    esperadas += [f"{lote1['pytorch_ms_por_lote']:.2f} ms".replace(".", ","), f"{lote1['onnx_ms_por_lote']:.2f} ms".replace(".", ",")]
    faltan = [e for e in esperadas if e not in readme]
    assert not faltan, f"Cifras de reports/ que no aparecen en el README: {faltan}"


def test_readme_significancia_coincide_con_bootstrap(readme: str) -> None:
    df = pd.read_csv(DIR_REPORTES / "bootstrap_diferencias.csv")
    m2m1 = df[(df["comparacion"] == "m2 − m1") & (df["metrica"] == "exactitud")].iloc[0]
    assert not m2m1["significativa"]
    assert "| M2 − M1 (CNN congelada vs. índices) | " in readme and "| **No** |" in readme
    texto = f"[{m2m1['ic95_inf'] * 100:+.2f}; {m2m1['ic95_sup'] * 100:+.2f}]".replace(".", ",").replace("-", "−")
    assert texto in readme


def test_readme_robustez_coincide_con_csv(readme: str) -> None:
    df = pd.read_csv(DIR_REPORTES / "robustez_perturbaciones.csv")
    fila = df[(df["perturbacion"] == "ruido_gaussiano") & (df["nivel"] == 10.0)].iloc[0]
    esperado = " | ".join(f"{fila[f'acc_m{i}'] * 100:.2f}".replace(".", ",") for i in range(1, 6))
    assert esperado in readme


def test_readme_particiones_y_formato_github(readme: str) -> None:
    assert "16.200 de entrenamiento · 5.400 de validación · 5.400 de prueba" in readme
    assert "carousel" not in readme  # GitHub no renderiza ````carousel
    bloque_mermaid = readme.split("```mermaid")[1].split("```")[0]
    assert "\\n" not in bloque_mermaid and "<br/>" in bloque_mermaid
    for figura in re.findall(r"\(reports/figures/([^)]+\.png)\)", readme):
        assert (DIR_REPORTES / "figures" / figura).exists(), figura


def test_readme_sin_cifras_antiguas(readme: str, ficha: str) -> None:
    for vieja in ["98,55 %", "4.022,50", "93,67 %", "τ = 0,65", "1.200 escenas", "Ensamble Calibrado"]:
        assert vieja not in readme, vieja
        assert vieja not in ficha, vieja


def test_ficha_coherente(resultados_json: dict, ficha: str) -> None:
    m5 = resultados_json["comparacion_modelos"][-1]
    assert _pct(m5["exactitud_top1"]) in ficha
    assert _usd(resultados_json["abstencion"]["prueba_m5"]["costo_total_por_1000_usd"]) in ficha
    assert "supuest" in ficha.lower()


def test_sin_rutas_personales_en_entregables() -> None:
    patron = re.compile("|".join([r"C:\\\\?Users", "/c/" + "Users", "gua" + "rn"]), re.IGNORECASE)
    rutas = list((RAIZ_PROYECTO / "notebooks").glob("*.ipynb")) + [
        RAIZ_PROYECTO / "README.md",
        RAIZ_PROYECTO / "MODEL_CARD.md",
        RAIZ_PROYECTO / "docs" / "GUIA_PASO_A_PASO.md",
        DIR_REPORTES / "resultados.json",
    ]
    for ruta in rutas:
        assert not patron.search(ruta.read_text(encoding="utf-8")), ruta.name


def test_sin_menciones_a_versiones_anteriores(readme: str, ficha: str) -> None:
    guia = (RAIZ_PROYECTO / "docs" / "GUIA_PASO_A_PASO.md").read_text(encoding="utf-8")
    for texto in (readme, ficha, guia):
        assert "versión anterior" not in texto.lower()


def test_confusion_anual_permanente_coincide_con_matrices(readme: str, ficha: str) -> None:
    """Los conteos del análisis cultivo anual vs. permanente salen de las matrices guardadas en reports/."""
    m1 = pd.read_csv(DIR_REPORTES / "matriz_confusion_espectral.csv", index_col=0)
    m5 = pd.read_csv(DIR_REPORTES / "matriz_confusion_mejor_modelo.csv", index_col=0)
    soporte = int(m5.loc[["AnnualCrop", "PermanentCrop"]].to_numpy().sum())
    assert soporte == 1_134
    for m, etiqueta in ((m1, "M1"), (m5, "M5")):
        a_p, p_a = int(m.loc["AnnualCrop", "PermanentCrop"]), int(m.loc["PermanentCrop", "AnnualCrop"])
        total = a_p + p_a
        pct = f"{total / soporte * 100:.2f}".replace(".", ",")
        assert f"| {a_p} | {p_a} | " in readme or f"{a_p} | {p_a}" in readme, etiqueta
        assert f"{total} ({pct} %)" in readme or f"**{total} ({pct} %)**" in readme, etiqueta
    assert "10 de 1.134" in ficha
