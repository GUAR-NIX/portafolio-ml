"""Figuras del README en reports/figures/ (matplotlib, backend Agg).

Cada serie conserva su color en todas las figuras: real (gris oscuro), línea base
estacional (naranja), modelo día siguiente (azul, principal), día siguiente sin
clima (amarillo) y modelo a 1 hora (aqua, comparación).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from demanda_citibike.config import FIGURAS
from demanda_citibike.evaluacion import wape

COLORES = {
    "real": "#333333",
    "base_estacional": "#eb6834",
    "pred_dia": "#2a78d6",
    "pred_dia_sin_clima": "#eda100",
    "pred_1h": "#1baf7a",
}
ETIQUETAS = {
    "base_estacional": "Base estacional (t−168)",
    "pred_dia": "LightGBM día siguiente (principal)",
    "pred_dia_sin_clima": "LightGBM día siguiente sin clima",
    "pred_1h": "LightGBM a 1 hora (comparación)",
}

NOMBRES_FEATURES = {
    "lag_1": "Rezago 1 h",
    "lag_2": "Rezago 2 h",
    "lag_3": "Rezago 3 h",
    "lag_24": "Rezago 24 h (ayer)",
    "lag_48": "Rezago 48 h (anteayer)",
    "lag_168": "Rezago 168 h (semana)",
    "lag_336": "Rezago 336 h (2 semanas)",
    "media_3": "Media móvil 3 h",
    "media_24": "Media móvil 24 h",
    "media_168": "Media móvil 168 h",
    "media_24_d1": "Media 24 h hasta ayer",
    "media_168_d1": "Media 168 h hasta ayer",
    "temperatura_c": "Temperatura (°C)",
    "precipitacion_mm": "Precipitación (mm)",
    "lluvia_mm": "Lluvia (mm)",
    "viento_kmh": "Viento (km/h)",
    "es_fin_de_semana": "Fin de semana",
    "es_feriado": "Feriado",
    "hora": "Hora del día",
    "dia_semana": "Día de semana",
    "mes": "Mes",
    "estacion_cat": "Estación",
}


def familia_feature(feature: str) -> str:
    """Agrupa las variables en familias para leer la importancia SHAP."""
    if feature.startswith("lag_") or feature.startswith("media_"):
        return "Historia de demanda"
    if feature in {"temperatura_c", "precipitacion_mm", "lluvia_mm", "viento_kmh"}:
        return "Clima"
    if feature == "estacion_cat":
        return "Estación"
    return "Calendario"


def _guardar(fig, nombre: str) -> Path:
    FIGURAS.mkdir(parents=True, exist_ok=True)
    ruta = FIGURAS / nombre
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return ruta


def _estilo(ax) -> None:
    ax.grid(alpha=0.25)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)


def graficar_demanda_horaria(df: pd.DataFrame) -> Path:
    """Salidas promedio por hora, separando día laboral y fin de semana."""
    tmp = df.groupby(["hora", "es_fin_de_semana"])["salidas"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    for valor, etiqueta, color in [(0, "Día laboral", "#2a78d6"), (1, "Fin de semana", "#eb6834")]:
        s = tmp[tmp["es_fin_de_semana"] == valor].sort_values("hora")
        ax.plot(s["hora"], s["salidas"], marker="o", markersize=5, linewidth=2, label=etiqueta, color=color)
    ax.set_xlabel("Hora del día")
    ax.set_ylabel("Salidas promedio por estación y hora")
    ax.set_title("Demanda promedio por hora — Citi Bike 2024 (250 estaciones)")
    ax.set_xticks(range(0, 24, 2))
    _estilo(ax)
    ax.legend(frameon=False)
    return _guardar(fig, "01_demanda_horaria.png")


def graficar_prediccion_vs_real(df_test: pd.DataFrame) -> Path:
    """Salidas diarias reales vs pronóstico día siguiente, 1 hora y base en el mes de test."""
    cols = ["pred_dia", "pred_1h", "base_estacional"]
    agg = df_test.groupby(df_test["fecha_hora"].dt.date)[["salidas", *cols]].sum().reset_index()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(agg["fecha_hora"], agg["salidas"], "o-", label="Real", color=COLORES["real"], linewidth=2.2, markersize=5)
    for c, estilo in zip(cols, ["s-", "d--", "^:"]):
        ax.plot(agg["fecha_hora"], agg[c], estilo, label=ETIQUETAS[c], color=COLORES[c], linewidth=1.8, markersize=5)
    ax.set_title("Salidas diarias de las 250 estaciones — mes de test (nunca visto)")
    ax.set_ylabel("Salidas totales del día")
    _estilo(ax)
    ax.legend(frameon=False)
    fig.autofmt_xdate()
    return _guardar(fig, "02_prediccion_vs_real.png")


def importancia_shap(modelo, X_sample: pd.DataFrame) -> pd.DataFrame:
    """Media de |SHAP| por variable (en salidas/hora) y su porcentaje del total."""
    import shap  # type: ignore

    valores = np.asarray(shap.TreeExplainer(modelo).shap_values(X_sample))
    if valores.ndim == 3:  # algunas versiones devuelven [salida, filas, features]
        valores = valores[0]
    imp = np.abs(valores).mean(axis=0)
    d = pd.DataFrame({"variable": X_sample.columns, "media_abs_shap": imp})
    d["pct"] = d["media_abs_shap"] / d["media_abs_shap"].sum() * 100
    d["familia"] = d["variable"].map(familia_feature)
    d["nombre"] = d["variable"].map(lambda c: NOMBRES_FEATURES.get(c, c))
    return d.sort_values("media_abs_shap", ascending=False).reset_index(drop=True)


def graficar_importancia(imp_dia: pd.DataFrame, imp_1h: pd.DataFrame) -> Path:
    """(a) Top variables del modelo principal; (b) cuota por familia en ambos modelos."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.3, 1]})
    top = imp_dia.head(12).iloc[::-1]
    ax1.barh(top["nombre"], top["media_abs_shap"], color=COLORES["pred_dia"], height=0.7)
    for y, (v, p) in enumerate(zip(top["media_abs_shap"], top["pct"])):
        ax1.text(v, y, f"  {p:.1f} %", va="center", fontsize=8, color="#555")
    ax1.set_xlabel("Media de |SHAP| (salidas por hora)")
    ax1.set_title("(a) Modelo día siguiente: top 12 variables")
    _estilo(ax1)

    familias = ["Historia de demanda", "Calendario", "Estación", "Clima"]
    fd = imp_dia.groupby("familia")["pct"].sum().reindex(familias).fillna(0)
    f1 = imp_1h.groupby("familia")["pct"].sum().reindex(familias).fillna(0)
    x = np.arange(len(familias))
    ax2.bar(x - 0.2, fd.values, 0.38, color=COLORES["pred_dia"], label="Día siguiente")
    ax2.bar(x + 0.2, f1.values, 0.38, color=COLORES["pred_1h"], label="1 hora")
    for xi, (a, b) in enumerate(zip(fd.values, f1.values)):
        ax2.text(xi - 0.2, a, f"{a:.1f}", ha="center", va="bottom", fontsize=8)
        ax2.text(xi + 0.2, b, f"{b:.1f}", ha="center", va="bottom", fontsize=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(familias, fontsize=9)
    ax2.set_ylabel("% de la media de |SHAP| total")
    ax2.set_title("(b) Cuota por familia de variables")
    _estilo(ax2)
    ax2.legend(frameon=False)
    fig.suptitle("Importancia de variables con SHAP (20 000 horas-estación del mes de test)")
    return _guardar(fig, "03_importancia_variables.png")


def graficar_error_por_hora(df_test: pd.DataFrame) -> Path:
    """WAPE por hora del día: base, día siguiente y 1 hora."""
    cols = ["base_estacional", "pred_dia", "pred_1h"]
    d = pd.DataFrame([
        {"hora": h, **{c: wape(g["salidas"].to_numpy(), g[c].to_numpy()) for c in cols}}
        for h, g in df_test.groupby("hora")
    ]).sort_values("hora")
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, c in enumerate(cols):
        ax.bar(x + (i - 1) * 0.28, d[c], 0.26, label=ETIQUETAS[c], color=COLORES[c])
    ax.set_xticks(x)
    ax.set_xticklabels(d["hora"].astype(int))
    ax.set_xlabel("Hora del día")
    ax.set_ylabel("WAPE (%)")
    ax.set_title("Error por hora del día en el mes de test")
    _estilo(ax)
    ax.legend(frameon=False)
    return _guardar(fig, "04_error_por_hora.png")


def graficar_intervalos(df_test: pd.DataFrame, cobertura_pct: float) -> Path:
    """Una semana de una estación con la banda p10–p90 del modelo día siguiente."""
    estacion = df_test.groupby("estacion")["salidas"].sum().idxmax()
    ini = df_test["fecha_hora"].min().normalize() + pd.Timedelta(days=7)
    sub = df_test[(df_test["estacion"] == estacion) & (df_test["fecha_hora"] >= ini)
                  & (df_test["fecha_hora"] < ini + pd.Timedelta(days=7))].sort_values("fecha_hora")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(sub["fecha_hora"], sub["pred_q10"], sub["pred_q90"], color=COLORES["pred_dia"], alpha=0.2,
                    label="Intervalo p10–p90 (nominal 80 %)")
    ax.plot(sub["fecha_hora"], sub["pred_dia"], color=COLORES["pred_dia"], linewidth=1.6, label="Pronóstico día siguiente")
    ax.plot(sub["fecha_hora"], sub["salidas"], color=COLORES["real"], linewidth=1.4, label="Real")
    ax.set_title(f"Intervalos por cuantiles, estación {estacion} — cobertura real en todo el test: {cobertura_pct:.1f} %")
    ax.set_ylabel("Salidas por hora")
    _estilo(ax)
    ax.legend(frameon=False, loc="upper left")
    fig.autofmt_xdate()
    return _guardar(fig, "05_intervalos_prediccion.png")


def tabla_lluvia(df_test: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """WAPE por condición (con lluvia, sin lluvia, total) para varias columnas de pronóstico."""
    grupos = {
        "Con lluvia": df_test[df_test["lluvia_mm"] > 0],
        "Sin lluvia": df_test[df_test["lluvia_mm"] == 0],
        "Total": df_test,
    }
    filas = []
    for etiqueta, g in grupos.items():
        fila = {"condicion": etiqueta, "horas_estacion": int(len(g))}
        for c in cols:
            fila[c] = round(wape(g["salidas"].to_numpy(), g[c].to_numpy()), 2)
        filas.append(fila)
    return pd.DataFrame(filas)


def graficar_error_lluvia(df_test: pd.DataFrame) -> Path:
    """Ablación de clima: WAPE con/sin lluvia para base, sin clima y con clima."""
    cols = ["base_estacional", "pred_dia_sin_clima", "pred_dia"]
    d = tabla_lluvia(df_test, cols)
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, c in enumerate(cols):
        barras = ax.bar(x + (i - 1) * 0.27, d[c], 0.25, label=ETIQUETAS[c], color=COLORES[c])
        ax.bar_label(barras, fmt="%.1f", fontsize=8, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n({n:,} horas-estación)".replace(",", " ") for c, n in zip(d["condicion"], d["horas_estacion"])])
    ax.set_ylabel("WAPE (%)")
    ax.set_title("Ablación de clima en el mes de test: mismo modelo con y sin variables de clima")
    _estilo(ax)
    ax.legend(frameon=False)
    return _guardar(fig, "06_error_lluvia.png")


def graficar_rebalanceo(stats: dict, nombres: dict[str, str]) -> Path:
    """Costo total (falsas alarmas + r × picos perdidos) según la razón de costos r."""
    razones = np.linspace(0, 10, 101)
    colores = {
        "linea_base_estacional": COLORES["base_estacional"],
        "dia_siguiente_media": COLORES["pred_dia"],
        "dia_siguiente_p90": "#6250d6",
    }
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for nombre, s in stats["estrategias"].items():
        if nombre not in colores:
            continue
        costo = s["falsas_alarmas"] + razones * s["picos_perdidos"]
        ax.plot(razones, costo, linewidth=2, color=colores[nombre],
                label=f"{nombres[nombre]} (FA {s['falsas_alarmas']:,}, PP {s['picos_perdidos']:,})".replace(",", " "))
    ax.set_xlabel("r = costo de un pico perdido / costo de una falsa alarma")
    ax.set_ylabel("Costo en 'viajes en vano' equivalentes")
    ax.set_title("Rebalanceo nocturno: costo de cada estrategia de alerta según r (mes de test)")
    _estilo(ax)
    ax.legend(frameon=False, fontsize=9)
    return _guardar(fig, "07_rebalanceo_costos.png")


def graficar_error_estaciones(error_estacion: pd.DataFrame) -> Path:
    """Mapa simple (longitud/latitud) del WAPE por estación en el mes de test."""
    d = error_estacion.dropna(subset=["latitud", "longitud", "WAPE_pct"])
    fig, ax = plt.subplots(figsize=(8, 9))
    tam = 15 + 120 * d["salidas"] / d["salidas"].max()
    sc = ax.scatter(d["longitud"], d["latitud"], c=d["WAPE_pct"], s=tam, cmap="Blues",
                    edgecolor="#333", linewidth=0.4, vmin=d["WAPE_pct"].quantile(0.02), vmax=d["WAPE_pct"].quantile(0.98))
    fig.colorbar(sc, ax=ax, shrink=0.7, label="WAPE día siguiente (%)")
    peores = d.nlargest(5, "WAPE_pct")
    for _, f in peores.iterrows():
        ax.annotate(str(f["nombre"])[:28], (f["longitud"], f["latitud"]), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.set_aspect(1 / np.cos(np.deg2rad(40.75)))
    ax.set_title("Error por estación (tamaño = salidas en el test; rotuladas las 5 peores)")
    _estilo(ax)
    return _guardar(fig, "08_error_por_estacion.png")


def graficar_error_por_dia(error_dia: pd.DataFrame) -> Path:
    """WAPE diario del modelo principal, marcando feriados y días de lluvia."""
    d = error_dia.sort_values("fecha")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(d["fecha"], d["WAPE_base_estacional"], "^:", color=COLORES["base_estacional"], label=ETIQUETAS["base_estacional"])
    ax.plot(d["fecha"], d["WAPE_pred_dia"], "s-", color=COLORES["pred_dia"], label=ETIQUETAS["pred_dia"])
    fer = d[d["es_feriado"] == 1]
    ax.scatter(fer["fecha"], fer["WAPE_pred_dia"], s=140, facecolor="none", edgecolor="#333", linewidth=1.5, zorder=5, label="Feriado")
    for _, f in d.nlargest(3, "WAPE_pred_dia").iterrows():
        ax.annotate(f"{pd.Timestamp(f['fecha']):%d-%b}\nlluvia {f['lluvia_mm_dia']:.1f} mm", (f["fecha"], f["WAPE_pred_dia"]),
                    fontsize=8, xytext=(5, 5), textcoords="offset points")
    ax.set_ylabel("WAPE del día (%)")
    ax.set_title("Error diario en el mes de test (círculos: feriados; rotulados los 3 peores días)")
    _estilo(ax)
    ax.legend(frameon=False)
    fig.autofmt_xdate()
    return _guardar(fig, "09_error_por_dia.png")
