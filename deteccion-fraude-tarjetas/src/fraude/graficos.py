"""Figuras de reports/figures/ (solo leen reports/; no entrenan nada).

Uso:  python -m fraude.graficos
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from fraude.config import DIR_FIGURAS, DIR_REPORTES, ETIQUETAS, FAMILIA

# Paleta categórica de referencia, asignada por familia (orden fijo) y tintas neutras para texto/ejes
COLOR_FAMILIA = {"Línea base": "#8a8985", "No supervisado": "#eb6834", "Supervisado": "#2a78d6", "Híbrido": "#1baf7a"}
SERIE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
TINTA, TINTA_2, REJILLA = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 130, "savefig.bbox": "tight", "font.size": 10,
    "axes.edgecolor": TINTA_2, "axes.labelcolor": TINTA, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": REJILLA, "grid.linewidth": 0.8,
    "xtick.color": TINTA_2, "ytick.color": TINTA_2, "legend.frameon": False, "lines.linewidth": 2,
})


def _guardar(fig, nombre: str) -> None:
    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    fig.savefig(DIR_FIGURAS / nombre, facecolor="white")
    plt.close(fig)
    print(f"  [fig] {nombre}")


def _res() -> dict:
    return json.loads((DIR_REPORTES / "resultados.json").read_text(encoding="utf-8"))


def fig_particion() -> None:
    ph = pd.read_csv(DIR_REPORTES / "transacciones_por_hora.csv")
    part = pd.read_csv(DIR_REPORTES / "particion_temporal.csv")
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True, gridspec_kw={"height_ratios": [2, 1.3]})
    tonos = {"entrenamiento": "#dbe8f8", "validacion": "#fde3d6", "prueba": "#d6f2e7"}
    for ax in axes:
        for _, r in part.iterrows():
            ax.axvspan(r["desde_h"], r["hasta_h"] + (1 if r["particion"] == "prueba" else 0), color=tonos[r["particion"]], zorder=0)
        ax.axvline(24, color=TINTA_2, lw=1, ls="--")
    axes[0].bar(ph["hora_abs"] + 0.5, ph["transacciones"], width=0.8, color="#8a8985")
    axes[0].set_ylabel("Transacciones / hora")
    axes[0].set_title("Partición temporal: el modelo solo ve el pasado (sin barajar)", loc="left")
    axes[1].bar(ph["hora_abs"] + 0.5, ph["fraudes"], width=0.8, color="#e34948")
    axes[1].set_ylabel("Fraudes / hora")
    axes[1].set_xlabel("Horas desde la primera transacción (2 días de septiembre de 2013)")
    nombres = {"entrenamiento": "Entrenamiento", "validacion": "Validación", "prueba": "Prueba"}
    for j, (_, r) in enumerate(part.iterrows()):
        axes[0].text((r["desde_h"] + r["hasta_h"]) / 2, ph["transacciones"].max() * (1.02 + 0.2 * (j == 1)),
                     f"{nombres[r['particion']]}\n{r['transacciones']:,} tx · {r['fraudes']} fraudes".replace(",", "."),
                     ha="center", va="bottom", fontsize=8.5, color=TINTA)
    axes[0].set_ylim(0, ph["transacciones"].max() * 1.5)
    axes[1].text(24.3, ph["fraudes"].max() * 0.9, "inicio del día 2", fontsize=8, color=TINTA_2)
    _guardar(fig, "01_particion_temporal.png")


def fig_auprc_ic() -> None:
    m = pd.read_csv(DIR_REPORTES / "metricas_prueba.csv").sort_values("auprc")
    tasa = pd.read_parquet(DIR_REPORTES / "predicciones_prueba.parquet", columns=["Class"])["Class"].mean()
    fig, ax = plt.subplots(figsize=(8.5, 5))
    y = np.arange(len(m))
    for i, (_, r) in enumerate(m.iterrows()):
        c = COLOR_FAMILIA[r["familia"]]
        ax.plot([r["auprc_ic_inf"], r["auprc_ic_sup"]], [i, i], color=c, lw=2.5, solid_capstyle="round")
        ax.plot(r["auprc"], i, "o", color=c, ms=8, mec="white", mew=2)
        ax.text(r["auprc_ic_sup"] + 0.015, i, f"{r['auprc']:.3f}".replace(".", ","), va="center", fontsize=8.5, color=TINTA)
    ax.axvline(tasa, color=TINTA_2, ls=":", lw=1)
    ax.text(tasa + 0.01, -0.9, f"azar = tasa de fraude ({tasa * 100:.2f} %)".replace(".", ","), fontsize=8, color=TINTA_2)
    ax.set_yticks(y, m["nombre"])
    ax.set_xlim(0, 1.02)
    ax.set_ylim(-1.3, len(m) - 0.4)
    ax.set_xlabel("AUPRC en prueba (IC 95 % bootstrap, 1.000 réplicas)")
    ax.set_title("AUPRC por modelo: los supervisados duplican al mejor no supervisado", loc="left")
    for fam, c in COLOR_FAMILIA.items():
        ax.plot([], [], "o", color=c, label=fam)
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    _guardar(fig, "02_auprc_ic_por_modelo.png")


def fig_curvas_pr() -> None:
    p = pd.read_parquet(DIR_REPORTES / "predicciones_prueba.parquet")
    res = _res()
    modelos = list(dict.fromkeys([res["modelo_servido"], "logistica", res["mejor_no_supervisado"], "autoencoder", "if_normales"]))
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for c, mod in zip(SERIE, modelos):
        pr, rc, _ = precision_recall_curve(p["Class"], p[f"s_{mod}"])
        ax.step(rc, pr, where="post", color=c, label=ETIQUETAS[mod])
    ax.axhline(p["Class"].mean(), color=TINTA_2, ls=":", lw=1)
    ax.set_xlabel("Recall (fraudes detectados / fraudes)")
    ax.set_ylabel("Precisión (fraudes / alertas)")
    ax.set_title("Curvas precisión-recall en prueba (75 fraudes)", loc="left")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend(loc="upper right", fontsize=8.5)
    _guardar(fig, "03_curvas_pr_prueba.png")


def fig_puntajes_anomalia() -> None:
    p = pd.read_parquet(DIR_REPORTES / "predicciones_prueba.parquet")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, mod in zip(axes, ["if_normales", "autoencoder", "lof"]):
        s = p[f"s_{mod}"].to_numpy()
        s = np.log10(s - s.min() + 1e-3) if mod != "if_normales" else s
        bins = np.linspace(np.quantile(s, 0.001), s.max(), 60)
        ax.hist(s[p["Class"] == 0], bins=bins, density=True, color="#8a8985", alpha=0.7, label="normales")
        ax.hist(s[p["Class"] == 1], bins=bins, density=True, color="#e34948", alpha=0.75, label="fraudes")
        ax.set_title(ETIQUETAS[mod], loc="left", fontsize=10)
        ax.set_xlabel("puntaje" if mod == "if_normales" else "log10(puntaje)")
        ax.set_yticks([])
    axes[0].legend()
    fig.suptitle("Los fraudes son raros, pero también lo son muchas compras legítimas: los puntajes se solapan", x=0.01, ha="left", fontweight="bold")
    _guardar(fig, "04_distribucion_puntajes_anomalia.png")


def fig_curva_costo() -> None:
    cur = pd.read_csv(DIR_REPORTES / "curvas_costo.csv")
    res = _res()
    costo = pd.DataFrame(res["costo_prueba"]).set_index("modelo")
    modelos = list(dict.fromkeys([res["modelo_servido"], "logistica", res["mejor_no_supervisado"], "regla_monto"]))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharey=False)
    for ax, part in zip(axes, ["validacion", "prueba"]):
        for c, mod in zip(SERIE, modelos):
            d = cur[(cur["modelo"] == mod) & (cur["particion"] == part)]
            ax.plot(d["alertas"], d["costo_eur"], color=c, label=ETIQUETAS[mod], lw=1.8)
            if part == "validacion":
                i = d["costo_eur"].idxmin()
                ax.plot(d.loc[i, "alertas"], d.loc[i, "costo_eur"], "o", color=c, ms=8, mec="white", mew=2)
            else:
                ax.plot(costo.loc[mod, "alertas"], costo.loc[mod, "costo_total_eur"], "o", color=c, ms=8, mec="white", mew=2)
        ax.set_xscale("symlog", linthresh=10)
        ax.set_xlabel("Alertas revisadas (las de mayor puntaje)")
        ax.set_ylabel("Costo total (EUR)")
        ax.set_title(("Validación: aquí se elige el umbral (●)" if part == "validacion" else "Prueba: el umbral de validación aplicado (●)"), loc="left")
    axes[0].legend(fontsize=8.5)
    _guardar(fig, "05_curva_costo.png")


def fig_sensibilidad() -> None:
    s = pd.read_csv(DIR_REPORTES / "sensibilidad_costo.csv")
    fracs = sorted(s["fraccion_perdida"].unique())
    fig, axes = plt.subplots(1, len(fracs), figsize=(13, 4), sharey=True)
    modelos = list(dict.fromkeys(s["modelo"]))
    for ax, fp in zip(axes, fracs):
        d = s[s["fraccion_perdida"] == fp]
        for c, mod in zip(SERIE, modelos):
            dm = d[d["modelo"] == mod]
            ax.plot(dm["costo_revision_eur"], dm["ahorro_vs_sin_modelo_pct"], "o-", color=c, ms=5, label=dm["nombre"].iloc[0])
        ax.axhline(0, color=TINTA_2, lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("Costo de revisar una alerta (EUR, escala log)")
        ax.set_title(f"Se pierde el {fp * 100:.0f} % del monto", loc="left")
    axes[0].set_ylabel("Ahorro frente a no usar modelo (%)")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Sensibilidad: el umbral se re-elige en validación para cada supuesto; el ahorro se mide en prueba", x=0.01, ha="left", fontweight="bold")
    _guardar(fig, "06_sensibilidad_costo.png")


def fig_pocas_etiquetas() -> None:
    g = pd.read_csv(DIR_REPORTES / "pocas_etiquetas_resumen.csv")
    res = _res()
    met = pd.read_csv(DIR_REPORTES / "metricas_prueba.csv").set_index("modelo")
    ref = res["pocas_etiquetas"]["referencia_no_supervisado"]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.axhspan(met.loc[ref, "auprc_ic_inf"], met.loc[ref, "auprc_ic_sup"], color="#fde3d6", zorder=0)
    ax.axhline(met.loc[ref, "auprc"], color="#eb6834", lw=2, label=f"{ETIQUETAS[ref]} (0 etiquetas de fraude), IC 95 %")
    colores = {"Regresión logística": COLOR_FAMILIA["Línea base"], "LightGBM con pesos": COLOR_FAMILIA["Supervisado"]}
    for mod, d in g.groupby("modelo", sort=True):
        c = colores[mod]
        d = d.sort_values("n_fraudes")
        ax.plot(d["n_fraudes"], d["media"], "o-", color=c, label=f"{mod} (media de 10 muestras)")
        ax.fill_between(d["n_fraudes"], (d["media"] - d["desv"]).clip(0), d["media"] + d["desv"], color=c, alpha=0.15, lw=0)
    ax.set_xscale("log")
    ax.set_xticks(g["n_fraudes"].unique(), [str(n) for n in g["n_fraudes"].unique()])
    ax.set_xlabel("Fraudes etiquetados en entrenamiento (todas las normales disponibles)")
    ax.set_ylabel("AUPRC en prueba")
    ax.set_ylim(0, 1)
    ax.set_title("¿Cuántas etiquetas necesita el supervisado para superar al no supervisado?", loc="left")
    ax.legend(fontsize=8.5, loc="upper left")
    _guardar(fig, "07_pocas_etiquetas.png")


def fig_shap() -> None:
    import shap

    d = np.load(DIR_REPORTES / "shap_muestra_prueba.npz")
    plt.figure()
    shap.summary_plot(d["shap"], d["X"], feature_names=list(d["nombres"]), max_display=12, show=False, plot_size=(8.5, 5.5))
    fig = plt.gcf()
    fig.axes[0].set_title("SHAP del modelo servido (prueba: 3.000 normales + 75 fraudes)\nV1–V28 son componentes PCA anónimas: indican *qué* pesa, no *por qué*",
                          loc="left", fontsize=10)
    _guardar(fig, "08_shap_resumen.png")


def fig_deriva() -> None:
    der = pd.read_csv(DIR_REPORTES / "deriva_variables.csv").head(12).iloc[::-1]
    res = _res()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={"wspace": 0.3})
    colores = ["#e34948" if v > 0.25 else ("#eda100" if v > 0.1 else "#8a8985") for v in der["psi"]]
    axes[0].barh(der["variable"], der["psi"], color=colores, height=0.7)
    axes[0].axvline(0.1, color=TINTA_2, ls=":", lw=1); axes[0].axvline(0.25, color=TINTA_2, ls="--", lw=1)
    axes[0].set_xlabel("PSI día 1 → día 2 (normales)  ·  0,10 moderada · 0,25 fuerte")
    axes[0].set_title(f"Deriva por variable (12 mayores)\nvalidación adversaria día 1 vs día 2: ROC-AUC {res['deriva']['adversaria']['roc_auc']:.3f}".replace(".", ","), loc="left")
    axes[0].grid(axis="y", visible=False)
    m = pd.read_csv(DIR_REPORTES / "metricas_prueba.csv").set_index("modelo")
    mods = list(dict.fromkeys([res["modelo_servido"], "logistica", res["mejor_no_supervisado"], "autoencoder"]))
    x = np.arange(len(mods))
    axes[1].bar(x - 0.18, m.loc[mods, "auprc_validacion"], width=0.34, color="#8a8985", label="validación (h 33–40)")
    axes[1].bar(x + 0.18, m.loc[mods, "auprc"], width=0.34, color="#2a78d6", label="prueba (h 40–48)")
    axes[1].set_xticks(x, [ETIQUETAS[k].replace(" (solo normales)", "\n(solo normales)").replace("LightGBM ", "LightGBM\n").replace("Regresión ", "Regresión\n") for k in mods], fontsize=8.5)
    axes[1].set_ylabel("AUPRC")
    axes[1].set_title("¿Se degrada el rendimiento con el tiempo?\nAUPRC en validación frente a prueba", loc="left")
    axes[1].legend(fontsize=8.5)
    axes[1].grid(axis="x", visible=False)
    _guardar(fig, "09_deriva_temporal.png")


def main() -> None:
    for f in (fig_particion, fig_auprc_ic, fig_curvas_pr, fig_puntajes_anomalia, fig_curva_costo, fig_sensibilidad,
              fig_pocas_etiquetas, fig_shap, fig_deriva):
        f()


if __name__ == "__main__":
    main()
