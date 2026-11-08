"""Gráficos de evaluación (PNG). Paleta validada para daltonismo, un solo eje Y, leyenda solo con ≥ 2 series,
textos en tinta neutra (el color identifica la serie, nunca el texto)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mtick  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config  # noqa: E402

AZUL, NARANJA, VERDE, AMARILLO = config.COLORES[:4]
GRIS, GRIS_CLARO = "#8a8984", "#c9c8c3"
TEXTO, TEXTO2, SUP = "#0b0b0b", "#52514e", "#fcfcfb"


def num(v: float, dec: int = 1) -> str:
    """Formato español: 12.345,6"""
    return f"{v:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def estilo():
    plt.rcParams.update({
        "figure.facecolor": SUP, "axes.facecolor": SUP, "savefig.facecolor": SUP, "savefig.dpi": 150,
        "font.size": 10, "text.color": TEXTO, "axes.labelcolor": TEXTO2, "xtick.color": TEXTO2,
        "ytick.color": TEXTO2, "axes.edgecolor": "#d8d7d2", "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": "#e8e7e3", "axes.axisbelow": True,
        "axes.titlesize": 12.5, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 28,
        "lines.linewidth": 2, "legend.frameon": False,
    })


def _titulo(ax, t, s=None):
    ax.set_title(t)
    if s:
        ax.annotate(s, xy=(0, 1), xycoords="axes fraction", xytext=(0, 8), textcoords="offset points",
                    fontsize=9, color=TEXTO2, va="bottom")


def _guardar(fig, nombre):
    fig.tight_layout()
    if nombre:
        config.FIGURAS.mkdir(parents=True, exist_ok=True)
        fig.savefig(config.FIGURAS / nombre, bbox_inches="tight")
    plt.close(fig)
    return fig


def _pct(ax, eje="y", dec=0, escala=1):
    f = mtick.FuncFormatter(lambda v, _: f"{num(v * escala, dec)} %")
    (ax.yaxis if eje == "y" else ax.xaxis).set_major_formatter(f)


# ---------------------------------------------------------------------------
def fuga_duracion(tabla: pd.DataFrame, tasa_por_duracion: pd.DataFrame, nombre="01_fuga_duracion.png"):
    """Izq.: tasa de éxito según la duración de la llamada. Der.: AUC con y sin la variable."""
    estilo()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.4, 1]})
    t = tasa_por_duracion
    a1.bar(range(len(t)), t["tasa"] * 100, color=AZUL, width=0.7)
    a1.set_xticks(range(len(t)), t.index.astype(str), rotation=0, fontsize=8.5)
    a1.set_xlabel("Duración de la llamada (segundos)")
    _pct(a1)
    for i, v in enumerate(t["tasa"] * 100):
        a1.text(i, v + 1, num(v, 0) + " %", ha="center", fontsize=8.5, color=TEXTO2)
    _titulo(a1, "La duración «predice» el resultado… porque ES el resultado",
            "Tasa de suscripción por duración: una llamada de 0 s nunca termina en sí")

    colores = [AZUL if "final" in i else GRIS for i in tabla.index]
    a2.barh(tabla.index, tabla["auc"], color=colores, height=0.55)
    a2.set_xlim(0.5, 1)
    for i, v in enumerate(tabla["auc"]):
        a2.text(v + 0.005, i, num(v, 3), va="center", fontsize=9.5, color=TEXTO)
    a2.invert_yaxis()
    a2.set_xlabel("AUC en prueba")
    a2.grid(axis="y", visible=False)
    _titulo(a2, "AUC con y sin la variable", "La cifra con fuga no existe en la vida real")
    return _guardar(fig, nombre)


def comparacion_desbalance(tabla: pd.DataFrame, nombre="02_comparacion_desbalance.png"):
    """Average precision (ordenar) y Brier (probabilidad real) por modelo × estrategia (fuera de pliegue)."""
    estilo()
    t = tabla[tabla["modelo"] != "regla"].copy()
    estrategias = list(dict.fromkeys(t["estrategia"]))
    modelos_ = ["logistica", "lightgbm"]
    colores = {"logistica": AZUL, "lightgbm": NARANJA}
    etiquetas = {"logistica": "Regresión logística", "lightgbm": "LightGBM"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    x = np.arange(len(estrategias))
    ancho = 0.36
    for ax, col, tit, sub, dec in [
        (axes[0], "average_precision", "Ordenar clientes: casi no cambia", "Average precision (más es mejor)", 3),
        (axes[1], "brier", "Probabilidad real: se rompe", "Brier sin calibrar (menos es mejor)", 3),
        (axes[2], "ganancia_umbral_teorico", "Ganancia si se llama con p > 10 %",
         "USD por cada 1.000 clientes, sin calibrar", 0),
    ]:
        for j, m in enumerate(modelos_):
            v = t[t["modelo"] == m].set_index("estrategia").loc[estrategias, col].to_numpy()
            if col.startswith("ganancia"):
                v = v / t["n"].iloc[0] * 1000
            ax.bar(x + (j - 0.5) * (ancho + 0.02), v, width=ancho, color=colores[m], label=etiquetas[m])
            for xi, vi in zip(x, v):
                ax.text(xi + (j - 0.5) * (ancho + 0.02), vi, num(vi, dec), ha="center", va="bottom",
                        fontsize=7.5, color=TEXTO2)
        ax.set_xticks(x, estrategias)
        ax.grid(axis="x", visible=False)
        _titulo(ax, tit, sub)
    lo = t["average_precision"].min()
    axes[0].set_ylim(lo - 0.05, t["average_precision"].max() + 0.02)
    axes[0].legend(loc="upper right", fontsize=8.5, ncol=2)
    axes[0].set_ylim(lo - 0.05, t["average_precision"].max() + 0.035)
    axes[2].yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: num(v, 0)))
    return _guardar(fig, nombre)


def calibracion(curvas: dict[str, pd.DataFrame], briers: dict[str, float], nombre="03_calibracion.png"):
    estilo()
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.plot([0, 1], [0, 1], color=GRIS_CLARO, lw=1.2, ls="--", label="Calibración perfecta")
    colores = [GRIS, AZUL, NARANJA, VERDE]
    for (k, c), color in zip(curvas.items(), colores):
        ancho, estilo_linea = (3.5, "-") if "final" in k else (1.6, (0, (4, 2)))
        ax.plot(c["prob_predicha"], c["tasa_real"], marker="o", ms=5, color=color, lw=ancho, ls=estilo_linea,
                label=f"{k} · Brier {num(briers[k], 4)}", zorder=3 if "final" in k else 4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _pct(ax, "x", escala=100)
    _pct(ax, "y", escala=100)
    ax.set_xlabel("Probabilidad predicha (promedio del decil)")
    ax.set_ylabel("Tasa real de suscripción")
    ax.legend(loc="upper left", fontsize=8.5)
    _titulo(ax, "Curva de calibración (prueba)", "Con pesos de clase el modelo promete ~3× lo que ocurre; sin pesos ya sale casi calibrado")
    return _guardar(fig, nombre)


def ganancia_umbral(curva: pd.DataFrame, umbral: float, todos: float, nombre="04_ganancia_umbral.png"):
    estilo()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(curva["umbral"] * 100, curva["ganancia"], color=AZUL)
    ax.axhline(todos, color=GRIS, lw=1.2, ls="--")
    ax.text(curva["umbral"].max() * 100, todos, "  llamar a todos", va="bottom", ha="right", fontsize=8.5,
            color=TEXTO2)
    ax.axhline(0, color=GRIS_CLARO, lw=1)
    fila = curva.iloc[(curva["umbral"] - umbral).abs().argmin()]
    ax.scatter([fila["umbral"] * 100], [fila["ganancia"]], s=60, color=NARANJA, zorder=3,
               edgecolor=SUP, linewidth=2)
    ax.annotate(f"umbral elegido {num(umbral * 100, 1)} %\n{num(fila['pct_llamados'], 0)} % llamados · "
                f"{num(fila['ganancia'], 0)} USD", (fila["umbral"] * 100, fila["ganancia"]),
                xytext=(18, -34), textcoords="offset points", fontsize=8.5, color=TEXTO)
    _pct(ax, "x")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: num(v, 0)))
    ax.set_xlabel("Umbral de probabilidad calibrada para llamar")
    ax.set_ylabel("Ganancia neta (USD)")
    _titulo(ax, "Ganancia de la campaña según el umbral (prueba)",
            f"Supuestos: {num(config.COSTO_LLAMADA, 0)} USD por cliente gestionado, "
            f"{num(config.BENEFICIO_EXITO, 0)} USD por depósito")
    return _guardar(fig, nombre)


def ganancia_acumulada(curvas: dict[str, pd.DataFrame], tasa_base: float, nombre="05_ganancia_acumulada.png"):
    estilo()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
    colores = [NARANJA, AZUL, GRIS]
    x = np.r_[0, curvas[next(iter(curvas))]["pct_llamados"].to_numpy()]
    ideal = np.minimum(100, x / (tasa_base * 100) * 100)
    a1.plot(x, ideal, color=GRIS_CLARO, lw=1.2, ls=":", label="Ideal")
    a1.plot([0, 100], [0, 100], color=GRIS_CLARO, lw=1.2, ls="--", label="Azar")
    for (k, c), color in zip(curvas.items(), colores):
        a1.plot(np.r_[0, c["pct_llamados"]], np.r_[0, c["pct_capturado"]], color=color, label=k)
        a2.plot(c["pct_llamados"], c["lift"], color=color, label=k)
    c0 = curvas[next(iter(curvas))]
    for p in (10, 30):
        f = c0.iloc[(c0["pct_llamados"] - p).abs().argmin()]
        a1.scatter([p], [f["pct_capturado"]], s=40, color=NARANJA, zorder=3, edgecolor=SUP, linewidth=2)
        a1.annotate(f"{p} % llamados → {num(f['pct_capturado'], 0)} %", (p, f["pct_capturado"]),
                    xytext=(10, -14), textcoords="offset points", fontsize=8.5)
    a2.axhline(1, color=GRIS_CLARO, lw=1.2, ls="--")
    for ax in (a1, a2):
        _pct(ax, "x")
        ax.set_xlabel("Clientes llamados (de mayor a menor probabilidad)")
    _pct(a1)
    a1.set_ylabel("Suscripciones capturadas")
    a2.set_ylabel("Lift (× vs. azar)")
    a1.legend(loc="lower right", fontsize=8.5)
    a2.legend(loc="upper right", fontsize=8.5)
    _titulo(a1, "Ganancia acumulada (prueba)", "Qué parte de las suscripciones se logra llamando solo al top X %")
    _titulo(a2, "Lift acumulado", "Cuántas veces mejor que llamar al azar")
    return _guardar(fig, nombre)


def importancia_shap(imp: pd.DataFrame, nombre="06_shap_importancia.png"):
    estilo()
    t = imp.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.barh(t["etiqueta"], t["shap_medio_abs"], color=AZUL, height=0.6)
    for i, v in enumerate(t["shap_medio_abs"]):
        ax.text(v, i, " " + num(v, 3), va="center", fontsize=8.5, color=TEXTO2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("|SHAP| medio (log-odds)")
    _titulo(ax, "¿Qué mira el modelo?", "Importancia global con SHAP, LightGBM, muestra de prueba")
    return _guardar(fig, nombre)


def beeswarm_shap(explicacion, nombre="07_shap_beeswarm.png"):
    import shap
    estilo()
    plt.figure(figsize=(9, 5.5))
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap.plots.beeswarm(explicacion, max_display=12, show=False, plot_size=None)
    fig = plt.gcf()
    ax = fig.axes[0]
    ax.set_yticks(ax.get_yticks(), [t.get_text().replace("Sum of", "Suma de").replace("other features", "variables más")
                                    for t in ax.get_yticklabels()])
    barra = fig.axes[-1]
    barra.set_ylabel("Valor de la variable (gris = categórica)", color=TEXTO2)
    barra.set_yticks(barra.get_yticks(), ["Bajo" if t.get_text() == "Low" else "Alto" for t in barra.get_yticklabels()])
    fig.axes[0].set_title("Efecto de cada variable en cada cliente (SHAP)", loc="left", pad=12)
    fig.axes[0].set_xlabel("Efecto en log-odds de suscribir (derecha = más probable)")
    return _guardar(fig, nombre)


def prueba_temporal(mensual: pd.DataFrame, nombre="08_prueba_temporal.png"):
    estilo()
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(mensual))
    ax.plot(x, mensual["tasa_real"] * 100, color=TEXTO, marker="o", ms=4, lw=1.6, label="Tasa real")
    ax.plot(x, mensual["prob_sin_recalibrar"] * 100, color=GRIS, marker="o", ms=4, label="Predicha · modelo fijo")
    ax.plot(x, mensual["prob_recalibrada"] * 100, color=NARANJA, marker="o", ms=4,
            label="Predicha · recalibrada con los 3 meses anteriores")
    ax.plot(x, mensual["prob_reentrenado"] * 100, color=AZUL, marker="o", ms=4,
            label="Predicha · re-entrenada cada mes")
    ax.set_xticks(x, mensual["periodo"], rotation=45, ha="right", fontsize=8)
    _pct(ax)
    ax.set_ylabel("Suscripción promedio del mes")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.set_ylim(0, 110)
    _titulo(ax, "Prueba de estrés temporal: entrenar hasta abr-2009, predecir el futuro",
            "La tasa real pasa de 6 % a ~50 %: el modelo fijo la subestima, recalibrar ayuda poco, re-entrenar la sigue")
    return _guardar(fig, nombre)
