"""Figuras PNG del proyecto (se leen de reports/*.csv). Paleta categórica validada para daltonismo en orden fijo,
un solo eje Y, leyenda con ≥ 2 series y textos en tinta neutra."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mtick  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config as C  # noqa: E402

PALETA = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GRIS, GRIS_CLARO = "#8a8984", "#c9c8c3"
TEXTO, TEXTO2, SUP = "#0b0b0b", "#52514e", "#fcfcfb"

# Cada modelo conserva su color en todas las figuras (el color sigue a la entidad, no al rango).
MODELOS = {
    "Popularidad global": GRIS_CLARO,
    "Popularidad reciente": GRIS,
    "Contenido (géneros + genome)": PALETA[2],
    "ALS implícito": PALETA[0],
    "Híbrido (ALS + contenido + recientes)": PALETA[1],
    "EASE (referencia)": PALETA[6],
    "Contenido solo con géneros elegidos": PALETA[3],
}
CORTOS = {
    "Popularidad global": "Pop. global", "Popularidad reciente": "Pop. reciente",
    "Contenido (géneros + genome)": "Contenido", "ALS implícito": "ALS", "Híbrido (ALS + contenido + recientes)": "Híbrido",
    "EASE (referencia)": "EASE (ref.)", "Contenido solo con géneros elegidos": "Solo géneros",
}
SEGMENTOS = ["nuevos (onboarding)", "pocos (1-20)", "medios (21-100)", "muchos (>100)"]


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
        C.FIGURAS.mkdir(parents=True, exist_ok=True)
        fig.savefig(C.FIGURAS / nombre, bbox_inches="tight")
    plt.close(fig)
    return fig


def _metricas() -> pd.DataFrame:
    return pd.read_csv(C.REPORTES / "metricas_prueba.csv")


# ---------------------------------------------------------------------------
def actividad_temporal(nombre="01_actividad_y_cortes.png"):
    """Calificaciones por mes y los dos cortes temporales."""
    import pyarrow.parquet as pq

    ts = pq.read_table(C.DATOS_PROCESADOS / "ratings.parquet", columns=["timestamp"]).column(0).to_numpy()
    mes = pd.Series(pd.to_datetime(ts, unit="s")).dt.to_period("M").value_counts().sort_index()
    estilo()
    fig, ax = plt.subplots(figsize=(10, 4))
    x = mes.index.to_timestamp()
    ax.fill_between(x, mes.values / 1e3, color=PALETA[0], alpha=0.15, lw=0)
    ax.plot(x, mes.values / 1e3, color=PALETA[0], lw=1.6)
    techo = mes.values.max() / 1e3
    for fecha, txt, ha, dx, y in [(C.CORTE_VALIDACION, "corte validación\n2017-07", "right", -6, 0.95),
                                  (C.CORTE_PRUEBA, "corte prueba\n2018-01", "left", 6, 0.80)]:
        ax.axvline(fecha, color=TEXTO2, lw=1, ls="--")
        ax.annotate(txt, (fecha, techo * y), xytext=(dx, 0), textcoords="offset points",
                    ha=ha, va="top", fontsize=8.5, color=TEXTO2)
    ax.axvspan(C.CORTE_PRUEBA, pd.Timestamp("2019-11-21"), color=PALETA[1], alpha=0.12, lw=0)
    ax.set_ylabel("miles de calificaciones por mes")
    _titulo(ax, "25 millones de calificaciones, 1995 → 2019",
            "Partición temporal global: se entrena con el pasado y se evalúa con lo que pasó después (zona naranja)")
    return _guardar(fig, nombre)


def ndcg_por_segmento(nombre="02_ndcg_por_segmento.png"):
    m = _metricas()
    modelos = [k for k in MODELOS if k != "Contenido solo con géneros elegidos" and k in set(m["modelo"])]
    estilo()
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ancho = 0.8 / len(modelos)
    for j, mod in enumerate(modelos):
        sub = m[m["modelo"] == mod].set_index("segmento").reindex(SEGMENTOS)
        x = np.arange(len(SEGMENTOS)) + (j - (len(modelos) - 1) / 2) * ancho
        err = np.vstack([sub["ndcg@10"] - sub["ndcg_ic95_bajo"], sub["ndcg_ic95_alto"] - sub["ndcg@10"]])
        ax.bar(x, sub["ndcg@10"], width=ancho * 0.9, color=MODELOS[mod], label=CORTOS[mod],
               edgecolor=SUP, linewidth=1)
        ax.errorbar(x, sub["ndcg@10"], yerr=err, fmt="none", ecolor=TEXTO2, elinewidth=0.8, capsize=0)
    usuarios = m[m["modelo"] == "ALS implícito"].set_index("segmento").reindex(SEGMENTOS)["usuarios"]
    ax.set_xticks(np.arange(len(SEGMENTOS)),
                  [f"{s}\n{num(u, 0)} usuarios" for s, u in zip(SEGMENTOS, usuarios.fillna(0))])
    ax.set_ylabel("NDCG@10 (prueba)")
    ax.legend(ncol=len(modelos), loc="upper center", bbox_to_anchor=(0.5, -0.17), fontsize=9)
    _titulo(ax, "NDCG@10 por segmento de usuario",
            "Segmento = positivos en entrenamiento · barras finas: IC 95 % bootstrap · nuevos: 5 primeras calificaciones como onboarding")
    return _guardar(fig, nombre)


def exactitud_vs_popularidad(nombre="03_exactitud_vs_popularidad.png"):
    """Dos paneles con el mismo eje X (NDCG): cobertura del catálogo y percentil de popularidad recomendado."""
    m = _metricas()
    m = m[m["segmento"] == "existentes (todos)"]
    estilo()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharex=True)
    for ax, col, etiqueta in [(axes[0], "cobertura_pct", "cobertura del catálogo (%)"),
                              (axes[1], "percentil_pop_medio", "percentil medio de popularidad recomendado")]:
        for _, f in m.iterrows():
            ax.scatter(f["ndcg@10"], f[col], s=70, color=MODELOS.get(f["modelo"], GRIS), edgecolor=SUP, lw=2, zorder=3)
            ax.annotate(CORTOS.get(f["modelo"], f["modelo"]), (f["ndcg@10"], f[col]), xytext=(6, 4),
                        textcoords="offset points", fontsize=8.5, color=TEXTO)
        ax.set_ylabel(etiqueta)
        ax.set_xlabel("NDCG@10 (usuarios existentes)")
    _titulo(axes[0], "Exactitud vs. cobertura y sesgo de popularidad",
            "Usuarios existentes del periodo de prueba · arriba-derecha (izq.) y abajo-derecha (der.) es mejor")
    return _guardar(fig, nombre)


def busqueda_als(nombre="04_busqueda_als.png"):
    r = pd.read_csv(C.REPORTES / "busqueda_als.csv").sort_values("ndcg_existentes")
    etiquetas = [f"{int(f.factores)} f · λ {f.regularizacion:g} · α {f.alpha:g}{' · BM25' if f.bm25 else ''}"
                 for f in r.itertuples()]
    estilo()
    fig, ax = plt.subplots(figsize=(9, 4.6))
    colores = [GRIS_CLARO] * (len(r) - 1) + [PALETA[0]]
    ax.barh(etiquetas, r["ndcg_existentes"], color=colores, height=0.65)
    for y, (v, s) in enumerate(zip(r["ndcg_existentes"], r["segundos_ajuste"])):
        ax.annotate(f"{num(v, 4)}  ({num(s, 0)} s)", (v, y), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=TEXTO2)
    ax.set_xlabel("NDCG@10 en validación (usuarios existentes)")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, r["ndcg_existentes"].max() * 1.25)
    _titulo(ax, "Búsqueda acotada de hiperparámetros del ALS",
            "12 configuraciones · validación 2017-07 → 2017-12 · entre paréntesis, segundos de ajuste")
    return _guardar(fig, nombre)


def hibrido_rejilla(nombre="05_hibrido_rejilla.png"):
    r = pd.read_csv(C.REPORTES / "busqueda_hibrido.csv")
    tabla = r.pivot(index="peso_reciente", columns="k", values="ndcg_todos").sort_index(ascending=False)
    estilo()
    fig, ax = plt.subplots(figsize=(7, 4))
    vals = tabla.values
    im = ax.imshow(np.ma.masked_invalid(vals), cmap="Blues", aspect="auto")
    ax.set_xticks(range(tabla.shape[1]), [f"{c:g}" for c in tabla.columns])
    ax.set_yticks(range(tabla.shape[0]), [f"{i:g}" for i in tabla.index])
    mn, mx = np.nanmin(vals), np.nanmax(vals)
    for i in range(tabla.shape[0]):
        for j in range(tabla.shape[1]):
            v = vals[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=9, color=GRIS)
                continue
            ax.text(j, i, num(v, 4), ha="center", va="center", fontsize=9,
                    color="white" if v > mn + 0.6 * (mx - mn) else TEXTO, fontweight="bold" if v == mx else "normal")
    ax.set_xlabel("k (positivos con los que ALS y contenido pesan igual)")
    ax.set_ylabel("β (peso de la popularidad reciente)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="NDCG@10 validación")
    _titulo(ax, "Ajuste del híbrido", "w_ALS = n / (n + k) · NDCG@10 de todos los usuarios de validación · — = no probado")
    return _guardar(fig, nombre)


def cola_larga(nombre="06_cola_larga.png"):
    """Curva de concentración: % de positivos que reciben el x % de películas más populares, y dónde cae cada modelo."""
    pel = pd.read_parquet(C.MODELOS / "peliculas.parquet")
    c = np.sort(pel["positivos"].to_numpy())[::-1]
    x = np.arange(1, len(c) + 1) / len(c) * 100
    y = np.cumsum(c) / c.sum() * 100
    estilo()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(x, y, color=PALETA[0])
    i20 = np.searchsorted(x, 20)
    ax.scatter([x[i20]], [y[i20]], color=PALETA[0], s=40, zorder=3, edgecolor=SUP, lw=2)
    ax.annotate(f"el 20 % más popular concentra el {num(y[i20], 0)} % de los positivos", (x[i20], y[i20]),
                xytext=(10, -18), textcoords="offset points", fontsize=9, color=TEXTO)
    ax.set_xlabel("% de películas del catálogo (de más a menos popular)")
    ax.set_ylabel("% acumulado de positivos")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    _titulo(ax, "La cola larga de MovieLens", f"{num(len(c), 0)} películas recomendables (modelo final, todos los datos)")
    return _guardar(fig, nombre)


def peso_als(nombre="07_peso_hibrido.png"):
    hp = json.loads((C.REPORTES / "hiperparametros.json").read_text(encoding="utf-8"))
    k = hp["hibrido"]["k"]
    n = np.arange(0, 301)
    estilo()
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(n, n / (n + k), color=PALETA[1], label="peso del ALS")
    ax.plot(n, k / (n + k), color=PALETA[2], label="peso del contenido")
    for x0, txt in [(C.N_ONBOARDING, "onboarding"), (20, "fin de «pocos»"), (100, "fin de «medios»")]:
        ax.axvline(x0, color=GRIS_CLARO, lw=1, ls="--")
        ax.annotate(txt, (x0, 1.02), fontsize=8, color=TEXTO2, ha="center")
    ax.set_xlabel("positivos del usuario (n)")
    ax.set_ylabel("peso en la mezcla")
    ax.set_ylim(0, 1.08)
    ax.legend(loc="center right")
    _titulo(ax, "Cuánto manda cada modelo en el híbrido", f"w = n / (n + k), k = {k:g} elegido en validación")
    return _guardar(fig, nombre)


def generar_todas():
    for f in (actividad_temporal, ndcg_por_segmento, exactitud_vs_popularidad, busqueda_als, hibrido_rejilla,
              cola_larga, peso_als):
        try:
            f()
            print("  figura:", f.__name__, flush=True)
        except Exception as e:  # una figura rota no debe tumbar el pipeline
            print(f"  (no se pudo generar {f.__name__}: {e})", flush=True)


if __name__ == "__main__":
    generar_todas()
