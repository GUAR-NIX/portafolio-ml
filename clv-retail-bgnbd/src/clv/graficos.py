"""Generación de las 9 figuras analíticas en reports/figures/ con estilo editorial unificado."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mtick  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config  # noqa: E402
from .modelos import ModeloBGNBD, ModeloGammaGamma  # noqa: E402

AZUL, NARANJA, VERDE, AMARILLO, ROSA, VERDE_OSC, VIOLETA, ROJO = config.COLORES
GRIS, GRIS_CLARO = "#8a8984", "#c9c8c3"
TEXTO, TEXTO2, SUP = "#0b0b0b", "#52514e", "#fcfcfb"


def num(v: float, dec: int = 1) -> str:
    """Formato numérico español: 12.345,6"""
    return f"{v:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def estilo() -> None:
    plt.rcParams.update({
        "figure.facecolor": SUP,
        "axes.facecolor": SUP,
        "savefig.facecolor": SUP,
        "savefig.dpi": 150,
        "font.size": 10,
        "text.color": TEXTO,
        "axes.labelcolor": TEXTO2,
        "xtick.color": TEXTO2,
        "ytick.color": TEXTO2,
        "axes.edgecolor": "#d8d7d2",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e8e7e3",
        "axes.axisbelow": True,
        "axes.titlesize": 12.0,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 26,
        "lines.linewidth": 2.0,
        "legend.frameon": False,
    })


def _titulo(ax, t: str, s: str | None = None) -> None:
    ax.set_title(t)
    if s:
        ax.annotate(
            s,
            xy=(0, 1),
            xycoords="axes fraction",
            xytext=(0, 7),
            textcoords="offset points",
            fontsize=8.8,
            color=TEXTO2,
            va="bottom",
        )


def _guardar(fig, nombre: str):
    fig.tight_layout()
    if nombre:
        config.FIGURAS.mkdir(parents=True, exist_ok=True)
        fig.savefig(config.FIGURAS / nombre, bbox_inches="tight")
    plt.close(fig)
    return fig


def _pct(ax, eje: str = "y", dec: int = 0, escala: float = 1.0) -> None:
    f = mtick.FuncFormatter(lambda v, _: f"{num(v * escala, dec)} %")
    (ax.yaxis if eje == "y" else ax.xaxis).set_major_formatter(f)


def fig_01_limpieza_y_corte(
    auditoria: pd.DataFrame,
    compras_diarias: pd.DataFrame,
    nombre: str = "01_limpieza_y_corte_temporal.png",
):
    """Izq.: auditoría de limpieza de UCI Online Retail II. Der.: ingreso mensual con corte de calibración y holdout."""
    estilo()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.2, 4.1), gridspec_kw={"width_ratios": [1.05, 1.35]})

    etiquetas = [
        "Crudas totales",
        "Sin Customer ID",
        "Cancelaciones (C*)",
        "Códigos no prod.",
        "Precio/cant. <= 0",
        "Duplicados",
        "Líneas limpias",
    ]
    valores = auditoria["filas"].to_numpy() / 1e3
    colores = [GRIS] + [ROJO] * 5 + [AZUL]
    y_pos = np.arange(len(etiquetas))
    a1.barh(y_pos, valores, color=colores, height=0.58)
    a1.invert_yaxis()
    a1.set_yticks(y_pos, etiquetas, fontsize=9)
    a1.set_xlabel("Miles de líneas de factura")
    a1.set_xlim(0, max(valores) * 1.22)
    a1.grid(axis="y", visible=False)
    for i, (v, pct) in enumerate(zip(valores, auditoria["pct_filas"])):
        a1.text(v + 15, i, f"{num(v, 1)} k ({num(pct, 1)} %)", va="center", fontsize=8.5, color=TEXTO)
    _titulo(
        a1,
        "Auditoría de limpieza contable",
        "Se descartan invitados, anulaciones y portes antes de modelar",
    )

    df_m = compras_diarias.copy()
    df_m["mes"] = df_m["fecha_dia"].dt.to_period("M").dt.to_timestamp()
    mensual = df_m.groupby("mes", as_index=False)["importe_dia"].sum()
    corte = pd.Timestamp(config.FECHA_CORTE_CAL)

    cal_m = mensual[mensual["mes"] <= corte]
    hold_m = mensual[mensual["mes"] >= corte - pd.Timedelta(days=15)]

    a2.plot(cal_m["mes"], cal_m["importe_dia"] / 1e3, color=AZUL, marker="o", markersize=4, label="Calibración (555 días)")
    a2.plot(hold_m["mes"], hold_m["importe_dia"] / 1e3, color=NARANJA, marker="o", markersize=4, label="Prueba / holdout (183 días)")
    a2.axvline(corte, color=TEXTO, linestyle="--", linewidth=1.3)
    a2.text(corte, a2.get_ylim()[1] * 0.92, "  Corte: 2011-06-09", fontsize=8.5, color=TEXTO, va="top")
    a2.set_ylabel("Ingreso mensual (miles de GBP)")
    a2.set_xlabel("Mes de operación")
    a2.legend(loc="upper left", fontsize=8.8)
    _titulo(
        a2,
        "Partición temporal estricta: 18 meses para calibrar → 6 meses de prueba",
        "El periodo de prueba incluye la campaña navideña de 2011 (183 días nunca vistos)",
    )
    return _guardar(fig, nombre)


def fig_02_independencia_frecuencia_ticket(
    df_clientes: pd.DataFrame,
    tabla_indep: pd.DataFrame,
    nombre: str = "02_independencia_frecuencia_ticket.png",
):
    """Verifica el supuesto de independencia de Gamma-Gamma antes y después de controlar mayoristas."""
    estilo()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.0, 4.0))

    rep = df_clientes[df_clientes["frecuencia"] >= 1].copy()
    rep["tramo_f"] = pd.cut(
        rep["frecuencia"],
        bins=[0, 1, 2, 3, 5, 8, 15, 1000],
        labels=["1", "2", "3", "4-5", "6-8", "9-15", "16+"],
    )
    res_tramo = rep.groupby("tramo_f", observed=False).agg(
        crudo=("ticket_medio", "mean"),
        acotado=("ticket_medio_acotado", "mean"),
        mediana=("ticket_medio", "median"),
    )

    x = np.arange(len(res_tramo))
    a1.plot(x, res_tramo["crudo"], marker="o", color=ROJO, label="Media cruda (con mayoristas > 3.000 GBP)")
    a1.plot(x, res_tramo["acotado"], marker="s", color=AZUL, label="Media acotada (p99,5 ≤ 3.000 GBP)")
    a1.plot(x, res_tramo["mediana"], marker="^", linestyle="--", color=VERDE, label="Mediana por tramo")
    a1.set_xticks(x, res_tramo.index.astype(str))
    a1.set_xlabel("Compras repetidas en calibración (x)")
    a1.set_ylabel("Ticket diario medio (GBP)")
    a1.legend(loc="upper left", fontsize=8.5)
    _titulo(
        a1,
        "Unos pocos mayoristas inflan el ticket en alta frecuencia",
        f"La mediana sube con la frecuencia: de {num(res_tramo['mediana'].iloc[0], 0)} a {num(res_tramo['mediana'].iloc[-1], 0)} GBP",
    )

    etiquetas = ["Crudo\n(con mayoristas)", "Acotado\n(tope 3.000 GBP)", "Solo minoristas\n(sin > 3.000 GBP)"]
    pos = np.arange(len(tabla_indep))
    ancho = 0.34
    b1 = a2.bar(pos - ancho / 2, tabla_indep["pearson_r"], width=ancho, color=NARANJA, label="Pearson (lineal)")
    b2 = a2.bar(pos + ancho / 2, tabla_indep["spearman_rho"], width=ancho, color=AZUL, label="Spearman (rangos)")
    a2.set_xticks(pos, etiquetas, fontsize=9)
    a2.set_ylim(0, max(tabla_indep["pearson_r"].max(), 0.35) * 1.25)
    a2.set_ylabel("Correlación entre frecuencia (x) y ticket (m_x)")
    a2.axhline(0.15, color=GRIS, linestyle=":", linewidth=1.2)
    for bar in list(b1) + list(b2):
        v = bar.get_height()
        a2.text(bar.get_x() + bar.get_width() / 2, v + 0.008, num(v, 2), ha="center", fontsize=8.5, color=TEXTO)
    a2.legend(loc="upper right", fontsize=8.8)
    _titulo(
        a2,
        "Supuesto de Gamma-Gamma: se cumple solo aproximadamente",
        f"Pearson {num(tabla_indep['pearson_r'].min(), 2)}–{num(tabla_indep['pearson_r'].max(), 2)}, pero Spearman ≈ {num(tabla_indep['spearman_rho'].mean(), 2)} en las tres variantes (p < 1e-47)",
    )
    return _guardar(fig, nombre)


def fig_03_ajuste_bgnbd_frecuencia(
    df_clientes: pd.DataFrame,
    bgnbd: ModeloBGNBD,
    cal_frec: pd.DataFrame,
    nombre: str = "03_ajuste_bgnbd_frecuencia.png",
):
    """Izq.: ajuste en calibración P(X(T)=x). Der.: compras esperadas vs reales en holdout según frecuencia x."""
    estilo()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.0, 4.0))

    max_x = 7
    frec_obs = np.minimum(df_clientes["frecuencia"].to_numpy(dtype=int), max_x)
    conteo_obs = np.bincount(frec_obs, minlength=max_x + 1) / len(frec_obs) * 100.0
    probs_teo = bgnbd.distribucion_frecuencia_teorica(df_clientes["edad_semanas"].to_numpy(), max_x=max_x) * 100.0

    pos = np.arange(max_x + 1)
    ancho = 0.38
    a1.bar(pos - ancho / 2, conteo_obs, width=ancho, color=GRIS, label="Observado en calibración")
    a1.bar(pos + ancho / 2, probs_teo, width=ancho, color=AZUL, label="Modelo BG/NBD")
    etiquetas_x = [str(k) for k in range(max_x)] + [f"{max_x}+"]
    a1.set_xticks(pos, etiquetas_x)
    a1.set_xlabel("Compras repetidas en calibración (x)")
    _pct(a1, "y", dec=0)
    a1.legend(loc="upper right", fontsize=8.8)
    _titulo(
        a1,
        "Ajuste en calibración: distribución de compras repetidas",
        "BG/NBD reproduce la proporción de compradores de una sola vez (x=0) y recurrentes",
    )

    pos2 = cal_frec["frecuencia_tramo"].to_numpy()
    etiquetas_f = [str(k) for k in pos2[:-1]] + [f"{pos2[-1]}+"]
    a2.plot(pos2, cal_frec["compras_real_media"], marker="o", color=TEXTO, label="Real en holdout (6 meses)")
    a2.plot(pos2, cal_frec["compras_pred_media"], marker="s", linestyle="--", color=AZUL, label="Predicho por BG/NBD")
    a2.set_xticks(pos2, etiquetas_f)
    a2.set_xlabel("Compras repetidas en calibración (x)")
    a2.set_ylabel("Compras medias en los 6 meses de prueba")
    a2.legend(loc="upper left", fontsize=8.8)
    _titulo(
        a2,
        "Pronóstico fuera de muestra por nivel de frecuencia",
        "Sin sobreajuste: la esperanza condicional sigue de cerca el conteo real futuro",
    )
    return _guardar(fig, nombre)


def fig_04_probabilidad_vivo_matriz(
    bgnbd: ModeloBGNBD,
    nombre: str = "04_probabilidad_vivo_matriz.png",
):
    """Izq.: matriz P(vivo | x, t_x, T=78 sem). Der.: paradoja de alta frecuencia ante inactividad prolongada."""
    estilo()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.2, 4.1), gridspec_kw={"width_ratios": [1.15, 1.1]})

    T_fijo = 78.0  # ~18 meses de antigüedad
    frec_grid = np.arange(1, 16)
    rec_grid = np.linspace(2.0, T_fijo, 40)
    matriz = np.zeros((len(frec_grid), len(rec_grid)))
    for i, fx in enumerate(frec_grid):
        matriz[i, :] = bgnbd.probabilidad_vivo(
            np.full_like(rec_grid, fx),
            rec_grid,
            np.full_like(rec_grid, T_fijo),
        )

    im = a1.imshow(
        matriz * 100.0,
        aspect="auto",
        origin="lower",
        extent=[float(rec_grid[0] * 7), float(rec_grid[-1] * 7), float(frec_grid[0]), float(frec_grid[-1])],
        cmap="YlGnBu",
        vmin=0,
        vmax=100,
    )
    cb = fig.colorbar(im, ax=a1, pad=0.02)
    cb.set_label("P(vivo) (%)")
    a1.set_xlabel("Recencia t_x (día de la última compra desde el alta, T = 546 d)")
    a1.set_ylabel("Compras repetidas (x)")
    _titulo(
        a1,
        "Superficie de supervivencia P(vivo | x, t_x, T)",
        "A igual recencia t_x baja, más frecuencia histórica implica MENOR probabilidad de seguir vivo",
    )

    dias_sin_comprar = np.linspace(0, 360, 70)
    T_sem = 78.0
    tx_sem = np.maximum(T_sem - dias_sin_comprar / 7.0, 0.5)
    for fx, col, et in [
        (1, VERDE, "x = 1 compra previa"),
        (3, AZUL, "x = 3 compras previas"),
        (6, NARANJA, "x = 6 compras previas"),
        (12, ROJO, "x = 12 compras previas"),
    ]:
        pv = bgnbd.probabilidad_vivo(
            np.full_like(tx_sem, fx),
            tx_sem,
            np.full_like(tx_sem, T_sem),
        ) * 100.0
        a2.plot(dias_sin_comprar, pv, color=col, label=et)

    a2.set_xlabel("Días transcurridos desde la última compra (T − t_x)")
    _pct(a2, "y", dec=0)
    a2.legend(loc="lower left", fontsize=8.5)
    _titulo(
        a2,
        "La paradoja del comprador frecuente en silencio",
        "Con 270 días sin comprar: x = 12 → P(vivo) ≈ 6 %; x = 1 → ≈ 89 % (T = 546 días)",
    )
    return _guardar(fig, nombre)


def fig_05_calibracion_vivo_y_compras(
    deciles: pd.DataFrame,
    nombre: str = "05_calibracion_vivo_y_compras.png",
):
    """Izq.: calibración de probabilidad de recompra por deciles. Der.: ingreso predicho vs real por decil."""
    estilo()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.0, 4.0))

    a1.plot([0, 100], [0, 100], linestyle=":", color=GRIS, label="Calibración perfecta")
    a1.plot(
        deciles["prob_recompra_media"] * 100.0,
        deciles["tasa_recompra_real"] * 100.0,
        marker="o",
        color=AZUL,
        label="BG/NBD P(Y(t) ≥ 1) por decil",
    )
    _pct(a1, "x", dec=0)
    _pct(a1, "y", dec=0)
    a1.set_xlabel("Probabilidad media predicha de recomprar en 6 meses")
    a1.set_ylabel("Porcentaje real que recompró en holdout")
    a1.legend(loc="upper left", fontsize=8.8)
    _titulo(
        a1,
        "Calibración de la probabilidad de recompra a 6 meses",
        "Cada punto agrupa ~430 clientes ordenados por valor esperado en el corte temporal",
    )

    x = deciles["decil"].to_numpy()
    ancho = 0.38
    a2.bar(x - ancho / 2, deciles["ingreso_real_medio"], width=ancho, color=TEXTO, label="Ingreso real (holdout)")
    a2.bar(x + ancho / 2, deciles["ingreso_pred_medio"], width=ancho, color=AZUL, label="Predicho (BG/NBD + Gamma-Gamma)")
    a2.set_xticks(x, [f"D{d}" for d in x])
    a2.set_xlabel("Decil de compras esperadas (D1 = menor, D10 = mayor)")
    a2.set_ylabel("Ingreso medio en 6 meses por cliente (GBP)")
    a2.legend(loc="upper left", fontsize=8.8)
    _titulo(
        a2,
        "Ingreso medio por cliente en prueba según decil",
        "El decil superior (D10) concentra más del 50 % de todo el ingreso futuro de la tienda",
    )
    return _guardar(fig, nombre)


def fig_06_shrinkage_gamma_gamma(
    df_clientes: pd.DataFrame,
    gg: ModeloGammaGamma,
    nombre: str = "06_shrinkage_gamma_gamma.png",
):
    """Izq.: contracción empírica de Bayes del ticket medio. Der.: peso de credibilidad w(x) según compras."""
    estilo()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.0, 4.0))

    m_grid = np.linspace(50, 1500, 80)
    mu_0 = gg.ticket_medio_poblacional
    a1.plot(m_grid, m_grid, linestyle=":", color=GRIS, label="Sin contracción (m_x observado)")
    a1.axhline(mu_0, linestyle="--", color=TEXTO2, linewidth=1.2, label=f"Media poblacional ({num(mu_0, 0)} GBP)")
    for fx, col in [(1, ROJO), (3, NARANJA), (10, AZUL)]:
        esp = gg.ticket_esperado(np.full_like(m_grid, fx), m_grid)
        a1.plot(m_grid, esp, color=col, label=f"Esperado con x = {fx} compras")
    a1.set_xlabel("Ticket medio observado del cliente m_x (GBP)")
    a1.set_ylabel("Ticket esperado E[M | x, m_x] (GBP)")
    a1.legend(loc="upper left", fontsize=8.5)
    _titulo(
        a1,
        "Contracción empírica de Bayes (shrinkage) de Gamma-Gamma",
        "Con 1 sola compra no se confía ciegamente en un pedido atípico; con 10 compras sí",
    )

    x_vals = np.arange(0, 16)
    pesos = np.where(
        x_vals >= 1,
        gg.peso_credibilidad(x_vals),
        gg.p / (gg.p + 2.0 * (gg.q - 1.0)),
    ) * 100.0
    a2.plot(x_vals, pesos, marker="o", color=AZUL)
    a2.set_xlabel("Compras repetidas del cliente (x)")
    _pct(a2, "y", dec=0)
    a2.set_ylim(0, 105)
    for xv in [1, 3, 6, 12]:
        yv = pesos[xv]
        a2.annotate(f"{num(yv, 0)} %", (xv, yv), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8.5)
    _titulo(
        a2,
        "Peso otorgado a la historia propia del cliente w(x)",
        f"Con p={num(gg.p, 2)} y q={num(gg.q, 2)}, a partir de x=3 el historial propio pesa más del 75 %",
    )
    return _guardar(fig, nombre)


def fig_07_comparacion_modelos_clv(
    metricas: pd.DataFrame,
    curvas: pd.DataFrame,
    nombre: str = "07_comparacion_modelos_clv.png",
):
    """Izq.: MAE y correlación de Spearman por modelo. Der.: curva acumulada de captura de ingreso futuro."""
    estilo()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.2, 4.1))

    etiquetas = [
        "Ritmo histórico\n(heurística)",
        "Tabla RFM\npor tramos",
        "LightGBM\nsupervisado",
        "BG/NBD +\nGamma-Gamma",
    ]
    pos = np.arange(len(metricas))
    colores = [GRIS, AMARILLO, NARANJA, AZUL]
    barras = a1.barh(pos, metricas["mae_ingreso_gbp"], color=colores, height=0.55)
    a1.invert_yaxis()
    a1.set_yticks(pos, etiquetas, fontsize=9)
    a1.set_xlabel("Error absoluto medio (MAE) del ingreso a 6 meses (GBP)")
    a1.set_xlim(0, metricas["mae_ingreso_gbp"].max() * 1.48)
    a1.grid(axis="y", visible=False)
    for bar, rho, top20 in zip(barras, metricas["spearman_ingreso"], metricas["captura_top20_pct"]):
        v = bar.get_width()
        a1.text(
            v + 10,
            bar.get_y() + bar.get_height() / 2,
            f"{num(v, 0)} GBP · ρ={num(rho, 3)} · Top20={num(top20, 1)} %",
            va="center",
            fontsize=8.3,
            color=TEXTO,
        )
    bg = metricas[metricas["modelo"] == "BG/NBD + Gamma-Gamma"].iloc[0]
    _titulo(
        a1,
        "Desempeño en los 6 meses de prueba (4.944 clientes)",
        f"BG/NBD + GG: ρ = {num(bg['spearman_ingreso'], 3)} y sesgo de cartera {num(bg['error_cartera_pct'], 1)} %",
    )

    col_modelos = [
        ("BG/NBD + Gamma-Gamma", AZUL, "-"),
        ("LightGBM supervisado", NARANJA, "--"),
        ("Tabla RFM por tramos", AMARILLO, "-."),
        ("Ritmo histórico (heurística)", GRIS, ":"),
    ]
    for nom, col, ls in col_modelos:
        if nom in curvas.columns:
            a2.plot(curvas["pct_clientes"], curvas[nom], color=col, linestyle=ls, label=nom)
    a2.plot(curvas["pct_clientes"], curvas["Azar"], color=GRIS_CLARO, linestyle=":", linewidth=1.2, label="Azar")
    _pct(a2, "x", dec=0)
    _pct(a2, "y", dec=0)
    a2.set_xlabel("Porcentaje de clientes priorizados")
    a2.set_ylabel("Porcentaje del ingreso futuro real capturado")
    a2.legend(loc="lower right", fontsize=8.3)
    _titulo(
        a2,
        "Curva de ganancia acumulada (ingreso real en holdout)",
        f"El top 20 % priorizado por BG/NBD + Gamma-Gamma concentra el {num(bg['captura_top20_pct'], 1)} % del ingreso futuro",
    )
    return _guardar(fig, nombre)


def fig_08_segmentos_y_campana_retencion(
    resumen_seg: pd.DataFrame,
    politicas: pd.DataFrame,
    nombre: str = "08_segmentos_y_campana_retencion.png",
):
    """Izq.: ingreso real en holdout por segmento. Der.: ganancia neta de las políticas de reactivación."""
    estilo()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.4, 4.1))

    y = np.arange(len(resumen_seg))
    ing_miles = resumen_seg["ingreso_real_holdout_gbp"] / 1e3
    colores_s = [AZUL, NARANJA, VERDE, AMARILLO, GRIS]
    a1.barh(y, ing_miles, color=colores_s[: len(y)], height=0.56)
    a1.invert_yaxis()
    a1.set_yticks(y, resumen_seg["segmento"], fontsize=9)
    a1.set_xlabel("Ingreso real en los 6 meses de prueba (miles de GBP)")
    a1.set_xlim(0, ing_miles.max() * 1.58)
    a1.grid(axis="y", visible=False)
    for i, (v, pct_i, n_cli) in enumerate(
        zip(ing_miles, resumen_seg["pct_ingreso_real"], resumen_seg["n_clientes"])
    ):
        a1.text(
            v + 35,
            i,
            f"{num(v, 0)} k GBP ({num(pct_i, 1)} % · {n_cli})",
            va="center",
            fontsize=8.2,
            color=TEXTO,
        )
    _titulo(
        a1,
        "Segmentación estratégica de la cartera",
        "Campeones y clientes en riesgo concentran el valor del negocio",
    )

    # Der.: validación con el holdout — tasa de rescate de equilibrio (menos es mejor) con IC 95 % bootstrap
    y2 = np.arange(len(politicas))
    tasa = politicas["tasa_rescate_equilibrio_pct"].to_numpy()
    err = np.vstack([
        tasa - politicas["tasa_rescate_equilibrio_ic95_inf"].to_numpy(),
        politicas["tasa_rescate_equilibrio_ic95_sup"].to_numpy() - tasa,
    ])
    colores_p = [GRIS, AMARILLO, NARANJA, AZUL]
    a2.barh(y2, tasa, xerr=err, color=colores_p[: len(y2)], height=0.55, error_kw={"ecolor": TEXTO2, "capsize": 3, "lw": 1})
    a2.invert_yaxis()
    etiquetas_p = [
        "1. Todos > 60 d inactivos",
        "2. Regla RFM clásica",
        "3. Top 20 % gasto histórico",
        "4. Zona de riesgo BG/NBD + GG",
    ]
    a2.set_yticks(y2, etiquetas_p[: len(y2)], fontsize=9)
    a2.set_xlabel("Tasa de rescate necesaria para cubrir el costo (%)")
    a2.set_xlim(0, float(politicas["tasa_rescate_equilibrio_ic95_sup"].max()) * 1.9)
    a2.grid(axis="y", visible=False)
    for i, (t, sup, rec, n_c) in enumerate(
        zip(tasa, politicas["tasa_rescate_equilibrio_ic95_sup"], politicas["tasa_recompra_real_pct"], politicas["clientes_contactados"])
    ):
        a2.text(
            sup + 0.25,
            i,
            f"{num(t, 1)} % · {n_c} contactos · {num(rec, 0)} % volvió solo",
            va="center",
            fontsize=8.2,
            color=TEXTO,
        )
    _titulo(
        a2,
        "Políticas de rescate validadas con el gasto real (holdout)",
        "% de los que no volvieron que habría que rescatar para pagar la campaña (IC 95 %)",
    )
    return _guardar(fig, nombre)


def fig_09_sensibilidad_escenario(
    sensibilidad: pd.DataFrame,
    nombre: str = "09_sensibilidad_escenario.png",
):
    """Escenario con supuestos: ganancia neta de cada política según la efectividad supuesta del rescate."""
    estilo()
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 3.9), sharey=True)
    colores_p = [GRIS, AMARILLO, NARANJA, AZUL]
    estilos = ["-", "-", "--", "-"]
    politicas = list(dict.fromkeys(sensibilidad["politica"]))
    for ax, efecto in zip(axes, sorted(sensibilidad["efecto_fuera_zona"].unique())):
        sub = sensibilidad[sensibilidad["efecto_fuera_zona"] == efecto]
        for pol, col, ls in zip(politicas, colores_p, estilos):
            d = sub[sub["politica"] == pol].sort_values("efectividad_rescate")
            ax.plot(d["efectividad_rescate"] * 100, d["ganancia_neta_gbp"] / 1e3, color=col, linestyle=ls, marker="o",
                    markersize=4, label=pol.split(" (")[0])
        ax.axhline(0, color=TEXTO2, linewidth=0.8)
        ax.set_xlabel("Efectividad supuesta del rescate en 0,20 ≤ P(vivo) < 0,70 (%)")
        _titulo(ax, f"Efecto fuera de la zona de riesgo: {num(efecto * 100, 0)} %",
                "Escenario con supuestos: margen recuperado calculado con el propio modelo")
    axes[0].set_ylabel("Ganancia neta del escenario (miles de GBP)")
    manejadores, etiquetas = axes[0].get_legend_handles_labels()
    fig.legend(manejadores, etiquetas, loc="lower center", ncol=4, fontsize=8.3, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    config.FIGURAS.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURAS / nombre, bbox_inches="tight")
    plt.close(fig)
    return fig
