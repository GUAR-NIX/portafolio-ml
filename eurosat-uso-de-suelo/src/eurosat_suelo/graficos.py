"""Generación de las 10 figuras del README (300 DPI) a partir de reports/ y models/."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

from eurosat_suelo.config import (
    ARCHIVO_PT,
    CLASES_EUROSAT,
    DIR_FIGURAS,
    DIR_MODELOS,
    DIR_MUESTRA,
    DIR_REPORTES,
    NOMBRES_ESPANOL,
)
from eurosat_suelo.datos import cargar_subconjunto_eurosat
from eurosat_suelo.evaluacion import PERTURBACIONES
from eurosat_suelo.modelos import cargar_red_finetuned, grad_cam, mapa_sensibilidad_oclusion

N_PRUEBA = 5400  # se actualiza con reports/resultados.json al generar las figuras

PALETA = {
    "primario": "#0f4c81",
    "secundario": "#0284c7",
    "exito": "#15803d",
    "alerta": "#dc2626",
    "ambar": "#d97706",
    "neutro": "#475569",
    "morado": "#6d28d9",
}


def configurar_estilo() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "figure.facecolor": "#fafafa",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#d1d5db",
            "axes.grid": True,
            "grid.color": "#e5e7eb",
            "grid.alpha": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
            "axes.titlesize": 12.0,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.0,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
        }
    )


def fig_01_mosaico_y_espectro(df_res27k: pd.DataFrame) -> Path:
    """Mosaico de las 10 clases Sentinel-2 y firma espectral/espacial sobre las 27.000 imágenes."""
    configurar_estilo()
    fig = plt.figure(figsize=(15.2, 6.2))
    gs = fig.add_gridspec(2, 7, width_ratios=[1, 1, 1, 1, 1, 0.25, 2.8], hspace=0.38, wspace=0.22)

    # Panel izquierdo: 10 parches reales de muestra (2 filas x 5 columnas)
    for idx, clase in enumerate(CLASES_EUROSAT):
        fila = idx // 5
        col = idx % 5
        ax_img = fig.add_subplot(gs[fila, col])
        ruta_png = DIR_MUESTRA / f"{idx:02d}_{clase}_01.png"
        img = plt.imread(ruta_png)
        ax_img.imshow(img)
        ax_img.set_title(f"{NOMBRES_ESPANOL[clase]}\n({int(df_res27k.loc[df_res27k['clase']==clase, 'imagenes_totales'].iloc[0]):,})", fontsize=8.4)
        ax_img.axis("off")

    # Panel derecho: Firma espectral ExG (vegetación) vs Índice de Agua y Rugosidad Espacial
    ax_der = fig.add_subplot(gs[:, 6])
    colores_macro = {
        "Agrícola": "#15803d",
        "Ecosistema natural": "#059669",
        "Infraestructura antrópica": "#dc2626",
        "Cuerpo de agua": "#0284c7",
    }
    offsets_custom = {
        "Zona industrial": (-8, -13),
        "Vegetación herbácea": (7, -3),
        "Zona residencial": (7, 4),
        "Cultivo anual": (7, -10),
        "Cultivo permanente": (7, 5),
    }
    for macro, sub in df_res27k.groupby("macro_categoria"):
        tamanos = sub["rugosidad_media"].to_numpy() * 9500.0
        ax_der.scatter(
            sub["exg_medio"],
            sub["indice_agua_medio"],
            s=tamanos,
            color=colores_macro.get(macro, PALETA["primario"]),
            alpha=0.85,
            edgecolors="#1f2937",
            linewidth=1.1,
            label=macro,
        )
        for _, r in sub.iterrows():
            dx, dy = offsets_custom.get(r["nombre_es"], (6, 4))
            ax_der.annotate(
                r["nombre_es"],
                (r["exg_medio"], r["indice_agua_medio"]),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=8.0,
                fontweight="bold",
            )

    ax_der.set_xlim(-0.032, 0.118)
    ax_der.set_ylim(-0.06, 0.38)
    ax_der.set_xlabel("Índice ExG medio de vegetación (2G − R − B) / (R + G + B)")
    ax_der.set_ylabel("Índice normalizado Azul-Rojo (B − R) / (B + R)")
    ax_der.set_title("Separabilidad espectral (27.000 escenas · tamaño = rugosidad)", loc="left", pad=12)
    ax_der.legend(loc="lower right", frameon=True, fontsize=8.2, markerscale=0.45)

    return _guardar(fig, "01_mosaico_clases_y_espectro_sentinel2.png")


ETIQUETAS_CORTAS = {
    "m1": "M1 · Espectral + GBDT",
    "m2": "M2 · CNN congelada + sonda",
    "m3": "M3 · Fusión CNN + espectral",
    "m4": "M4 · Fine-tuning D4",
    "m5": "M5 · Ensamble M3 + M4",
}
COLORES_MODELO = {
    "m1": PALETA["ambar"],
    "m2": PALETA["secundario"],
    "m3": PALETA["primario"],
    "m4": PALETA["morado"],
    "m5": PALETA["exito"],
}


def _guardar(fig: plt.Figure, nombre: str) -> Path:
    ruta = DIR_FIGURAS / nombre
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)
    return ruta


def fig_02_comparacion_arquitecturas(df_comp: pd.DataFrame) -> Path:
    """Exactitud Top-1 con IC 95 % bootstrap y costo (supuesto) de errores por 1.000 parcelas de los 5 modelos."""
    configurar_estilo()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 5.4))
    y = np.arange(len(df_comp))
    etiquetas = [ETIQUETAS_CORTAS[m] for m in df_comp["id"]]
    colores = [COLORES_MODELO[m] for m in df_comp["id"]]

    acc = df_comp["exactitud_top1"].to_numpy() * 100.0
    err = np.vstack(
        [acc - df_comp["exactitud_ic95_inf"].to_numpy() * 100.0, df_comp["exactitud_ic95_sup"].to_numpy() * 100.0 - acc]
    )
    ax1.barh(y, acc, height=0.55, color=colores, xerr=err, capsize=4, error_kw={"elinewidth": 1.3, "ecolor": "#111827"})
    ax1.set_yticks(y)
    ax1.set_yticklabels(etiquetas)
    ax1.invert_yaxis()
    ax1.set_xlim(max(0.0, acc.min() - 8.0), 100.0)
    ax1.set_xlabel("Exactitud Top-1 en prueba (%) · barras = IC 95 % bootstrap pareado")
    ax1.set_title(f"Exactitud en {N_PRUEBA:,} escenas de prueba".replace(",", "."), loc="left", pad=12)
    for i in range(len(df_comp)):
        ax1.text(df_comp["exactitud_ic95_sup"].iloc[i] * 100.0 + 0.25, y[i], f"{acc[i]:.2f} %".replace(".", ","), va="center", fontsize=8.6, fontweight="bold")

    costos = df_comp["costo_por_1000_usd"].to_numpy()
    err_c = np.vstack([costos - df_comp["costo_1000_ic95_inf"].to_numpy(), df_comp["costo_1000_ic95_sup"].to_numpy() - costos])
    ax2.barh(y, costos, height=0.55, color=colores, xerr=err_c, capsize=4, error_kw={"elinewidth": 1.3, "ecolor": "#111827"})
    ax2.set_yticks(y)
    ax2.set_yticklabels([m.upper() for m in df_comp["id"]])
    ax2.invert_yaxis()
    ax2.set_xlim(0, df_comp["costo_1000_ic95_sup"].max() * 1.35)
    ax2.set_xlabel("Costo SUPUESTO de errores por 1.000 parcelas (USD), sin abstención")
    ax2.set_title("Costo de errores (35 USD misma macro · 140 USD crítico)", loc="left", pad=12)
    for i, c in enumerate(costos):
        ax2.text(df_comp["costo_1000_ic95_sup"].iloc[i] + costos.max() * 0.02, y[i], f"{c:,.0f} USD · {df_comp['tasa_error_critico_pct'].iloc[i]:.2f} % crítico".replace(",", "X").replace(".", ",").replace("X", "."), va="center", fontsize=8.4)
    plt.tight_layout()
    return _guardar(fig, "02_comparacion_arquitecturas_cpu.png")


def fig_03_curva_few_shot(df_fs: pd.DataFrame) -> Path:
    """Curva de eficiencia de datos de 5 a 1.000 imágenes por clase."""
    configurar_estilo()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.6, 5.4))

    x = df_fs["ejemplos_por_clase"].to_numpy()
    ax1.plot(x, df_fs["acc_espectral"] * 100.0, "o--", color=PALETA["ambar"], linewidth=2.2, label="Espectral + textura (GBDT)")
    ax1.plot(x, df_fs["acc_cnn_probe"] * 100.0, "s-", color=PALETA["secundario"], linewidth=2.3, label="MobileNetV3 congelada (sonda lineal)")
    ax1.plot(x, df_fs["acc_hibrido"] * 100.0, "D-", color=PALETA["exito"], linewidth=2.5, label="Embeddings CNN + índices (sonda lineal)")
    ax1.set_xscale("log")
    ax1.set_xticks(x)
    ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax1.set_xlabel("Imágenes de entrenamiento etiquetadas por clase (escala log)")
    ax1.set_ylabel(f"Exactitud Top-1 en prueba (%, n = {N_PRUEBA:,})".replace(",", "."))
    ax1.set_title("Eficiencia Few-Shot", loc="left", pad=14)
    ax1.legend(loc="lower right", frameon=True, fontsize=8.3)

    ganancia = (df_fs["acc_hibrido"] - df_fs["acc_espectral"]) * 100.0
    ax2.bar([str(v) for v in x], ganancia, color=PALETA["primario"], width=0.55)
    ax2.set_xlabel("Imágenes etiquetadas por clase")
    ax2.set_ylabel("Ventaja de exactitud del híbrido (puntos)")
    ax2.set_title("Ventaja de los rasgos preentrenados según el volumen de etiquetas", loc="left", pad=14)
    ax2.set_ylim(min(0, ganancia.min() * 1.2), max(ganancia) * 1.22)
    for i, g in enumerate(ganancia):
        ax2.text(i, g + 0.4, f"{g:+.1f}".replace(".", ","), ha="center", fontsize=8.5, fontweight="bold")
    plt.tight_layout()
    return _guardar(fig, "03_curva_few_shot_transfer_learning.png")


def fig_04_f1_por_clase(df_clases: pd.DataFrame) -> Path:
    """Desglose de F1 por clase: clasificador espectral (M1) frente al ensamble (M5)."""
    configurar_estilo()
    df_ord = df_clases.sort_values("ganancia_f1_transfer", ascending=False).reset_index(drop=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.8, 5.6))

    y = np.arange(len(df_ord))
    h = 0.36
    ax1.barh(y - h / 2, df_ord["f1_espectral_clasico"] * 100.0, height=h, color=PALETA["ambar"], label="M1 · Espectral + textura")
    ax1.barh(y + h / 2, df_ord["f1"] * 100.0, height=h, color=PALETA["exito"], label="M5 · Ensamble calibrado")
    ax1.set_yticks(y)
    ax1.set_yticklabels(df_ord["nombre_es"])
    ax1.invert_yaxis()
    ax1.set_xlim(max(0.0, df_ord["f1_espectral_clasico"].min() * 100.0 - 10.0), 104)
    ax1.set_xlabel("F1 por clase en prueba (%)")
    ax1.set_title(f"F1 por clase ({N_PRUEBA:,} escenas de prueba)".replace(",", "."), loc="left", pad=22)
    ax1.legend(loc="lower right", bbox_to_anchor=(1.0, 1.01), ncol=2, frameon=False, fontsize=8.8)

    gan = df_ord["ganancia_f1_transfer"].to_numpy() * 100.0
    colores_gan = [PALETA["primario"] if g >= 0 else PALETA["neutro"] for g in gan]
    ax2.barh(y, gan, height=0.54, color=colores_gan)
    ax2.axvline(0.0, color="#374151", linewidth=1.0)
    ax2.set_yticks(y)
    ax2.set_yticklabels(df_ord["clase"])
    ax2.invert_yaxis()
    ax2.set_xlim(min(-2.0, gan.min() * 1.3), max(gan) * 1.28)
    ax2.set_xlabel("Diferencia de F1: M5 − M1 (puntos)")
    ax2.set_title("Dónde aporta más la CNN", loc="left", pad=14)
    for i, g in enumerate(gan):
        ax2.text(max(g, 0) + 0.3, i, f"{g:+.1f}".replace(".", ","), va="center", fontsize=8.4, fontweight="bold")
    plt.tight_layout()
    return _guardar(fig, "04_f1_por_clase_y_ganancia_transfer.png")


def fig_05_matrices_confusion(cm_spec: pd.DataFrame, cm_mejor: pd.DataFrame) -> Path:
    """Matrices de confusión normalizadas por fila (sensibilidad) de M1 y M5, con conteos."""
    configurar_estilo()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 6.2))
    etiq = [NOMBRES_ESPANOL[c][:16] for c in CLASES_EUROSAT]

    for ax, cm_df, titulo in [
        (ax1, cm_spec, "M1 · Espectral + textura (GBDT)"),
        (ax2, cm_mejor, "M5 · Ensamble M3 + M4 calibrado"),
    ]:
        mat = cm_df.to_numpy()
        pct = mat / mat.sum(axis=1, keepdims=True)
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
        ax.grid(False)
        ax.set_xticks(range(len(etiq)))
        ax.set_yticks(range(len(etiq)))
        ax.set_xticklabels(etiq, rotation=45, ha="right", fontsize=8.0)
        ax.set_yticklabels(etiq, fontsize=8.0)
        ax.set_xlabel("Clase predicha")
        ax.set_ylabel("Clase real")
        ax.set_title(titulo + " · color = % de la fila", loc="left", pad=12, fontsize=10.8)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if mat[i, j] > 0:
                    ax.text(j, i, str(int(mat[i, j])), ha="center", va="center", fontsize=7.2,
                            color="white" if pct[i, j] > 0.5 else "#111827", fontweight="bold" if i == j else "normal")
    plt.tight_layout()
    return _guardar(fig, "05_matrices_confusion_espectral_vs_transfer.png")


def fig_06_calibracion_y_abstencion(df_cal: pd.DataFrame, df_fiab: pd.DataFrame, df_abst: pd.DataFrame, abst: dict) -> Path:
    """ECE antes/después del escalado por temperatura, diagrama de fiabilidad de M5 y costo vs tau (val y prueba)."""
    configurar_estilo()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17.5, 5.2))

    y = np.arange(len(df_cal))
    ax1.barh(y - 0.2, df_cal["ece_antes"], height=0.38, color="#cbd5e1", label="Sin calibrar (T = 1)")
    ax1.barh(y + 0.2, df_cal["ece_despues"], height=0.38, color=[COLORES_MODELO[m] for m in df_cal["modelo"]], label="Con T ajustada en validación")
    ax1.set_yticks(y)
    ax1.set_yticklabels([f"{ETIQUETAS_CORTAS[m]}\nT = {t:.2f}".replace(".", ",") for m, t in zip(df_cal["modelo"], df_cal["temperatura_validacion"])], fontsize=8.2)
    ax1.invert_yaxis()
    ax1.set_xlabel("ECE en prueba (menor es mejor)")
    ax1.set_title("Escalado por temperatura", loc="left", pad=12)
    ax1.legend(loc="lower right", fontsize=8.0)

    for etapa, estilo, color in (("antes", "o--", "#94a3b8"), ("despues", "o-", PALETA["exito"])):
        sub = df_fiab[(df_fiab["modelo"] == "m5") & (df_fiab["etapa"] == etapa)]
        ax2.plot(sub["confianza_media"], sub["exactitud"], estilo, color=color, linewidth=2.0,
                 label="M5 sin calibrar" if etapa == "antes" else "M5 calibrado")
    ax2.plot([0, 1], [0, 1], ":", color="#374151", linewidth=1.0, label="Calibración perfecta")
    ax2.set_xlim(0.2, 1.0)
    ax2.set_ylim(0.2, 1.0)
    ax2.set_xlabel("Confianza media del intervalo")
    ax2.set_ylabel("Exactitud observada")
    ax2.set_title("Diagrama de fiabilidad (prueba)", loc="left", pad=12)
    ax2.legend(loc="upper left", fontsize=8.2)

    tau_val = abst["tau_validacion_m5"]
    for particion, estilo, color in (("validacion", "s--", PALETA["neutro"]), ("prueba", "D-", PALETA["alerta"])):
        sub = df_abst[df_abst["particion"] == particion]
        ax3.plot(sub["umbral_confianza"], sub["costo_total_por_1000_usd"], estilo, color=color, linewidth=2.0,
                 label=f"Costo en {'validación' if particion == 'validacion' else 'prueba'}")
    costo_test = abst["prueba_m5"]["costo_total_por_1000_usd"]
    ax3.axvline(tau_val, color=PALETA["exito"], linestyle="--", linewidth=1.6,
                label=f"τ elegido en validación = {tau_val:.2f}".replace(".", ","))
    ax3.scatter([tau_val], [costo_test], color=PALETA["exito"], s=80, zorder=5)
    ax3.set_xlabel("Umbral de confianza mínima τ (debajo → fotointérprete)")
    ax3.set_ylabel("Costo SUPUESTO por 1.000 parcelas (USD)")
    ax3.set_title("Abstención: τ se elige en validación", loc="left", pad=12)
    ax3.legend(loc="lower left", fontsize=8.2)
    plt.tight_layout()
    return _guardar(fig, "06_calibracion_temperatura_y_abstencion.png")


def fig_07_robustez(df_rob: pd.DataFrame) -> Path:
    """Exactitud de los 5 modelos bajo velo simulado, ruido gaussiano y desenfoque, con ejemplos visuales."""
    configurar_estilo()
    fig = plt.figure(figsize=(17.0, 8.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.5, 1.0], hspace=0.35, wspace=0.22)
    titulos = {"velo_simulado": "Velo / bruma SIMULADO", "ruido_gaussiano": "Ruido gaussiano", "desenfoque": "Desenfoque gaussiano"}

    datos = cargar_subconjunto_eurosat()
    imgs_te = datos["imagenes"][datos["particion"] == "test"]
    lbls_te = datos["etiquetas"][datos["particion"] == "test"]
    escena = imgs_te[int(np.where(lbls_te == CLASES_EUROSAT.index("River"))[0][0])][None, ...]

    for col, (pert, (funcion, niveles, etiqueta_eje)) in enumerate(PERTURBACIONES.items()):
        ax = fig.add_subplot(gs[0, col])
        sub = df_rob[df_rob["perturbacion"] == pert]
        for m in ("m1", "m2", "m3", "m4", "m5"):
            ax.plot(sub["nivel"], sub[f"acc_{m}"] * 100.0, "o-", color=COLORES_MODELO[m],
                    linewidth=2.6 if m == "m5" else 1.8, label=ETIQUETAS_CORTAS[m])
        ax.set_xlabel(etiqueta_eje)
        ax.set_ylabel("Exactitud en prueba (%)" if col == 0 else "")
        ax.set_ylim(0, 100)
        ax.set_title(titulos[pert], loc="left", pad=10)
        if col == 0:
            ax.legend(loc="lower left", fontsize=8.0, frameon=True)

        ax_img = fig.add_subplot(gs[1, col])
        niveles_vis = [0.0, niveles[1], niveles[-1]]
        tira = np.hstack([funcion(escena, nv, semilla=42)[0] for nv in niveles_vis])
        ax_img.imshow(tira)
        ax_img.set_xticks([32, 96, 160])
        ax_img.set_xticklabels([f"nivel {nv:g}".replace(".", ",") for nv in niveles_vis])
        ax_img.set_yticks([])
        ax_img.grid(False)
        ax_img.set_title("Escena de río en prueba (64×64 px)", fontsize=9.2, loc="left")
    return _guardar(fig, "07_robustez_velo_ruido_desenfoque.png")


def fig_08_mapas_oclusion_espacial() -> Path:
    """Mapas de sensibilidad por oclusión (64x64) del modelo 4 en 4 clases representativas."""
    configurar_estilo()
    red = cargar_red_finetuned(DIR_MODELOS / ARCHIVO_PT)
    datos = cargar_subconjunto_eurosat()
    imgs_te = datos["imagenes"][datos["particion"] == "test"]
    lbls_te = datos["etiquetas"][datos["particion"] == "test"]

    clases_demo = ["Highway", "River", "Residential", "Forest"]
    fig, axes = plt.subplots(2, 4, figsize=(14.4, 6.6))
    for col, nombre_c in enumerate(clases_demo):
        cid = CLASES_EUROSAT.index(nombre_c)
        candidatos = np.where(lbls_te == cid)[0][:8]
        probs_cand = [mapa_sensibilidad_oclusion(red, imgs_te[int(c)], clase_objetivo=cid, parche=16, paso=16)["probabilidad_base"] for c in candidatos]
        img = imgs_te[int(candidatos[int(np.argmax(probs_cand))])]
        res_ocl = mapa_sensibilidad_oclusion(red, img, clase_objetivo=cid, parche=14, paso=6)

        axes[0, col].imshow(img)
        axes[0, col].set_title(f"{NOMBRES_ESPANOL[nombre_c]}\nP({nombre_c}) = {res_ocl['probabilidad_base']*100:.1f} %".replace(".", ","), fontsize=9.2)
        axes[0, col].axis("off")
        axes[1, col].imshow(img)
        axes[1, col].imshow(res_ocl["mapa_importancia"], cmap="jet", alpha=0.52, vmin=0.0, vmax=1.0)
        axes[1, col].set_title("Caída de probabilidad al ocluir", fontsize=8.8)
        axes[1, col].axis("off")
    plt.tight_layout()
    return _guardar(fig, "08_mapas_oclusion_espacial.png")


def seleccionar_ejemplos_grad_cam(pred_df: pd.DataFrame, n: int = 4) -> tuple[list[int], list[int]]:
    """Índices de prueba: n aciertos seguros de clases distintas y n errores del modelo 4 con más confianza."""
    aciertos: list[int] = []
    for clase in ("Highway", "River", "Industrial", "PermanentCrop", "Pasture", "SeaLake"):
        cid = CLASES_EUROSAT.index(clase)
        cand = pred_df[(pred_df["y_real"] == cid) & (pred_df["pred_m4"] == cid)].sort_values("conf_m4", ascending=False)
        if len(cand):
            aciertos.append(int(cand.index[0]))
        if len(aciertos) == n:
            break
    fallos_df = pred_df[pred_df["y_real"] != pred_df["pred_m4"]].sort_values("conf_m4", ascending=False)
    fallos = list(fallos_df.drop_duplicates("y_real").index[:n].astype(int))
    return aciertos, fallos


def fig_09_grad_cam(pred_df: pd.DataFrame) -> Path:
    """Grad-CAM del modelo 4 (bloque features[8], 6x6) sobre aciertos y errores de prueba."""
    configurar_estilo()
    red = cargar_red_finetuned(DIR_MODELOS / ARCHIVO_PT)
    datos = cargar_subconjunto_eurosat()
    imgs_te = datos["imagenes"][datos["particion"] == "test"]
    aciertos, fallos = seleccionar_ejemplos_grad_cam(pred_df)

    fig, axes = plt.subplots(2, 8, figsize=(19.0, 5.6))
    for fila, (indices, tipo) in enumerate(((aciertos, "Acierto"), (fallos, "Error"))):
        for j, idx in enumerate(indices):
            img = imgs_te[idx]
            gc = grad_cam(red, img)
            real = CLASES_EUROSAT[int(pred_df.loc[idx, "y_real"])]
            ax_img, ax_cam = axes[fila, 2 * j], axes[fila, 2 * j + 1]
            ax_img.imshow(img)
            ax_img.set_title(f"{tipo} · real: {NOMBRES_ESPANOL[real].split(' /')[0]}", fontsize=8.4,
                             color=PALETA["exito"] if tipo == "Acierto" else PALETA["alerta"])
            ax_cam.imshow(img)
            ax_cam.imshow(gc["mapa"], cmap="jet", alpha=0.5, vmin=0, vmax=1)
            ax_cam.set_title(f"Grad-CAM → {NOMBRES_ESPANOL[gc['nombre_clase']].split(' /')[0]}\nP = {gc['probabilidad']*100:.1f} %".replace(".", ","), fontsize=8.2)
            for a in (ax_img, ax_cam):
                a.axis("off")
    fig.suptitle("Grad-CAM del modelo 4 (MobileNetV3 fine-tuned): arriba aciertos, abajo errores con alta confianza", fontsize=11.5, fontweight="bold", x=0.01, ha="left")
    plt.tight_layout()
    return _guardar(fig, "09_grad_cam_aciertos_y_errores.png")


def fig_10_bootstrap(df_dif: pd.DataFrame) -> Path:
    """Diferencias entre modelos con IC 95 % bootstrap pareado: exactitud (puntos) y costo por 1.000 parcelas."""
    configurar_estilo()
    acc = df_dif[df_dif["metrica"] == "exactitud"].reset_index(drop=True)
    cos = df_dif[df_dif["metrica"] == "costo_1000"].reset_index(drop=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 4.8), gridspec_kw={"width_ratios": [1.1, 1.0]})

    for ax, df, escala, xlabel in (
        (ax1, acc, 100.0, "Diferencia de exactitud (puntos) · IC 95 %"),
        (ax2, cos, 1.0, "Diferencia de costo por 1.000 parcelas (USD) · IC 95 %"),
    ):
        y = np.arange(len(df))
        for i, r in df.iterrows():
            color = PALETA["exito"] if r["significativa"] else PALETA["neutro"]
            ax.plot([r["ic95_inf"] * escala, r["ic95_sup"] * escala], [i, i], color=color, linewidth=3)
            ax.plot(r["diferencia"] * escala, i, "o", color=color, markersize=7)
        ax.axvline(0, color="#111827", linewidth=1.0)
        ax.set_yticks(y)
        ax.set_yticklabels([c.replace("_sin_abstencion", " sin abst.").replace("_con_abstencion", " con abst.").upper().replace("ABST.", "abst.").replace("SIN", "sin").replace("CON", "con") for c in df["comparacion"]], fontsize=8.6)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
    ax1.set_title("¿Qué mejoras son significativas? (verde = IC sin 0)", loc="left", pad=12)
    ax2.set_title("Costo: modelos y políticas de abstención", loc="left", pad=12)
    plt.tight_layout()
    return _guardar(fig, "10_bootstrap_diferencias.png")


def generar_todas_las_figuras() -> list[Path]:
    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    resultados = json.loads((DIR_REPORTES / "resultados.json").read_text(encoding="utf-8"))
    global N_PRUEBA
    N_PRUEBA = int(resultados["particiones"]["prueba"])
    leer = lambda nombre, **kw: pd.read_csv(DIR_REPORTES / nombre, **kw)  # noqa: E731
    pred_df = pd.read_parquet(DIR_REPORTES / "predicciones_test.parquet")

    rutas = [
        fig_01_mosaico_y_espectro(leer("resumen_clases_27000.csv")),
        fig_02_comparacion_arquitecturas(leer("comparacion_modelos.csv")),
        fig_03_curva_few_shot(leer("curva_few_shot_eurosat.csv")),
        fig_04_f1_por_clase(leer("metricas_por_clase.csv")),
        fig_05_matrices_confusion(leer("matriz_confusion_espectral.csv", index_col=0), leer("matriz_confusion_mejor_modelo.csv", index_col=0)),
        fig_06_calibracion_y_abstencion(leer("calibracion_temperatura.csv"), leer("curva_fiabilidad.csv"),
                                        leer("curva_abstencion_fotointerpretacion.csv"), resultados["abstencion"]),
        fig_07_robustez(leer("robustez_perturbaciones.csv")),
        fig_08_mapas_oclusion_espacial(),
        fig_09_grad_cam(pred_df),
        fig_10_bootstrap(leer("bootstrap_diferencias.csv")),
    ]
    print(f"[ok] {len(rutas)} figuras generadas en reports/figures/.")
    return rutas


if __name__ == "__main__":
    generar_todas_las_figuras()
