"""Descarga, auditoría radiométrica de las 27.000 imágenes Sentinel-2 (EuroSAT) y particionado estratificado."""

from __future__ import annotations

import io
import json
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from eurosat_suelo.config import (
    CLASES_EUROSAT,
    DIR_DATOS,
    DIR_MUESTRA,
    DIR_RAW,
    DIR_REPORTES,
    MACRO_CATEGORIA,
    NOMBRES_ESPANOL,
    SEMILLA,
)

URLS_PARQUET_EUROSAT: dict[str, str] = {
    "train": "https://huggingface.co/datasets/timm/eurosat-rgb/resolve/main/data/train-00000-of-00001.parquet",
    "val": "https://huggingface.co/datasets/timm/eurosat-rgb/resolve/main/data/validation-00000-of-00001.parquet",
    "test": "https://huggingface.co/datasets/timm/eurosat-rgb/resolve/main/data/test-00000-of-00001.parquet",
}


def _parquet_completo(ruta: Path) -> bool:
    """Un Parquet termina con la marca PAR1: un archivo cortado no la tiene."""
    if not ruta.exists() or ruta.stat().st_size < 12:
        return False
    with open(ruta, "rb") as f:
        f.seek(-4, 2)
        return f.read(4) == b"PAR1"


def _descargar_archivo(url: str, destino: Path) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    if _parquet_completo(destino):
        return
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (EuroSAT-Portfolio/1.0)"})
    parcial = destino.with_suffix(destino.suffix + ".part")
    with urllib.request.urlopen(req, timeout=180) as resp, open(parcial, "wb") as f:
        while True:
            bloque = resp.read(1024 * 256)
            if not bloque:
                break
            f.write(bloque)
    if not _parquet_completo(parcial):
        raise RuntimeError(f"La descarga de {destino.name} quedó incompleta; vuelve a ejecutarla")
    parcial.replace(destino)


def _decodificar_imagen_rgb64(celda_imagen: Any) -> np.ndarray:
    """Convierte una celda de imagen de Parquet (dict con bytes o bytes directos) a uint8 (64, 64, 3)."""
    if isinstance(celda_imagen, dict):
        raw_bytes = celda_imagen.get("bytes")
    elif isinstance(celda_imagen, (bytes, bytearray)):
        raw_bytes = celda_imagen
    else:
        raise TypeError(f"Formato de celda de imagen no soportado: {type(celda_imagen)}")

    with Image.open(io.BytesIO(raw_bytes)) as img:
        img_rgb = img.convert("RGB")
        if img_rgb.size != (64, 64):
            img_rgb = img_rgb.resize((64, 64), Image.Resampling.BILINEAR)
        return np.asarray(img_rgb, dtype=np.uint8)


def calcular_descriptores_espectrales_lote(imagenes_uint8: np.ndarray) -> dict[str, np.ndarray]:
    """Calcula índices espectrales visibles y textura espacial sobre un lote (N, 64, 64, 3) en [0, 255]."""
    arr = imagenes_uint8.astype(np.float32) / 255.0
    r = arr[:, :, :, 0]
    g = arr[:, :, :, 1]
    b = arr[:, :, :, 2]
    eps = 1e-6

    media_r = r.mean(axis=(1, 2))
    media_g = g.mean(axis=(1, 2))
    media_b = b.mean(axis=(1, 2))
    std_r = r.std(axis=(1, 2))
    std_g = g.std(axis=(1, 2))
    std_b = b.std(axis=(1, 2))

    suma_rgb = media_r + media_g + media_b + eps
    # Excess Green Index (ExG = (2G - R - B) / (R + G + B))
    exg = (2.0 * media_g - media_r - media_b) / suma_rgb
    # Visible Atmospherically Resistant Index (VARI)
    vari = (media_g - media_r) / (media_g + media_r - media_b + eps)
    # Índice normalizado Azul-Rojo (proxy de cuerpos de agua y sombras húmedas)
    indice_agua = (media_b - media_r) / (media_b + media_r + eps)

    # Luminancia y contraste global
    gris = 0.299 * r + 0.587 * g + 0.114 * b
    brillo = gris.mean(axis=(1, 2))
    contraste = gris.std(axis=(1, 2))

    # Rugosidad espacial (magnitud media de diferencias finitas horizontales y verticales)
    gx = np.abs(np.diff(gris, axis=2)).mean(axis=(1, 2))
    gy = np.abs(np.diff(gris, axis=1)).mean(axis=(1, 2))
    rugosidad = 0.5 * (gx + gy)

    # Anomalías radiométricas: velo atmosférico/nube delgada (alto brillo) o escena plana sin contraste
    velo_atmosferico = (brillo > 0.62) | (contraste < 0.032)

    return {
        "media_r": media_r,
        "media_g": media_g,
        "media_b": media_b,
        "std_r": std_r,
        "std_g": std_g,
        "std_b": std_b,
        "exg": exg,
        "vari": vari,
        "indice_agua": indice_agua,
        "brillo": brillo,
        "contraste": contraste,
        "rugosidad": rugosidad,
        "velo_atmosferico": velo_atmosferico,
    }


def extraer_vector_handcrafted_imagen(imagenes_uint8: np.ndarray) -> np.ndarray:
    """Extrae un vector de 54 descriptores espectrales, de color HSV/RGB y de textura espacial por imagen.

    Se utiliza para el Modelo 1 (Baseline clásico de Teledetección con Índices Espectrales + Textura).
    """
    arr = imagenes_uint8.astype(np.float32) / 255.0
    n = arr.shape[0]
    r = arr[:, :, :, 0]
    g = arr[:, :, :, 1]
    b = arr[:, :, :, 2]
    eps = 1e-6

    # 1. Estadísticos por canal (media, std, p10, p25, p50, p75, p90) -> 3 * 7 = 21 atributos
    feats: list[np.ndarray] = []
    for canal in (r, g, b):
        plano = canal.reshape(n, -1)
        feats.append(plano.mean(axis=1, keepdims=True))
        feats.append(plano.std(axis=1, keepdims=True))
        qs = np.percentile(plano, [10, 25, 50, 75, 90], axis=1).T
        feats.append(qs)

    # 2. Índices espectrales por píxel (media y std de ExG, NGRDI, NGBDI, NDBI-proxy) -> 8 atributos
    suma = r + g + b + eps
    exg_px = (2.0 * g - r - b) / suma
    ngrdi_px = (g - r) / (g + r + eps)
    ngbdi_px = (g - b) / (g + b + eps)
    ibw_px = (b - r) / (b + r + eps)
    for idx_map in (exg_px, ngrdi_px, ngbdi_px, ibw_px):
        pl = idx_map.reshape(n, -1)
        feats.append(pl.mean(axis=1, keepdims=True))
        feats.append(pl.std(axis=1, keepdims=True))

    # 3. Correlaciones inter-banda (R-G, R-B, G-B) -> 3 atributos
    r_f = r.reshape(n, -1)
    g_f = g.reshape(n, -1)
    b_f = b.reshape(n, -1)
    for c1, c2 in ((r_f, g_f), (r_f, b_f), (g_f, b_f)):
        c1_c = c1 - c1.mean(axis=1, keepdims=True)
        c2_c = c2 - c2.mean(axis=1, keepdims=True)
        num = (c1_c * c2_c).mean(axis=1, keepdims=True)
        den = c1_c.std(axis=1, keepdims=True) * c2_c.std(axis=1, keepdims=True) + eps
        feats.append(num / den)

    # 4. Gradientes espaciales (Sobel aproximado, energía direccional H/V/D y anisotropía) -> 10 atributos
    gris = 0.299 * r + 0.587 * g + 0.114 * b
    gx = np.diff(gris, axis=2)[:, :-1, :]
    gy = np.diff(gris, axis=1)[:, :, :-1]
    mag = np.sqrt(gx**2 + gy**2 + eps)
    mag_f = mag.reshape(n, -1)
    feats.append(mag_f.mean(axis=1, keepdims=True))
    feats.append(mag_f.std(axis=1, keepdims=True))
    feats.append(np.percentile(mag_f, [50, 90, 99], axis=1).T)
    energia_h = np.abs(gx).mean(axis=(1, 2), keepdims=True).reshape(n, 1)
    energia_v = np.abs(gy).mean(axis=(1, 2), keepdims=True).reshape(n, 1)
    anisotropia = np.abs(energia_h - energia_v) / (energia_h + energia_v + eps)
    feats.extend([energia_h, energia_v, anisotropia])

    # Contraste centro vs periferia (útil para autopistas, ríos y polígonos industriales) -> 2 atributos
    centro = gris[:, 16:48, 16:48].reshape(n, -1)
    feats.append(centro.mean(axis=1, keepdims=True) - gris.reshape(n, -1).mean(axis=1, keepdims=True))
    feats.append(centro.std(axis=1, keepdims=True) - gris.reshape(n, -1).std(axis=1, keepdims=True))

    # 5. Histograma grueso de luminancia y saturación (12 atributos)
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    sat = (cmax - cmin) / (cmax + eps)
    for mapa in (gris, sat):
        pl = mapa.reshape(n, -1)
        feats.append(np.percentile(pl, [5, 20, 40, 60, 80, 95], axis=1).T)

    matriz = np.hstack(feats).astype(np.float32)
    return np.nan_to_num(matriz, nan=0.0, posinf=0.0, neginf=0.0)


NOMBRE_NPZ = "eurosat_27000_rgb64.npz"


def preparar_datos_eurosat(
    n_train_por_clase: int | None = None,
    n_val_por_clase: int | None = None,
    n_test_por_clase: int | None = None,
    forzar: bool = False,
) -> dict[str, Any]:
    """Descarga los 3 splits oficiales de EuroSAT (27.000 imágenes), audita el corpus completo y guarda el archivo .npz.

    Por defecto se usan las particiones oficiales completas (16.200 / 5.400 / 5.400). Un cupo por clase
    permite volver a un subconjunto estratificado más pequeño si se quiere entrenar más rápido.
    """
    DIR_RAW.mkdir(parents=True, exist_ok=True)
    DIR_MUESTRA.mkdir(parents=True, exist_ok=True)
    DIR_REPORTES.mkdir(parents=True, exist_ok=True)

    ruta_npz = DIR_RAW / NOMBRE_NPZ
    ruta_resumen_csv = DIR_REPORTES / "resumen_clases_27000.csv"
    ruta_auditoria_json = DIR_DATOS / "cifras_control_eurosat.json"
    ruta_muestra_csv = DIR_DATOS / "catalogo_muestra.csv"

    if (
        not forzar
        and ruta_npz.exists()
        and ruta_resumen_csv.exists()
        and ruta_auditoria_json.exists()
        and ruta_muestra_csv.exists()
    ):
        return json.loads(ruta_auditoria_json.read_text(encoding="utf-8"))

    # 1. Descargar los 3 archivos Parquet oficiales a data/raw/
    rutas_parquet: dict[str, Path] = {}
    for split, url in URLS_PARQUET_EUROSAT.items():
        destino = DIR_RAW / f"eurosat_{split}.parquet"
        _descargar_archivo(url, destino)
        rutas_parquet[split] = destino

    # 2. Decodificar las 27.000 imágenes y auditar calidad radiométrica
    rng = np.random.default_rng(SEMILLA)
    registros_resumen: list[pd.DataFrame] = []

    sub_imagenes: list[np.ndarray] = []
    sub_etiquetas: list[np.ndarray] = []
    sub_particion: list[np.ndarray] = []
    sub_filenames: list[np.ndarray] = []

    cupos_por_split = {
        "train": n_train_por_clase,
        "val": n_val_por_clase,
        "test": n_test_por_clase,
    }

    total_oficial = 0
    conteo_por_split: dict[str, int] = {}

    for split, ruta_pq in rutas_parquet.items():
        df_pq = pd.read_parquet(ruta_pq)
        n_split = len(df_pq)
        total_oficial += n_split
        conteo_por_split[split] = n_split

        imgs_split = np.stack([_decodificar_imagen_rgb64(x) for x in df_pq["image"]], axis=0)
        labels_split = df_pq["label"].to_numpy(dtype=np.int64)
        if "filename" in df_pq.columns:
            fnames_split = df_pq["filename"].astype(str).to_numpy()
        else:
            fnames_split = np.array([f"{CLASES_EUROSAT[lbl]}_{split}_{i}.jpg" for i, lbl in enumerate(labels_split)])

        desc = calcular_descriptores_espectrales_lote(imgs_split)
        df_meta = pd.DataFrame(
            {
                "split_oficial": split,
                "clase_id": labels_split,
                "clase": [CLASES_EUROSAT[i] for i in labels_split],
                "media_r": desc["media_r"],
                "media_g": desc["media_g"],
                "media_b": desc["media_b"],
                "exg": desc["exg"],
                "vari": desc["vari"],
                "indice_agua": desc["indice_agua"],
                "brillo": desc["brillo"],
                "contraste": desc["contraste"],
                "rugosidad": desc["rugosidad"],
                "velo_atmosferico": desc["velo_atmosferico"],
            }
        )
        registros_resumen.append(df_meta)

        # Partición oficial completa (cupo None) o selección estratificada determinística por clase
        cupo = cupos_por_split[split]
        for clase_id in range(len(CLASES_EUROSAT)):
            idx_clase = np.where(labels_split == clase_id)[0]
            if cupo is None:
                elegidos = idx_clase.copy()
            else:
                elegidos = rng.choice(idx_clase, size=min(cupo, len(idx_clase)), replace=False)
            elegidos.sort()
            sub_imagenes.append(imgs_split[elegidos])
            sub_etiquetas.append(labels_split[elegidos])
            sub_particion.append(np.full(len(elegidos), split, dtype="<U8"))
            sub_filenames.append(fnames_split[elegidos])

    # 3. Consolidar estadísticas de las 27.000 imágenes oficiales
    df_todo = pd.concat(registros_resumen, ignore_index=True)
    resumen_clases = (
        df_todo.groupby(["clase_id", "clase"], as_index=False)
        .agg(
            imagenes_totales=("media_r", "count"),
            media_r=("media_r", "mean"),
            media_g=("media_g", "mean"),
            media_b=("media_b", "mean"),
            exg_medio=("exg", "mean"),
            vari_medio=("vari", "mean"),
            indice_agua_medio=("indice_agua", "mean"),
            brillo_medio=("brillo", "mean"),
            contraste_medio=("contraste", "mean"),
            rugosidad_media=("rugosidad", "mean"),
            pct_velo_o_bajo_contraste=("velo_atmosferico", lambda s: float(np.mean(s) * 100.0)),
        )
        .sort_values("clase_id")
    )
    resumen_clases["nombre_es"] = resumen_clases["clase"].map(NOMBRES_ESPANOL)
    resumen_clases["macro_categoria"] = resumen_clases["clase"].map(MACRO_CATEGORIA)
    resumen_clases["pct_corpus"] = resumen_clases["imagenes_totales"] / float(total_oficial) * 100.0
    resumen_clases.to_csv(ruta_resumen_csv, index=False, encoding="utf-8")

    # 4. Guardar archivo .npz con las imágenes de modelado (sin comprimir: se carga más rápido)
    arr_imgs = np.concatenate(sub_imagenes, axis=0)
    arr_lbls = np.concatenate(sub_etiquetas, axis=0)
    arr_part = np.concatenate(sub_particion, axis=0)
    arr_fnames = np.concatenate(sub_filenames, axis=0)

    np.savez(
        ruta_npz,
        imagenes=arr_imgs,
        etiquetas=arr_lbls,
        particion=arr_part,
        filenames=arr_fnames,
    )

    # 5. Guardar muestra ligera versionable (50 imágenes PNG, 5 por clase del split test)
    filas_muestra: list[dict[str, Any]] = []
    mask_test = arr_part == "test"
    imgs_test = arr_imgs[mask_test]
    lbls_test = arr_lbls[mask_test]
    fnames_test = arr_fnames[mask_test]

    for clase_id, nombre_clase in enumerate(CLASES_EUROSAT):
        idxs = np.where(lbls_test == clase_id)[0][:5]
        for j, idx_local in enumerate(idxs):
            nombre_png = f"{clase_id:02d}_{nombre_clase}_{j+1:02d}.png"
            ruta_png = DIR_MUESTRA / nombre_png
            Image.fromarray(imgs_test[idx_local]).save(ruta_png, format="PNG")
            filas_muestra.append(
                {
                    "archivo_muestra": nombre_png,
                    "filename_original": str(fnames_test[idx_local]),
                    "clase_id": int(clase_id),
                    "clase": nombre_clase,
                    "nombre_es": NOMBRES_ESPANOL[nombre_clase],
                    "macro_categoria": MACRO_CATEGORIA[nombre_clase],
                }
            )
    pd.DataFrame(filas_muestra).to_csv(ruta_muestra_csv, index=False, encoding="utf-8")

    auditoria = {
        "dataset": "EuroSAT RGB (Sentinel-2 MSI, 64x64 px, 10 m/px)",
        "fuente_oficial": "Helber et al. (2019), IEEE JSTARS / HuggingFace timm/eurosat-rgb",
        "licencia": "MIT",
        "total_imagenes_oficial": int(total_oficial),
        "conteo_splits_oficial": conteo_por_split,
        "numero_clases": len(CLASES_EUROSAT),
        "total_subconjunto_modelado": int(len(arr_imgs)),
        "n_train": int(np.sum(arr_part == "train")),
        "n_val": int(np.sum(arr_part == "val")),
        "n_test": int(np.sum(arr_part == "test")),
        "pct_velo_atmosferico_global": round(float(df_todo["velo_atmosferico"].mean() * 100.0), 2),
        "muestra_versionada_png": len(filas_muestra),
    }
    ruta_auditoria_json.write_text(json.dumps(auditoria, indent=2, ensure_ascii=False), encoding="utf-8")
    return auditoria


def cargar_subconjunto_eurosat() -> dict[str, np.ndarray]:
    """Carga las imágenes de modelado (por defecto las 27.000 oficiales) desde data/raw/eurosat_27000_rgb64.npz."""
    ruta_npz = DIR_RAW / NOMBRE_NPZ
    if not ruta_npz.exists():
        preparar_datos_eurosat(forzar=False)
    datos = np.load(ruta_npz, allow_pickle=False)
    return {
        "imagenes": datos["imagenes"],
        "etiquetas": datos["etiquetas"],
        "particion": datos["particion"],
        "filenames": datos["filenames"],
    }
