"""Evaluación en periodo de prueba (holdout): calibración de recompra, error de conteo e ingreso,
verificación del supuesto de independencia de Gamma-Gamma y curvas de captura acumulada."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import brier_score_loss, mean_absolute_error, root_mean_squared_error, roc_auc_score


def verificar_supuesto_gamma_gamma(df: pd.DataFrame) -> pd.DataFrame:
    """
    El modelo Gamma-Gamma supone que el valor monetario medio m_x es independiente de la
    frecuencia de compra x (para clientes con x >= 1). En UCI Online Retail II existen unos
    pocos distribuidores mayoristas con tickets > 3.000 GBP que distorsionan la correlación
    lineal de Pearson. Esta tabla compara la correlación antes y después de acotar o excluir
    esos mayoristas extremos.
    """
    rep = df[df["frecuencia"] >= 1].copy()
    sin_mayoristas = rep[rep["es_mayorista_extremo"] == 0]

    filas = []
    for nombre, sub, col_m in [
        ("Crudo (incluye mayoristas > 3.000 GBP)", rep, "ticket_medio"),
        ("Con tope p99,5 (ticket acotado a 3.000 GBP)", rep, "ticket_medio_acotado"),
        ("Solo minoristas (excluye mayoristas > 3.000 GBP)", sin_mayoristas, "ticket_medio"),
    ]:
        r_p, p_p = pearsonr(sub["frecuencia"], sub[col_m])
        r_s, p_s = spearmanr(sub["frecuencia"], sub[col_m])
        filas.append({
            "variante": nombre,
            "n_clientes_repetidores": len(sub),
            "ticket_medio_gbp": float(sub[col_m].mean()),
            "ticket_mediana_gbp": float(sub[col_m].median()),
            "ticket_max_gbp": float(sub[col_m].max()),
            "pearson_r": float(r_p),
            "pearson_pval": float(p_p),
            "spearman_rho": float(r_s),
            "spearman_pval": float(p_s),
        })
    return pd.DataFrame(filas)


def _captura_en_top(y_real: np.ndarray, y_score: np.ndarray, fraccion: float) -> float:
    n = len(y_real)
    k = max(int(round(n * fraccion)), 1)
    orden = np.argsort(-y_score, kind="mergesort")
    total = float(np.sum(y_real))
    if total <= 0:
        return 0.0
    return float(np.sum(y_real[orden[:k]]) / total * 100.0)


def evaluar_modelos_holdout(
    df_eval: pd.DataFrame,
    predicciones: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    """
    Compara todos los modelos en el periodo de prueba (holdout de 183 días = 6 meses):
      - Clasificación de recompra (AUC y Brier frente a vivo_holdout)
      - Conteo de compras (MAE, RMSE, sesgo agregado y Spearman)
      - Ingreso futuro en 6 meses (MAE, RMSE, Spearman, error de cartera y captura en top 10 % / 20 %)
    """
    y_vivo = df_eval["vivo_holdout"].to_numpy(dtype=int)
    y_compras = df_eval["compras_holdout"].to_numpy(dtype=float)
    y_ingreso = df_eval["gasto_holdout_acotado"].to_numpy(dtype=float)

    filas = []
    for nombre, pred in predicciones.items():
        # Si el modelo distingue entre P(vivo en T) y P(recompra en H), usamos prob_recompra frente a vivo_holdout
        prob_rec = np.clip(pred.get("prob_recompra", pred["prob_vivo"]), 1e-6, 1.0 - 1e-6)
        compras_pred = np.asarray(pred["compras_esperadas"], dtype=float)
        ingreso_pred = np.asarray(pred["ingreso_esperado"], dtype=float)

        auc_v = float(roc_auc_score(y_vivo, prob_rec))
        brier_v = float(brier_score_loss(y_vivo, prob_rec))

        mae_c = float(mean_absolute_error(y_compras, compras_pred))
        rmse_c = float(root_mean_squared_error(y_compras, compras_pred))
        sesgo_c = float((np.sum(compras_pred) - np.sum(y_compras)) / max(np.sum(y_compras), 1.0) * 100.0)
        rho_c = float(spearmanr(y_compras, compras_pred).statistic)

        mae_i = float(mean_absolute_error(y_ingreso, ingreso_pred))
        rmse_i = float(root_mean_squared_error(y_ingreso, ingreso_pred))
        sesgo_i = float((np.sum(ingreso_pred) - np.sum(y_ingreso)) / max(np.sum(y_ingreso), 1.0) * 100.0)
        rho_i = float(spearmanr(y_ingreso, ingreso_pred).statistic)

        top10 = _captura_en_top(y_ingreso, ingreso_pred, 0.10)
        top20 = _captura_en_top(y_ingreso, ingreso_pred, 0.20)

        filas.append({
            "modelo": nombre,
            "auc_recompra": auc_v,
            "brier_recompra": brier_v,
            "mae_compras": mae_c,
            "rmse_compras": rmse_c,
            "sesgo_compras_pct": sesgo_c,
            "spearman_compras": rho_c,
            "mae_ingreso_gbp": mae_i,
            "rmse_ingreso_gbp": rmse_i,
            "error_cartera_pct": sesgo_i,
            "spearman_ingreso": rho_i,
            "captura_top10_pct": top10,
            "captura_top20_pct": top20,
        })
    return pd.DataFrame(filas)


def calibracion_por_deciles(
    df_eval: pd.DataFrame,
    pred_bgnbd: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Agrupa a los clientes en 10 deciles según las compras esperadas por BG/NBD y compara real vs predicho."""
    aux = pd.DataFrame({
        "prob_vivo": pred_bgnbd["prob_vivo"],
        "prob_recompra": pred_bgnbd.get("prob_recompra", pred_bgnbd["prob_vivo"]),
        "vivo_real": df_eval["vivo_holdout"].to_numpy(dtype=float),
        "compras_pred": pred_bgnbd["compras_esperadas"],
        "compras_real": df_eval["compras_holdout"].to_numpy(dtype=float),
        "ingreso_pred": pred_bgnbd["ingreso_esperado"],
        "ingreso_real": df_eval["gasto_holdout_acotado"].to_numpy(dtype=float),
    })
    aux["decil"] = pd.qcut(aux["compras_pred"].rank(method="first"), 10, labels=range(1, 11)).astype(int)
    res = (
        aux.groupby("decil", as_index=False)
        .agg(
            n_clientes=("vivo_real", "count"),
            prob_vivo_media=("prob_vivo", "mean"),
            prob_recompra_media=("prob_recompra", "mean"),
            tasa_recompra_real=("vivo_real", "mean"),
            compras_pred_media=("compras_pred", "mean"),
            compras_real_media=("compras_real", "mean"),
            ingreso_pred_medio=("ingreso_pred", "mean"),
            ingreso_real_medio=("ingreso_real", "mean"),
            ingreso_real_total=("ingreso_real", "sum"),
        )
        .sort_values("decil")
    )
    res["pct_ingreso_real"] = res["ingreso_real_total"] / max(float(res["ingreso_real_total"].sum()), 1.0) * 100.0
    return res


def calibracion_por_frecuencia(
    df_eval: pd.DataFrame,
    pred_bgnbd: dict[str, np.ndarray],
    max_frec: int = 10,
) -> pd.DataFrame:
    """Compara compras esperadas vs compras reales en holdout según la frecuencia histórica x = 0, 1, ..., 10+."""
    aux = pd.DataFrame({
        "frecuencia_tramo": np.minimum(df_eval["frecuencia"].to_numpy(dtype=int), max_frec),
        "compras_pred": pred_bgnbd["compras_esperadas"],
        "compras_real": df_eval["compras_holdout"].to_numpy(dtype=float),
        "prob_recompra": pred_bgnbd.get("prob_recompra", pred_bgnbd["prob_vivo"]),
        "vivo_real": df_eval["vivo_holdout"].to_numpy(dtype=float),
    })
    return (
        aux.groupby("frecuencia_tramo", as_index=False)
        .agg(
            n_clientes=("compras_real", "count"),
            compras_pred_media=("compras_pred", "mean"),
            compras_real_media=("compras_real", "mean"),
            prob_recompra_media=("prob_recompra", "mean"),
            tasa_recompra_real=("vivo_real", "mean"),
        )
    )


def curvas_captura_ingreso(
    df_eval: pd.DataFrame,
    predicciones: dict[str, dict[str, np.ndarray]],
    n_puntos: int = 20,
) -> pd.DataFrame:
    """Calcula la curva de Lorenz / ganancia acumulada de ingreso real en holdout para cada modelo."""
    y_ingreso = df_eval["gasto_holdout_acotado"].to_numpy(dtype=float)
    total = max(float(np.sum(y_ingreso)), 1.0)
    fracciones = np.linspace(0.0, 1.0, n_puntos + 1)

    filas = []
    for f in fracciones:
        fila: dict[str, float] = {"pct_clientes": float(f * 100.0)}
        for nombre, pred in predicciones.items():
            if f == 0.0:
                fila[nombre] = 0.0
            else:
                fila[nombre] = _captura_en_top(y_ingreso, pred["ingreso_esperado"], float(f))
        # Oráculo (ordenando por el gasto real futuro)
        fila["Oráculo"] = 0.0 if f == 0.0 else _captura_en_top(y_ingreso, y_ingreso, float(f))
        fila["Azar"] = float(f * 100.0)
        filas.append(fila)
    return pd.DataFrame(filas)


def _metricas_ranking(
    y_vivo: np.ndarray,
    y_compras: np.ndarray,
    y_ingreso: np.ndarray,
    pred: dict[str, np.ndarray],
) -> dict[str, float]:
    prob = np.clip(np.asarray(pred.get("prob_recompra", pred["prob_vivo"]), dtype=float), 1e-6, 1.0 - 1e-6)
    ingreso = np.asarray(pred["ingreso_esperado"], dtype=float)
    return {
        "auc_recompra": float(roc_auc_score(y_vivo, prob)),
        "brier_recompra": float(brier_score_loss(y_vivo, prob)),
        "spearman_compras": float(spearmanr(y_compras, pred["compras_esperadas"]).statistic),
        "spearman_ingreso": float(spearmanr(y_ingreso, ingreso).statistic),
        "captura_top20_pct": _captura_en_top(y_ingreso, ingreso, 0.20),
    }


def bootstrap_comparacion(
    df_eval: pd.DataFrame,
    pred_a: dict[str, np.ndarray],
    pred_b: dict[str, np.ndarray],
    nombre_a: str = "BG/NBD + Gamma-Gamma",
    nombre_b: str = "LightGBM supervisado",
    n_bootstrap: int = 1000,
    semilla: int = 42,
) -> pd.DataFrame:
    """
    Bootstrap pareado (se remuestrean clientes, los mismos para ambos modelos) de la diferencia A − B en
    AUC, Brier, Spearman y captura del top 20 %. La diferencia es significativa al 95 % si el intervalo
    percentil no contiene el 0.
    """
    y_vivo = df_eval["vivo_holdout"].to_numpy(dtype=int)
    y_compras = df_eval["compras_holdout"].to_numpy(dtype=float)
    y_ingreso = df_eval["gasto_holdout_acotado"].to_numpy(dtype=float)
    pa = {k: np.asarray(v) for k, v in pred_a.items()}
    pb = {k: np.asarray(v) for k, v in pred_b.items()}

    base_a = _metricas_ranking(y_vivo, y_compras, y_ingreso, pa)
    base_b = _metricas_ranking(y_vivo, y_compras, y_ingreso, pb)
    rng = np.random.default_rng(semilla)
    n = len(y_vivo)
    difs: dict[str, list[float]] = {k: [] for k in base_a}
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        if y_vivo[idx].min() == y_vivo[idx].max():
            continue
        ma = _metricas_ranking(y_vivo[idx], y_compras[idx], y_ingreso[idx], {k: v[idx] for k, v in pa.items()})
        mb = _metricas_ranking(y_vivo[idx], y_compras[idx], y_ingreso[idx], {k: v[idx] for k, v in pb.items()})
        for k in difs:
            difs[k].append(ma[k] - mb[k])

    filas = []
    for k, valores in difs.items():
        ic_inf, ic_sup = np.percentile(valores, [2.5, 97.5])
        filas.append({
            "metrica": k,
            "modelo_a": nombre_a,
            "modelo_b": nombre_b,
            "valor_a": base_a[k],
            "valor_b": base_b[k],
            "diferencia_a_menos_b": base_a[k] - base_b[k],
            "ic95_inf": float(ic_inf),
            "ic95_sup": float(ic_sup),
            "significativa_95": bool(ic_inf > 0 or ic_sup < 0),
            "n_bootstrap": len(valores),
        })
    return pd.DataFrame(filas)
