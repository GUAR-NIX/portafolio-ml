"""Modelos probabilísticos BG/NBD y Gamma-Gamma (Fader, Hardie & Lee, 2005),
heurísticas de línea base y modelo supervisado LightGBM (Hurdle/Tweedie)."""
from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import betaln, gammaln, hyp2f1, roots_legendre

from . import config


@dataclass
class ModeloBGNBD:
    """
    Beta-Geometric / Negative Binomial Distribution (Fader, Hardie & Lee, 2005).
    Parámetros:
        r, alpha : forma y escala de la distribución Gamma de la tasa de compra lambda.
        a, b     : parámetros de la distribución Beta de la probabilidad de abandono p tras cada compra.
    """
    penalizacion: float = 0.001
    r: float = 1.0
    alpha: float = 1.0
    a: float = 1.0
    b: float = 1.0
    log_verosimilitud_: float = float("nan")
    exito_optimizacion_: bool = False

    @staticmethod
    def _log_verosimilitud_individual(
        params_log: np.ndarray,
        x: np.ndarray,
        t_x: np.ndarray,
        T: np.ndarray,
    ) -> np.ndarray:
        r, alpha, a, b = np.exp(params_log)
        ln_a1 = gammaln(r + x) - gammaln(r) + r * np.log(alpha)
        ln_a2 = gammaln(a + b) + gammaln(b + x) - gammaln(b) - gammaln(a + b + x)
        ln_a3 = -(r + x) * np.log(alpha + T)

        tiene_rep = x > 0
        ln_a4 = np.full_like(ln_a3, -np.inf, dtype=float)
        if np.any(tiene_rep):
            xm = x[tiene_rep]
            txm = t_x[tiene_rep]
            ln_a4[tiene_rep] = np.log(a) - np.log(b + xm - 1.0) - (r + xm) * np.log(alpha + txm)

        ln_a3_a4 = np.logaddexp(ln_a3, ln_a4)
        return ln_a1 + ln_a2 + ln_a3_a4

    def ajustar(self, x: np.ndarray, t_x: np.ndarray, T: np.ndarray) -> "ModeloBGNBD":
        x_arr = np.asarray(x, dtype=float)
        tx_arr = np.asarray(t_x, dtype=float)
        T_arr = np.asarray(T, dtype=float)

        def objetivo(params_log: np.ndarray) -> float:
            ll = self._log_verosimilitud_individual(params_log, x_arr, tx_arr, T_arr)
            if not np.all(np.isfinite(ll)):
                return 1e12
            reg = self.penalizacion * float(np.sum(np.exp(params_log) ** 2))
            return -float(np.sum(ll)) + reg

        mejores_params = np.zeros(4)
        mejor_val = np.inf
        exito = False
        puntos_inicio = [
            np.log([0.5, 4.0, 0.2, 1.5]),
            np.log([0.8, 6.0, 0.5, 2.5]),
            np.log([0.3, 2.0, 0.1, 1.0]),
        ]
        limites = [(-6.0, 6.0), (-6.0, 8.0), (-6.0, 6.0), (-6.0, 8.0)]
        for p0 in puntos_inicio:
            res = minimize(objetivo, p0, method="L-BFGS-B", bounds=limites)
            if res.fun < mejor_val:
                mejor_val = float(res.fun)
                mejores_params = res.x
                exito = bool(res.success)

        self.r, self.alpha, self.a, self.b = [float(v) for v in np.exp(mejores_params)]
        ll_final = self._log_verosimilitud_individual(mejores_params, x_arr, tx_arr, T_arr)
        self.log_verosimilitud_ = float(np.sum(ll_final))
        self.exito_optimizacion_ = exito
        return self

    def probabilidad_vivo(self, x: np.ndarray, t_x: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Probabilidad condicional P(vivo en T | x, t_x, T) (Fader, Hardie & Lee 2005, Ec. 11)."""
        x_arr = np.asarray(x, dtype=float)
        tx_arr = np.asarray(t_x, dtype=float)
        T_arr = np.asarray(T, dtype=float)

        prob = np.ones_like(x_arr, dtype=float)
        m = x_arr > 0
        if np.any(m):
            xm = x_arr[m]
            txm = tx_arr[m]
            Tm = T_arr[m]
            log_termino = (
                np.log(self.a)
                - np.log(self.b + xm - 1.0)
                + (self.r + xm) * (np.log(self.alpha + Tm) - np.log(self.alpha + txm))
            )
            log_termino = np.clip(log_termino, -50.0, 50.0)
            prob[m] = 1.0 / (1.0 + np.exp(log_termino))
        return prob

    def probabilidad_recompra(
        self,
        t: float | np.ndarray,
        x: np.ndarray,
        t_x: np.ndarray,
        T: np.ndarray,
    ) -> np.ndarray:
        """
        Probabilidad exacta de realizar al menos 1 compra en el horizonte (T, T + t]:
            P(Y(t) >= 1 | x, t_x, T) = P(vivo en T | x, t_x, T) * [1 - ((alpha + T) / (alpha + T + t))^(r + x)]
        A diferencia de P(vivo en T) (que vale 1 para x = 0 porque la baja en BG/NBD ocurre tras una compra),
        P(Y(t) >= 1) refleja también el decaimiento de la tasa posterior lambda cuando pasa el tiempo sin comprar.
        """
        x_arr = np.asarray(x, dtype=float)
        tx_arr = np.asarray(t_x, dtype=float)
        T_arr = np.asarray(T, dtype=float)
        t_arr = np.asarray(t, dtype=float)
        p_vivo = self.probabilidad_vivo(x_arr, tx_arr, T_arr)
        p_cero_dado_vivo = ((self.alpha + T_arr) / (self.alpha + T_arr + t_arr)) ** (self.r + x_arr)
        return p_vivo * (1.0 - p_cero_dado_vivo)

    def compras_esperadas(
        self,
        t: float | np.ndarray,
        x: np.ndarray,
        t_x: np.ndarray,
        T: np.ndarray,
        usar_cuadratura: bool = False,
    ) -> np.ndarray:
        """
        Número esperado de compras E[Y(t) | x, t_x, T] en el intervalo (T, T + t].
        Usa la solución analítica exacta con 2F1 (Fader, Hardie & Lee 2005, Ec. 10), válida por
        continuación analítica para todo a > 0 (a != 1). Si usar_cuadratura=True o |a - 1| < 1e-4,
        integra exactamente mediante el cambio de variable p = u^(1/a) que elimina la singularidad en p=0.
        """
        x_arr = np.asarray(x, dtype=float)
        tx_arr = np.asarray(t_x, dtype=float)
        T_arr = np.asarray(T, dtype=float)
        t_arr = np.asarray(t, dtype=float)

        p_vivo = self.probabilidad_vivo(x_arr, tx_arr, T_arr)

        if not usar_cuadratura and abs(self.a - 1.0) > 1e-4:
            z = t_arr / (self.alpha + T_arr + t_arr)
            h = hyp2f1(self.r + x_arr, self.b + x_arr, self.a + self.b + x_arr - 1.0, z)
            factor = (self.a + self.b + x_arr - 1.0) / (self.a - 1.0)
            corchete = 1.0 - ((self.alpha + T_arr) / (self.alpha + T_arr + t_arr)) ** (self.r + x_arr) * h
            return factor * corchete * p_vivo

        # Cuadratura de Gauss-Legendre de 64 nodos (con sustitución p = u^(1/a) solo si a < 1)
        nodos, pesos = roots_legendre(64)
        u_grid = 0.5 * (nodos + 1.0)
        w_grid = 0.5 * pesos
        exponente = (1.0 / self.a) if self.a < 1.0 else 1.0
        p = (u_grid ** exponente)[None, :]

        ratio_t = (t_arr / (self.alpha + T_arr))[..., None]
        rx = (self.r + x_arr)[..., None]
        bx = (self.b + x_arr)[..., None]

        integrando = np.where(
            p > 1e-8,
            (1.0 - (1.0 + ratio_t * p) ** (-rx)) / p,
            rx * ratio_t,
        )
        if self.a < 1.0:
            log_peso_beta = (bx - 1.0) * np.log(np.maximum(1.0 - p, 1e-15)) - np.log(self.a) - betaln(self.a, bx)
        else:
            log_peso_beta = (
                (self.a - 1.0) * np.log(np.maximum(p, 1e-15))
                + (bx - 1.0) * np.log(np.maximum(1.0 - p, 1e-15))
                - betaln(self.a, bx)
            )
        esperado_cond_vivo = np.sum(integrando * np.exp(log_peso_beta) * w_grid[None, :], axis=-1)
        return esperado_cond_vivo * p_vivo

    def distribucion_frecuencia_teorica(self, T: np.ndarray, max_x: int = 7) -> np.ndarray:
        """
        Calcula la distribución marginal esperada P(X(T) = x) promediada sobre las edades T
        observadas (Fader, Hardie & Lee 2005, Ec. 5), agrupando la cola en >= max_x.
        """
        T_arr = np.asarray(T, dtype=float)
        probs = np.zeros(max_x + 1, dtype=float)
        acumulado = np.zeros_like(T_arr, dtype=float)

        for k in range(max_x):
            ln_t1 = (
                gammaln(self.r + k)
                - gammaln(self.r)
                - gammaln(k + 1)
                + self.r * np.log(self.alpha)
                - (self.r + k) * np.log(self.alpha + T_arr)
                + betaln(self.a, self.b + k)
                - betaln(self.a, self.b)
            )
            t1 = np.exp(ln_t1)
            if k == 0:
                pk = t1
            else:
                j = np.arange(0, k)[:, None]  # (k, 1)
                T_row = np.maximum(T_arr[None, :], 1e-8)  # (1, N), evita 0 * log(0) cuando T=0
                ln_suma = (
                    gammaln(self.r + j)
                    - gammaln(self.r)
                    - gammaln(j + 1)
                    + self.r * np.log(self.alpha)
                    + np.where(j == 0, 0.0, j * np.log(T_row))
                    - (self.r + j) * np.log(self.alpha + T_row)
                )
                coef_beta = np.exp(betaln(self.a + 1.0, self.b + k - 1.0) - betaln(self.a, self.b))
                pk = t1 + coef_beta * (1.0 - np.sum(np.exp(ln_suma), axis=0))
            pk = np.clip(pk, 0.0, 1.0)
            probs[k] = float(np.mean(pk))
            acumulado += pk

        probs[max_x] = float(np.mean(np.clip(1.0 - acumulado, 0.0, 1.0)))
        probs /= max(float(np.sum(probs)), 1e-12)
        return probs


@dataclass
class ModeloGammaGamma:
    """
    Modelo Gamma-Gamma de valor monetario (Fader, Hardie & Lee, 2005).
    Parámetros:
        p     : parámetro de forma de cada compra individual.
        q     : parámetro de forma de la distribución Gamma de la tasa latente nu.
        gamma : parámetro de escala de la distribución Gamma de nu.
    """
    penalizacion: float = 0.001
    p: float = 1.0
    q: float = 2.5
    gamma: float = 100.0
    log_verosimilitud_: float = float("nan")
    exito_optimizacion_: bool = False

    @staticmethod
    def _log_verosimilitud_individual(
        params_log: np.ndarray,
        x: np.ndarray,
        m_x: np.ndarray,
    ) -> np.ndarray:
        p, q_menos_1, gamma = np.exp(params_log)
        q = 1.05 + q_menos_1  # Garantiza q > 1 para que exista la media poblacional E[M]
        return (
            gammaln(p * x + q)
            - gammaln(p * x)
            - gammaln(q)
            + q * np.log(gamma)
            + (p * x - 1.0) * np.log(m_x)
            + (p * x) * np.log(x)
            - (p * x + q) * np.log(gamma + x * m_x)
        )

    def ajustar(self, x: np.ndarray, m_x: np.ndarray) -> "ModeloGammaGamma":
        x_arr = np.asarray(x, dtype=float)
        m_arr = np.asarray(m_x, dtype=float)
        mask = (x_arr >= 1) & (m_arr > 0)
        x_fit = x_arr[mask]
        m_fit = m_arr[mask]

        def objetivo(params_log: np.ndarray) -> float:
            ll = self._log_verosimilitud_individual(params_log, x_fit, m_fit)
            if not np.all(np.isfinite(ll)):
                return 1e12
            reg = self.penalizacion * float(np.sum(np.exp(params_log[:2]) ** 2))
            return -float(np.sum(ll)) + reg

        media_obs = float(np.mean(m_fit))
        p0 = np.log([4.0, 3.0, max(media_obs, 10.0)])
        limites = [(-4.0, 6.0), (-4.0, 6.0), (0.0, 12.0)]
        res = minimize(objetivo, p0, method="L-BFGS-B", bounds=limites)

        p_opt, q_m1_opt, gamma_opt = np.exp(res.x)
        self.p = float(p_opt)
        self.q = float(1.05 + q_m1_opt)
        self.gamma = float(gamma_opt)
        self.log_verosimilitud_ = float(
            np.sum(self._log_verosimilitud_individual(res.x, x_fit, m_fit))
        )
        self.exito_optimizacion_ = bool(res.success)
        return self

    @property
    def ticket_medio_poblacional(self) -> float:
        """Media a priori E[M] = (p * gamma) / (q - 1)."""
        return float((self.p * self.gamma) / max(self.q - 1.0, 1e-6))

    def peso_credibilidad(self, x: np.ndarray) -> np.ndarray:
        """Peso de contracción empírica de Bayes w(x) = (p * x) / (p * x + q - 1)."""
        x_arr = np.asarray(x, dtype=float)
        return np.where(
            x_arr >= 1,
            (self.p * x_arr) / (self.p * x_arr + self.q - 1.0),
            0.0,
        )

    def ticket_esperado(self, x: np.ndarray, m_x: np.ndarray) -> np.ndarray:
        """
        Valor monetario esperado condicional E[M | x, m_x].
        Para clientes con x == 0 (sin compras repetidas), usa contracción parcial con su primera compra
        (equivalente a media bayesiana con 1 observación) para aprovechar la señal de su pedido inicial.
        """
        x_arr = np.asarray(x, dtype=float)
        m_arr = np.asarray(m_x, dtype=float)
        mu_0 = self.ticket_medio_poblacional
        w = self.peso_credibilidad(x_arr)
        # Para x == 0 usamos peso w_0 = p / (p + 2*(q - 1)) sobre la primera compra observada
        w_0 = self.p / (self.p + 2.0 * (self.q - 1.0))
        peso_efectivo = np.where(x_arr >= 1, w, w_0)
        return (1.0 - peso_efectivo) * mu_0 + peso_efectivo * m_arr


def calcular_clv(
    bgnbd: ModeloBGNBD,
    gg: ModeloGammaGamma,
    x: np.ndarray,
    t_x: np.ndarray,
    T: np.ndarray,
    m_x: np.ndarray,
    horizonte_meses: int = config.HORIZONTE_MESES,
    margen_bruto: float = config.MARGEN_BRUTO,
    tasa_descuento_mensual: float = config.TASA_DESCUENTO_MENSUAL,
) -> dict[str, np.ndarray]:
    """
    Calcula para cada cliente:
      - prob_vivo          : P(vivo en T | x, t_x, T)
      - compras_esperadas  : E[Y(H) | x, t_x, T] en el horizonte total
      - ticket_esperado    : E[M | x, m_x]
      - ingreso_esperado   : compras_esperadas * ticket_esperado (sin descontar)
      - clv_neto           : valor presente neto del margen descontado mes a mes
    """
    semanas_por_mes = (config.HORIZONTE_DIAS / 6.0) / config.DIAS_POR_SEMANA
    clv_acum = np.zeros_like(np.asarray(x, dtype=float))
    compras_prev = np.zeros_like(clv_acum)
    ticket_esp = gg.ticket_esperado(x, m_x)

    for mes in range(1, horizonte_meses + 1):
        t_sem = mes * semanas_por_mes
        compras_acum = bgnbd.compras_esperadas(t_sem, x, t_x, T)
        compras_mes = np.maximum(compras_acum - compras_prev, 0.0)
        compras_prev = compras_acum
        factor_descuento = (1.0 + tasa_descuento_mensual) ** mes
        clv_acum += (compras_mes * ticket_esp * margen_bruto) / factor_descuento

    t_total = horizonte_meses * semanas_por_mes
    compras_totales = bgnbd.compras_esperadas(t_total, x, t_x, T)
    ingreso_bruto = compras_totales * ticket_esp
    prob_vivo = bgnbd.probabilidad_vivo(x, t_x, T)
    prob_recompra = bgnbd.probabilidad_recompra(t_total, x, t_x, T)

    return {
        "prob_vivo": prob_vivo,
        "prob_recompra": prob_recompra,
        "compras_esperadas": compras_totales,
        "ticket_esperado": ticket_esp,
        "ingreso_esperado": ingreso_bruto,
        "clv_neto": clv_acum,
    }


class ReglaRitmoHistorico:
    """
    Heurística 1 (Extrapolación ingenua del ritmo histórico):
    Supone que quien compró en los últimos 180 días sigue activo y mantiene su ritmo
    diario histórico (n_compras / (edad_dias + 30)).
    """
    def predecir(self, df: pd.DataFrame, horizonte_dias: int = config.HORIZONTE_DIAS) -> dict[str, np.ndarray]:
        prob_vivo = np.clip(1.0 - df["inactividad_dias"].to_numpy() / 270.0, 0.05, 0.95)
        ritmo_diario = df["n_compras"].to_numpy() / np.maximum(df["edad_dias"].to_numpy() + 30.0, 30.0)
        compras_esp = ritmo_diario * horizonte_dias * prob_vivo
        ticket_esp = df["ticket_medio_acotado"].to_numpy()
        ingreso_esp = compras_esp * ticket_esp
        return {
            "prob_vivo": prob_vivo,
            "compras_esperadas": compras_esp,
            "ingreso_esperado": ingreso_esp,
        }


class ReglaQuintilesRFM:
    """
    Heurística 2 (Tabla dinámica por tramos de Recencia × Frecuencia):
    Aprende en la ventana previa la tasa de recompra, compras medias e ingreso medio
    de cada celda (tramo_inactividad, tramo_frecuencia).
    """
    def __init__(self) -> None:
        self.tabla_: pd.DataFrame | None = None
        self.media_global_: dict[str, float] = {}

    @staticmethod
    def _tramos(df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        out["tramo_inact"] = pd.cut(
            df["inactividad_dias"],
            bins=[-1, 30, 90, 180, 300, 10000],
            labels=["0-30d", "31-90d", "91-180d", "181-300d", ">300d"],
        )
        out["tramo_frec"] = pd.cut(
            df["frecuencia"],
            bins=[-1, 0, 1, 3, 7, 10000],
            labels=["0", "1", "2-3", "4-7", "8+"],
        )
        return out

    def ajustar(self, df_hist: pd.DataFrame) -> "ReglaQuintilesRFM":
        tr = self._tramos(df_hist)
        aux = tr.copy()
        aux["vivo_holdout"] = df_hist["vivo_holdout"]
        aux["compras_holdout"] = df_hist["compras_holdout"]
        aux["gasto_holdout_acotado"] = df_hist["gasto_holdout_acotado"]
        self.tabla_ = (
            aux.groupby(["tramo_inact", "tramo_frec"], observed=False)
            .agg(
                prob_vivo=("vivo_holdout", "mean"),
                compras_esperadas=("compras_holdout", "mean"),
                ingreso_esperado=("gasto_holdout_acotado", "mean"),
            )
            .reset_index()
        )
        self.media_global_ = {
            "prob_vivo": float(df_hist["vivo_holdout"].mean()),
            "compras_esperadas": float(df_hist["compras_holdout"].mean()),
            "ingreso_esperado": float(df_hist["gasto_holdout_acotado"].mean()),
        }
        return self

    def predecir(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        tr = self._tramos(df).merge(self.tabla_, on=["tramo_inact", "tramo_frec"], how="left")
        return {
            "prob_vivo": tr["prob_vivo"].fillna(self.media_global_["prob_vivo"]).to_numpy(),
            "compras_esperadas": tr["compras_esperadas"].fillna(self.media_global_["compras_esperadas"]).to_numpy(),
            "ingreso_esperado": tr["ingreso_esperado"].fillna(self.media_global_["ingreso_esperado"]).to_numpy(),
        }


class ModeloSupervisadoCLV:
    """
    Modelo ML supervisado en dos etapas (Hurdle LightGBM) entrenado sobre una ventana
    histórica previa (sin fuga de información):
      1) LGBMClassifier para P(vivo_holdout = 1)
      2) LGBMRegressor para compras_holdout e ingreso en holdout
    """
    VARIABLES = [
        "frecuencia",
        "recencia_dias",
        "edad_dias",
        "inactividad_dias",
        "ticket_medio_acotado",
        "gasto_total_cal_acotado",
        "desv_ticket",
        "unidades_medias",
        "items_medios",
        "compras_ultimos_90d",
        "gasto_ultimos_90d",
        "es_reino_unido",
    ]

    def __init__(self, semilla: int = config.SEMILLA) -> None:
        self.semilla = semilla
        self.clf_vivo = lgb.LGBMClassifier(
            n_estimators=120,
            max_depth=4,
            num_leaves=12,
            learning_rate=0.04,
            min_child_samples=40,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=semilla,
            verbose=-1,
        )
        self.reg_compras = lgb.LGBMRegressor(
            objective="poisson",
            n_estimators=120,
            max_depth=4,
            num_leaves=12,
            learning_rate=0.04,
            min_child_samples=40,
            random_state=semilla,
            verbose=-1,
        )
        self.reg_ticket = lgb.LGBMRegressor(
            objective="regression_l1",
            n_estimators=100,
            max_depth=4,
            num_leaves=10,
            learning_rate=0.04,
            min_child_samples=40,
            random_state=semilla,
            verbose=-1,
        )

    def ajustar(self, df_hist: pd.DataFrame) -> "ModeloSupervisadoCLV":
        X = df_hist[self.VARIABLES]
        self.clf_vivo.fit(X, df_hist["vivo_holdout"])
        self.reg_compras.fit(X, df_hist["compras_holdout"])
        activos = df_hist["compras_holdout"] > 0
        self.reg_ticket.fit(X.loc[activos], df_hist.loc[activos, "ticket_holdout_acotado"])
        return self

    def predecir(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        X = df[self.VARIABLES]
        prob_vivo = self.clf_vivo.predict_proba(X)[:, 1]
        compras_esp = np.clip(self.reg_compras.predict(X), 0.0, None)
        ticket_esp = np.clip(self.reg_ticket.predict(X), 10.0, None)
        ingreso_esp = compras_esp * ticket_esp
        return {
            "prob_vivo": prob_vivo,
            "compras_esperadas": compras_esp,
            "ticket_esperado": ticket_esp,
            "ingreso_esperado": ingreso_esp,
        }
