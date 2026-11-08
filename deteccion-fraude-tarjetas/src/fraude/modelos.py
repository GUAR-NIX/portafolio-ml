"""Modelos: líneas base, detectores de anomalías no supervisados, LightGBM y el híbrido.

Convención: todo `puntaje(...)` devuelve un número **mayor cuanto más sospechosa** es la transacción.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fraude.config import SEMILLA

IDX_LOG_MONTO = -1  # última columna de COLUMNAS_MODELO


# ---------------------------------------------------------------- líneas base
def puntaje_regla_monto(X: np.ndarray) -> np.ndarray:
    """Regla de negocio ingenua: revisar primero los montos más altos."""
    return X[:, IDX_LOG_MONTO].copy()


def entrenar_logistica(X: np.ndarray, y: np.ndarray):
    modelo = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))
    return modelo.fit(X, y)


# ---------------------------------------------------------------- no supervisados
def entrenar_isolation_forest(X: np.ndarray, semilla: int = SEMILLA) -> IsolationForest:
    return IsolationForest(n_estimators=200, max_samples=256, random_state=semilla, n_jobs=-1).fit(X)


def puntaje_isolation_forest(modelo: IsolationForest, X: np.ndarray) -> np.ndarray:
    return -modelo.score_samples(X)


def entrenar_lof(X_normales: np.ndarray, n_max: int = 40_000, semilla: int = SEMILLA):
    """LOF en modo *novelty* (aprende la densidad de lo normal y puntúa datos nuevos).

    Se submuestrea a 40.000 normales: LOF guarda todo el conjunto y busca vecinos en cada predicción.
    """
    rng = np.random.default_rng(semilla)
    idx = rng.choice(len(X_normales), size=min(n_max, len(X_normales)), replace=False)
    return make_pipeline(StandardScaler(), LocalOutlierFactor(n_neighbors=35, novelty=True, n_jobs=-1)).fit(X_normales[idx])


def puntaje_lof(modelo, X: np.ndarray) -> np.ndarray:
    return -modelo.score_samples(X)


@dataclass
class Autoencoder:
    """Autoencoder denso 29→20→10→20→29 entrenado en PyTorch (CPU) y servido en NumPy.

    Los pesos se guardan en un .npz (~10 KB) y la inferencia es NumPy puro, así que la API no
    necesita importar torch. El puntaje es el error cuadrático medio de reconstrucción.
    """

    media: np.ndarray
    desv: np.ndarray
    pesos: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    historial: list[dict] = field(default_factory=list)

    def puntaje(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self.media) / self.desv
        h = Z
        for i, (W, b) in enumerate(self.pesos):
            h = h @ W.T + b
            if i < len(self.pesos) - 1:
                h = np.tanh(h)
        return np.mean((h - Z) ** 2, axis=1)

    def guardar(self, ruta) -> None:
        arr = {"media": self.media, "desv": self.desv}
        for i, (W, b) in enumerate(self.pesos):
            arr[f"W{i}"], arr[f"b{i}"] = W, b
        np.savez_compressed(ruta, **arr)

    @classmethod
    def cargar(cls, ruta) -> "Autoencoder":
        d = np.load(ruta)
        n = sum(1 for k in d.files if k.startswith("W"))
        return cls(media=d["media"], desv=d["desv"], pesos=[(d[f"W{i}"], d[f"b{i}"]) for i in range(n)])


def entrenar_autoencoder(
    X_normales: np.ndarray,
    capas: tuple[int, ...] = (20, 10, 20),
    epocas: int = 40,
    lote: int = 512,
    lr: float = 1e-3,
    paciencia: int = 5,
    semilla: int = SEMILLA,
    verbose: bool = True,
) -> Autoencoder:
    """Entrena solo con transacciones normales; el 10 % de ellas sirve para la parada temprana.

    La parada temprana mira el error de reconstrucción de normales, no etiquetas de fraude.
    """
    import torch
    from torch import nn

    torch.manual_seed(semilla)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    rng = np.random.default_rng(semilla)
    media, desv = X_normales.mean(axis=0), X_normales.std(axis=0) + 1e-6
    Z = ((X_normales - media) / desv).astype(np.float32)
    idx = rng.permutation(len(Z))
    corte = int(0.9 * len(Z))
    Zt, Zv = torch.from_numpy(Z[idx[:corte]]), torch.from_numpy(Z[idx[corte:]])

    dims = [Z.shape[1], *capas, Z.shape[1]]
    modulos: list[nn.Module] = []
    for i in range(len(dims) - 1):
        modulos.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            modulos.append(nn.Tanh())
    red = nn.Sequential(*modulos)
    opt = torch.optim.Adam(red.parameters(), lr=lr)
    perdida = nn.MSELoss()

    mejor, mejor_estado, sin_mejora, historial = np.inf, None, 0, []
    for ep in range(1, epocas + 1):
        t0 = time.time()
        red.train()
        perm = torch.randperm(len(Zt))
        total = 0.0
        for i in range(0, len(Zt), lote):
            b = Zt[perm[i : i + lote]]
            opt.zero_grad()
            l = perdida(red(b), b)
            l.backward()
            opt.step()
            total += l.item() * len(b)
        red.eval()
        with torch.no_grad():
            lv = perdida(red(Zv), Zv).item()
        historial.append({"epoca": ep, "mse_entrenamiento": total / len(Zt), "mse_validacion_normales": lv, "segundos": time.time() - t0})
        if verbose and (ep == 1 or ep % 5 == 0):
            print(f"  [AE] época {ep:2d} · MSE ent {total / len(Zt):.4f} · MSE normales reservadas {lv:.4f}")
        if lv < mejor - 1e-5:
            mejor, sin_mejora = lv, 0
            mejor_estado = {k: v.detach().clone() for k, v in red.state_dict().items()}
        else:
            sin_mejora += 1
            if sin_mejora >= paciencia:
                break
    red.load_state_dict(mejor_estado)
    lineales = [m for m in red if isinstance(m, nn.Linear)]
    pesos = [(m.weight.detach().numpy().astype(np.float64), m.bias.detach().numpy().astype(np.float64)) for m in lineales]
    return Autoencoder(media=media, desv=desv, pesos=pesos, historial=historial)


# ---------------------------------------------------------------- supervisados
PARAMS_LGBM = dict(
    objective="binary",
    learning_rate=0.03,
    num_leaves=15,
    min_child_samples=40,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    n_jobs=-1,
    verbose=-1,
)


def submuestrear(X, y, razon_normales_por_fraude: int, semilla: int = SEMILLA):
    """Submuestreo aleatorio de normales: conserva todos los fraudes y `razon` normales por cada uno."""
    rng = np.random.default_rng(semilla)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    neg_sel = rng.choice(neg, size=min(len(neg), razon_normales_por_fraude * len(pos)), replace=False)
    idx = np.sort(np.r_[pos, neg_sel])
    return X[idx], y[idx]


def entrenar_lgbm(
    X, y, X_val=None, y_val=None, peso_positivos: float = 1.0, n_estimadores: int = 3000,
    semilla: int = SEMILLA, **extra,
) -> lgb.LGBMClassifier:
    """LightGBM con parada temprana por AUPRC de **validación** (si se pasa)."""
    params = {**PARAMS_LGBM, **extra}
    modelo = lgb.LGBMClassifier(n_estimators=n_estimadores, scale_pos_weight=peso_positivos, random_state=semilla, **params)
    if X_val is not None:
        modelo.fit(X, y, eval_set=[(X_val, y_val)], eval_metric="average_precision",
                   callbacks=[lgb.early_stopping(200, first_metric_only=True, verbose=False)])
    else:
        modelo.fit(X, y)
    return modelo


def puntaje_sup(modelo, X) -> np.ndarray:
    return modelo.predict_proba(X)[:, 1]


# ---------------------------------------------------------------- híbrido
def puntajes_anomalia_oof(X: np.ndarray, y: np.ndarray, n_pliegues: int = 3, semilla: int = SEMILLA, epocas_ae: int = 25):
    """Puntajes de IF y AE *fuera de pliegue* para el entrenamiento del híbrido.

    Si el autoencoder se entrena con las normales de entrenamiento y luego se usan sus errores como
    variable en esas mismas filas, las normales salen con error artificialmente bajo (ya las vio) y los
    fraudes no: el supervisado aprendería una separación que no existe en datos nuevos. Por eso cada
    normal se puntúa con un modelo que no la vio. Los fraudes nunca entran al ajuste de los detectores.
    """
    oof_if = np.zeros(len(X))
    oof_ae = np.zeros(len(X))
    normales = np.flatnonzero(y == 0)
    fraudes = np.flatnonzero(y == 1)
    acum_if_fr = np.zeros(len(fraudes))
    acum_ae_fr = np.zeros(len(fraudes))
    kf = KFold(n_splits=n_pliegues, shuffle=True, random_state=semilla)
    for k, (tr, te) in enumerate(kf.split(normales)):
        Xn = X[normales[tr]]
        mif = entrenar_isolation_forest(Xn, semilla + k)
        mae = entrenar_autoencoder(Xn, epocas=epocas_ae, semilla=semilla + k, verbose=False)
        oof_if[normales[te]] = puntaje_isolation_forest(mif, X[normales[te]])
        oof_ae[normales[te]] = mae.puntaje(X[normales[te]])
        acum_if_fr += puntaje_isolation_forest(mif, X[fraudes]) / n_pliegues
        acum_ae_fr += mae.puntaje(X[fraudes]) / n_pliegues
        print(f"  [OOF] pliegue {k + 1}/{n_pliegues} listo")
    oof_if[fraudes] = acum_if_fr
    oof_ae[fraudes] = acum_ae_fr
    return oof_if, oof_ae


def con_anomalias(X: np.ndarray, p_if: np.ndarray, p_ae: np.ndarray) -> np.ndarray:
    """Añade los puntajes de anomalía como dos columnas más (log del error del AE: cola muy larga)."""
    return np.column_stack([X, p_if, np.log(p_ae + 1e-9)])
