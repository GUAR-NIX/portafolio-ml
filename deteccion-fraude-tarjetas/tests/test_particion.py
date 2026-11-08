"""Partición temporal sin fuga: rangos que no se solapan, sin barajar y con empates de Time juntos."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraude.datos import particion_temporal, preparar


def _sintetico(n: int = 1000, semilla: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)
    t = np.sort(rng.integers(0, 172_800, n)).astype(float)
    df = pd.DataFrame({"Time": t, "Amount": rng.exponential(50, n), "Class": (rng.random(n) < 0.01).astype(int)})
    for i in range(1, 29):
        df[f"V{i}"] = rng.normal(size=n)
    return preparar(df)


def test_rangos_no_se_solapan_y_cubren_todo() -> None:
    df = _sintetico()
    p = particion_temporal(df)
    assert p.entrenamiento["Time"].max() < p.validacion["Time"].min()
    assert p.validacion["Time"].max() < p.prueba["Time"].min()
    assert len(p.entrenamiento) + len(p.validacion) + len(p.prueba) == len(df)


def test_empates_de_time_quedan_en_la_misma_particion() -> None:
    df = _sintetico()
    t_corte = float(df["Time"].quantile(0.6))
    extra = df.iloc[[0, 1, 2]].copy()
    extra["Time"] = t_corte  # tres transacciones exactamente en el segundo del corte
    df = pd.concat([df, extra], ignore_index=True)
    p = particion_temporal(df)
    partes = [p.entrenamiento, p.validacion, p.prueba]
    for i, d in enumerate(partes):
        otros = pd.concat([x for j, x in enumerate(partes) if j != i])
        assert not set(d["Time"]) & set(otros["Time"]), "Un segundo de Time cruza particiones"


def test_no_baraja_aunque_la_entrada_este_desordenada() -> None:
    df = _sintetico().sample(frac=1.0, random_state=1)
    p = particion_temporal(df)
    for d in (p.entrenamiento, p.validacion, p.prueba):
        assert d["Time"].is_monotonic_increasing


def test_predicciones_guardadas_respetan_el_orden_temporal(pred_validacion, pred_prueba, resultados) -> None:
    assert pred_validacion["Time"].max() < pred_prueba["Time"].min()
    part = {r["particion"]: r for r in resultados["particion"]}
    assert len(pred_prueba) == part["prueba"]["transacciones"]
    assert int(pred_prueba["Class"].sum()) == part["prueba"]["fraudes"]
    assert len(pred_validacion) == part["validacion"]["transacciones"]
    assert part["entrenamiento"]["hasta_h"] <= part["validacion"]["desde_h"]


def test_particion_real_sin_fuga(datos_completos) -> None:
    p = particion_temporal(datos_completos)
    assert p.entrenamiento["Time"].max() < p.validacion["Time"].min()
    assert p.validacion["Time"].max() < p.prueba["Time"].min()
    # los duplicados exactos nunca cruzan particiones porque comparten Time
    cols = [c for c in datos_completos.columns if c not in ("dia", "hora", "log_monto")]
    claves = [set(map(tuple, d[cols].to_numpy())) for d in (p.entrenamiento, p.validacion, p.prueba)]
    assert not (claves[0] & claves[1]) and not (claves[1] & claves[2]) and not (claves[0] & claves[2])
