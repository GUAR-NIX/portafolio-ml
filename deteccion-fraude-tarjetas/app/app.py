"""App Streamlit: puntúa transacciones de la muestra de prueba y muestra cómo cambia la decisión con los costos.

Arranque:  streamlit run app/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from fraude.predecir import Puntuador  # noqa: E402

st.set_page_config(page_title="Fraude con tarjetas", page_icon="💳", layout="wide")


@st.cache_resource
def cargar_puntuador() -> Puntuador:
    return Puntuador.cargar(RAIZ / "models")


@st.cache_data
def cargar_muestra() -> pd.DataFrame:
    return pd.read_csv(RAIZ / "data" / "muestra" / "transacciones_prueba_muestra.csv")


try:
    p = cargar_puntuador()
except FileNotFoundError:
    st.error("Faltan los artefactos de models/. Ejecuta `python -m fraude.entrenar`.")
    st.stop()
muestra = cargar_muestra()
res = json.loads((RAIZ / "reports" / "resultados.json").read_text(encoding="utf-8"))

st.title("💳 Detección de fraude: ¿revisar o dejar pasar?")
st.caption(f"Modelo servido: **{p.meta['nombre_servido']}** · umbral elegido en validación · "
           f"muestra de la partición de prueba ({len(muestra)} transacciones, {int(muestra.Class.sum())} fraudes reales).")

with st.sidebar:
    st.header("Supuestos de costo")
    cr = st.slider("Costo de revisar una alerta (EUR)", 0.5, 50.0, float(p.meta["costo_revision_eur"]), 0.5)
    fp = st.slider("Fracción del monto que se pierde si no se detecta", 0.1, 1.0, float(p.meta["fraccion_perdida"]), 0.05)
    solo_fraudes = st.checkbox("Mostrar solo fraudes reales", value=False)

pred = pd.DataFrame(p.puntuar(muestra.to_dict("records"), cr, fp))
tabla = pd.concat([muestra[["Time", "Amount", "Class"]].reset_index(drop=True), pred], axis=1)
tabla["hora"] = (tabla["Time"] / 3600).round(2)

c1, c2, c3, c4 = st.columns(4)
for col, politica, nombre in [(c1, "alerta_umbral_costo", "Umbral por costo"), (c2, "alerta_perdida_esperada", "Pérdida esperada")]:
    al = tabla[politica]
    detect = tabla.loc[al & (tabla.Class == 1), "Amount"].sum()
    perdido = tabla.loc[~al & (tabla.Class == 1), "Amount"].sum() * fp
    col.metric(nombre, f"{int(al.sum())} alertas", f"{int((al & (tabla.Class == 1)).sum())} fraudes detectados")
    col.caption(f"Costo en la muestra: {cr * al.sum() + perdido:,.0f} EUR (monto protegido {detect:,.0f} EUR)")
c3.metric("AUPRC en prueba", f"{next(m['auprc'] for m in res['metricas_prueba'] if m['modelo'] == res['modelo_servido']):.3f}")
c4.metric("Sin modelo (se pierde todo)", f"{tabla.loc[tabla.Class == 1, 'Amount'].sum() * fp:,.0f} EUR")

st.subheader("Transacciones puntuadas")
vista = tabla[tabla.Class == 1] if solo_fraudes else tabla
st.dataframe(
    vista[["hora", "Amount", "Class", "puntaje_modelo", "probabilidad_fraude_calibrada", "perdida_esperada_eur",
           "alerta_umbral_costo", "alerta_perdida_esperada", "percentil_anomalia_ae"]]
    .sort_values("puntaje_modelo", ascending=False),
    width="stretch", hide_index=True, height=420,
)
st.info("La regla de pérdida esperada deja pasar fraudes de pocos euros (revisarlos cuesta más de lo que se perdería) "
        "y revisa transacciones grandes aunque su probabilidad sea moderada. Mueve el costo de revisión para verlo.")
