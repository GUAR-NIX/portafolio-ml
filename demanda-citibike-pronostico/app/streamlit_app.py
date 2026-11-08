"""App Streamlit: pronóstico por estación y día, mapa de estaciones y análisis de errores."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st

from demanda_citibike.clima import cargar_clima
from demanda_citibike.config import MODELOS, PROCESADOS, REPORTES
from demanda_citibike.features import FEATURES_1H, FEATURES_DIA, construir_caracteristicas
from demanda_citibike.modelos import cargar_modelo

st.set_page_config(page_title="Pronóstico Citi Bike", page_icon="🚲", layout="wide")


@st.cache_resource(show_spinner=False)
def cargar():
    modelo = cargar_modelo(MODELOS / "modelo_lgbm.joblib")
    ruta_1h = MODELOS / "modelo_1h.joblib"
    modelo_1h = cargar_modelo(ruta_1h) if ruta_1h.exists() else None
    demanda = pd.read_parquet(PROCESADOS / "demanda_horaria.parquet")
    features = construir_caracteristicas(demanda, cargar_clima())
    meta = pd.read_csv(PROCESADOS / "estaciones.csv", dtype={"estacion": str})
    return modelo, modelo_1h, features, meta


@st.cache_data(show_spinner=False)
def cargar_reportes():
    est = pd.read_csv(REPORTES / "error_por_estacion.csv", dtype={"estacion": str})
    dia = pd.read_csv(REPORTES / "error_por_dia.csv", parse_dates=["fecha"])
    return est, dia


def tab_pronostico(modelo, modelo_1h, features, nombres) -> None:
    estaciones = sorted(features["estacion"].astype(str).unique())
    c1, c2 = st.columns(2)
    estacion = c1.selectbox("Estación", estaciones, format_func=lambda x: nombres.get(x, x))
    sub_est = features[features["estacion"].astype(str) == estacion]
    fechas = sorted(sub_est["fecha_hora"].dt.date.unique())
    fecha = c2.selectbox("Fecha", fechas, index=len(fechas) - 1)

    sub = sub_est[sub_est["fecha_hora"].dt.date == fecha].sort_values("fecha_hora")
    d = pd.DataFrame({
        "hora": sub["fecha_hora"].dt.hour.to_numpy(),
        "salidas reales": sub["salidas"].to_numpy(),
        "día siguiente (principal)": np.clip(modelo.predict(sub[FEATURES_DIA]), 0, None),
        "base estacional (t-168)": sub["lag_168"].to_numpy(),
    })
    if modelo_1h is not None:
        d["1 hora (comparación)"] = np.clip(modelo_1h.predict(sub[FEATURES_1H]), 0, None)
    series = [c for c in d.columns if c != "hora"]
    fig = px.line(
        d, x="hora", y=series, markers=True,
        color_discrete_map={
            "salidas reales": "#333333", "día siguiente (principal)": "#2a78d6",
            "base estacional (t-168)": "#eb6834", "1 hora (comparación)": "#1baf7a",
        },
        labels={"value": "Salidas", "variable": "Serie", "hora": "Hora del día"},
        title=f"Pronóstico horario — {nombres.get(estacion, estacion)} · {fecha}",
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "El modelo **día siguiente** solo usa información disponible a la medianoche anterior "
        "(rezagos ≥ 24 h, calendario y clima); el de **1 hora** usa la hora anterior y no "
        "serviría para planificar el rebalanceo nocturno."
    )
    m1, m2 = st.columns(2)
    m1.metric("Salidas reales del día", int(d["salidas reales"].sum()))
    m2.metric("Pronóstico día siguiente", int(round(d["día siguiente (principal)"].sum())))
    st.dataframe(d.round(1), width="stretch", hide_index=True)


def tab_mapa(err_est: pd.DataFrame) -> None:
    st.markdown("Error del modelo **día siguiente** por estación en el test final (28 días: del 3 de diciembre a las 23:00 al 31 de diciembre de 2024). "
                "Tamaño = salidas reales; color = WAPE.")
    d = err_est.dropna(subset=["latitud", "longitud", "WAPE_pct"]).copy()
    # Color: de azul claro (WAPE bajo) a azul oscuro (WAPE alto), según los percentiles 5–95
    bajo, alto = d["WAPE_pct"].quantile([0.05, 0.95])
    t = ((d["WAPE_pct"] - bajo) / max(alto - bajo, 1e-9)).clip(0, 1)
    d["color"] = [[int(198 - 190 * v), int(219 - 171 * v), int(239 - 132 * v), 200] for v in t]
    d["radio"] = 25 + 90 * np.sqrt(d["salidas"] / d["salidas"].max())
    for col in ["WAPE_pct", "WAPE_base_pct", "sesgo_pct"]:
        d[col] = d[col].round(1)
    capa = pdk.Layer(
        "ScatterplotLayer", data=d[["longitud", "latitud", "color", "radio", "nombre", "salidas",
                                    "WAPE_pct", "WAPE_base_pct", "sesgo_pct"]],
        get_position="[longitud, latitud]", get_fill_color="color", get_radius="radio",
        pickable=True, stroked=True, get_line_color=[40, 40, 40, 120], line_width_min_pixels=0.5,
    )
    vista = pdk.ViewState(latitude=float(d["latitud"].mean()), longitude=float(d["longitud"].mean()), zoom=11.3)
    st.pydeck_chart(pdk.Deck(
        layers=[capa], initial_view_state=vista, map_style=pdk.map_styles.CARTO_LIGHT, height=620,
        tooltip={"text": "{nombre}\nWAPE {WAPE_pct} % · base {WAPE_base_pct} %\nSalidas {salidas} · sesgo {sesgo_pct} %"},
    ))
    st.caption(f"Color: WAPE de {bajo:.0f} % (claro) a {alto:.0f} % (oscuro). Mapa base © OpenStreetMap contributors © CARTO.")


def tab_errores(err_est: pd.DataFrame, err_dia: pd.DataFrame) -> None:
    sin_salidas = err_est[err_est["WAPE_pct"].isna()]
    if len(sin_salidas):
        st.info(f"{len(sin_salidas)} estaciones no tuvieron salidas en el mes de test (cerradas o con ID cambiado): "
                + ", ".join(sin_salidas["nombre"].astype(str)) + ". Su WAPE no está definido.")
    err_est = err_est.dropna(subset=["WAPE_pct"])
    c1, c2 = st.columns(2)
    c1.subheader("Estaciones con más error")
    c1.dataframe(err_est[["nombre", "salidas", "WAPE_pct", "WAPE_base_pct", "sesgo_pct"]].head(15),
                 width="stretch", hide_index=True)
    c2.subheader("Estaciones con menos error")
    c2.dataframe(err_est[["nombre", "salidas", "WAPE_pct", "WAPE_base_pct", "sesgo_pct"]].tail(15).iloc[::-1],
                 width="stretch", hide_index=True)

    fig = px.scatter(err_est, x="salidas", y="WAPE_pct", hover_name="nombre", log_x=True,
                     labels={"salidas": "Salidas en el mes de test (escala log)", "WAPE_pct": "WAPE %"},
                     title="¿Las estaciones con menos volumen tienen más error relativo?")
    fig.update_traces(marker=dict(color="#2a78d6"))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Error por día: feriados y lluvia")
    fig2 = px.bar(err_dia, x="fecha", y="WAPE_pred_dia", color=err_dia["es_feriado"].map({0: "Normal", 1: "Feriado"}),
                  hover_data=["lluvia_mm_dia", "temperatura_media_c", "salidas", "WAPE_base_estacional"],
                  color_discrete_map={"Normal": "#2a78d6", "Feriado": "#eb6834"},
                  labels={"WAPE_pred_dia": "WAPE día siguiente %", "fecha": "Fecha", "color": "Tipo de día"})
    st.plotly_chart(fig2, width="stretch")


def main() -> None:
    st.title("🚲 Pronóstico de demanda de bicicletas por estación y hora")
    st.markdown(
        "Modelo global **LightGBM** entrenado con los viajes de Citi Bike de 2024 (250 estaciones más activas) "
        "y clima horario de Open-Meteo. Pronóstico con **horizonte de negocio**: cada noche, para el día siguiente."
    )
    try:
        modelo, modelo_1h, features, meta = cargar()
        err_est, err_dia = cargar_reportes()
    except Exception as e:  # noqa: BLE001
        st.error(f"Ejecuta primero el pipeline (`python -m demanda_citibike.pipeline`). Detalle: {e}")
        return

    nombres = dict(zip(meta["estacion"], meta["nombre"].fillna(meta["estacion"])))
    t1, t2, t3 = st.tabs(["📈 Pronóstico", "🗺️ Mapa de estaciones", "🔎 Análisis de errores"])
    with t1:
        tab_pronostico(modelo, modelo_1h, features, nombres)
    with t2:
        tab_mapa(err_est)
    with t3:
        tab_errores(err_est, err_dia)


if __name__ == "__main__":
    main()
