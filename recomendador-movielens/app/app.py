"""App Streamlit del recomendador.

    streamlit run app/app.py --server.port 8611
"""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from recomendador import config as C
from recomendador.demo import PaqueteIncompleto
from recomendador.servicio import GENEROS_ES, cargar

st.set_page_config(page_title="Recomendador MovieLens", page_icon="🎬", layout="wide")


@st.cache_resource
def recomendador():
    return cargar()


try:
    r = recomendador()
except PaqueteIncompleto as e:
    st.error(str(e))
    st.stop()
METODOS = {"Híbrido": "hibrido", "ALS (colaborativo)": "als", "Contenido": "contenido", "Populares ahora": "popular"}


def tabla(recs: list[dict], porque: bool = False) -> pd.DataFrame:
    df = pd.DataFrame(recs)
    if df.empty:
        return df
    df["generos"] = df["generos"].map(lambda g: ", ".join(GENEROS_ES.get(x, x) for x in g))
    cols = ["titulo", "anio", "generos", "popularidad", "puntaje"] + (["porque_te_gusto"] if porque and "porque_te_gusto" in df else [])
    nombres = {"titulo": "Película", "anio": "Año", "generos": "Géneros", "popularidad": "Positivos (popularidad)",
               "puntaje": "Puntaje", "porque_te_gusto": "Porque te gustó…"}
    df = df[cols].rename(columns=nombres)
    df.index = range(1, len(df) + 1)
    return df


def selector_peliculas(etiqueta: str, clave: str, por_defecto: list[int] | None = None) -> list[int]:
    texto = st.text_input(f"Buscar {etiqueta}", key=f"buscar_{clave}", placeholder="p. ej. matrix, toy story, amélie")
    opciones = {}
    for m in (por_defecto or []):
        if m in r.idx:
            opciones[m] = r.peliculas.iloc[r.idx[m]]["titulo"]
    if texto:
        for x in r.buscar(texto, 15):
            opciones[x["movieId"]] = x["titulo"]
    elegidas = st.session_state.get(f"elegidas_{clave}", por_defecto or [])
    for m in elegidas:
        opciones.setdefault(m, r.peliculas.iloc[r.idx[m]]["titulo"])
    return st.multiselect(etiqueta.capitalize(), list(opciones), default=[m for m in elegidas if m in opciones],
                          format_func=lambda m: opciones[m], key=f"elegidas_{clave}")


st.title("🎬 Recomendador de películas · MovieLens 25M")
st.caption("ALS implícito + contenido (géneros y Tag Genome) + popularidad reciente · arranque en frío por géneros o con pocas películas")

pest = st.tabs(["👤 Usuario existente", "🆕 Usuario nuevo", "🔗 Películas similares", "📊 Resultados"])

with pest[0]:
    c1, c2, c3 = st.columns([2, 2, 1])
    usuarios = sorted(r.demo_usuarios)
    u = c1.selectbox("Usuario (paquete de demostración)", usuarios, index=min(7, len(usuarios) - 1))
    metodo = c2.radio("Método", list(METODOS), horizontal=True, key="m_usuario")
    n = c3.number_input("Cuántas", 5, 50, 10, key="n_usuario")
    res = r.recomendar_usuario(int(u), int(n), METODOS[metodo])
    st.markdown(f"**{res['positivos_en_historial']} películas le gustaron** (≥ 4★) · segmento *{res['segmento']}* · "
                f"en el híbrido el ALS pesa **{res['peso_als']:.0%}** y el contenido {1 - res['peso_als']:.0%}")
    st.dataframe(tabla(res["recomendaciones"], porque=True), width="stretch")
    with st.expander("Algunas películas que le gustaron"):
        fila = r.demo_usuarios[int(u)]
        ids = r.demo_positivos[fila].indices[:30]
        st.write(", ".join(r.peliculas.iloc[ids]["titulo"].tolist()))

with pest[1]:
    st.subheader("Opción A · solo géneros (registro)")
    elegidos = st.multiselect("¿Qué géneros te gustan?", list(GENEROS_ES), default=["Sci-Fi", "Animation"],
                              format_func=lambda g: GENEROS_ES[g])
    if elegidos:
        st.dataframe(tabla(r.usuario_nuevo(elegidos, 10)["recomendaciones"]), width="stretch")
    st.subheader("Opción B · algunas películas que te gustaron")
    gustos = selector_peliculas("películas que te gustaron", "nuevo", por_defecto=[1, 2571, 4993])
    metodo_b = st.radio("Método", list(METODOS), horizontal=True, key="m_nuevo")
    if gustos:
        res = r.recomendar_historial(gustos, 10, METODOS[metodo_b])
        st.markdown(f"Con {res['positivos_en_historial']} películas el ALS pesa **{res['peso_als']:.0%}** en el híbrido "
                    "(sus factores se calculan al vuelo, sin re-entrenar).")
        st.dataframe(tabla(res["recomendaciones"], porque=True), width="stretch")

with pest[2]:
    c1, c2 = st.columns([3, 2])
    with c1:
        texto = st.text_input("Busca una película", value="toy story")
        encontradas = r.buscar(texto, 15) if texto else []
    if encontradas:
        peli = c1.selectbox("Película", [x["movieId"] for x in encontradas],
                            format_func=lambda m: next(x["titulo"] for x in encontradas if x["movieId"] == m))
        metodo_s = c2.radio("Similitud por", ["Híbrido", "ALS (colaborativo)", "Contenido"], horizontal=True)
        res = r.similares(int(peli), 10, {"Híbrido": "hibrido", "ALS (colaborativo)": "als", "Contenido": "contenido"}[metodo_s])
        st.dataframe(tabla(res["similares"]), width="stretch")
        st.caption("Contenido = mismo «ADN» de etiquetas (Tag Genome) y géneros · ALS = las ven los mismos usuarios")
    else:
        st.info("No encontré películas con ese texto en el catálogo recomendable.")

with pest[3]:
    meta = r.metadata
    st.markdown(f"Prueba: entrenar con todo lo anterior a **{meta['corte_prueba']}** y evaluar lo que cada usuario "
                "vio después (hasta 2019-11). Relevante = calificó ≥ 4★. Top-10.")
    m = pd.DataFrame(meta["metricas_prueba"])
    cols = ["modelo", "segmento", "usuarios", "ndcg@10", "precision@10", "recall@10", "map@10", "cobertura_pct",
            "novedad_bits", "diversidad_generos", "percentil_pop_medio"]
    st.dataframe(m[cols], width="stretch", hide_index=True)
    for f in sorted(C.FIGURAS.glob("*.png")):
        st.image(str(f), width="stretch")
    st.caption("Hiperparámetros: " + json.dumps(meta["hiperparametros"], ensure_ascii=False))
