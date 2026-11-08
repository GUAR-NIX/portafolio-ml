"""App Streamlit: subir un parche satelital y ver la clase, las probabilidades y el Grad-CAM.

    streamlit run app/app.py --server.port 8616

La clasificación usa el mismo motor que la API (eurosat_suelo.predecir, ONNX Runtime + joblib).
El Grad-CAM se calcula sobre la red fine-tuned en PyTorch (models/mobilenetv3_eurosat_finetuned.pt)
para la clase que predice el ensamble. No necesita los datos crudos.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from eurosat_suelo.config import ARCHIVO_PT, CLASES_EUROSAT, DIR_MODELOS, DIR_MUESTRA, NOMBRES_ESPANOL  # noqa: E402
from eurosat_suelo.predecir import (  # noqa: E402
    ArtefactoNoEncontrado,
    decodificar_entrada_a_rgb64,
    predecir_parche_satelital,
)

MUESTRA_INICIAL = "06_PermanentCrop_05.png"  # confianza 0,79: muestra la derivación a fotointérprete

st.set_page_config(page_title="Uso de suelo · EuroSAT", page_icon="🛰️", layout="wide")


@st.cache_resource
def red_grad_cam():
    from eurosat_suelo.modelos import cargar_red_finetuned

    ruta = DIR_MODELOS / ARCHIVO_PT
    if not ruta.exists():
        raise ArtefactoNoEncontrado(f"Falta models/{ARCHIVO_PT}: el Grad-CAM necesita los pesos PyTorch.")
    return cargar_red_finetuned(ruta)


def figura_grad_cam(img: np.ndarray, mapa: np.ndarray) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.3))
    axes[0].imshow(img)
    axes[0].set_title("Parche (64×64 px)", fontsize=9)
    axes[1].imshow(img)
    axes[1].imshow(mapa, cmap="jet", alpha=0.5, vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM (modelo 4)", fontsize=9)
    for a in axes:
        a.axis("off")
    fig.tight_layout()
    return fig


st.title("🛰️ Clasificación de uso de suelo (Sentinel-2 · EuroSAT)")
st.caption(
    "Ensamble MobileNetV3-Small fine-tuned + fusión espectral, calibrado con temperatura en validación. "
    "Funciona con parches RGB de Sentinel-2 a 10 m/píxel; otras imágenes se reescalan a 64×64 y el resultado "
    "no es fiable fuera de ese dominio."
)

muestras = sorted(p.name for p in DIR_MUESTRA.glob("*.png"))
with st.sidebar:
    st.header("Imagen")
    subida = st.file_uploader("Sube un parche PNG/JPG", type=["png", "jpg", "jpeg", "tif", "tiff"])
    elegida = st.selectbox("…o elige una muestra de prueba", muestras, index=muestras.index(MUESTRA_INICIAL) if MUESTRA_INICIAL in muestras else 0)
    propio = st.checkbox("Cambiar el umbral τ elegido en validación", value=False)
    umbral = st.slider("Umbral de confianza τ (debajo → fotointérprete)", 0.0, 0.95, value=0.5, step=0.05) if propio else None

entrada = subida.getvalue() if subida is not None else elegida
origen = subida.name if subida is not None else f"muestra {elegida} (partición de prueba)"

try:
    img = decodificar_entrada_a_rgb64(entrada)
    res = predecir_parche_satelital(img, umbral_confianza=umbral)
except ArtefactoNoEncontrado as exc:
    st.error(str(exc))
    st.stop()
except (ValueError, OSError) as exc:
    st.error(f"No se pudo leer la imagen: {exc}")
    st.stop()

st.subheader(f"Imagen: {origen}")
c1, c2, c3 = st.columns(3)
c1.metric("Clase predicha", res["nombre_es"], res["clase_predicha"])
c2.metric("Confianza calibrada", f"{res['confianza'] * 100:.1f} %".replace(".", ","))
auto = res["decision_operativa"] == "CLASIFICACION_AUTOMATICA"
c3.metric("Decisión", "Automática" if auto else "Revisión humana", f"τ = {res['umbral_aplicado']:.2f}".replace(".", ","),
          delta_color="normal" if auto else "inverse")
if res["motivo_triaje"]:
    st.warning(res["motivo_triaje"])

izq, der = st.columns([1.0, 1.2])
with izq:
    probs = pd.DataFrame(
        {"clase": [NOMBRES_ESPANOL[c] for c in CLASES_EUROSAT], "probabilidad": [res["probabilidades"][c] for c in CLASES_EUROSAT]}
    ).sort_values("probabilidad", ascending=False)
    st.markdown("**Probabilidades por clase**")
    st.bar_chart(probs, x="clase", y="probabilidad", horizontal=True, sort="-probabilidad", height=330)
with der:
    st.markdown("**Dónde mira la red**")
    try:
        from eurosat_suelo.modelos import grad_cam

        gc = grad_cam(red_grad_cam(), img, clase_objetivo=CLASES_EUROSAT.index(res["clase_predicha"]))
        st.pyplot(figura_grad_cam(img, gc["mapa"]), clear_figure=True)
        st.caption("Grad-CAM sobre el bloque features[8] (rejilla 6×6) para la clase predicha por el ensamble.")
    except ArtefactoNoEncontrado as exc:
        st.info(str(exc))

with st.expander("Telemetría espectral y respuesta completa (igual que la API)"):
    st.json(res)
