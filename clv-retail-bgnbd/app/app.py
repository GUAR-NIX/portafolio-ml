"""App Streamlit: buscar un cliente y ver su CLV, P(vivo), segmento y acción recomendada.

    streamlit run app/app.py --server.port 8615

Reutiliza el modelo guardado (models/modelo_clv.joblib) a través de clv.predecir y las tablas de reports/.
No necesita los datos crudos.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from clv import config, predecir  # noqa: E402

CLIENTE_EJEMPLO = "13093"


def num(v: float, dec: int = 0) -> str:
    """Formato numérico español: 12.345,6"""
    return f"{v:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")


@st.cache_data
def cargar_cartera() -> pd.DataFrame:
    return pd.read_csv(config.REPORTES / "cartera_clientes.csv", dtype={"customer_id": str})


@st.cache_data
def cargar_tabla(nombre: str) -> pd.DataFrame:
    return pd.read_csv(config.REPORTES / nombre)


def buscar_cliente(cartera: pd.DataFrame, id_cliente: str) -> pd.Series | None:
    fila = cartera[cartera["customer_id"] == str(id_cliente).strip()]
    return None if fila.empty else fila.iloc[0]


def ficha_cliente(fila: pd.Series) -> dict:
    """Valora al cliente con el motor de la API (clv.predecir) a partir de su historia hasta el corte."""
    return predecir.valorar_cliente(
        frecuencia=int(fila["frecuencia"]),
        recencia_dias=float(fila["recencia_dias"]),
        edad_dias=float(fila["edad_dias"]),
        ticket_medio=float(fila["ticket_medio_acotado"]),
    )


def mostrar_valoracion(v: dict) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CLV neto a 6 meses", f"{num(v['clv_neto_gbp'], 2)} GBP")
    c2.metric("P(vivo) al corte", f"{num(v['prob_vivo'] * 100, 1)} %")
    c3.metric("P(recomprar en 6 meses)", f"{num(v['prob_recompra'] * 100, 1)} %")
    c4.metric("Compras esperadas (días)", num(v["compras_esperadas"], 2))
    c5, c6, c7 = st.columns([1, 1, 2])
    c5.metric("Ticket esperado (shrinkage)", f"{num(v['ticket_esperado_gbp'], 2)} GBP")
    c6.metric("Ingreso esperado 6 m", f"{num(v['ingreso_esperado_gbp'], 2)} GBP")
    c7.metric("Segmento", v["segmento"])
    st.info(f"**Acción recomendada:** {v['accion_recomendada']}")
    if v["conviene_contactar_rescate"]:
        st.warning(
            f"Entra en la campaña de rescate (política 4). Beneficio esperado: {num(v['beneficio_neto_rescate_gbp'], 2)} GBP "
            f"**bajo el supuesto** de {num(config.EFECTIVIDAD_RESCATE * 100)} % de efectividad, que no está validado con un experimento."
        )


def main() -> None:
    st.set_page_config(page_title="CLV retail", page_icon="💎", layout="wide")
    cartera = cargar_cartera()
    meta = predecir.cargar_metadata()

    with st.sidebar:
        st.header("💎 CLV retail")
        st.caption("BG/NBD + Gamma-Gamma sobre UCI Online Retail II")
        st.markdown(
            f"- Corte: **{config.FECHA_CORTE_CAL}** ({num(len(cartera))} clientes)\n"
            f"- Prueba: 2011-06-10 → {config.FECHA_FIN} (6 meses)\n"
            f"- Margen bruto: {num(config.MARGEN_BRUTO * 100)} % · descuento {num(config.TASA_DESCUENTO_MENSUAL * 100)} % mensual\n"
            f"- Costo por contacto: {num(config.COSTO_CONTACTO_RETENCION, 2)} GBP"
        )
        p = meta.get("parametros_bgnbd", {})
        st.caption(f"BG/NBD: r={p.get('r')}, α={p.get('alpha_semanas')}, a={p.get('a')}, b={p.get('b')}")

    st.title("¿Quién sigue vivo y cuánto vale?")
    tab_buscar, tab_simular, tab_cartera = st.tabs(["🔎 Buscar cliente", "🧪 Simular cliente", "📊 Cartera y políticas"])

    with tab_buscar:
        col_id, col_seg = st.columns([1, 2])
        id_cliente = col_id.text_input("ID de cliente (Customer ID)", value=CLIENTE_EJEMPLO)
        ejemplos = cartera.groupby("segmento")["customer_id"].first()
        col_seg.caption("Ejemplos por segmento: " + " · ".join(f"{s}: `{c}`" for s, c in ejemplos.items()))
        fila = buscar_cliente(cartera, id_cliente)
        if fila is None:
            st.error(f"No existe el cliente {id_cliente!r} entre los {num(len(cartera))} clientes con compras antes del corte.")
        else:
            st.subheader(f"Cliente {fila['customer_id']} · {fila['pais']}")
            st.caption(
                f"Historia hasta el corte: {int(fila['frecuencia'])} compras repetidas · primera compra hace "
                f"{num(fila['edad_dias'])} días · última hace {num(fila['inactividad_dias'])} días · ticket medio "
                f"{num(fila['ticket_medio_acotado'], 2)} GBP · gasto total {num(fila['gasto_total_cal_acotado'], 2)} GBP"
            )
            mostrar_valoracion(ficha_cliente(fila))
            with st.expander("¿Qué pasó realmente en los 6 meses de prueba?"):
                st.write(
                    f"Compró en **{int(fila['compras_holdout'])}** días distintos y gastó "
                    f"**{num(fila['gasto_holdout_acotado'], 2)} GBP** (ticket acotado a 3.000 GBP). "
                    "El modelo no vio estos datos: sirven para juzgar la predicción."
                )

    with tab_simular:
        st.caption("Valora un cliente hipotético con el mismo motor que la API (`POST /valorar`).")
        c1, c2, c3, c4 = st.columns(4)
        edad = c3.slider("Días desde la primera compra (T)", 1, 555, 490)
        frecuencia = c1.slider("Compras repetidas (x)", 0, 60, 6)
        recencia = c2.slider("Días entre primera y última compra (t_x)", 0, edad, min(240, edad)) if frecuencia > 0 else 0
        ticket = c4.number_input("Ticket medio (GBP)", 1.0, 40_000.0, 520.0, step=10.0)
        mostrar_valoracion(predecir.valorar_cliente(frecuencia, float(recencia), float(edad), ticket))

    with tab_cartera:
        seg = cargar_tabla("resumen_segmentos.csv")
        st.subheader("Segmentos")
        st.dataframe(
            seg[["segmento", "n_clientes", "prob_vivo_media", "tasa_recompra_real", "clv_neto_medio_gbp",
                 "ingreso_real_holdout_gbp", "pct_ingreso_real", "accion_recomendada"]],
            hide_index=True, use_container_width=True,
        )
        st.subheader("Políticas de rescate validadas con el gasto real del periodo de prueba")
        pol = cargar_tabla("politicas_retencion.csv")
        st.dataframe(
            pol[["politica", "clientes_contactados", "costo_campana_gbp", "tasa_recompra_real_pct",
                 "margen_real_por_contacto_gbp", "tasa_rescate_equilibrio_pct",
                 "tasa_rescate_equilibrio_ic95_inf", "tasa_rescate_equilibrio_ic95_sup"]].round(2),
            hide_index=True, use_container_width=True,
        )
        st.caption(
            "Tasa de rescate de equilibrio: % de los contactados que no volvieron que habría que recuperar "
            "(a su ritmo de gasto histórico) para pagar la campaña. Sin experimento A/B no se mide el efecto causal."
        )
        rescate = cartera[cartera["conviene_contactar_rescate"]].sort_values("beneficio_neto_rescate_gbp", ascending=False)
        st.download_button(
            f"Descargar lista de rescate ({len(rescate)} clientes, CSV)",
            rescate.to_csv(index=False).encode("utf-8"),
            file_name="lista_rescate.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
