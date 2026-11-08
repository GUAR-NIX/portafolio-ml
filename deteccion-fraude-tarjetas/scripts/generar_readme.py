"""Genera README.md y MODEL_CARD.md a partir de reports/ (ninguna cifra se escribe a mano).

Uso:  python scripts/generar_readme.py
La prueba tests/test_readme_cifras.py comprueba que las cifras de ambos archivos salen de reports/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from fraude.config import ALERTAS_POR_DIA, ETIQUETAS  # noqa: E402

R = RAIZ / "reports"


# ---------------------------------------------------------------- formato es-EC (coma decimal, punto de miles)
def num(x: float, d: int = 3) -> str:
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pct(x: float, d: int = 1) -> str:
    """x ya en porcentaje → '57,6 %'."""
    return f"{num(x, d)} %"


def eur(x: float) -> str:
    return f"{num(x, 0)} EUR"


def ic(lo: float, hi: float, d: int = 3) -> str:
    return f"[{num(lo, d)}; {num(hi, d)}]"


def cargar() -> dict:
    res = json.loads((R / "resultados.json").read_text(encoding="utf-8"))
    return {
        "res": res,
        "met": pd.read_csv(R / "metricas_prueba.csv").set_index("modelo"),
        "costo": pd.read_csv(R / "costo_prueba.csv").set_index("modelo"),
        "dif": pd.read_csv(R / "bootstrap_diferencias.csv"),
        "sens": pd.read_csv(R / "sensibilidad_costo.csv"),
        "part": pd.read_csv(R / "particion_temporal.csv").set_index("particion"),
        "pe": pd.read_csv(R / "pocas_etiquetas_resumen.csv"),
        "shap": pd.read_csv(R / "shap_importancia.csv"),
        "der": pd.read_csv(R / "deriva_variables.csv"),
        "ent": res["entrenamiento"],
    }


def tabla_resultados(c: dict) -> str:
    met, costo, k = c["met"], c["costo"], c["res"]["capacidad"]["k_prueba"]
    filas = [
        "| Familia | Modelo | AUPRC [IC 95 %] | ROC-AUC | Precisión@" + str(k) + " | Recall a precisión ≥ 80 % | Costo en prueba [IC 95 %] | Ahorro vs. sin modelo |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for m in met.sort_values("auprc", ascending=False).index:
        r, cc = met.loc[m], costo.loc[m]
        nombre = f"**{ETIQUETAS[m]}** ⭐" if m == c["res"]["modelo_servido"] else ETIQUETAS[m]
        filas.append(
            f"| {r['familia']} | {nombre} | {num(r['auprc'])} {ic(r['auprc_ic_inf'], r['auprc_ic_sup'])} | {num(r['roc_auc'])} | "
            f"{pct(100 * r[f'precision_at_{k}'])} | {pct(100 * r['recall_a_precision_80'])} | "
            f"{eur(cc['costo_total_eur'])} {ic(cc['costo_ic_inf'], cc['costo_ic_sup'], 0)} | {pct(cc['ahorro_vs_sin_modelo_pct'])} |"
        )
    return "\n".join(filas)


def tabla_politicas(c: dict) -> str:
    costo, srv = c["costo"], c["res"]["modelo_servido"]
    filas = [
        "| Política (modelo servido) | Alertas | Fraudes detectados / 75 | Falsas alertas | Monto protegido | Costo total [IC 95 %] | Ahorro |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for m, nombre in [(srv, "Umbral por costo elegido en validación"), ("perdida_esperada", "Pérdida esperada: p calibrada × monto > 5 EUR"),
                      ("capacidad", f"Capacidad fija: top {c['res']['capacidad']['k_prueba']} ({ALERTAS_POR_DIA} alertas/día)")]:
        r = costo.loc[m]
        filas.append(f"| {nombre} | {int(r['alertas'])} | {int(r['fraudes_detectados'])} | {int(r['falsas_alertas'])} | "
                     f"{eur(r['monto_fraude_detectado_eur'])} | {eur(r['costo_total_eur'])} {ic(r['costo_ic_inf'], r['costo_ic_sup'], 0)} | "
                     f"{pct(r['ahorro_vs_sin_modelo_pct'])} |")
    return "\n".join(filas)


def tabla_sensibilidad(c: dict) -> str:
    s = c["sens"][c["sens"]["fraccion_perdida"] == 1.0]
    modelos = list(dict.fromkeys(s["modelo"]))
    nombres = {m: s[s.modelo == m]["nombre"].iloc[0] for m in modelos}
    filas = ["| Costo de revisar | " + " | ".join(nombres[m] for m in modelos) + " |", "| ---: |" + " ---: |" * len(modelos)]
    for cr, d in s.groupby("costo_revision_eur"):
        d = d.set_index("modelo")
        mejor = d["ahorro_vs_sin_modelo_pct"].idxmax()
        celdas = [(f"**{pct(d.loc[m, 'ahorro_vs_sin_modelo_pct'])}**" if m == mejor else pct(d.loc[m, "ahorro_vs_sin_modelo_pct"])) for m in modelos]
        filas.append(f"| {num(cr, 1)} EUR | " + " | ".join(celdas) + " |")
    return "\n".join(filas)


def tabla_pocas(c: dict) -> str:
    g = c["pe"].pivot(index="n_fraudes", columns="modelo", values="media")
    d = c["pe"].pivot(index="n_fraudes", columns="modelo", values="desv")
    cols = list(g.columns)
    filas = ["| Fraudes etiquetados | " + " | ".join(f"{m} (AUPRC media ± d.e.)" for m in cols) + " |", "| ---: |" + " ---: |" * len(cols)]
    for n in g.index:
        celdas = [num(g.loc[n, m]) + ("" if pd.isna(d.loc[n, m]) else f" ± {num(d.loc[n, m])}") for m in cols]
        filas.append(f"| {n} | " + " | ".join(celdas) + " |")
    return "\n".join(filas)


def diferencia(c: dict, a: str, b: str, metrica: str = "auprc") -> pd.Series:
    d = c["dif"]
    return d[(d.modelo_a == a) & (d.modelo_b == b) & (d.metrica == metrica)].iloc[0]


def _ejemplo_api(muestra: pd.DataFrame) -> str:
    """Solicitud y respuesta reales de la API para el primer fraude de la muestra."""
    from fraude.predecir import Puntuador

    fila = muestra[muestra.Class == 1].iloc[0]
    r = Puntuador.cargar(RAIZ / "models").puntuar([fila.to_dict()])[0]
    sol = f'{{"transacciones": [{{"V1": {fila.V1:.3f}, "V2": {fila.V2:.3f}, "...": "...", "V28": {fila.V28:.3f}, "Amount": {fila.Amount:.2f}}}]}}'
    resp = json.dumps({"resultados": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}]}, ensure_ascii=False, indent=1)
    return sol + "\n→ " + resp


def readme(c: dict) -> str:
    res, met, costo, part, ent = c["res"], c["met"], c["costo"], c["part"], c["ent"]
    srv, nosup, k = res["modelo_servido"], res["mejor_no_supervisado"], res["capacidad"]["k_prueba"]
    datos = res["datos"]
    ms, mn, ml = met.loc[srv], met.loc[nosup], met.loc["logistica"]
    d_srv_nosup = diferencia(c, srv, nosup)
    d_hib = diferencia(c, "hibrido", ent["esquema_desbalance_elegido"])
    d_pes_sub = diferencia(c, "lgbm_pesos", "lgbm_submuestreo")
    d_pes_sin = diferencia(c, "lgbm_pesos", "lgbm_sin_ajuste")
    d_log_nosup = diferencia(c, "logistica", nosup)
    cruces = res["pocas_etiquetas"]["n_fraudes_para_superar"]
    pe = c["pe"].set_index(["modelo", "n_fraudes"])
    sens1 = c["sens"][c["sens"]["fraccion_perdida"] == 1.0].set_index(["costo_revision_eur", "modelo"])
    cr_min, cr_max = sens1.index.get_level_values(0).min(), sens1.index.get_level_values(0).max()
    adv = res["deriva"]["adversaria"]
    der = c["der"]
    shap_top = c["shap"].sort_values("shap_medio_abs_fraudes", ascending=False).head(3)["variable"].tolist()
    sin_modelo = costo.loc[srv, "costo_sin_modelo_eur"]
    pr, va = part.loc["prueba"], part.loc["validacion"]
    tr = part.loc["entrenamiento"]
    tiempos = ent["tiempos"]
    por_hora = pd.read_csv(R / "transacciones_por_hora.csv")
    pred_te = pd.read_parquet(R / "predicciones_prueba.parquet", columns=["Amount", "Class"])
    max_fraude_te = float(pred_te.loc[pred_te.Class == 1, "Amount"].max())
    muestra = pd.read_csv(RAIZ / "data" / "muestra" / "transacciones_prueba_muestra.csv")
    medio_ic = (ms["auprc_ic_sup"] - ms["auprc_ic_inf"]) / 2
    ahorros = [costo.loc[m, "ahorro_vs_sin_modelo_pct"] for m in (srv, "perdida_esperada", "capacidad")]
    lgbm_pe = c["pe"][c["pe"].modelo == "LightGBM con pesos"].set_index("n_fraudes")["media"]
    caidas = [int(n) for n in lgbm_pe.index if n > (cruces["LightGBM con pesos"]["primer_n"] or 0) and lgbm_pe[n] <= res["pocas_etiquetas"]["auprc_referencia"]]
    ejemplo = _ejemplo_api(muestra)

    return f"""# 💳 Detección de fraude con tarjetas: anomalías no supervisadas frente a modelos supervisados

**Datos:** {num(datos['filas'], 0)} transacciones reales de titulares europeos (ULB / Worldline, dos días de septiembre de 2013), con **{datos['fraudes']} fraudes ({pct(datos['tasa_fraude_pct'], 2)})**.

¿Hace falta etiquetar fraudes o basta con buscar lo **raro**? Este proyecto compara, con una **partición temporal** (entrenar con el pasado y evaluar con el futuro):
- **Líneas base:** una regla por monto y una regresión logística.
- **No supervisados:** Isolation Forest (sin etiquetas y solo con normales), LOF en modo *novelty* y un **autoencoder** en PyTorch (CPU).
- **Supervisados:** LightGBM sin ajuste, con **pesos** y con **submuestreo**.
- **Híbrido:** LightGBM con los puntajes de anomalía como variables.

Las decisiones se toman con el **costo de cada error** usando el `Amount` real: el umbral se elige en validación y se reporta en prueba, con intervalos bootstrap y análisis de sensibilidad. Incluye una curva de **pocas etiquetas**, **SHAP**, **deriva** entre días, una API FastAPI (`/puntuar`) y una app Streamlit.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-supervisado-2a78d6)
![PyTorch CPU](https://img.shields.io/badge/PyTorch-autoencoder_CPU-EE4C2C?logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-IF_%C2%B7_LOF-F7931E?logo=scikitlearn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B?logo=streamlit&logoColor=white)

---

## 🏆 Resultados en prueba (últimas {num(pr['horas'], 1)} horas: {num(pr['transacciones'], 0)} transacciones, {int(pr['fraudes'])} fraudes)

Métrica principal: **AUPRC** (área bajo la curva precisión-recall). Con {pct(100 * pr['tasa_fraude_pct'] / 100, 2)} de fraude en prueba, un modelo al azar obtiene AUPRC ≈ {num(pr['tasa_fraude_pct'] / 100, 4)} y «todo es normal» tiene una exactitud del {pct(100 - pr['tasa_fraude_pct'], 2)}: por eso **nunca se reporta exactitud**. Precisión@{k} = alertas que puede revisar un equipo de {ALERTAS_POR_DIA} alertas al día en esas {num(pr['horas'], 1)} horas. IC 95 %: bootstrap pareado con 1.000 réplicas. El costo usa el umbral elegido **en validación** (supuestos en la sección de costo).

{tabla_resultados(c)}

⭐ Modelo servido en la API: el de mayor AUPRC **en validación** ({num(ms['auprc_validacion'])}). En prueba, los tres LightGBM y el híbrido no se distinguen entre sí (ver diferencias pareadas).

| AUPRC con IC 95 % por modelo | Curvas precisión-recall en prueba |
| :---: | :---: |
| ![AUPRC por modelo](reports/figures/02_auprc_ic_por_modelo.png) | ![Curvas PR](reports/figures/03_curvas_pr_prueba.png) |

### Hallazgos clave

1. **El supervisado gana con holgura.** {ETIQUETAS[srv]}: AUPRC {num(ms['auprc'])} frente a {num(mn['auprc'])} del mejor no supervisado, {ETIQUETAS[nosup]}. La diferencia pareada es {num(d_srv_nosup['diferencia_media'])} {ic(d_srv_nosup['ic_inf'], d_srv_nosup['ic_sup'])}. Incluso la regresión logística ({num(ml['auprc'])}) supera al mejor no supervisado por {num(d_log_nosup['diferencia_media'])} {ic(d_log_nosup['ic_inf'], d_log_nosup['ic_sup'])}.
2. **Raro no es lo mismo que fraudulento.** Isolation Forest ({num(met.loc['if_normales', 'auprc'])}) y el autoencoder ({num(met.loc['autoencoder', 'auprc'])}) tienen ROC-AUC de {num(met.loc['if_normales', 'roc_auc'])} y {num(met.loc['autoencoder', 'roc_auc'])}, que parece buena, pero sus primeras alertas son compras legítimas inusuales: precisión@{k} de {pct(100 * met.loc['if_normales', f'precision_at_{k}'])} y {pct(100 * met.loc['autoencoder', f'precision_at_{k}'])}. LOF, que mide densidad *local*, sí encuentra los grupos de fraude ({pct(100 * mn[f'precision_at_{k}'])}).
3. **Con solo {cruces['Regresión logística']['primer_n']} fraudes etiquetados**, una regresión logística ya supera al mejor no supervisado (AUPRC {num(pe.loc[('Regresión logística', cruces['Regresión logística']['primer_n']), 'media'])} frente a {num(mn['auprc'])}). LightGBM necesita muchas más etiquetas: pasa la referencia por primera vez con {cruces['LightGBM con pesos']['primer_n']} y solo de forma estable con todos los {int(tr['fraudes'])}.
4. **El manejo del desbalance no cambia el ranking.** Pesos − submuestreo: {num(d_pes_sub['diferencia_media'])} {ic(d_pes_sub['ic_inf'], d_pes_sub['ic_sup'])}. Pesos − sin ajuste: {num(d_pes_sin['diferencia_media'])} {ic(d_pes_sin['ic_inf'], d_pes_sin['ic_sup'])}. Los pesos sí bajan la ROC-AUC ({num(met.loc['lgbm_pesos', 'roc_auc'])} frente a {num(met.loc['lgbm_sin_ajuste', 'roc_auc'])}) y descalibran las probabilidades. El **híbrido** tampoco aporta: {num(d_hib['diferencia_media'])} {ic(d_hib['ic_inf'], d_hib['ic_sup'])}.
5. **Ordenar por monto es peor que el azar** (AUPRC {num(met.loc['regla_monto', 'auprc'])}, ROC-AUC {num(met.loc['regla_monto', 'roc_auc'])}): la mediana de un fraude es {num(datos['monto_fraude_mediana_eur'], 2)} EUR y la de una compra normal {num(datos['monto_normal_mediana_eur'], 2)} EUR. Los defraudadores prueban con montos pequeños.

---

## 📌 El problema y las reglas de evaluación

- **Desbalance extremo:** 1 fraude por cada {num(ent['razon_normales_por_fraude_entrenamiento'], 0)} transacciones normales en entrenamiento.
- **El futuro no se ve.** La partición es **temporal** por `Time`, sin barajar. Se corta por *valores* de tiempo, así que todas las transacciones del mismo segundo caen juntas, incluidos los {num(datos['duplicados_exactos'], 0)} duplicados exactos:

| Partición | Horas | Transacciones | Fraudes | Tasa | Monto defraudado |
| :--- | :---: | ---: | ---: | ---: | ---: |
| Entrenamiento | 0 → {num(tr['hasta_h'], 1)} | {num(tr['transacciones'], 0)} | {int(tr['fraudes'])} | {pct(tr['tasa_fraude_pct'], 3)} | {eur(tr['monto_fraude_eur'])} |
| Validación | {num(va['desde_h'], 1)} → {num(va['hasta_h'], 1)} | {num(va['transacciones'], 0)} | {int(va['fraudes'])} | {pct(va['tasa_fraude_pct'], 3)} | {eur(va['monto_fraude_eur'])} |
| Prueba | {num(pr['desde_h'], 1)} → {num(pr['hasta_h'], 1)} | {num(pr['transacciones'], 0)} | {int(pr['fraudes'])} | {pct(pr['tasa_fraude_pct'], 3)} | {eur(pr['monto_fraude_eur'])} |

![Partición temporal](reports/figures/01_particion_temporal.png)

¿Por qué no una partición aleatoria? Los fraudes llegan en **ráfagas** (hasta {int(por_hora['fraudes'].max())} en una hora, mismo patrón de ataque repetido en minutos). Si se baraja, el modelo ve en entrenamiento «hermanos» casi idénticos de los fraudes de prueba y la métrica se infla. Además, en producción el modelo siempre puntúa transacciones posteriores a las que aprendió.

**Todo lo que se decide usa validación:** la parada temprana de LightGBM, el esquema de desbalance, el modelo servido, la calibración de Platt y el umbral. La prueba solo se usa para reportar.

---

## 💶 Costo por error (con `Amount` real)

**Supuestos explícitos** (en `src/fraude/config.py`):
- Revisar una alerta cuesta **5 EUR**: unos 6 minutos de un analista a ~50 EUR/h. Se paga sea fraude o no.
- Un fraude **no detectado** cuesta el **100 % de su `Amount`** (sin contracargo).
- Un fraude alertado se bloquea a tiempo y no genera pérdida (supuesto optimista).
- **Sin modelo**, en prueba se pierden {eur(sin_modelo)}.

El umbral se elige minimizando el costo **en validación** y se aplica tal cual en prueba. La **regla de pérdida esperada** alerta si `p_calibrada × Amount > 5 EUR`, con la probabilidad calibrada por Platt en validación (2 parámetros, robusta con solo {int(va['fraudes'])} fraudes).

{tabla_politicas(c)}

| Curva de costo: validación (se elige) y prueba (se aplica) | Sensibilidad a los supuestos |
| :---: | :---: |
| ![Curva de costo](reports/figures/05_curva_costo.png) | ![Sensibilidad](reports/figures/06_sensibilidad_costo.png) |

**Lectura:**
- Con 5 EUR por alerta, las tres políticas del modelo servido ahorran entre {pct(min(ahorros))} y {pct(max(ahorros))} frente a no usar modelo.
- Los IC del costo son **anchos y se solapan**: con {int(pr['fraudes'])} fraudes, unos pocos de montos altos deciden el resultado (el mayor fraude de prueba, {eur(max_fraude_te)}, equivale a {num(max_fraude_te / 5, 0)} revisiones).
- La regla de pérdida esperada detecta menos fraudes ({int(costo.loc['perdida_esperada', 'fraudes_detectados'])} frente a {int(costo.loc[srv, 'fraudes_detectados'])}), pero deja pasar sobre todo fraudes de pocos euros: revisarlos cuesta más de lo que se perdería.

**Sensibilidad** (se pierde el 100 % del monto; el umbral se re-elige en validación para cada costo; en negrita el mejor):

{tabla_sensibilidad(c)}

Cuando revisar es caro ({num(cr_max, 0)} EUR por alerta), la regla de pérdida esperada mantiene un ahorro del {pct(sens1.loc[(cr_max, 'perdida_esperada'), 'ahorro_vs_sin_modelo_pct'])}, mientras que la regresión logística llega a {pct(sens1.loc[(cr_max, 'logistica'), 'ahorro_vs_sin_modelo_pct'])} y LOF a {pct(sens1.loc[(cr_max, nosup), 'ahorro_vs_sin_modelo_pct'])}. Una política que usa el monto de **cada** transacción es más robusta a los supuestos que un umbral único sobre el puntaje. El archivo `reports/sensibilidad_costo.csv` incluye también los escenarios en que se pierde el 25 % o el 50 % del monto (con contracargo parcial).

---

## 🏷️ ¿Cuántas etiquetas necesita el supervisado?

Se entrena con **todas las normales** y solo *n* fraudes de entrenamiento, elegidos al azar 10 veces. No hay parada temprana: con pocas etiquetas tampoco habría una validación etiquetada confiable. La referencia es el mejor no supervisado, que usa 0 etiquetas de fraude ({ETIQUETAS[nosup]}: AUPRC {num(mn['auprc'])} {ic(mn['auprc_ic_inf'], mn['auprc_ic_sup'])}).

{tabla_pocas(c)}

Con n = {int(tr['fraudes'])} (todos los fraudes) el LightGBM obtiene {num(pe.loc[('LightGBM con pesos', int(tr['fraudes'])), 'media'])} y no {num(met.loc['lgbm_pesos', 'auprc'])} como en la tabla principal, porque aquí usa 300 árboles fijos sin parada temprana.

![Pocas etiquetas](reports/figures/07_pocas_etiquetas.png)

**Lectura:**
- Pocas etiquetas bastan si el modelo es **simple**. La logística con {cruces['Regresión logística']['primer_n']} fraudes ya supera al mejor no supervisado, porque aprende una *dirección* de fraude y usa todas las normales como referencia.
- LightGBM con pocos positivos (y pesos de hasta {num((tr['transacciones'] - tr['fraudes']) / 5, 0)}:1 con 5 fraudes) sobreajusta y varía mucho entre muestras{(': con ' + ' y '.join(map(str, caidas)) + ' fraudes vuelve a caer por debajo de la referencia') if caidas else ''}.
- Conclusión práctica: los detectores de anomalías sirven **antes** de tener etiquetas, para priorizar qué revisar y así conseguirlas. En cuanto hay unas decenas de fraudes confirmados, conviene pasar a un modelo supervisado simple.

---

## 🔍 Interpretabilidad (SHAP) y deriva temporal

![SHAP](reports/figures/08_shap_resumen.png)

- En los fraudes, las variables que más pesan son **{', '.join(shap_top)}**. `V1`–`V28` son **componentes PCA anónimas**: SHAP dice *qué* componente empuja la decisión, pero no puede traducirse a «país», «comercio» u «hora». Sirve para auditar estabilidad, no para explicarle al cliente por qué se bloqueó su tarjeta.
- **Deriva:** un clasificador distingue el día 1 del día 2 con ROC-AUC **{num(adv['roc_auc'])}** usando solo `V1`–`V28` y el monto. Las componentes que más derivan son {der['variable'].iloc[0]} (PSI {num(der['psi'].iloc[0], 2)}), {der['variable'].iloc[1]} ({num(der['psi'].iloc[1], 2)}) y {der['variable'].iloc[2]} ({num(der['psi'].iloc[2], 2)}). El PCA se ajustó sobre los dos días juntos y recoge la tendencia temporal.
- Aun así, la AUPRC del modelo servido pasa de {num(ms['auprc_validacion'])} en validación a {num(ms['auprc'])} en prueba: las variables que más derivan no son las que más separan el fraude. Con más días, la deriva habría que vigilarla (PSI semanal y reentrenamiento).

![Deriva](reports/figures/09_deriva_temporal.png)

---

## 🧭 Arquitectura

```mermaid
flowchart LR
    A[OpenML 1597<br/>descarga reanudable] --> B[Partición temporal<br/>60/20/20 por Time]
    B --> C1[Líneas base<br/>monto · logística]
    B --> C2[No supervisados<br/>IF · LOF · autoencoder<br/>solo normales]
    B --> C3[LightGBM<br/>sin ajuste · pesos · submuestreo]
    C2 -- puntajes fuera de pliegue --> C4[Híbrido]
    C3 --> C4
    C1 & C2 & C3 & C4 --> D[reports/predicciones<br/>validación y prueba]
    D --> E[evaluar.py<br/>AUPRC + IC bootstrap<br/>costo · sensibilidad<br/>pocas etiquetas · SHAP · deriva]
    E --> F[README y MODEL_CARD<br/>generados desde reports/]
    E --> G[models/<br/>LightGBM + AE .npz + metadata]
    G --> H[API FastAPI /puntuar]
    G --> I[App Streamlit]
```

---

## 🧠 Decisiones

- **AUPRC como métrica principal.** La ROC-AUC premia ordenar bien decenas de miles de normales fáciles. Isolation Forest tiene ROC-AUC {num(met.loc['if_normales', 'roc_auc'])} y, aun así, ningún fraude entre sus primeras {k} alertas. La AUPRC mide lo que importa: cuántas alertas son fraude.
- **No supervisados entrenados solo con normales** (*novelty detection*). En la práctica las normales «etiquetadas» son gratis: son las transacciones sin reclamo. También se entrena Isolation Forest sin ninguna etiqueta: con 0,2 % de contaminación, el resultado es casi igual ({num(met.loc['if_sin_etiquetas', 'auprc'])} frente a {num(met.loc['if_normales', 'auprc'])}).
- **Autoencoder en PyTorch CPU, servido en NumPy.** Torch se instaló sin problema. Es una red densa 29 → 20 → 10 → 20 → 29 con activación tanh, entrenada {ent['autoencoder']['epocas']} épocas con parada temprana sobre normales reservadas (sin mirar fraudes). Los pesos se guardan en un `.npz` de ~10 KB y la inferencia es NumPy puro: la API no importa torch.
- **Puntajes de anomalía fuera de pliegue para el híbrido.** Si el autoencoder puntúa las mismas normales con las que se entrenó, su error sale artificialmente bajo y el supervisado aprende una separación que no existe en datos nuevos. Por eso cada normal se puntúa con un detector que no la vio (3 pliegues).
- **Parada temprana por AUPRC de validación** en LightGBM (paciencia 200, tasa 0,03).
- **Submuestreo 1:20 frente a pesos ~{num(ent['razon_normales_por_fraude_entrenamiento'], 0)}:1.** El submuestreo entrena en {num(tiempos['lgbm_submuestreo']['entrenamiento_s'], 1)} s frente a {num(tiempos['lgbm_pesos']['entrenamiento_s'], 1)} s y rinde igual. Ambos distorsionan las probabilidades, así que la regla de pérdida esperada usa la calibración de Platt (Dal Pozzolo et al. 2015 estudian justamente esta corrección).
- **Precisión@k desde la capacidad del equipo**: {ALERTAS_POR_DIA} alertas/día × {num(pr['horas'], 2)} h / 24 = **{k} alertas** en prueba.
- **Bootstrap con pesos por fila.** Cada réplica es un vector de conteos y el orden por puntaje se calcula una sola vez por modelo: 10 modelos × 1.000 réplicas en segundos. Con pesos = 1 coincide con scikit-learn (lo verifica una prueba).

## ⚠️ Limitaciones

- **Solo dos días y {int(pr['fraudes'])} fraudes en prueba.** Los IC de la AUPRC del modelo servido miden ±{num(medio_ic, 2)}, y los del costo son aún más anchos. Ninguna diferencia entre los LightGBM es significativa.
- **V1–V28 son PCA ya hechas por el banco**, y el PCA se ajustó con los dos días (incluido el «futuro»). Es una pequeña fuga de información que no se puede deshacer y que no existiría en producción.
- **Supuestos de costo simplificados**: no incluyen la fricción del cliente legítimo bloqueado, los contracargos ni que un fraude alertado pueda no recuperarse. Por eso hay análisis de sensibilidad, que no reemplaza a datos reales del negocio.
- **Duplicados exactos** ({num(datos['duplicados_exactos'], 0)} filas): se conservan porque pueden ser cobros repetidos legítimos. Al cortar por segundo no cruzan particiones.
- **Rendimiento bajo del autoencoder y de Isolation Forest.** Con componentes PCA decorrelacionadas les queda poca estructura que explotar. No se ajustó su arquitectura con las etiquetas de validación, para no convertirlos en modelos supervisados encubiertos.
- **La API sirve un solo modelo** (el LightGBM elegido en validación). El híbrido no se empaqueta porque no mejora.

---

## ▶️ Cómo reproducir

```bash
python -m venv .venv
.venv\\Scripts\\activate            # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
pip install -e .
python descargar_datos.py          # ~73 MB desde OpenML, reanudable y validado (284.807 filas, 492 fraudes)
python -m fraude.entrenar          # entrena ({num(ent['segundos_entrenamiento'] / 60, 1)} min en CPU) y llama a fraude.evaluar
python -m fraude.evaluar           # solo evaluación desde reports/predicciones_*.parquet (~3 min)
python -m fraude.graficos          # figuras de reports/figures/
python scripts/generar_notebooks.py
python scripts/generar_readme.py   # README.md y MODEL_CARD.md desde reports/
pytest -q -p no:cacheprovider
```

**API** (funciona en un clon sin descargar datos: los modelos versionados pesan < 3 MB):

```bash
uvicorn api.main:app --port 8000
```

Ejemplo real con el primer fraude de la muestra de prueba (`V3`–`V27` abreviadas):

```json
POST /puntuar
{ejemplo}
```

**App:** `streamlit run app/app.py`. Puntúa una muestra de prueba ({num(len(muestra), 0)} transacciones, con los {int(muestra.Class.sum())} fraudes de prueba) y permite mover el costo de revisión para ver cómo cambian las alertas.

## 📁 Estructura

```
├── descargar_datos.py          # descarga reanudable (HTTP Range) desde OpenML
├── src/fraude/                 # config, datos, modelos, metricas, costo, deriva, entrenar, evaluar, graficos, predecir
├── api/main.py                 # FastAPI /salud y /puntuar
├── app/app.py                  # Streamlit
├── models/                     # lgbm_servido.txt, autoencoder.npz, metadata.json
├── reports/                    # métricas, costo, sensibilidad, pocas etiquetas, SHAP, deriva, predicciones, figures/
├── notebooks/                  # 01 datos y deriva · 02 modelos y pocas etiquetas · 03 costo, SHAP y API
├── scripts/                    # generar_notebooks.py, generar_readme.py
├── tests/                      # partición sin fuga, métricas, costo, modelos, API, app, cifras del README
├── MODEL_CARD.md
└── docs/GUIA_PASO_A_PASO.md
```

## 📚 Datos, licencia y cita

- **Datos:** *Credit Card Fraud Detection*, Machine Learning Group de la Université Libre de Bruxelles (ULB) y Worldline. Se descargan de OpenML (dataset 1597, «creditcard»). Licencia de los datos: **Open Database License (ODbL) 1.0**, según la publicación original del MLG-ULB. Los datos crudos no se versionan; `data/muestra/` contiene {num(len(muestra), 0)} filas de la partición de prueba para la app y las pruebas, bajo la misma licencia.
- **Cita:** Dal Pozzolo, A., Caelen, O., Johnson, R. A. y Bontempi, G. (2015). *Calibrating Probability with Undersampling for Unbalanced Classification*. IEEE Symposium Series on Computational Intelligence (SSCI).
- **Código:** licencia MIT (ver `LICENSE`).
"""


def model_card(c: dict) -> str:
    res, met, costo, part, ent = c["res"], c["met"], c["costo"], c["part"], c["ent"]
    srv, nosup, k = res["modelo_servido"], res["mejor_no_supervisado"], res["capacidad"]["k_prueba"]
    ms, pr = met.loc[srv], part.loc["prueba"]
    cs, cp = costo.loc[srv], costo.loc["perdida_esperada"]
    meta = json.loads((RAIZ / "models" / "metadata.json").read_text(encoding="utf-8"))
    return f"""# Ficha del modelo: {ETIQUETAS[srv]} para detección de fraude con tarjetas

## Resumen
- **Tarea:** puntuar transacciones con tarjeta según su probabilidad de fraude y decidir si se envían a revisión manual.
- **Modelo:** LightGBM binario sobre `V1`–`V28` (componentes PCA anónimas) y `log(1 + Amount)`. Pesos de clase ~{num(ent['razon_normales_por_fraude_entrenamiento'], 0)}:1 y parada temprana por AUPRC de validación. Se eligió entre tres LightGBM y un híbrido por la mayor AUPRC **en validación** ({num(ms['auprc_validacion'])}).
- **Artefactos:** `models/lgbm_servido.txt` (LightGBM), `models/autoencoder.npz` (señal de anomalía complementaria, NumPy puro) y `models/metadata.json` (umbral, calibración de Platt y supuestos).

## Datos
- ULB Credit Card Fraud (OpenML 1597): {num(res['datos']['filas'], 0)} transacciones de dos días de septiembre de 2013, {res['datos']['fraudes']} fraudes ({pct(res['datos']['tasa_fraude_pct'], 2)}). Licencia de los datos: ODbL 1.0.
- **Partición temporal sin barajar:** entrenamiento de 0 a {num(part.loc['entrenamiento', 'hasta_h'], 1)} h, validación de {num(part.loc['validacion', 'desde_h'], 1)} a {num(part.loc['validacion', 'hasta_h'], 1)} h y prueba de {num(pr['desde_h'], 1)} a {num(pr['hasta_h'], 1)} h ({num(pr['transacciones'], 0)} transacciones, {int(pr['fraudes'])} fraudes).

## Rendimiento en prueba (IC 95 % bootstrap, 1.000 réplicas)
| Métrica | Valor |
| :--- | ---: |
| AUPRC | {num(ms['auprc'])} {ic(ms['auprc_ic_inf'], ms['auprc_ic_sup'])} |
| ROC-AUC | {num(ms['roc_auc'])} {ic(ms['roc_auc_ic_inf'], ms['roc_auc_ic_sup'])} |
| Precisión@{k} (capacidad de {ALERTAS_POR_DIA} alertas/día) | {pct(100 * ms[f'precision_at_{k}'])} |
| Recall con precisión ≥ 80 % | {pct(100 * ms['recall_a_precision_80'])} |
| Recall con precisión ≥ 90 % | {pct(100 * ms['recall_a_precision_90'])} |
| Mejor no supervisado ({ETIQUETAS[nosup]}), AUPRC | {num(met.loc[nosup, 'auprc'])} |

## Uso previsto y decisión
- **Umbral por costo** (elegido en validación con 5 EUR por alerta y pérdida del 100 % del monto): puntaje ≥ {num(meta['umbral_costo_validacion'], 4)}. En prueba genera {int(cs['alertas'])} alertas, detecta {int(cs['fraudes_detectados'])} de {int(pr['fraudes'])} fraudes y cuesta {eur(cs['costo_total_eur'])} frente a {eur(cs['costo_sin_modelo_eur'])} sin modelo (ahorro {pct(cs['ahorro_vs_sin_modelo_pct'])}).
- **Pérdida esperada** (alternativa recomendada si cambian los costos): alertar si `p_calibrada × Amount > costo de revisión`, con Platt `a = {num(meta['platt']['a'], 4)}` y `b = {num(meta['platt']['b'], 4)}`. En prueba: {int(cp['alertas'])} alertas y costo de {eur(cp['costo_total_eur'])}. Es más robusta en el análisis de sensibilidad.
- Es una herramienta de **priorización para analistas**, no de bloqueo automático.

## Usos no previstos
- Explicar al cliente por qué se marcó su transacción: las variables son PCA anónimas.
- Aplicarlo a otros bancos, países o años sin reentrenar: la validación adversaria separa incluso el día 1 del día 2 (ROC-AUC {num(res['deriva']['adversaria']['roc_auc'])}).

## Limitaciones y riesgos
- Solo dos días de datos y {int(pr['fraudes'])} fraudes en prueba: los IC son anchos y las diferencias entre LightGBM no son significativas.
- El PCA del banco se ajustó con los dos días (pequeña fuga de información imposible de deshacer).
- Las probabilidades crudas están descalibradas por los pesos de clase; para decisiones con monto hay que usar la calibración de Platt.
- Los supuestos de costo no incluyen la fricción del cliente legítimo bloqueado.
- **Monitoreo recomendado:** PSI semanal de las variables más importantes, tasa de alertas y precisión de las alertas revisadas; reentrenar si el PSI supera 0,25 o la precisión de las alertas cae.

## Ética
- Los falsos positivos bloquean compras legítimas. Con el umbral por costo, {int(cs['falsas_alertas'])} de {int(cs['alertas'])} alertas en prueba son legítimas: la revisión humana es obligatoria antes de cualquier bloqueo.
- El dataset no tiene variables demográficas, así que no se puede auditar la equidad entre grupos de clientes.
"""


def main() -> None:
    c = cargar()
    (RAIZ / "README.md").write_text(readme(c), encoding="utf-8")
    (RAIZ / "MODEL_CARD.md").write_text(model_card(c), encoding="utf-8")
    print("[ok] README.md y MODEL_CARD.md generados desde reports/")


if __name__ == "__main__":
    main()
