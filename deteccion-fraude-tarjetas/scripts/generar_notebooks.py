"""Genera y ejecuta los 3 notebooks (nbformat + nbclient). Leen reports/ y data/; no reentrenan.

Uso:  python scripts/generar_notebooks.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

RAIZ = Path(__file__).resolve().parents[1]
DIR_NB = RAIZ / "notebooks"

PREAMBULO = (
    "import json, os, sys, warnings\n"
    "from pathlib import Path\n"
    "warnings.filterwarnings('ignore')\n"
    "RAIZ = Path.cwd() if (Path.cwd() / 'src').is_dir() else Path.cwd().parent  # raíz del proyecto o notebooks/\nos.chdir(RAIZ)\n"
    "sys.path.insert(0, str(RAIZ / 'src'))\n"
    "import numpy as np, pandas as pd\n"
    "from IPython.display import Image, display\n"
    "pd.set_option('display.float_format', lambda x: f'{x:,.4f}')\n"
    "pd.set_option('display.width', 160)\n"
    "R = RAIZ / 'reports'\n"
    "res = json.loads((R / 'resultados.json').read_text(encoding='utf-8'))"
)


def md(t: str):
    return new_markdown_cell(t)


def code(t: str):
    return new_code_cell(t)


def nb_01():
    return [
        md("# 01 · Datos, partición temporal y deriva\n\n"
           "284.807 transacciones de titulares europeos en **dos días** de septiembre de 2013 (ULB / Worldline). "
           "`V1`–`V28` son componentes PCA anónimas; solo `Time` (segundos desde la primera transacción) y `Amount` (EUR) "
           "conservan su significado. El fraude es el **0,17 %**."),
        code(PREAMBULO),
        code("from fraude.datos import cargar, particion_temporal\n"
             "df = cargar()\n"
             "print(f\"{len(df):,} transacciones · {int(df.Class.sum())} fraudes · tasa {df.Class.mean()*100:.3f} %\")\n"
             "df.groupby('Class')['Amount'].describe()"),
        md("## Partición temporal (sin barajar)\n\n"
           "En producción el modelo puntúa transacciones **futuras** con lo aprendido del **pasado**. Una partición aleatoria "
           "mezclaría minutos del día 2 en el entrenamiento y sobreestimaría el rendimiento, porque los fraudes llegan en ráfagas "
           "(el mismo patrón de ataque se repite en minutos). Se corta por *valores* de `Time`: 60 % / 20 % / 20 %."),
        code("part = particion_temporal(df)\n"
             "display(part.resumen())\n"
             "# Comprobaciones de fuga: los rangos de tiempo no se solapan y ninguna fila está en dos particiones\n"
             "assert part.entrenamiento.Time.max() < part.validacion.Time.min() <= part.validacion.Time.max() < part.prueba.Time.min()\n"
             "assert len(part.entrenamiento) + len(part.validacion) + len(part.prueba) == len(df)\n"
             "print('Sin solapamiento temporal ✔')"),
        code("display(Image(filename=str(R / 'figures' / '01_particion_temporal.png')))"),
        md("## Deriva entre el día 1 y el día 2\n\n"
           "PSI y KS por variable sobre transacciones normales, más una *validación adversaria*: un clasificador intenta "
           "adivinar el día a partir de `V1`–`V28` y el monto."),
        code("display(pd.read_csv(R / 'deriva_variables.csv').head(8))\n"
             "adv = res['deriva']['adversaria']\n"
             "print(f\"Validación adversaria ROC-AUC = {adv['roc_auc']:.3f}; variables que más delatan el día: {list(adv['top_variables'])}\")\n"
             "print('|corr(V, Time)| en normales:', {k: round(v, 2) for k, v in res['deriva']['corr_time_top'].items()})\n"
             "display(pd.read_csv(R / 'fraude_por_dia.csv'))"),
        code("display(Image(filename=str(R / 'figures' / '09_deriva_temporal.png')))"),
        md("**Lectura:** los dos días son casi perfectamente separables (ROC-AUC ≈ 0,995), sobre todo por `V1`, `V3` y `V28`: "
           "el PCA se ajustó sobre los dos días juntos y varias componentes recogen el ciclo y la tendencia temporal. "
           "Aun así, el rendimiento del modelo supervisado no cae entre validación y prueba, porque las variables que más "
           "derivan no son las que más separan el fraude (V14, V4, V12 según SHAP)."),
    ]


def nb_02():
    return [
        md("# 02 · Modelos, métricas para desbalance y pocas etiquetas\n\n"
           "Diez modelos: dos líneas base, cuatro detectores no supervisados (entrenados sin etiquetas o solo con normales), "
           "tres LightGBM con distintos manejos del desbalance y un híbrido. Métrica principal: **AUPRC** en prueba, "
           "con IC 95 % bootstrap pareado (1.000 réplicas). Nunca exactitud: «todo es normal» acierta el 99,87 % en prueba."),
        code(PREAMBULO),
        code("m = pd.read_csv(R / 'metricas_prueba.csv')\n"
             "k = res['capacidad']['k_prueba']\n"
             "cols = ['nombre', 'familia', 'auprc', 'auprc_ic_inf', 'auprc_ic_sup', 'roc_auc', f'precision_at_{k}', 'recall_a_precision_80', 'auprc_validacion']\n"
             "m[cols].sort_values('auprc', ascending=False)"),
        md("### Las métricas se recalculan desde las predicciones guardadas y coinciden con scikit-learn"),
        code("from sklearn.metrics import average_precision_score, roc_auc_score\n"
             "p = pd.read_parquet(R / 'predicciones_prueba.parquet')\n"
             "for _, r in m.iterrows():\n"
             "    s = p[f\"s_{r['modelo']}\"]\n"
             "    assert abs(average_precision_score(p.Class, s) - r['auprc']) < 1e-9\n"
             "    assert abs(roc_auc_score(p.Class, s) - r['roc_auc']) < 1e-9\n"
             "print('AUPRC y ROC-AUC idénticas a sklearn ✔')"),
        code("display(Image(filename=str(R / 'figures' / '02_auprc_ic_por_modelo.png')))\n"
             "display(Image(filename=str(R / 'figures' / '03_curvas_pr_prueba.png')))"),
        md("### ¿Qué diferencias son significativas? (bootstrap pareado)"),
        code("d = pd.read_csv(R / 'bootstrap_diferencias.csv')\n"
             "d[d.metrica == 'auprc'][['comparacion', 'diferencia_media', 'ic_inf', 'ic_sup', 'significativa']]"),
        md("### ¿Por qué rinden tan poco Isolation Forest y el autoencoder?\n\n"
           "Detectan lo **raro**, no lo **fraudulento**. Muchas compras legítimas son raras (montos altos, combinaciones poco "
           "comunes) y muchos fraudes se parecen a compras normales de pocos euros. Además `V1`–`V28` ya son componentes PCA "
           "decorrelacionadas: al autoencoder le queda poca estructura que aprender, y su error se parece a una distancia al centro."),
        code("display(Image(filename=str(R / 'figures' / '04_distribucion_puntajes_anomalia.png')))"),
        md("## Pocas etiquetas: ¿cuántos fraudes etiquetados hacen falta?"),
        code("g = pd.read_csv(R / 'pocas_etiquetas_resumen.csv')\n"
             "display(g.pivot(index='n_fraudes', columns='modelo', values='media'))\n"
             "pe = res['pocas_etiquetas']\n"
             "print(f\"Referencia: {pe['referencia_no_supervisado']} con AUPRC {pe['auprc_referencia']:.3f}\")\n"
             "print('Cruces:', pe['n_fraudes_para_superar'])"),
        code("display(Image(filename=str(R / 'figures' / '07_pocas_etiquetas.png')))"),
    ]


def nb_03():
    return [
        md("# 03 · Costo por error, interpretabilidad y API\n\n"
           "**Supuestos:** revisar una alerta cuesta 5 EUR; un fraude no detectado cuesta el 100 % de su `Amount`; un fraude "
           "alertado se bloquea sin pérdida. El umbral se elige **en validación** (mínimo costo) y se aplica tal cual en prueba."),
        code(PREAMBULO),
        code("c = pd.read_csv(R / 'costo_prueba.csv')\n"
             "c[['nombre', 'alertas', 'fraudes_detectados', 'fraudes_perdidos', 'costo_total_eur', 'costo_ic_inf', 'costo_ic_sup', 'ahorro_vs_sin_modelo_pct']]"),
        md("### Recalcular el umbral del modelo servido desde las predicciones de validación"),
        code("from fraude.costo import elegir_umbral, costo_total\n"
             "pv = pd.read_parquet(R / 'predicciones_validacion.parquet')\n"
             "pt = pd.read_parquet(R / 'predicciones_prueba.parquet')\n"
             "srv = res['modelo_servido']\n"
             "u, curva = elegir_umbral(pv.Class, pv.Amount, pv[f's_{srv}'], 5.0, 1.0)\n"
             "fila = c.set_index('modelo').loc[srv]\n"
             "assert abs(u - fila['umbral']) < 1e-12\n"
             "assert abs(costo_total(pt.Class, pt.Amount, pt[f's_{srv}'] >= u, 5.0) - fila['costo_total_eur']) < 1e-6\n"
             "print(f'Umbral {u:.4f} → {int((pt[f\"s_{srv}\"] >= u).sum())} alertas en prueba, costo {fila.costo_total_eur:,.2f} EUR ✔')"),
        code("display(Image(filename=str(R / 'figures' / '05_curva_costo.png')))\n"
             "display(Image(filename=str(R / 'figures' / '06_sensibilidad_costo.png')))"),
        code("s = pd.read_csv(R / 'sensibilidad_costo.csv')\n"
             "s.pivot_table(index=['fraccion_perdida', 'costo_revision_eur'], columns='modelo', values='ahorro_vs_sin_modelo_pct').round(1)"),
        md("## SHAP del modelo servido\n\n"
           "V1–V28 son componentes PCA **anónimas**: SHAP dice *qué* componente empuja la decisión, pero no puede traducirse a "
           "«país», «comercio» u «hora». Sirve para auditar estabilidad y deriva, no para explicar al cliente."),
        code("display(pd.read_csv(R / 'shap_importancia.csv').head(10))\n"
             "display(Image(filename=str(R / 'figures' / '08_shap_resumen.png')))"),
        md("## API `/puntuar` (FastAPI, probada con `TestClient`)"),
        code("from fastapi.testclient import TestClient\n"
             "from api.main import app\n"
             "cliente = TestClient(app)\n"
             "muestra = pd.read_csv(RAIZ / 'data' / 'muestra' / 'transacciones_prueba_muestra.csv')\n"
             "ejemplo = pd.concat([muestra[muestra.Class == 1].head(3), muestra[muestra.Class == 0].head(3)])\n"
             "r = cliente.post('/puntuar', json={'transacciones': ejemplo.drop(columns=['Time', 'Class']).to_dict('records')})\n"
             "print(r.status_code)\n"
             "pd.concat([ejemplo[['Amount', 'Class']].reset_index(drop=True), pd.DataFrame(r.json()['resultados'])], axis=1)"),
    ]


def ejecutar(nombre: str, celdas) -> None:
    nb = new_notebook(cells=celdas)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(RAIZ)}}).execute()
    DIR_NB.mkdir(exist_ok=True)
    with open(DIR_NB / nombre, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"[ok] {nombre}")


if __name__ == "__main__":
    ejecutar("01_datos_particion_temporal_y_deriva.ipynb", nb_01())
    ejecutar("02_modelos_metricas_y_pocas_etiquetas.ipynb", nb_02())
    ejecutar("03_costo_shap_y_api.ipynb", nb_03())
