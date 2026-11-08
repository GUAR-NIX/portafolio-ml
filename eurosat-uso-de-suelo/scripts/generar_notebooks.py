"""Genera y ejecuta los 3 notebooks de ML_06_satelite_eurosat usando nbformat y nbclient."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

RAIZ = Path(__file__).resolve().parents[1]
DIR_NB = RAIZ / "notebooks"


def ejecutar_y_guardar(nb: nbformat.NotebookNode, ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    cliente = NotebookClient(
        nb,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(RAIZ)}},
    )
    cliente.execute()
    with open(ruta, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"[ok] Notebook ejecutado: {ruta.name}")


def crear_nb_01() -> nbformat.NotebookNode:
    nb = new_notebook()
    nb.cells = [
        new_markdown_cell(
            "# 01 · Imágenes Satelitales Sentinel-2 (EuroSAT): Auditoría Radiométrica e Índices Espectrales\n\n"
            "Este cuaderno audita las **27.000 escenas oficiales de Sentinel-2 MSI (64×64 píxeles, 10 m/píxel)** "
            "distribuidas en 10 clases de uso y cobertura de suelo (*EuroSAT*, Helber et al., 2019), "
            "calcula índices espectrales en las bandas visibles ($R, G, B$) y analiza la separabilidad física "
            "frente a la rugosidad espacial."
        ),
        new_code_cell(
            "import json\n"
            "from pathlib import Path\nimport os\nif not Path('reports').is_dir():  # Jupyter abre el cuaderno en notebooks/: se trabaja desde la raíz\n    os.chdir('..')\n"
            "import pandas as pd\n"
            "from IPython.display import display, Image\n\n"
            "auditoria = json.loads(Path('data/cifras_control_eurosat.json').read_text(encoding='utf-8'))\n"
            "print(json.dumps(auditoria, indent=2, ensure_ascii=False))"
        ),
        new_code_cell(
            "df_27k = pd.read_csv('reports/resumen_clases_27000.csv')\n"
            "cols = ['clase_id', 'clase', 'nombre_es', 'macro_categoria', 'imagenes_totales', "
            "'exg_medio', 'indice_agua_medio', 'brillo_medio', 'rugosidad_media', 'pct_velo_o_bajo_contraste']\n"
            "display(df_27k[cols].round(4))"
        ),
        new_markdown_cell(
            "## Mosaico de las 10 clases de cobertura terrestre y espacio espectral (`ExG` vs. Índice de Agua)"
        ),
        new_code_cell("display(Image('reports/figures/01_mosaico_clases_y_espectro_sentinel2.png'))"),
    ]
    return nb


def crear_nb_02() -> nbformat.NotebookNode:
    nb = new_notebook()
    nb.cells = [
        new_markdown_cell(
            "# 02 · Transfer Learning en CPU (`MobileNetV3-Small`) vs. descriptores espectrales, IC bootstrap y Few-Shot\n\n"
            "Cinco modelos entrenados en **CPU** con las particiones oficiales de EuroSAT: **16.200** imágenes de "
            "entrenamiento, **5.400** de validación (calibración, peso del ensamble y umbral) y **5.400** de prueba "
            "(solo para reportar). Los intervalos son bootstrap pareado (2.000 réplicas) sobre la prueba."
        ),
        new_code_cell(
            "import warnings\n"
            "warnings.filterwarnings('ignore')\n"
            "import json\n"
            "from pathlib import Path\nimport os\nif not Path('reports').is_dir():  # Jupyter abre el cuaderno en notebooks/: se trabaja desde la raíz\n    os.chdir('..')\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "from IPython.display import display, Image\n\n"
            "res = json.loads(Path('reports/resultados.json').read_text(encoding='utf-8'))\n"
            "print('Particiones:', res['particiones'])\n"
            "df_comp = pd.read_csv('reports/comparacion_modelos.csv')\n"
            "display(df_comp[['id', 'modelo', 'exactitud_top1', 'exactitud_ic95_inf', 'exactitud_ic95_sup', 'f1_macro', "
            "'log_loss', 'ece_calibracion', 'errores_inter_macro', 'costo_por_1000_usd', 'latencia_ms_img']])"
        ),
        new_markdown_cell("## Recalcular la exactitud desde las predicciones guardadas (sin reentrenar)"),
        new_code_cell(
            "pred = pd.read_parquet('reports/predicciones_test.parquet')\n"
            "for m in ['m1', 'm2', 'm3', 'm4', 'm5']:\n"
            "    print(m, round(float((pred[f'pred_{m}'] == pred['y_real']).mean()), 4))"
        ),
        new_markdown_cell("## ¿Qué diferencias son significativas? (IC 95 % del bootstrap pareado)"),
        new_code_cell("display(pd.read_csv('reports/bootstrap_diferencias.csv'))"),
        new_code_cell(
            "display(pd.read_csv('reports/curva_few_shot_eurosat.csv'))\n"
            "df_clases = pd.read_csv('reports/metricas_por_clase.csv')\n"
            "display(df_clases[['clase', 'nombre_es', 'soporte', 'f1_espectral_clasico', 'f1', 'ganancia_f1_transfer', 'principal_confusion_es', 'casos_principal_confusion']])"
        ),
        new_code_cell(
            "for f in ['02_comparacion_arquitecturas_cpu', '10_bootstrap_diferencias', '03_curva_few_shot_transfer_learning',\n"
            "          '04_f1_por_clase_y_ganancia_transfer', '05_matrices_confusion_espectral_vs_transfer']:\n"
            "    display(Image(f'reports/figures/{f}.png'))"
        ),
    ]
    return nb


def crear_nb_03() -> nbformat.NotebookNode:
    nb = new_notebook()
    nb.cells = [
        new_markdown_cell(
            "# 03 · Calibración, abstención, robustez, ONNX, Grad-CAM y API\n\n"
            "Todo lo que se ajusta (temperaturas, peso del ensamble y umbral τ) se elige en **validación**; aquí se "
            "muestran sus efectos en **prueba**. Los costos (18, 35 y 140 USD) son **supuestos** y se analiza su "
            "sensibilidad. El velo atmosférico es **simulado**."
        ),
        new_code_cell(
            "import warnings\n"
            "warnings.filterwarnings('ignore')\n"
            "import json\n"
            "from pathlib import Path\nimport os\nif not Path('reports').is_dir():  # Jupyter abre el cuaderno en notebooks/: se trabaja desde la raíz\n    os.chdir('..')\n"
            "import pandas as pd\n"
            "from IPython.display import display, Image\n\n"
            "res = json.loads(Path('reports/resultados.json').read_text(encoding='utf-8'))\n"
            "display(pd.read_csv('reports/calibracion_temperatura.csv'))\n"
            "print('Peso del ensamble (elegido en validación):', res['ensamble'])"
        ),
        new_markdown_cell("## Umbral de abstención: elegido en validación, evaluado en prueba"),
        new_code_cell(
            "a = res['abstencion']\n"
            "print('tau elegido en validación (M5):', a['tau_validacion_m5'], '| tau que minimizaría la prueba:', a['tau_que_minimizaria_prueba_m5'])\n"
            "display(pd.DataFrame([a['validacion_m5'], a['prueba_m5']], index=['validación', 'prueba']))\n"
            "display(pd.read_csv('reports/comparacion_abstencion.csv'))"
        ),
        new_markdown_cell("## Sensibilidad a los costos supuestos (τ re-elegido en validación en cada escenario)"),
        new_code_cell(
            "sens = pd.read_csv('reports/sensibilidad_costos.csv')\n"
            "display(sens[sens['un_solo_cambio']])"
        ),
        new_markdown_cell("## Robustez: velo simulado, ruido gaussiano y desenfoque (5 modelos)"),
        new_code_cell("display(pd.read_csv('reports/robustez_perturbaciones.csv'))"),
        new_markdown_cell("## ONNX: salidas iguales a PyTorch y latencia en CPU"),
        new_code_cell(
            "o = res['onnx']\n"
            "print('Diferencia máxima de logits:', o['diferencia_max_abs_logits'], '| misma clase:', o['pct_misma_clase_predicha'], '%')\n"
            "display(pd.DataFrame(o['latencias']))"
        ),
        new_markdown_cell("## API FastAPI (inferencia con ONNX Runtime)"),
        new_code_cell(
            "from fastapi.testclient import TestClient\n"
            "from api.main import app\n\n"
            "cliente = TestClient(app)\n"
            "resp = cliente.post('/clasificar_lote', json={'parcelas': [\n"
            "    {'id_parcela': 'ESCENA_BOSQUE', 'archivo_muestra': '01_Forest_01.png'},\n"
            "    {'id_parcela': 'ESCENA_AUTOPISTA', 'archivo_muestra': '03_Highway_01.png'},\n"
            "    {'id_parcela': 'ESCENA_RIO', 'archivo_muestra': '08_River_01.png'}\n"
            "]}).json()\n"
            "for r in resp['resultados']:\n"
            "    print(r['id_parcela'], '->', r['clase_predicha'], r['confianza'], r['decision_operativa'])"
        ),
        new_code_cell(
            "for f in ['06_calibracion_temperatura_y_abstencion', '07_robustez_velo_ruido_desenfoque',\n"
            "          '09_grad_cam_aciertos_y_errores', '08_mapas_oclusion_espacial']:\n"
            "    display(Image(f'reports/figures/{f}.png'))"
        ),
    ]
    return nb


def main() -> None:
    ejecutar_y_guardar(crear_nb_01(), DIR_NB / "01_datos_sentinel2_indices_espectrales_y_calidad.ipynb")
    ejecutar_y_guardar(crear_nb_02(), DIR_NB / "02_transfer_learning_cpu_vs_espectral_y_fewshot.ipynb")
    ejecutar_y_guardar(crear_nb_03(), DIR_NB / "03_calibracion_abstencion_oclusion_y_api.ipynb")


if __name__ == "__main__":
    main()
