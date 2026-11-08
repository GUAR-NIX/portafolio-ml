"""Las cifras del README y de la ficha salen de reports/ (no se escriben a mano).

1. README.md y MODEL_CARD.md son exactamente lo que genera scripts/generar_readme.py con los reports actuales.
2. Las cifras clave aparecen con el formato del texto y se recalculan desde las predicciones guardadas.
3. Higiene: sin rutas personales, sin enlaces a otras carpetas, LICENSE MIT y guía con ≥ 8 preguntas.
"""

from __future__ import annotations

import importlib.util
import re

import pytest
from sklearn.metrics import average_precision_score

from fraude.config import RAIZ_PROYECTO


@pytest.fixture(scope="module")
def gen():
    spec = importlib.util.spec_from_file_location("generar_readme", RAIZ_PROYECTO / "scripts" / "generar_readme.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def readme() -> str:
    return (RAIZ_PROYECTO / "README.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def ficha() -> str:
    return (RAIZ_PROYECTO / "MODEL_CARD.md").read_text(encoding="utf-8")


def test_readme_y_ficha_estan_al_dia(gen, readme, ficha) -> None:
    c = gen.cargar()
    assert readme == gen.readme(c), "README.md desactualizado: ejecuta python scripts/generar_readme.py"
    assert ficha == gen.model_card(c), "MODEL_CARD.md desactualizado: ejecuta python scripts/generar_readme.py"


def test_cifras_clave_en_readme(gen, readme, resultados, pred_prueba) -> None:
    y = pred_prueba["Class"]
    for m in resultados["metricas_prueba"]:
        auprc = average_precision_score(y, pred_prueba[f"s_{m['modelo']}"])  # recalculada, no leída
        assert f"{gen.num(auprc)} {gen.ic(m['auprc_ic_inf'], m['auprc_ic_sup'])}" in readme, m["modelo"]
    for c in resultados["costo_prueba"]:
        assert gen.eur(c["costo_total_eur"]) in readme, c["modelo"]
    srv = next(m for m in resultados["metricas_prueba"] if m["modelo"] == resultados["modelo_servido"])
    assert f"AUPRC {gen.num(srv['auprc'])} frente a" in readme
    pe = resultados["pocas_etiquetas"]["n_fraudes_para_superar"]
    assert f"Con solo {pe['Regresión logística']['primer_n']} fraudes etiquetados" in readme
    assert f"ROC-AUC **{gen.num(resultados['deriva']['adversaria']['roc_auc'])}**" in readme
    part = {r["particion"]: r for r in resultados["particion"]}
    assert f"{gen.num(part['prueba']['transacciones'], 0)} transacciones, {part['prueba']['fraudes']} fraudes" in readme


def test_cifras_clave_en_ficha(gen, ficha, resultados) -> None:
    srv = next(m for m in resultados["metricas_prueba"] if m["modelo"] == resultados["modelo_servido"])
    assert f"{gen.num(srv['auprc'])} {gen.ic(srv['auprc_ic_inf'], srv['auprc_ic_sup'])}" in ficha
    costo = next(c for c in resultados["costo_prueba"] if c["modelo"] == resultados["modelo_servido"])
    assert f"cuesta {gen.eur(costo['costo_total_eur'])}" in ficha
    assert f"detecta {costo['fraudes_detectados']} de" in ficha


def test_readme_sin_exactitud_como_metrica(readme) -> None:
    assert "**nunca se reporta exactitud**" in readme
    assert "| Exactitud" not in readme


ARCHIVOS_TEXTO = (".py", ".md", ".ipynb", ".json", ".csv", ".txt", ".toml")


def _versionables():
    for p in RAIZ_PROYECTO.rglob("*"):
        partes = set(p.relative_to(RAIZ_PROYECTO).parts)
        if p.is_file() and p.suffix in ARCHIVOS_TEXTO and not partes & {".venv", "raw", "__pycache__", ".pytest_cache", "fraude.egg-info"}:
            yield p


def test_sin_rutas_personales_ni_enlaces_a_otras_carpetas() -> None:
    usuarios = "Us" + "ers"  # partido para que este archivo no coincida consigo mismo
    patron = re.compile(rf"[A-Za-z]:[\\/]+{usuarios}[\\/]|/c/{usuarios}/|\]\(\.\./", re.IGNORECASE)
    malos = [str(p.relative_to(RAIZ_PROYECTO)) for p in _versionables() if patron.search(p.read_text(encoding="utf-8", errors="ignore"))]
    assert not malos, f"Rutas personales o enlaces relativos externos en: {malos}"


def test_sin_menciones_a_versiones_anteriores(readme, ficha) -> None:
    guia = (RAIZ_PROYECTO / "docs" / "GUIA_PASO_A_PASO.md").read_text(encoding="utf-8")
    for texto in (readme, ficha, guia):
        assert "versión anterior" not in texto.lower()


def test_licencia_y_guia() -> None:
    lic = (RAIZ_PROYECTO / "LICENSE").read_text(encoding="utf-8")
    assert "MIT License" in lic and "Erick Guarnizo" in lic
    guia = (RAIZ_PROYECTO / "docs" / "GUIA_PASO_A_PASO.md").read_text(encoding="utf-8")
    preguntas = re.findall(r"^\*\*\d+\. .+\?\*\*$", guia, flags=re.MULTILINE)
    assert len(preguntas) >= 8
