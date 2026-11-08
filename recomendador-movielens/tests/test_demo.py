"""Paquete incompleto: sin los archivos con datos de MovieLens (no se redistribuyen) el error debe ser claro."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

import api.main as api
from recomendador import demo
from recomendador.servicio import Recomendador


def test_mensaje_dice_como_generar_los_archivos(tmp_path):
    for a in demo.ARCHIVOS_VERSIONADOS:  # carpeta con lo versionado pero sin los datos de MovieLens
        (tmp_path / a).write_bytes(b"")
    msg = demo.mensaje_faltantes(tmp_path)
    assert "peliculas.parquet" in msg and "demo_positivos.npz" in msg and "demo_vistos.npz" in msg
    assert "python descargar_datos.py" in msg and "python generar_demo.py" in msg
    assert "metadata.json" not in msg


def test_paquete_completo_no_da_mensaje(tmp_path):
    for a in (*demo.ARCHIVOS_VERSIONADOS, *demo.ARCHIVOS_DEMO):
        (tmp_path / a).write_bytes(b"")
    assert demo.faltantes(tmp_path) == [] and demo.mensaje_faltantes(tmp_path) == ""


def test_recomendador_sin_archivos_lanza_error_claro(tmp_path):
    with pytest.raises(demo.PaqueteIncompleto, match="generar_demo.py"):
        Recomendador(tmp_path)


def test_api_responde_503_con_el_mensaje(monkeypatch, tmp_path):
    def sin_paquete():
        Recomendador(tmp_path)

    monkeypatch.setattr(api, "cargar", sin_paquete)
    r = TestClient(api.app).get("/salud")
    assert r.status_code == 503 and "python generar_demo.py" in r.json()["detail"]


def test_matrices_demo_conservan_las_filas_pedidas(tmp_path):
    """guardar_matrices_demo guarda exactamente las filas de los usuarios de demostración, en ese orden."""
    import scipy.sparse as sp

    class Parte:
        positivos = sp.csr_matrix(np.eye(4, dtype=np.float32))
        vistos = sp.csr_matrix(np.ones((4, 4), dtype=np.float32))

    demo.guardar_matrices_demo(Parte, np.array([3, 1]), tmp_path)
    pos = sp.load_npz(tmp_path / "demo_positivos.npz").toarray()
    assert pos.shape == (2, 4) and pos[0, 3] == 1 and pos[1, 1] == 1 and pos.sum() == 2
    assert sp.load_npz(tmp_path / "demo_vistos.npz").shape == (2, 4)
