"""Pipeline completo en un comando:

    python -m recomendador.entrenar            # ~1-2 h en una PC de 6 núcleos compartida (reanuda si se corta)

1. Parquet + partición temporal de validación → búsqueda acotada de hiperparámetros (ALS, contenido, híbrido, EASE).
2. Partición de prueba (corte 2018-01-01) → todos los modelos re-ajustados con los hiperparámetros elegidos,
   evaluados una sola vez por segmento de usuario.
3. Modelo final re-entrenado con TODOS los datos → paquete liviano en models/ para la API y la app.
   (peliculas.parquet y demo_positivos/demo_vistos.npz llevan datos de MovieLens: no se versionan, ver demo.py)
"""
from __future__ import annotations

import gc
import json
import sys
import time

import numpy as np
import pandas as pd

from . import __version__
from . import busqueda as B
from . import config as C
from . import datos as D
from . import evaluacion as E
from . import modelos as Mo
from .demo import guardar_matrices_demo, tabla_peliculas


def _vec_generos(generos: np.ndarray) -> np.ndarray:
    return generos / np.maximum(np.linalg.norm(generos, axis=1, keepdims=True), 1e-9)


def fase_validacion(cal, peliculas, genome) -> dict:
    print("== Validación: entrenar < 2017-07-01, evaluar 2017-07 → 2017-12", flush=True)
    part = D.particionar(cal, C.CORTE_VALIDACION, C.CORTE_PRUEBA)
    print(" ", part.resumen, flush=True)
    gen, gnm = D.caracteristicas_items(part.items, peliculas, genome)
    ex, nu = E.usuarios_existentes(part, 4000), E.usuarios_nuevos(part, 3000)
    rutas = {n: C.REPORTES / f"busqueda_{n}.csv" for n in ("als", "contenido", "hibrido", "ease")}

    p_als, _ = B.buscar_als(part, ex, nu, rutas["als"])
    print("  mejor ALS:", p_als, flush=True)
    p_cont, _ = B.buscar_contenido(part, gen, gnm, ex, nu, rutas["contenido"])
    print("  mejor contenido:", p_cont, flush=True)
    als = Mo.ALS(**p_als).ajustar(part.positivos)
    cont = Mo.Contenido(**p_cont).ajustar(part.positivos, generos=gen, genome=gnm)
    p_hib, _ = B.buscar_hibrido(part, als, cont, ex, nu, rutas["hibrido"])
    print("  mejor híbrido:", p_hib, flush=True)
    del als, cont
    gc.collect()
    lam, _ = B.buscar_ease(part, ex, nu, rutas["ease"])
    print("  mejor λ de EASE:", lam, flush=True)
    return {"als": p_als, "contenido": p_cont, "hibrido": p_hib, "ease_lambda": lam, "particion": part.resumen,
            "usuarios_validacion": {"existentes": len(ex), "nuevos": len(nu)}}


def fase_prueba(cal, peliculas, genome, hp: dict) -> dict:
    print("== Prueba: entrenar < 2018-01-01, evaluar 2018-01 → 2019-11 (una sola vez)", flush=True)
    part = D.particionar(cal, C.CORTE_PRUEBA, C.FIN_PRUEBA)
    print(" ", part.resumen, flush=True)
    gen, gnm = D.caracteristicas_items(part.items, peliculas, genome)
    vg = _vec_generos(gen)
    ex, nu = E.usuarios_existentes(part), E.usuarios_nuevos(part)
    hist = E.historiales_de(part, ex, nu)

    tiempos = {}
    pop = Mo.Popularidad().ajustar(part.positivos)
    rec = Mo.PopularidadReciente().ajustar(part.positivos, recientes=part.recientes)
    als, tiempos["ALS"] = E.cronometrar(Mo.ALS(**hp["als"]).ajustar, part.positivos)
    cont, tiempos["Contenido"] = E.cronometrar(Mo.Contenido(**hp["contenido"]).ajustar, part.positivos,
                                               generos=gen, genome=gnm)
    hib = Mo.Hibrido(als, cont, recientes=part.recientes, **hp["hibrido"])
    gen_eleg = Mo.GenerosElegidos(cont)
    modelos = [pop, rec, als, cont, hib]

    resumenes, por_usuario = [], []
    for m in modelos:
        (df, listas), seg = E.cronometrar(E.evaluar, m, part, ex, nu)
        print(f"  {m.nombre}: NDCG@10 = {df['ndcg'].mean():.4f} ({seg:.0f} s)", flush=True)
        resumenes.append(E.resumir(m.nombre, df, listas, part, vg, hist))
        por_usuario.append(df.assign(modelo=m.nombre))
        if m is hib:
            ejemplos = listas
    # Usuario nuevo que solo marca géneros (solo tiene sentido para nuevos)
    df, listas = E.evaluar(gen_eleg, part, np.array([], dtype=np.int64), nu)
    resumenes.append(E.resumir(gen_eleg.nombre, df, listas, part, vg, hist[len(ex):]))
    por_usuario.append(df.assign(modelo=gen_eleg.nombre))
    print(f"  {gen_eleg.nombre}: NDCG@10 = {df['ndcg'].mean():.4f}", flush=True)
    # EASE: referencia fuerte (solo evaluación)
    ease, tiempos["EASE"] = E.cronometrar(Mo.EASE(hp["ease_lambda"]).ajustar, part.positivos)
    df, listas = E.evaluar(ease, part, ex, nu)
    resumenes.append(E.resumir(ease.nombre, df, listas, part, vg, hist))
    por_usuario.append(df.assign(modelo=ease.nombre))
    print(f"  {ease.nombre}: NDCG@10 = {df['ndcg'].mean():.4f}", flush=True)
    del ease
    gc.collect()

    resumen = pd.concat(resumenes, ignore_index=True)
    resumen.round(5).to_csv(C.REPORTES / "metricas_prueba.csv", index=False)
    pd.concat(por_usuario, ignore_index=True).to_parquet(C.REPORTES / "metricas_por_usuario.parquet", index=False)

    # Peso del ALS en el híbrido por segmento (para explicar el comportamiento)
    n_pos = np.concatenate([part.n_positivos_usuario[ex], [len(part.eval_nuevos[u]["onboarding"]) for u in nu]])
    pesos = pd.DataFrame({"n_positivos": n_pos, "peso_als": hib.peso_als(n_pos)})

    # Ejemplo legible para el notebook/README: un usuario con muchos positivos
    idx_items = part.items
    titulos = peliculas.set_index("movieId")["titulo"]
    u0 = int(ex[np.argmax(part.n_positivos_usuario[ex] > 100)])
    ejemplo = {
        "usuario": int(part.usuarios[u0]),
        "le_gusto_reciente": titulos.reindex(idx_items[part.positivos[u0].indices[-8:]]).tolist(),
        "recomendado_hibrido": titulos.reindex(idx_items[ejemplos[int(np.flatnonzero(ex == u0)[0])]]).tolist(),
        "acerto_en_prueba": titulos.reindex(idx_items[part.eval_existentes[u0]][:10]).tolist(),
    }
    return {
        "particion": part.resumen, "usuarios_evaluados": {"existentes": len(ex), "nuevos": len(nu)},
        "segundos_ajuste": {k: round(v, 1) for k, v in tiempos.items()},
        "varianza_explicada_genome_svd": round(cont.varianza_explicada, 4),
        "peso_als_mediano_por_segmento": pesos.assign(
            segmento=[C.segmento_de(int(n)) for n in pesos["n_positivos"][: len(ex)]] + ["nuevos (onboarding)"] * len(nu)
        ).groupby("segmento")["peso_als"].median().round(3).to_dict(),
        "ejemplo": ejemplo,
    }


def fase_final(cal, peliculas, genome, hp: dict) -> dict:
    """Re-entrena con todos los datos y guarda el paquete que usan la API y la app (< 50 MB)."""
    print("== Modelo final con todos los datos (hasta 2019-11-21)", flush=True)
    fin = pd.Timestamp("2019-12-01")
    part = D.particionar(cal, fin, fin + pd.Timedelta(days=1))
    gen, gnm = D.caracteristicas_items(part.items, peliculas, genome)
    als = Mo.ALS(**hp["als"]).ajustar(part.positivos)
    cont = Mo.Contenido(**hp["contenido"]).ajustar(part.positivos, generos=gen, genome=gnm)

    C.MODELOS.mkdir(exist_ok=True)
    tabla_peliculas(peliculas, part, cont.tiene_genome).to_parquet(C.MODELOS / "peliculas.parquet", index=False)
    np.savez_compressed(
        C.MODELOS / "factores.npz",
        factores_item=als.factores_item.astype(np.float32),
        idf_bm25=getattr(als, "idf", np.ones(len(part.items), np.float32)),
        genome_reducido=cont.genome_red.astype(np.float16),
        generos=gen.astype(np.uint8),
    )
    # Usuarios de demostración: 1.500 al azar estratificados por segmento (factores + lo que ya vio)
    rng = np.random.default_rng(C.SEMILLA)
    n_pos = part.n_positivos_usuario
    demo = []
    for minimo, maximo, _ in C.SEGMENTOS:
        cand = np.flatnonzero((n_pos >= minimo) & (n_pos <= maximo))
        demo.extend(rng.choice(cand, size=min(500, len(cand)), replace=False).tolist())
    demo = np.sort(np.array(demo))
    guardar_matrices_demo(part, demo, C.MODELOS)
    np.savez_compressed(C.MODELOS / "demo_usuarios.npz", usuarios=part.usuarios[demo].astype(np.int32),
                        factores=als.factores_usuario[demo].astype(np.float32))
    return {"usuarios": int(len(part.usuarios)), "items_catalogo": int(len(part.items)),
            "largo_medio_bm25": float(getattr(als, "largo_medio", 1.0)),
            "positivos": int(part.positivos.nnz), "usuarios_demo": int(len(demo))}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # la consola de Windows (cp1252) no imprime → ni tildes raras
    t0 = time.perf_counter()
    C.REPORTES.mkdir(exist_ok=True)
    C.FIGURAS.mkdir(parents=True, exist_ok=True)
    D.convertir_a_parquet()
    cal = D.cargar_calificaciones()
    peliculas = D.cargar_peliculas()
    genome, _ = D.cargar_genome()

    ruta_hp = C.REPORTES / "hiperparametros.json"
    if ruta_hp.exists():
        val = json.loads(ruta_hp.read_text(encoding="utf-8"))
        print("Hiperparámetros ya elegidos:", {k: val[k] for k in ("als", "contenido", "hibrido", "ease_lambda")})
    else:
        val = fase_validacion(cal, peliculas, genome)
        ruta_hp.write_text(json.dumps(val, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    gc.collect()
    prueba = fase_prueba(cal, peliculas, genome, val)
    gc.collect()
    final = fase_final(cal, peliculas, genome, val)

    metadata = {
        "version": __version__, "k": C.K, "umbral_positivo": C.UMBRAL_POSITIVO,
        "corte_validacion": str(C.CORTE_VALIDACION.date()), "corte_prueba": str(C.CORTE_PRUEBA.date()),
        "hiperparametros": {k: val[k] for k in ("als", "contenido", "hibrido", "ease_lambda")},
        "prueba": prueba, "modelo_final": final,
        "metricas_prueba": pd.read_csv(C.REPORTES / "metricas_prueba.csv").round(4).to_dict("records"),
    }
    (C.MODELOS / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, default=float),
                                             encoding="utf-8")
    (C.REPORTES / "resultados.json").write_text(json.dumps(prueba, indent=2, ensure_ascii=False, default=float),
                                                encoding="utf-8")
    from .graficos import generar_todas

    generar_todas()
    print(f"Listo en {(time.perf_counter() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
