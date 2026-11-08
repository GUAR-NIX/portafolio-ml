"""Genera los archivos de models/ que contienen datos de MovieLens (la licencia no permite redistribuirlos).

    python descargar_datos.py     # primero, si aún no tienes data/raw/ml-25m/
    python generar_demo.py
"""
from recomendador.demo import main

if __name__ == "__main__":
    main()
