# Guía paso a paso (Windows)

El proyecto **ya viene entrenado y ejecutado**: los factores del modelo están en `models/` (12 MB), los notebooks
tienen resultados y las figuras y tablas están en `reports/`. Esta guía sirve para volver a correrlo, por ejemplo para
demostrarlo en una entrevista.

> **Licencia de MovieLens:** no permite redistribuir el dataset, así que tres archivos de `models/` que llevan sus
> datos (`peliculas.parquet`, `demo_positivos.npz`, `demo_vistos.npz`) no están en el repositorio. Hay que generarlos
> una vez en el paso 2 (descarga de ~262 MB + ~2 min).

## 1 · Crear el entorno e instalar dependencias (3-5 min)

```powershell
cd recomendador-movielens
py -3.14 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

> `implicit` trae rueda precompilada para Windows y Python 3.14: no hace falta compilador.
> Si PowerShell bloquea la activación, ejecuta una vez: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

## 2 · Datos de MovieLens y archivos de demostración (≈ 5 min, 262 MB → 1,2 GB)

```powershell
python descargar_datos.py
python generar_demo.py
```

`descargar_datos.py` baja `ml-25m.zip` del sitio oficial de GroupLens, verifica el **MD5** publicado y deja los CSV en
`data/raw/ml-25m/`. `generar_demo.py` reconstruye en `models/` los tres archivos con datos de MovieLens (idénticos a
los del modelo entrenado, porque usa los mismos usuarios de `demo_usuarios.npz`). Si te los saltas, la API responde
`503` con un mensaje que dice exactamente esto, y la app lo muestra en pantalla.

## 3 · Pruebas (≈ 1 min)

```powershell
pytest
```

Resultado esperado: todas pasan. Las pruebas de métricas, partición y modelos usan datos sintéticos de resultado
conocido; las de la API usan el paquete de `models/` y se omiten si faltan los archivos del paso 2.

## 4 · La API

```powershell
uvicorn api.main:app --reload
```

Abre `http://localhost:8000/docs` y prueba:

1. `GET /usuarios_demo` → copia un usuario y llama `GET /recomendar/{usuario}`. Mira `segmento`, `peso_als` y la
   columna `porque_te_gusto` de cada recomendación.
2. En el mismo endpoint cambia `metodo` a `popular`, `als` y `contenido` y compara.
3. `POST /recomendar` con `{"le_gustaron": [1, 3114, 4886]}` (Toy Story 1 y 2, Monsters Inc.): un usuario que no
   existía recibe recomendaciones colaborativas sin re-entrenar (*fold-in*).
4. `POST /usuario_nuevo` con `{"generos": ["Terror"]}` o `{"generos": ["ciencia ficcion", "Animación"]}`: arranque en
   frío solo con géneros (acepta nombres en español, con o sin tildes).
5. `GET /similares/4993` (El Señor de los Anillos 1) con `metodo` = `contenido`, `als` o `hibrido`.
6. `GET /buscar?q=matrix` para encontrar el `movieId` de una película.

Desde otra ventana de PowerShell:

```powershell
Invoke-RestMethod "http://localhost:8000/similares/2571?n=5"
Invoke-RestMethod -Method Post -Uri http://localhost:8000/usuario_nuevo -ContentType "application/json" -Body '{"generos": ["Comedia", "Romance"]}'
```

Para cerrarla: **Ctrl + C**.

## 5 · La app

```powershell
streamlit run app/app.py --server.port 8611
```

Pestañas: usuario existente (con el «porque te gustó…»), usuario nuevo (por géneros o eligiendo películas), películas
similares y resultados.

## 6 · Re-entrenar todo (≈ 1 h en 6 núcleos compartidos)

```powershell
python -m recomendador.entrenar
```

1. Convierte los CSV a Parquet por bloques (pico de RAM bajo) en `data/processed/`.
2. **Validación** (corte 2017-07-01): 12 configuraciones de ALS, 8 de contenido, 20 del híbrido y 3 de EASE. Cada
   resultado se guarda en `reports/busqueda_*.csv` al terminar: si se corta, al relanzar continúa donde quedó.
3. **Prueba** (corte 2018-01-01): re-ajusta todo con los hiperparámetros elegidos y evalúa una sola vez →
   `reports/metricas_prueba.csv` y `reports/metricas_por_usuario.parquet`.
4. **Modelo final** con todos los datos → `models/` (lo que usan la API y la app; incluye los 3 archivos con datos de MovieLens).
5. Figuras → `reports/figures/`.

RAM: pico ≈ 3 GB (EASE invierte una matriz de 16 mil × 16 mil). Para volver a correr solo la prueba, deja
`reports/hiperparametros.json`; para repetir la búsqueda, bórralo junto con `reports/busqueda_*.csv`.

## 7 · Notebooks

```powershell
jupyter lab notebooks
```

1. `01_datos_y_particion_temporal`: los datos, cuánto futuro se cuela con *leave-last-one*, los segmentos.
2. `02_modelos_y_busqueda`: los modelos, la búsqueda de hiperparámetros, similitud por contenido vs. colaborativa, fold-in.
3. `03_evaluacion_por_segmento_y_api`: métricas por segmento con bootstrap pareado (guarda `reports/bootstrap_pareado.csv`), más allá de la exactitud, API.

---

## Preguntas de entrevista (con respuesta)

**1. ¿Por qué no hiciste una partición aleatoria?**
Porque filtra el futuro: el modelo aprende de calificaciones de 2019 para «predecir» una de 2005 y sabe qué películas
se van a volver populares. Incluso *leave-last-one* por usuario filtra: en el notebook 1 mido que, en promedio,
el 53,8 % del entrenamiento ocurre después del momento que se quiere predecir. Con un corte global por fecha todos comparten
el mismo reloj, que es como funciona en producción.

**2. ¿Por qué tratar las calificaciones como señal implícita?**
Porque el objetivo es ordenar qué va a ver y disfrutar, no predecir la nota (el RMSE del premio Netflix no mide eso).
Calificar ≥ 4 es «lo vio y le gustó», que es lo que tiene la mayoría de las empresas (reproducciones, compras, clics).

**3. ¿Qué hace ALS y qué es α?**
Factoriza la matriz usuario × película en vectores de usuario y de película. Cada positivo tiene confianza α y los
ceros (no vistos) confianza 1: no son «no le gusta», son «no sabemos». Alterna: fija películas y resuelve usuarios
por mínimos cuadrados, y al revés. La librería `implicit` lo hace en paralelo con gradiente conjugado.

**4. ¿Cómo recomiendas a un usuario que no estaba en el entrenamiento?**
Tres niveles: (a) solo géneros → contenido con un prior de popularidad; (b) algunas películas → *fold-in* del ALS
(un paso de mínimos cuadrados con los factores de las películas fijos, milisegundos) mezclado con contenido; (c) con
historia larga, el híbrido le da casi todo el peso al ALS. El peso es w = n/(n + k), con k elegido en validación.

**5. ¿Por qué el híbrido lleva popularidad reciente?**
Porque ALS y contenido no saben qué es tendencia *ahora*: en una partición temporal la gente ve estrenos, y la
popularidad de los últimos 180 días es una línea base sorprendentemente difícil de ganar. Sumarla con un peso β
(elegido en validación) mejora el NDCG de validación de 0,234 a 0,250 (con k = 30).

**6. ¿Qué es NDCG@10 y por qué no solo precisión?**
NDCG premia los aciertos según la posición (1/log2(posición + 1)) y se normaliza por la mejor lista posible: acertar
en el puesto 1 vale más que en el 10. Precision@10 y Recall@10 no distinguen el orden. MAP también premia el orden,
pero cae muy rápido con cada fallo.

**7. ¿Por qué mides cobertura, novedad y sesgo de popularidad?**
Porque recomendar siempre los mismos éxitos da buen NDCG (a la gente le gustan los éxitos) pero no aporta nada: el
usuario ya los conoce y el 80 % del catálogo nunca se muestra. La popularidad reciente cubre el 1,4 % del catálogo
y el híbrido el 6,2 % (el ALS, 12,7 %): el híbrido gana exactitud a costa de descubrimiento, y bajar β lo reequilibra.

**8. ¿Por qué evaluar por segmento?**
Porque el promedio esconde a quién le sirve cada modelo. Los usuarios con muchos positivos dominan el promedio de los
«existentes», pero el problema difícil (y donde el contenido aporta) es el arranque en frío.

**9. ¿Qué es EASE y por qué no lo usas en la API?**
Es un autoencoder lineal item-item con solución cerrada: B = I − P·diag(1/diag(P)), P = (XᵀX + λI)⁻¹. Suele ganarle
a ALS en MovieLens y aquí empata con el híbrido en usuarios existentes (NDCG 0,167 vs. 0,166), aunque pierde en usuarios nuevos (0,385 vs. 0,415). Pero B es catálogo × catálogo (≈ 1 GB en float32) y no admite catálogos de
cientos de miles de ítems; lo dejo como referencia de cuánto se podría ganar.

**10. ¿Qué fugas de información quedan?**
El Tag Genome se calculó en 2019 con etiquetas y calificaciones de todo el periodo, así que el modelo de contenido ve
algo del futuro (lo documento en la ficha). La alternativa limpia sería reconstruir los vectores con las etiquetas de
`tags.csv` anteriores al corte.

**11. ¿Cómo lo pondrías en producción?**
Re-entrenar ALS cada noche (≈ 2 min con todo MovieLens), servir factores en memoria (o en un índice ANN como FAISS si
el catálogo crece), *fold-in* en línea para usuarios nuevos y eventos del día, y una prueba A/B midiendo
reproducciones, no NDCG offline. Vigilar cobertura y sesgo de popularidad como métricas de salud.
