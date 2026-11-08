# Guía Paso a Paso — Pronóstico de demanda de bicicletas (Citi Bike + clima)

Esta guía explica cómo reproducir el proyecto de punta a punta, ejecutar las pruebas y preparar preguntas de entrevista sobre pronóstico de series temporales a escala.

---

## 1 · Requisitos

- **Python** `>= 3.10` (el proyecto usa Python 3.14).
- **RAM**: 8 GB (la ingesta de 44,3 M de viajes se hace con DuckDB en streaming).
- **Disco**: ~17,5 GB (8,7 GB de zips + 8,7 GB de CSV extraídos; se pueden borrar los zips tras extraer) y ~8 MB del Parquet procesado.
- Sin GPU: LightGBM entrena en CPU.

---

## 2 · Instalación

```powershell
# En la carpeta raíz del proyecto (demanda-citibike-pronostico)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

---

## 3 · Descarga de datos (reanudable)

```powershell
python descargar_datos.py
```

- Descarga **12 meses** de viajes de Citi Bike (`https://s3.amazonaws.com/tripdata/AAAAMM-citibike-tripdata.zip`, 8,7 GB comprimidos) y los extrae a `data/raw/` (otros 8,7 GB de CSV).
- Descarga el **clima horario** de Open-Meteo para NYC (temperatura, precipitación, lluvia, viento) a `data/raw/clima_nyc_horario.csv`.
- Es **idempotente y reanudable**: salta archivos ya presentes y escribe con `.parcial` (atómico).

---

## 4 · Pipeline completo

```powershell
python -m demanda_citibike.pipeline
```

Qué hace, en orden:

1. **Ingesta**: DuckDB agrega los 44,3 M de viajes de 2024 a salidas/llegadas por estación × hora, elige las 250 estaciones más activas y guarda `data/procesados/demanda_horaria.parquet`.
2. **Características**: dos conjuntos sin fuga de información. *Día siguiente* (principal): rezagos de 24, 48, 168 y 336 h y medias móviles hasta ayer; *1 hora* (comparación): además la hora anterior. Ambos con calendario (hora, día, mes, feriado) y clima.
3. **Backtesting**: 4 cortes semanales (origen móvil) + test final = últimos 28 días (del 3 de diciembre a las 23:00 al 31 de diciembre de 2024), nunca vistos.
4. **Métricas**: MAE, RMSE, WAPE y sesgo del modelo frente a las líneas base.
5. **Intervalos**: modelos por cuantil p10/p90 (cobertura nominal 80 %).
6. **Ablación de clima** (el modelo día siguiente con y sin clima), **importancia SHAP** y **análisis de negocio** (alertas de rebalanceo con costos).
7. Guarda `reports/resultados.json`, `reports/figures/*.png`, predicciones y el modelo en `models/modelo_lgbm.joblib`.

---

## 5 · Pruebas

```powershell
pytest -q
```

Las 41 pruebas corren sin internet: las de código con **datos sintéticos**: validan que los rezagos no filtren información del futuro (incluido que el modelo día siguiente no use las últimas 24 h), la partición temporal, las métricas, las líneas base y el análisis de rebalanceo; las de `test_readme_cifras.py` recalculan las métricas desde `reports/` y comprueban que las cifras del README y la ficha coinciden.

---

## 6 · API y app

```powershell
uvicorn api.main:app --reload            # API en http://127.0.0.1:8000
streamlit run app/streamlit_app.py       # App en http://localhost:8501
```

Endpoints de la API:

- `GET /salud` → estado.
- `GET /estaciones` → lista de estaciones.
- `GET /pronostico/{estacion}?fecha=2024-12-20` → pronóstico horario del día.

La API y la app funcionan en un clon sin descargar los viajes: el repositorio incluye el modelo, `data/procesados/demanda_horaria.parquet` y el clima. `tests/test_api.py` prueba `/salud`, `/estaciones` y `/pronostico` con `TestClient`.

---

## 7 · Resolución de problemas

| Síntoma | Causa | Solución |
|---|---|---|
| `FileNotFoundError: No hay CSVs de viajes` | No se ejecutó la descarga | Ejecuta `python descargar_datos.py` |
| `FileNotFoundError: clima` | Falta el clima | Se descarga con `descargar_datos.py` |
| Error 404 al descargar viajes | Nombre de archivo antiguo (`.csv.zip`) | El patrón correcto es `AAAAMM-citibike-tripdata.zip` (sin `.csv`) |
| API responde `503` o `/salud` dice `no_listo` | Falta el modelo o la demanda procesada | Ejecuta `python descargar_datos.py` y luego `python -m demanda_citibike.pipeline` |
| Memoria alta | Cargar los CSV en pandas | La ingesta usa DuckDB; nunca se carga el CSV completo en pandas |

---

## 8 · Preguntas de entrevista (con respuesta)

### 1. ¿Por qué un solo modelo global para 250 estaciones en lugar de 250 modelos?

Un modelo global comparte información entre estaciones (las que se parecen se benefician mutuamente), es más simple de mantener y se entrena una sola vez. La estación se codifica como variable categórica, así el modelo puede aprender patrones por estación. Además evita entrenar 250 modelos (costoso e inestable para estaciones con pocos datos). La contrapartida es que el modelo debe ser suficientemente expresivo; se compensa con más árboles y la interacción estación × hora que LightGBM aprende implícitamente.

### 2. ¿Cómo evitas la fuga de información en los rezagos?

Los rezagos y las medias móviles se calculan con `groupby(estación).shift(lag)` **antes** de dividir en entrenamiento/prueba, de modo que para la hora `t` solo se usan horas `< t`. La media móvil se calcula sobre la serie *desplazada* (`shift(1).rolling(...)`) para que no incluya la hora objetivo. En validación, las particiones son temporales estrictas: entrenar con el pasado y evaluar solo el futuro.

### 3. ¿Por qué WAPE en lugar de MAPE?

El MAPE divide cada error por el valor real `|y−ŷ|/y`; cuando una estación tiene 0 salidas en una hora (muy común de madrugada), el MAPE explota. El WAPE pondera por el volumen total (`Σ|y−ŷ| / Σy`), es robusto a ceros y da un porcentaje interpretable para el negocio.

### 4. ¿Por qué backtesting de origen móvil y no k-fold aleatorio?

En series temporales, el k-fold aleatorio filtra información del futuro hacia el entrenamiento (fuga). El backtesting de origen móvil entrena solo con datos anteriores a cada corte y evalúa la semana siguiente, replicando el uso real (pronosticar mañana con lo de hasta hoy). El mes final se reserva y **nunca** participa en ningún entrenamiento.

### 5. ¿Qué aporta el clima al modelo y cómo lo mediste?

Con una **ablación**: entreno el mismo modelo día siguiente con y sin las cuatro variables de clima, con los mismos cortes temporales, y comparo el error. Sin clima el WAPE del test sube de 50,6 % a 58,2 % (y de 61,9 % a 90,9 % en horas con lluvia). La importancia de variables (SHAP o *gain*) no responde a esta pregunta: dice cuánto mueve una variable la predicción, no si mejora el error. Ojo: en producción se usaría un **pronóstico** del tiempo; aquí se usa el clima real y se documenta.

### 6. ¿Cómo generas intervalos de predicción con LightGBM?

Se entrena un modelo por cuantil (`objective="quantile"`, `alpha=0.1` y `0.9`). La diferencia entre ambos define un intervalo nominal del 80 %. Luego se mide la **cobertura real** (fracción de observaciones que caen dentro) y se reporta; si está cerca del 80 %, el intervalo está calibrado.

### 7. ¿Cómo cuantificas el valor de negocio (rebalanceo)?

Un **pico** es una hora en que las salidas superan el percentil 90 de esa estación en el entrenamiento. Cada noche se alerta cada estación-hora de mañana cuyo pronóstico supera ese umbral. Hay dos errores: **falsa alarma** (un viaje de camión en vano) y **pico perdido** (estación vacía). Como no tengo los costos reales, defino costo = falsas alarmas + r × picos perdidos y analizo la sensibilidad a r. Alertando con la media, el modelo domina a la base (menos falsas alarmas y menos picos perdidos) pero detecta solo el 28,7 % de los picos; alertando con el cuantil p90 detecta el 58,1 % y es más barato si un pico perdido cuesta más de 3,5 falsas alarmas. Contar solo las falsas alarmas evitadas como "viajes ahorrados" sería engañoso.

### 8. ¿Por qué DuckDB para la ingesta y no pandas?

Citi Bike 2024 tiene 44,3 M de viajes en CSV (8,7 GB). Cargarlos en pandas agotaría la RAM. DuckDB lee los CSV en streaming y agrega `GROUP BY estación, hora` con SQL vectorizado, en pocos minutos y con memoria acotada (`SET memory_limit`). El resultado (250 estaciones × 8 784 horas de 2024, año bisiesto = 2 196 000 filas) ya cabe en pandas para el modelado.

### 9. ¿Cómo elegiste las 250 estaciones?

Por **salidas totales** en 2024: las 250 con más salidas (de 2 378) concentran el 41,0 % de los viajes. Deja fuera estaciones pequeñas o nuevas (limitación explícita). Como la selección usa el año completo, comprobé que con solo enero–noviembre coinciden 247 de 250: la fuga por selección es despreciable.

### 10. ¿Qué métrica usarías para decidir "el modelo sirve" frente a la línea base?

La mejora relativa del WAPE sobre la línea base estacional (misma hora de la semana anterior). La línea base estacional es fuerte en series con fuerte periodicidad semanal; si el modelo no la supera de forma consistente (en el test final y en los 4 cortes de backtesting), no aporta valor. También se mira el sesgo (para no sobre/subestimar sistemáticamente la flota) y la cobertura de los intervalos.

### 11. ¿Cómo manejarías el arranque en frío de una estación nueva (sin historia de rezagos)?

Con una estación nueva no hay `lag_168`. Opciones: usar la media de estaciones similares (por zona o perfil de demanda), un modelo de *cold start* basado solo en calendario/clima/ubicación, o degradar a la línea base global de la ciudad hasta acumular historia. Aquí no se modela porque las 250 estaciones seleccionadas tienen historia completa.

### 12. ¿Por qué la estación se trata como categórica y no como numérica?

La identidad de estación no tiene orden: la estación 42 no es "el doble" de la 21. Tratarla como numérica impone una relación ordinal inexistente. Como categórica, LightGBM aprende divisiones óptimas por estación y puede combinarla con la hora (interacción estación × hora) para capturar patrones locales.

### 13. ¿Por qué el resultado principal no es el modelo con menor error?

Porque el modelo a 1 hora (WAPE 42,3 %) usa la demanda de la hora anterior, que no existe cuando se planifica el rebalanceo la noche antes. El horizonte lo fija la decisión de negocio: el modelo principal pronostica las 24 horas de mañana con información hasta la medianoche (solo rezagos de 24 h o más) y logra 50,6 %. La diferencia entre ambos es el costo de anticiparse un día; presentar el de 1 hora como resultado sería una fuga de información respecto al uso real.

### 14. ¿Afinaste los hiperparámetros de LightGBM?

No: son fijos (`PARAMS_LGBM` en `modelos.py`: tasa de aprendizaje 0,05, 63 hojas, 600 árboles como máximo, `subsample` y `colsample_bytree` 0,8, semilla 42). Lo único que se ajusta solo es el número de árboles, con *early stopping* sobre los 14 días previos a cada corte. Lo documento a propósito: no usé el test ni el backtesting para elegir parámetros, así que las cifras no están infladas por la búsqueda, pero tampoco son el mejor resultado posible. El siguiente paso sería una búsqueda (Optuna) con validación de origen móvil dentro del entrenamiento, y reportar cuánto mejora el WAPE frente a la configuración fija.
