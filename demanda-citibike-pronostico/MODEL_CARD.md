# MODEL_CARD · Pronóstico de demanda de bicicletas (Citi Bike)

## Resumen

Modelo de **regresión global LightGBM** que predice el número de **salidas de bicicletas por estación y hora** en el sistema Citi Bike de Nueva York, **para el día siguiente**. Un único modelo sirve para las **250 estaciones** con más actividad (la estación es una característica categórica), combinando rezagos de demanda de 24 h o más, calendario, feriados y clima horario.

- **Tarea**: regresión (demanda horaria, objetivo no negativo; las predicciones se recortan a 0).
- **Horizonte**: el pronóstico se emite a medianoche del día D para las 24 horas de D+1 (horizonte de 1 a 24 h). Solo usa información disponible a esa hora.
- **Tipo de modelo**: LightGBM (gradient boosting), un modelo global + dos modelos por cuantil (p10/p90) para intervalos.
- **Modelos de comparación** (mismos cortes y filas): el mismo modelo sin clima (ablación) y un modelo a 1 hora que usa también la hora anterior.
- **Datos**: 44 302 799 viajes de Citi Bike de 2024 (12 meses) + clima horario de Open-Meteo.
- **Granularidad**: estación × hora; 250 estaciones × 8 784 horas = 2 196 000 filas (2 112 000 de modelado tras descartar las dos primeras semanas sin historia).
- **Artefactos**: `models/modelo_lgbm.joblib` (día siguiente, el que sirven la API y la app) y `models/modelo_1h.joblib` (comparación). La API y la app también leen `data/procesados/demanda_horaria.parquet`, `data/procesados/estaciones.csv` y `data/raw/clima_nyc_horario.csv`, que se versionan (8 MB en total).

## Uso previsto

Planificación nocturna del **rebalanceo de bicicletas**: anticipar en qué estaciones y horas de mañana habrá picos de salidas, para llevar bicicletas antes de que la estación se vacíe.

**No usar** para: predecir una bicicleta o un usuario individual; estaciones fuera de las 250 modeladas o estaciones nuevas; decisiones con impacto legal o de seguridad. El modelo a 1 hora no sirve para la planificación nocturna (necesita la demanda de la hora anterior).

## Entrenamiento y evaluación

- **Partición temporal estricta**: backtesting de origen móvil con 4 cortes semanales (5 nov – 3 dic de 2024) + test final = 28 días: del 3 de diciembre de 2024 a las 23:00 al 31 de diciembre de 2024 (673 horas por estación, 168 250 horas-estación), nunca visto. Validación de 14 días antes de cada corte para *early stopping*.
- **Sin fuga de información**: todas las variables de demanda tienen un rezago mínimo de 24 h. Una prueba automática altera las 24 h previas a una hora y verifica que las variables del modelo no cambian.
- **Hiperparámetros fijos**: LightGBM con `PARAMS_LGBM` de `src/demanda_citibike/modelos.py` (tasa de aprendizaje 0,05, 63 hojas, hasta 600 árboles, `subsample` y `colsample_bytree` 0,8, semilla 42; 400 árboles en los modelos de cuantil). No hubo búsqueda de hiperparámetros; solo el número de árboles se ajusta con *early stopping* en los 14 días previos a cada corte. Una búsqueda sobre los cortes del backtesting podría mejorar algo el resultado.
- **Líneas base** (también disponibles la noche anterior): estacional (misma hora de la semana anterior, t−168) y "ayer" (t−24).
- **Métricas**: MAE, RMSE, WAPE (robusto a ceros) y sesgo; por hora, por estación, por día y en horas de lluvia; cobertura del intervalo 80 %.
- **Clima**: se usa el clima **real** como sustituto del pronóstico meteorológico a 24 h.

## Características

| Grupo | Variables |
|---|---|
| Rezagos (≥ 24 h) | `lag_24`, `lag_48`, `lag_168`, `lag_336` |
| Medias móviles hasta ayer | `media_24_d1`, `media_168_d1` (sobre la serie desplazada 24 h) |
| Calendario | `hora`, `dia_semana` (categóricas), `mes`, `es_fin_de_semana`, `es_feriado` |
| Clima | `temperatura_c`, `precipitacion_mm`, `lluvia_mm`, `viento_kmh` |
| Identidad | `estacion_cat` (categórica, 250 niveles) |

## Resultados clave

Test final (28 días: del 3 de diciembre de 2024 a las 23:00 al 31 de diciembre de 2024). Fuente: `reports/resultados.json`.

| Modelo | MAE | WAPE | Sesgo |
|---|---:|---:|---:|
| Base estacional (t−168) | 3,64 | 72,9 % | +7,4 % |
| Base ayer (t−24) | 3,10 | 62,0 % | +1,7 % |
| Día siguiente sin clima (ablación) | 2,91 | 58,2 % | +22,6 % |
| **Día siguiente (este modelo)** | **2,53** | **50,6 %** | +7,4 % |
| A 1 hora (comparación, otro horizonte) | 2,12 | 42,3 % | +3,7 % |

- **Mejora frente a la base estacional**: −30,6 % de WAPE en el test; en el backtesting 38,4 % frente a 54,4 % (gana en las 4 semanas).
- **Aporte del clima (ablación)**: sin clima el WAPE sube de 50,6 % a 58,2 %; en horas con lluvia, de 61,9 % a 90,9 %.
- **Intervalo p10–p90**: cobertura real **75,1 %** (nominal 80 %).
- **Importancia SHAP** (media de |SHAP|): historia de demanda 48,5 %, clima 24,6 %, calendario 20,0 %, estación 6,9 %. La variable más influyente es la temperatura (17,4 %).
- **Rebalanceo** (pico = salidas por encima del percentil 90 de la estación): alertando con la media, 717 falsas alarmas y 2 191 picos perdidos frente a 2 768 y 2 230 de la base (recall 28,7 %); alertando con el cuantil p90, recall 58,1 % con 3 919 falsas alarmas. Conviene la regla p90 si un pico perdido cuesta más de 3,54 falsas alarmas.
- Antes de recortar a 0, el 2,61 % de las predicciones del test eran negativas.

## Limitaciones y sesgos

- Las **250 estaciones más activas** (41,0 % de los viajes) no representan a las estaciones pequeñas o nuevas (sesgo hacia zonas de alta demanda). Se eligieron con el año completo; con enero–noviembre coinciden 247 de 250.
- El **clima real** se usa como pronóstico; con un pronóstico meteorológico real el error sería mayor, sobre todo en días de lluvia.
- Parte del peso de la temperatura es estacional (frío de diciembre); el modelo no vio ningún diciembre en el entrenamiento.
- No se modela el **stock de bicicletas por estación** (solo salidas); el análisis de rebalanceo usa supuestos explícitos y una razón de costos *r* desconocida.
- **Feriados atípicos** (Nochebuena y Navidad: WAPE diario de 190 % y 271 %), cierres de estaciones (3 estaciones sin salidas en diciembre) y eventos extraordinarios no están en las variables.
- El modelo tiende a la media: subestima los picos, lo que reduce el recall de las alertas con la predicción media.
- Datos de una sola ciudad (NYC) y un solo año (2024): la generalización requiere reentrenamiento.

## Entorno

- Python 3.14; versiones fijadas en `requirements.txt` (`lightgbm` 4.7.0, `shap` 0.54.1, `duckdb` 2.0.0, `pandas` 3.1.0, `holidays` 0.107, `fastapi`, `streamlit`).
- Entrenamiento en CPU (Ryzen 5 4600G, 32 GB RAM): el pipeline completo (3 conjuntos de variables × 5 entrenamientos + cuantiles + SHAP) tarda ~8 minutos.
