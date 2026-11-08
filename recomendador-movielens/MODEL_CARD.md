# Ficha del modelo · Recomendador híbrido de películas v1.0.0

## Qué hace

Dado un usuario (su historial de películas que le gustaron, solo unas pocas películas o solo los géneros que marcó al registrarse), devuelve:

- las **N películas** que más probablemente verá y calificará con ≥ 4★, sin repetir las ya vistas;
- para cada una, la película de su historial más parecida (**«porque te gustó…»**);
- el **segmento** del usuario y cuánto pesa el componente colaborativo en su mezcla.

También devuelve **películas similares** a una dada, por contenido, por co-consumo (ALS) o mezclando ambas.

## Uso previsto

- **Sí:** filas de «recomendado para ti» y «si te gustó X» en un catálogo de cine, onboarding de usuarios nuevos y demostraciones de portafolio.
- **No:** decisiones con consecuencias para las personas (crédito, empleo, precios), ni inferir atributos sensibles a partir de los gustos.
- **Fuera de contexto:** otro catálogo, otro país u otra época (los datos terminan en noviembre de 2019). Hay que re-entrenar, y la popularidad reciente en especial caduca en semanas.

## Datos

| | |
|---|---|
| Fuente | MovieLens 25M (GroupLens, Harper y Konstan 2015). Uso de investigación y educación, sin redistribución: los datos y los tres archivos del paquete que los contienen no se versionan (ver Reproducibilidad) |
| Tamaño | 25.000.095 calificaciones, 162.541 usuarios, 59.047 películas calificadas, enero 1995 → noviembre 2019 |
| Contenido | 19 géneros + Tag Genome (relevancia de 1.128 etiquetas para 13.816 películas) |
| Señal | **positivo implícito = calificación ≥ 4★** (49,8 % de las calificaciones) |
| Catálogo recomendable | películas con ≥ 5 positivos antes del corte: 17.102 en prueba, 19.937 en el modelo final |
| Validación | entrenar < 2017-07-01 · evaluar 2017-07 → 2017-12 (3.907 existentes + 3.557 nuevos) |
| Prueba | entrenar < 2018-01-01 (22,5 M calificaciones, 11,2 M positivos, 150.543 usuarios) · evaluar 2018-01 → 2019-11, una sola vez: **6.267 usuarios existentes** y **11.710 nuevos** |
| Modelo servido | re-entrenado con todos los datos (12,4 M positivos) y los mismos hiperparámetros |

**Partición temporal global:** el mismo corte de fecha para todos los usuarios, así que ningún dato posterior al corte entra al entrenamiento. Con *leave-last-one* por usuario, el 53,8 % del entrenamiento sería posterior al momento evaluado. **Usuarios nuevos:** sin historia antes del corte; sus 5 primeras calificaciones hacen de onboarding y se evalúa con el resto.

Los datos son anónimos: un id numérico por usuario, sin datos demográficos.

## Modelo

| Componente | Detalle |
|---|---|
| ALS implícito (`implicit` 0.7) | 128 factores, λ = 1, 15 iteraciones, confianza α = 2 sobre pesos BM25 (K1 = 100, B = 0,8; IDF y largo medio del entrenamiento). Usuarios no vistos: *fold-in* exacto por mínimos cuadrados |
| Contenido | vector por película = [Tag Genome centrado y reducido con SVD a 64 dimensiones (72,8 % de la varianza) · géneros] con peso de géneros 0,2; perfil = promedio de lo que le gustó; puntaje = coseno + 0,3·z(log popularidad) |
| Popularidad reciente | log de positivos en los 180 días previos al corte |
| **Híbrido** | `w·z(ALS) + (1 − w)·z(contenido) + 2·z(recientes)`, con `w = n / (n + 30)` y n = positivos del usuario |
| Solo géneros | coseno entre los géneros elegidos y los de cada película + popularidad |

**Selección:** búsqueda acotada en validación (12 configuraciones de ALS, 8 de contenido, 20 del híbrido, 3 de EASE), maximizando NDCG@10. El ALS se eligió con los usuarios existentes y el contenido y el híbrido con todos. **EASE** (λ = 1.000) es solo una referencia de evaluación: su matriz película × película pesa 1,2 GB.

## Métricas en prueba (top-10)

| Segmento (usuarios) | NDCG@10 híbrido [IC 95 %] | ALS | Popularidad reciente | EASE (ref.) |
|---|---:|---:|---:|---:|
| Nuevos, onboarding de 5 (11.710) | **0,415** [0,410–0,420] | 0,367 | 0,364 | 0,385 |
| Pocos, 1-20 positivos (479) | **0,211** [0,190–0,233] | 0,168 | 0,186 | 0,191 |
| Medios, 21-100 (1.977) | 0,160 [0,151–0,169] | 0,150 | 0,110 | 0,165 |
| Muchos, > 100 (3.811) | 0,163 [0,157–0,170] | 0,145 | 0,124 | 0,164 |
| **Existentes, todos (6.267)** | **0,166** [0,161–0,171] | 0,148 | 0,125 | 0,167 |

Híbrido en usuarios existentes: Precision@10 0,142 · Recall@10 0,072 · MAP@10 0,092 · hit rate@10 57 % · cobertura del catálogo 6,2 % · novedad 3,9 bits · diversidad de géneros 0,67 · percentil de popularidad medio 98,0.

**Mejora frente al ALS** (bootstrap pareado por usuario, IC 95 %): nuevos +0,049 [+0,046; +0,051] · pocos +0,044 [+0,031; +0,058] · medios +0,010 [+0,006; +0,014] · muchos +0,019 [+0,016; +0,022]. Todos los intervalos (también frente a popularidad reciente y contenido) están en `reports/bootstrap_pareado.csv`.

**Arranque en frío:** con 5 películas el híbrido logra 0,415. Con solo géneros, 0,225, peor que mostrar lo popular reciente (0,364). Por eso conviene pedir películas, no géneros.

Para contrastar las cifras: con 23 meses de ventana, los usuarios tienen muchos relevantes (mediana de 10 a 15 en existentes y 46 en nuevos), y por eso Precision@10 es relativamente alta y Recall@10 baja. Las cifras no son comparables con artículos que usan particiones aleatorias.

## Limitaciones y riesgos

1. **Fuga leve por el Tag Genome.** GroupLens lo calculó en 2019 con etiquetas, calificaciones y reseñas de todo el periodo, así que el contenido «ve» algo del futuro. El contenido solo es el modelo más débil, así que el efecto en el híbrido debería ser chico, pero no está medido. **Mitigación:** reconstruir los vectores con las etiquetas de `tags.csv` anteriores al corte.
2. **Sesgo de popularidad.** Todos los modelos recomiendan películas más populares que las que el usuario consume (+6,1 percentiles el híbrido, +4,7 el ALS). El término de popularidad reciente (β = 2) sube el NDCG pero baja la cobertura del 12,7 % (ALS) al 6,2 %. Para nuevos, el híbrido cubre solo el 1,6 % del catálogo. Esto puede reforzar a los éxitos («rico se hace más rico») y perjudicar al cine independiente o local.
3. **β en el borde de la rejilla.** El mejor β (2) es el mayor probado, con retorno decreciente (+0,004 de 1 a 2). No se siguió subiendo para no sobreajustar la validación.
4. **Sesgo de quién vuelve.** Solo el 4 % de los usuarios con historia vuelve después del corte, y suelen ser los más activos. El segmento «pocos» tiene 479 usuarios: su intervalo es ancho (±0,02).
5. **Popularidad reciente fija.** En el paquete servido, «reciente» son los 180 días antes de noviembre de 2019. En producción hay que recalcularla a diario.
6. **Cobertura de catálogo.** Las películas con menos de 5 positivos no se recomiendan (el 6,3 % de los positivos de prueba fueron a películas fuera del catálogo). Las nuevas sin genome entran solo por géneros.
7. **Lo que no mide la evaluación offline:** si la persona vio la película *por* la recomendación. La métrica real es una prueba A/B de reproducciones y satisfacción.
8. **Equidad:** no hay datos demográficos, así que no se puede auditar el desempeño por grupo. Solo hay segmentos por actividad.

## Reproducibilidad

- `python descargar_datos.py` (verifica el MD5 oficial) y luego `python -m recomendador.entrenar` (semilla 42, ~1 h la primera vez en 6 núcleos; reanuda si se corta).
- Paquete servido en `models/`: `factores.npz` (factores ALS, IDF BM25, genome reducido, géneros), `demo_usuarios.npz` (factores de 1.500 usuarios anónimos) y `metadata.json` con hiperparámetros y métricas, que sí se versionan (12 MB). `peliculas.parquet`, `demo_positivos.npz` y `demo_vistos.npz` llevan títulos y calificaciones de MovieLens, que no se pueden redistribuir: se generan con `python generar_demo.py` tras `python descargar_datos.py`. Sin ellos la API responde `503` con ese mensaje.
- 43 pruebas automáticas: métricas calculadas a mano, partición sin fuga, modelos con datos sintéticos de estructura conocida (incluido el fold-in con BM25), el aviso por paquete incompleto y la API con el paquete completo.

## Mantenimiento recomendado

Re-entrenar el ALS cada noche (≈ 1 min con todo MovieLens). Recalcular la popularidad reciente cada día. Vigilar la cobertura del catálogo, el percentil de popularidad medio y la tasa de clics por segmento. Re-ajustar k y β si cambia la mezcla de usuarios nuevos y antiguos.
