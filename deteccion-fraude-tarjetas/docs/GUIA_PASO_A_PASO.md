# Guía paso a paso: detección de fraude con tarjetas

Esta guía explica qué hace el proyecto, en qué orden y por qué. Al final hay preguntas de entrevista con sus respuestas. Las cifras exactas están en el README (que se genera desde `reports/`).

## 1. Preparar el entorno

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

PyTorch se instala en su versión CPU (índice de `download.pytorch.org/whl/cpu` en `requirements.txt`). No hace falta GPU: el autoencoder entrena en menos de un minuto.

## 2. Descargar los datos

```bash
python descargar_datos.py
```

- Baja el Parquet público de OpenML (dataset 1597, ~73 MB) a `data/raw/`.
- Es **reanudable**: escribe en un archivo `.part` y, si la conexión se corta, el siguiente intento pide solo los bytes que faltan (cabecera HTTP `Range`). Si el Parquet no está disponible, usa el ARFF de OpenML.
- **No usa `fetch_openml`**, porque OpenML marca `Time` como identificador de fila y lo descarta, y sin `Time` no hay partición temporal. Es un buen ejemplo de por qué conviene revisar qué columnas devuelve una API de datos.
- Valida 284.807 filas, 492 fraudes y las 31 columnas esperadas.

## 3. Partición temporal

`src/fraude/datos.py` corta por cuantiles de `Time` en 60 % / 20 % / 20 %, **sin barajar**. El corte se hace por *valores* de tiempo, no por posición, así que todas las transacciones del mismo segundo (incluidos los duplicados exactos) caen en la misma partición. Las pruebas `tests/test_particion.py` verifican que los rangos no se solapan.

## 4. Entrenar

```bash
python -m fraude.entrenar
```

1. **Líneas base:** ordenar por monto y una regresión logística estandarizada.
2. **No supervisados:**
   - Isolation Forest con todo el entrenamiento (sin etiquetas) y solo con normales.
   - LOF en modo `novelty=True` con 40.000 normales.
   - Autoencoder 29 → 20 → 10 → 20 → 29 en PyTorch, entrenado solo con normales y con parada temprana sobre normales reservadas.
3. **LightGBM** de tres formas: sin ajuste, con `scale_pos_weight` = normales/fraudes y con submuestreo 1:20. La parada temprana usa la AUPRC de validación.
4. **Híbrido:** LightGBM con dos variables más, el puntaje de Isolation Forest y el log del error del autoencoder, calculados **fuera de pliegue** para el entrenamiento.
5. Guarda los puntajes de validación y prueba de los 10 modelos en `reports/predicciones_*.parquet` y el modelo servido en `models/`.

## 5. Evaluar (sin reentrenar)

```bash
python -m fraude.evaluar
```

Lee las predicciones guardadas y calcula:
- AUPRC, ROC-AUC, precisión@k y recall a precisión fija, con IC bootstrap pareado.
- Costo con umbral elegido en validación, la regla de pérdida esperada, la curva de costo y la sensibilidad.
- La curva de pocas etiquetas, SHAP del booster guardado y la deriva entre días.

Al final escribe `reports/resultados.json` y `models/metadata.json`.

## 6. Figuras, notebooks, README

```bash
python -m fraude.graficos
python scripts/generar_notebooks.py
python scripts/generar_readme.py
pytest -q -p no:cacheprovider
```

El README y la ficha del modelo se **generan** desde `reports/`. Una prueba comprueba que sus cifras coinciden con las que se recalculan desde las predicciones.

## 7. Servir

```bash
uvicorn api.main:app --port 8000      # POST /puntuar, GET /salud, docs en /docs
streamlit run app/app.py
```

La API solo necesita `models/` (menos de 3 MB): funciona en un clon sin descargar datos.

---

## Preguntas de entrevista

**1. ¿Por qué AUPRC y no ROC-AUC como métrica principal?**
La ROC-AUC mide qué tan bien se ordena un fraude frente a una normal al azar, y la tasa de falsos positivos se divide entre ~57.000 normales. Un modelo puede generar cientos de falsas alertas y seguir con ROC-AUC de 0,95. La AUPRC promedia la **precisión** (qué fracción de las alertas es fraude) a lo largo del recall, que es justo lo que le cuesta al equipo de revisión. En este proyecto, Isolation Forest tiene ROC-AUC ≈ 0,95 y ningún fraude entre sus primeras 80 alertas. Su referencia al azar es la tasa de fraude (0,13 % en prueba), no 0,5.

**2. ¿Por qué nunca exactitud?**
Porque «todo es normal» acierta el 99,87 % en prueba y no detecta ningún fraude.

**3. ¿Por qué una partición temporal y no aleatoria?**
- En producción el modelo puntúa transacciones **futuras**.
- Los fraudes llegan en ráfagas: el mismo atacante repite un patrón en minutos. Con una partición aleatoria, el entrenamiento contiene «hermanos» de los fraudes de prueba y la métrica se infla.
- Además, todo lo que se decide (parada temprana, umbral, calibración) usa validación, que está en el tiempo entre entrenamiento y prueba.

**4. ¿Cómo funciona Isolation Forest y por qué rinde mal aquí?**
Construye árboles con cortes aleatorios. Un punto raro queda aislado en pocas divisiones, así que la profundidad promedio es el puntaje (menos profundidad, más anómalo). Detecta lo **raro**, no lo **fraudulento**: muchas compras legítimas son raras (montos altos, combinaciones inusuales de componentes) y muchos fraudes son compras pequeñas que se parecen a las normales. Por eso su ROC-AUC es alta, pero sus primeras alertas son legítimas.

**5. ¿Cómo se usa un autoencoder para detectar anomalías?**
Se entrena para reconstruir solo transacciones normales, pasando por un cuello de botella de 10 dimensiones. Lo que se parece a lo normal se reconstruye bien; lo distinto, mal. El **error de reconstrucción** es el puntaje. Aquí rinde poco porque `V1`–`V28` ya son componentes PCA decorrelacionadas: el banco ya quitó la estructura lineal, y el error termina pareciéndose a una distancia al centro. Se sirve en NumPy (los pesos están en un `.npz`) para que la API no dependa de torch.

**6. ¿Qué diferencia hay entre entrenar «sin etiquetas» y «solo con normales»?**
- *Sin etiquetas* (outlier detection): se ajusta con todo y se asume que la contaminación es pequeña.
- *Solo normales* (novelty detection): se ajusta con transacciones que se sabe que son legítimas, que en un banco son casi gratis (las que nadie reclamó).

Con 0,2 % de fraude, la diferencia práctica es mínima: las dos variantes de Isolation Forest dan casi la misma AUPRC.

**7. ¿Pesos o submuestreo para el desbalance?**
- **Pesos:** usan todas las normales y multiplican el gradiente de los fraudes por ~474.
- **Submuestreo:** descarta normales (aquí se dejan 20 por fraude), así que entrena más rápido pero pierde información.

En este proyecto ninguno supera de forma significativa a LightGBM sin ajuste en AUPRC, porque el ranking casi no cambia. Los dos **distorsionan las probabilidades**: si se van a usar como probabilidades (por ejemplo, p × monto), hay que calibrarlas. Dal Pozzolo et al. (2015), los autores del dataset, derivan la corrección exacta para el submuestreo.

**8. ¿Cómo eliges el umbral?**
Con el **costo** y en **validación**:
- Cada alerta cuesta 5 EUR de revisión.
- Cada fraude no detectado cuesta su `Amount`.

Se recorre el ranking de validación, se toma el umbral de mínimo costo y se aplica tal cual en prueba. Elegirlo en prueba sería sobreajustar el umbral a los datos con los que se reporta.

**9. ¿Qué es la regla de pérdida esperada y por qué es más robusta?**
Alerta si `p_calibrada × Amount > costo de revisión`, que es la decisión de Bayes con costos que dependen de cada transacción. Una compra de 2.000 EUR con 1 % de probabilidad se revisa, y una de 2 EUR con 50 % no. En la sensibilidad mantiene el mayor ahorro cuando revisar es caro, porque concentra las revisiones donde hay dinero en juego. Requiere probabilidades calibradas: aquí se usa Platt (2 parámetros) ajustado en validación.

**10. ¿Cuántas etiquetas necesita el supervisado para superar al no supervisado?**
Una regresión logística con solo **5 fraudes** etiquetados (y todas las normales) ya supera al mejor no supervisado (LOF). LightGBM necesita muchas más y es inestable con pocas: varía mucho entre muestras. Lección: los detectores de anomalías sirven para **arrancar** y priorizar revisiones; en cuanto hay unas decenas de fraudes confirmados, conviene un supervisado simple.

**11. ¿Por qué los puntajes de anomalía del híbrido se calculan fuera de pliegue?**
Si el autoencoder se entrena con las normales de entrenamiento y luego puntúa esas mismas filas, su error sale bajo porque ya las vio. Los fraudes, que nunca vio, salen con error alto. El supervisado aprendería «error alto = fraude» con una separación que en datos nuevos es menor. Fuera de pliegue, cada normal se puntúa con un modelo que no la vio. Aun así, el híbrido no mejora: el LightGBM ya captura esa información con `V1`–`V28`.

**12. ¿Qué te dice SHAP si las variables son PCA anónimas?**
Dice **qué** componente empuja cada decisión (aquí `V14`, `V4` y `V12` en los fraudes), pero no **por qué** en términos de negocio. Sirve para auditar estabilidad y vigilar la deriva de las variables importantes. No sirve para explicarle al cliente por qué se bloqueó su compra.

**13. ¿Cómo mediste la deriva y qué encontraste?**
- PSI y KS por variable entre el día 1 y el día 2, solo en normales.
- Una **validación adversaria**: un clasificador intenta adivinar el día.

Con ROC-AUC ≈ 0,995, los días son casi perfectamente separables, sobre todo por `V1`, `V3` y `V28`. Aun así, la AUPRC del modelo servido apenas cambia entre validación y prueba, porque lo que deriva no es lo que separa el fraude. En producción: PSI semanal y reentrenar si supera 0,25.

**14. ¿Cómo calculas el IC bootstrap de forma eficiente?**
Cada réplica es un vector de **pesos** (cuántas veces salió cada fila). El orden por puntaje se calcula una sola vez por modelo, y las métricas son sumas acumuladas ponderadas. Es **pareado**: todos los modelos usan las mismas filas en cada réplica, así que las diferencias tienen IC propio. Con pesos = 1 coincide con scikit-learn (lo verifica una prueba).

**15. ¿Qué cambiarías en producción?**
- Variables de comportamiento (velocidad de transacciones por tarjeta, distancia al comercio habitual), que el dataset no trae.
- Etiquetas con retraso: los contracargos llegan semanas después.
- Reentrenamiento periódico con monitoreo de PSI y de la precisión de las alertas revisadas.
- Medir la fricción del cliente legítimo bloqueado como un costo más.
