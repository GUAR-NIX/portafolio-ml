# Ficha del modelo · Priorización de llamadas v1.0.0

## Qué hace

Dado un cliente del banco (datos personales básicos, productos que tiene, su historia en campañas anteriores, el tipo de teléfono, el momento planificado de la llamada y el contexto económico del mes), devuelve:

- la **probabilidad calibrada** de que contrate un depósito a plazo si se le llama;
- **si conviene llamarlo** (probabilidad ≥ umbral de costo/beneficio, 11 % por defecto);
- el **valor esperado** de la llamada en USD (`p × beneficio − costo`);
- las **3 razones** principales (SHAP).

## Uso previsto

- **Priorizar** a quién llamar en una campaña de telemercadeo de depósitos a plazo y decidir **cuántos** llamar según el costo de la llamada o la capacidad del call center.
- **No** sirve para decidir créditos, precios ni ningún trato diferente a un cliente: solo ordena una lista de llamadas comerciales.
- Fuera de contexto (otro banco, otro país, otro producto, otro momento económico) hay que **re-entrenar**: el modelo depende mucho del régimen económico de 2008–2010.

## Datos

| | |
|---|---|
| Fuente | UCI Bank Marketing, `bank-additional-full.csv` (Moro, Cortez y Rita, 2014) · CC BY 4.0 |
| Periodo | mayo 2008 → noviembre 2010 (el año se reconstruye del orden de las filas) |
| Tamaño | 41.188 filas − 12 duplicados exactos = **41.176 clientes** · 11,3 % aceptan |
| Entrenamiento | 32.940 (80 %, estratificado, semilla 42), con 5 pliegues estratificados para todas las decisiones |
| Prueba | 8.236 (20 %), usada una sola vez |
| Estrés temporal | pasado hasta 2009-04 (30.421, 6,3 % éxito) → futuro 2009-05 a 2010-11 (10.755, 25,4 %) |

Los datos están anonimizados: no identifican a ninguna persona.

> **Ojo:** las métricas de abajo vienen de la partición estratificada, optimista frente a la prueba de estrés temporal (AUC 0,70 con el modelo fijo; ver Limitaciones).

### Variables excluidas por fuga de información

| Variable | Por qué |
|---|---|
| `duration` (duración de la llamada) | Se conoce cuando la llamada terminó; 0 s ⇒ «no» seguro. Con ella el AUC sería 0,952 (irreal) |
| `campaign` (nº de llamadas en esta campaña, incluida la última) | Depende del resultado (se deja de llamar cuando acepta) y no se conoce antes de empezar |

## Modelo

- **Variables (19):** edad, ocupación, estado civil, educación, crédito en mora, hipoteca, préstamo personal, tipo de teléfono, mes y día de la semana del contacto, resultado de la campaña anterior, contactado antes (bandera derivada de `pdays = 999`), días desde el último contacto, contactos previos, variación del empleo, IPC, confianza del consumidor, euríbor a 3 meses, nº de empleados.
- **Algoritmo:** LightGBM (200 árboles, 15 hojas, lr 0,03, `min_child_samples` 100, submuestreo 0,8 de filas y columnas), categóricas nativas, **sin pesos de clase ni remuestreo**.
- **Calibración:** Platt sobre el logit, ajustado con las predicciones fuera de pliegue del entrenamiento (como `CalibratedClassifierCV(ensemble=False)`). Elegido frente a isotónica por Brier con validación cruzada (0,07695 vs 0,07710).
- **Umbral:** 11 %, el que maximiza la ganancia fuera de pliegue (teórico: costo/beneficio = 8/80 = 10 %).
- **Selección:** 2 modelos (logística, LightGBM) × 4 estrategias de desbalance (ninguna, pesos, submuestreo, SMOTE-NC) comparados por ganancia calibrada fuera de pliegue; además una regla de negocio como línea base.

## Supuestos de negocio

| Supuesto | Valor | Base |
|---|---:|---|
| Costo por cliente gestionado | 8 USD | 2,6 llamadas promedio por cliente × ~3 USD por llamada |
| Margen por depósito | 80 USD | 5.000 USD a 1 año × 1,6 % de margen neto de intereses |

Son supuestos razonables pero **no** datos del banco. La API permite cambiarlos por petición (`costo_llamada`, `beneficio_exito`) y el umbral se recalcula como su cociente sin re-entrenar.

## Métricas en prueba (8.236 clientes)

| Métrica | LightGBM calibrado | Regresión logística | Regla por segmento |
|---|---:|---:|---:|
| AUC | **0,815** | 0,801 | 0,675 |
| Average precision | **0,490** | 0,444 | 0,246 |
| Brier | **0,0751** | 0,0785 | 0,0898 |
| Brier skill | **0,249** | 0,215 | 0,102 |
| Log loss | **0,265** | — | — |
| Prob. media / tasa real | 11,2 % / 11,3 % | 11,2 % / 11,3 % | 11,3 % / 11,3 % |
| Capturado en el top 10 % / 20 % / 30 % | **47 % / 66 % / 74 %** | 44 % / 64 % / 73 % | 28 % / 38 % / 49 % |

**Intervalos de confianza 95 % (bootstrap pareado, 2.000 remuestras, `reports/bootstrap_metricas_prueba.csv`):** AUC de LightGBM 0,815 [0,798; 0,832], AP 0,490 [0,455; 0,525], Brier 0,0751 [0,0706; 0,0793]. Diferencia LightGBM − logística: AUC +0,014 [+0,008; +0,020], AP +0,046 [+0,033; +0,059], Brier −0,0034 [−0,0044; −0,0025]: significativa pero pequeña. Los intervalos reflejan solo el muestreo de la prueba, no el cambio de régimen.

**Decisión con umbral 11 %:** se llama al 18,8 % de los clientes, se captura el 65,5 % de los depósitos, conversión 39,3 % (vs 11,3 %), ganancia **36.256 USD** vs 8.352 USD llamando a todos (×4,3).

**Desbalance (fuera de pliegue, LightGBM):** AP sin ajuste 0,464 · pesos 0,464 · submuestreo 0,451 · SMOTE-NC 0,416. Brier sin calibrar 0,077 · 0,155 · 0,163 · 0,134. Pesos y remuestreo no mejoran el orden y rompen la probabilidad.

## Limitaciones y riesgos

1. **Cambio de régimen.** Entrenado con 70 % de datos de 2008 (tasas de 3–6 %); en 2009–2010 la tasa fue 25–57 %. En la prueba de estrés el modelo fijo cae a AUC 0,70 y subestima la tasa (predice 11,7 % en promedio cuando es 25,4 %, la mitad). La partición estratificada de la evaluación principal es optimista frente a esta prueba. **Mitigación:** re-entrenar cada mes (AUC 0,79, ganancia 137 k vs 133 k llamando a todos) y vigilar tasa real vs. predicha.
2. **El mes mezcla estacionalidad y régimen.** Meses como octubre o marzo casi solo aparecen en el periodo de alta tasa; el modelo puede atribuir al mes lo que es contexto económico. No hay que leer el efecto del mes como estacionalidad pura.
3. **Selección de clientes por el banco.** Tras 2008 el banco llamó a muchos menos clientes, probablemente ya seleccionados; los datos reflejan a quién decidió llamar el banco, no a la población completa (sesgo de selección).
4. **Variables personales.** Edad, estado civil y ocupación influyen poco (edad es la 6.ª variable). Antes de usarlo habría que revisar que la priorización no excluya sistemáticamente a grupos protegidos; aquí solo ordena llamadas comerciales, pero el riesgo existe.
5. **Supuestos económicos.** La ganancia en USD depende de los supuestos de costo y margen; la tabla de sensibilidad del README muestra que el modelo gana en 8 de 9 escenarios y solo empata cuando llamar es baratísimo.
6. **SHAP explica el modelo, no la causalidad:** «euríbor bajo sube la probabilidad» describe una asociación del periodo, no un efecto causal.

## Licencia

El **código** es MIT (ver `LICENSE`). Los **datos** (UCI Bank Marketing) siguen su propia licencia, CC BY 4.0.

## Reproducibilidad

`python descargar_datos.py` y luego `python -m campana.entrenar` (≈ 4 min, semilla 42). Versiones: Python 3.14, LightGBM 4.7, scikit-learn 1.9, imbalanced-learn 0.14, pandas 3.0, numpy 2.5, SHAP 0.54. Si `joblib.load` falla con otras versiones, basta con re-entrenar.
