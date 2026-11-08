# Ficha del modelo: LightGBM con pesos para detección de fraude con tarjetas

## Resumen
- **Tarea:** puntuar transacciones con tarjeta según su probabilidad de fraude y decidir si se envían a revisión manual.
- **Modelo:** LightGBM binario sobre `V1`–`V28` (componentes PCA anónimas) y `log(1 + Amount)`. Pesos de clase ~474:1 y parada temprana por AUPRC de validación. Se eligió entre tres LightGBM y un híbrido por la mayor AUPRC **en validación** (0,800).
- **Artefactos:** `models/lgbm_servido.txt` (LightGBM), `models/autoencoder.npz` (señal de anomalía complementaria, NumPy puro) y `models/metadata.json` (umbral, calibración de Platt y supuestos).

## Datos
- ULB Credit Card Fraud (OpenML 1597): 284.807 transacciones de dos días de septiembre de 2013, 492 fraudes (0,17 %). Licencia de los datos: ODbL 1.0.
- **Partición temporal sin barajar:** entrenamiento de 0 a 33,4 h, validación de 33,4 a 40,4 h y prueba de 40,4 a 48,0 h (56.962 transacciones, 75 fraudes).

## Rendimiento en prueba (IC 95 % bootstrap, 1.000 réplicas)
| Métrica | Valor |
| :--- | ---: |
| AUPRC | 0,782 [0,682; 0,865] |
| ROC-AUC | 0,928 [0,878; 0,969] |
| Precisión@80 (capacidad de 250 alertas/día) | 72,5 % |
| Recall con precisión ≥ 80 % | 76,0 % |
| Recall con precisión ≥ 90 % | 74,7 % |
| Mejor no supervisado (LOF novelty (solo normales)), AUPRC | 0,489 |

## Uso previsto y decisión
- **Umbral por costo** (elegido en validación con 5 EUR por alerta y pérdida del 100 % del monto): puntaje ≥ 0,0088. En prueba genera 157 alertas, detecta 59 de 75 fraudes y cuesta 3.651 EUR frente a 7.729 EUR sin modelo (ahorro 52,8 %).
- **Pérdida esperada** (alternativa recomendada si cambian los costos): alertar si `p_calibrada × Amount > costo de revisión`, con Platt `a = 0,4710` y `b = -1,9530`. En prueba: 70 alertas y costo de 3.278 EUR. Es más robusta en el análisis de sensibilidad.
- Es una herramienta de **priorización para analistas**, no de bloqueo automático.

## Usos no previstos
- Explicar al cliente por qué se marcó su transacción: las variables son PCA anónimas.
- Aplicarlo a otros bancos, países o años sin reentrenar: la validación adversaria separa incluso el día 1 del día 2 (ROC-AUC 0,995).

## Limitaciones y riesgos
- Solo dos días de datos y 75 fraudes en prueba: los IC son anchos y las diferencias entre LightGBM no son significativas.
- El PCA del banco se ajustó con los dos días (pequeña fuga de información imposible de deshacer).
- Las probabilidades crudas están descalibradas por los pesos de clase; para decisiones con monto hay que usar la calibración de Platt.
- Los supuestos de costo no incluyen la fricción del cliente legítimo bloqueado.
- **Monitoreo recomendado:** PSI semanal de las variables más importantes, tasa de alertas y precisión de las alertas revisadas; reentrenar si el PSI supera 0,25 o la precisión de las alertas cae.

## Ética
- Los falsos positivos bloquean compras legítimas. Con el umbral por costo, 98 de 157 alertas en prueba son legítimas: la revisión humana es obligatoria antes de cualquier bloqueo.
- El dataset no tiene variables demográficas, así que no se puede auditar la equidad entre grupos de clientes.
