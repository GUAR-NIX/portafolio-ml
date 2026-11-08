# Ficha del modelo · Valor de Vida del Cliente (BG/NBD + Gamma-Gamma) v1.1.0

## Qué hace

A partir del resumen transaccional de un cliente en un comercio minorista no contractual (`frecuencia` de compras repetidas $x$, `recencia_dias` $t_x$, `edad_dias` $T$ y `ticket_medio` $m_x$), estima:

1. **$P(\text{vivo en } T \mid x, t_x, T)$**: probabilidad de que el cliente siga activo en el negocio y no haya abandonado en silencio.
2. **$P(Y(t) \ge 1 \mid x, t_x, T)$**: probabilidad calibrada de que realice al menos una compra en los próximos $t$ meses.
3. **$\mathbb{E}[Y(t) \mid x, t_x, T]$**: número esperado de días de compra futuros en el horizonte.
4. **$\mathbb{E}[M \mid x, m_x]$**: ticket diario esperado aplicando contracción empírica de Bayes (*shrinkage*) entre la media poblacional y el historial del cliente.
5. **CLV neto descontado (GBP)**: valor presente neto del margen bruto futuro descontado mes a mes.
6. **Decisión de rescate comercial**: segmento estratégico y si cae en la zona de riesgo ($0,20 \le P(\text{vivo}) < 0,70$ y $x \ge 2$) con beneficio esperado mayor que el costo de contacto. El beneficio usa un supuesto de efectividad (25 %) que no está validado con un experimento.

## Uso previsto

- **Priorización comercial y asignación de presupuesto de retención** en comercio electrónico o retail sin suscripción contractual.
- **Valoración financiera de la cartera de clientes** (cohorte activa, margen futuro esperado y concentración de ingresos).
- **No** debe utilizarse para decisiones de crédito, precios discriminatorios individuales ni contextos contractuales (telecomunicaciones, seguros o SaaS con fecha de baja explícita, donde corresponden modelos de supervivencia censurada o clasificación supervisada de *churn*).

## Datos y partición temporal

| Concepto | Detalle |
|---|---|
| Fuente | **UCI Online Retail II** (Chen, Sain & Guo, 2012) · CC BY 4.0 |
| Periodo total | `2009-12-01` → `2011-12-09` (738 días, tienda británica de artículos de regalo y hogar) |
| Limpieza contable | 1.067.371 filas crudas → se excluyen 243.007 sin `Customer ID`, 18.744 cancelaciones (`C*`), 2.925 códigos contables/envío (`POST`, `M`, `DOT`, `BANK CHARGES`...), 61 con precio/cantidad $\le 0$ y 26.055 duplicados exactos → **776.579 líneas limpias** consolidadas en **32.878 compras diarias** |
| Ventana de calibración | `2009-12-01` → `2011-06-09` (**555 días** $\approx$ 79,3 semanas) · **4.944 clientes** (3.336 repetidores con $x \ge 1$) |
| Ventana de prueba (*holdout*) | `2011-06-10` → `2011-12-09` (**183 días = 6 meses**) · 52,0 % de los clientes vuelve a comprar |
| Ventana previa sin fuga (para LightGBM y tabla RFM) | Corte en `2010-12-09` (373 días) con objetivo en los 182 días siguientes (`2010-12-10` → `2011-06-09`) |
| Ventana estacional (robustez de LightGBM) | Corte en `2010-06-09` (190 días) con objetivo en la misma estación que la prueba (`2010-06-10` → `2010-12-09`) |

## Formulación y parámetros estimados

### 1. Proceso de conteo y abandono: BG/NBD (Fader, Hardie & Lee, 2005)
- Mientras está activo, cada cliente compra según un proceso de Poisson con tasa $\lambda \sim \text{Gamma}(r, \alpha)$.
- Tras cada compra, el cliente se da de baja en silencio con probabilidad $p \sim \text{Beta}(a, b)$.
- **Parámetros estimados (MLE):**
  - $r = 0,6830$, $\alpha = 9,4806\text{ semanas}$ ($\implies \mathbb{E}[\lambda] = 0,0720\text{ compras/semana} \approx 3,7\text{ compras/año}$).
  - $a = 0,1518$, $b = 3,3057$ ($\implies \mathbb{E}[p] = 4,39\%$ de probabilidad media de baja tras cada compra).
  - Log-verosimilitud en calibración: $-58.257,1$.

### 2. Valor monetario por compra: Gamma-Gamma (Fader, Hardie & Lee, 2005)
- El ticket medio observado $m_x$ condicionada la tasa latente $\nu$ sigue una distribución Gamma de forma $p x$, con $\nu \sim \text{Gamma}(q, \gamma)$.
- **Control de mayoristas extremos:** el 0,5 % superior de los tickets supera las `3.000 GBP` (llegando a `38.663 GBP` en distribuidores mayoristas). Acotar el ticket diario en `3.000 GBP` evita sesgar la escala poblacional $\gamma$.
- **Supuesto de independencia frecuencia–ticket: se cumple solo aproximadamente.** Tras acotar, la correlación lineal es baja (Pearson $r = 0,159$), pero la de rangos es $\rho = 0,249$ (Spearman, $p \approx 3 \times 10^{-48}$) y no cambia al excluir mayoristas: los clientes frecuentes tienen tickets algo mayores, y el modelo los contrae de más hacia la media.
- **Parámetros estimados (MLE sobre 3.336 repetidores):**
  - $p = 2,2995$, $q = 3,8383$, $\gamma = 484,87$.
  - Ticket medio poblacional a priori: $\mu_0 = \frac{p \gamma}{q - 1} = 392,81\text{ GBP}$.

## Desempeño en el periodo de prueba (4.944 clientes, 6 meses nunca vistos)

El AUC y el Brier se calculan sobre la probabilidad de **volver a comprar en los 6 meses** ($P(Y(t) \ge 1)$), no sobre $P(\text{vivo})$, que no es observable.

| Métrica | Ritmo histórico (heurística) | Tabla RFM por tramos | LightGBM supervisado (Hurdle) | **BG/NBD + Gamma-Gamma** |
|---|---:|---:|---:|---:|
| AUC de recompra en 6 meses | 0,762 | 0,774 | 0,800 | **0,810** |
| Brier de recompra (menos es mejor) | 0,2178 | 0,2006 | 0,1897 | **0,1774** |
| MAE de compras en 6 meses | 1,064 | 1,191 | 1,097 | **1,045** |
| RMSE de compras en 6 meses | 1,861 | 2,742 | 2,074 | **1,784** |
| Sesgo total de compras de cartera | −3,2 % | −21,8 % | −26,9 % | **−1,4 %** |
| Spearman de compras | 0,591 | 0,566 | 0,607 | **0,624** |
| Spearman de ingreso futuro (GBP) | 0,608 | 0,546 | 0,603 | **0,636** |
| Error agregado de ingreso de cartera | −12,4 % | −23,8 % | −40,9 % | **−10,0 %** |
| Captura de ingreso en el Top 10 % | 55,2 % | 51,0 % | 56,5 % | **57,1 %** |
| Captura de ingreso en el Top 20 % | 70,7 % | 64,6 % | 71,6 % | **72,0 %** |

**Significancia frente a LightGBM** (bootstrap pareado, 1.000 remuestreos, IC 95 % de la diferencia BG/NBD − LightGBM): AUC +0,010 [+0,004; +0,017], Brier −0,0124 [−0,0158; −0,0086], Spearman de compras +0,018 [+0,008; +0,029] y de ingreso +0,033 [+0,021; +0,046] son significativas; la captura del Top 20 % (+0,4 pp [−0,6; +1,2]) **no** lo es.

**Ventana de LightGBM:** se entrena con la ventana previa (objetivo diciembre–junio, sin temporada navideña), de ahí su −40,9 % de ingreso. Con la ventana de la misma estación (objetivo junio–diciembre de 2010, solo 190 días de historia) empeora: AUC 0,774, Brier 0,2580 y +100,5 % de ingreso.

## Políticas de retención validadas con el periodo de prueba

No hubo campañas en los datos, así que el efecto causal no se puede medir. Con el holdout se valida a quién elige cada política y qué % de los contactados que no volvieron habría que rescatar (a su ritmo de gasto histórico) para cubrir 6 GBP por contacto:

| Política | Contactados | Volvió sin campaña | Rescate necesario (IC 95 %) |
|---|---:|---:|---:|
| 1. Todos con > 60 días sin comprar | 3.362 | 39,1 % | 8,6 % [7,9; 9,4] |
| 2. Regla RFM clásica | 1.016 | 64,7 % | 6,8 % [5,5; 8,3] |
| 3. Top 20 % gasto histórico con > 60 d sin comprar | 334 | 69,8 % | 3,0 % [2,3; 3,9] |
| 4. BG/NBD + GG (zona de riesgo, beneficio esperado > costo) | 190 | 40,0 % | 3,6 % [2,5; 5,0] |

Las políticas 3 y 4 no se distinguen estadísticamente; la 4 elige clientes realmente en riesgo y la 3 en buena parte a quienes iban a volver igual. El ROI en dinero no se puede medir con estos datos (no hubo campañas) y calcularlo con el margen predicho por el propio modelo sería circular; por eso se reporta solo como escenario con sensibilidad: de +6,1 % a +748,5 % para la política 4 según la efectividad supuesta (5 % a 40 %).

## Supuestos económicos y limitaciones

1. **Estacionalidad navideña:** BG/NBD asume una tasa $\lambda$ estacionaria por cliente. En el comercio de regalos, septiembre–noviembre presenta un pico estacional que contribuye al −10,0 % de diferencia en el ingreso agregado del segundo semestre (mientras que LightGBM entrenado sobre el semestre anterior subestima un −40,9 %).
2. **Compradores de una sola vez ($x = 0$):** en la formulación estándar de BG/NBD la baja ocurre después de una compra, por lo que $P(\text{vivo en } T \mid x=0) = 1$. Por ello, para decisiones operativas y calibración a 6 meses se expone explícitamente **$P(Y(t) \ge 1)$**, que sí decae con la antigüedad $T$ ($32,9\%$ medio vs $30,5\%$ real en el segmento de nuevos/una compra).
3. **Distribuidores mayoristas:** los clientes con pedidos > 3.000 GBP tienen dinámicas de cuenta clave B2B y deben gestionarse con un ejecutivo comercial dedicado en lugar de campañas automatizadas minoristas.
4. **Supuestos de campaña:** margen bruto del 30 % y costo de 6 GBP por cliente contactado. La efectividad del rescate (25 % en la zona de riesgo) es un supuesto sin validar: solo se usa para elegir a quién contactar y para el escenario. Todos los parámetros son configurables por petición en la API.
5. **«Inactivos / perdidos» subestimados:** $P(\text{vivo})$ media de 7,5 % (recompra predicha 6,2 %), pero el 20,5 % volvió a comprar y gastaron 4,0 veces lo esperado (14.498 frente a 3.603 GBP). Son 39 clientes (0,4 % del ingreso), probablemente estacionales.

## Reproducibilidad

`python descargar_datos.py` y `python -m clv.entrenar` (tarda ~20 segundos en CPU, semilla 42). Pruebas automáticas: `pytest` (31 pruebas que verifican la limpieza, las identidades matemáticas de ${}_2F_1$, la paradoja de BG/NBD, el *shrinkage* de Gamma-Gamma, la validación de políticas con el holdout, el bootstrap, los endpoints de la API y la app Streamlit).
