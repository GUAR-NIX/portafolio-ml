# Guía paso a paso (Windows)

El proyecto **ya viene entrenado y ejecutado**: artefacto en `models/modelo_clv.joblib`, metadatos en `models/metadata.json`, notebooks ejecutados en `notebooks/`, las 9 figuras, la captura de la app y las tablas en `reports/`. Esta guía muestra cómo reproducirlo desde cero o demostrarlo en vivo durante una entrevista técnica.

## 1 · Crear el entorno e instalar dependencias (2 min)

```powershell
cd clv-retail-bgnbd          # entra a la carpeta del proyecto (donde la hayas clonado)
py -3.14 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

> Si PowerShell bloquea la activación del entorno virtual, ejecuta una sola vez:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

## 2 · Ejecutar las pruebas automáticas (~10 s)

```powershell
pytest -v
```

Resultado esperado: **`31 passed`**. Las pruebas verifican:
- las 5 reglas de limpieza contable y la consolidación diaria de facturas del mismo día;
- las propiedades matemáticas exactas de **BG/NBD** ($P(\text{vivo}\mid x=0)=1$, monotonía temporal, la paradoja del comprador frecuente en silencio y la coincidencia entre la solución cerrada con la función hipergeométrica ${}_2F_1$ y la cuadratura de Gauss-Legendre de 64 nodos);
- la contracción empírica de Bayes (*shrinkage*) de **Gamma-Gamma**;
- que las políticas de rescate se validan con el **gasto real del periodo de prueba** (no con predicciones del modelo) y que el escenario responde a sus supuestos;
- el **bootstrap pareado** (un modelo contra sí mismo no es significativo; contra el azar sí);
- la validación y el ordenamiento de los endpoints de **FastAPI**;
- que la **app Streamlit** se importa, se ejecuta sin errores y muestra las mismas cifras que `reports/cartera_clientes.csv`.

## 3 · Probar la API en vivo

```powershell
uvicorn api.main:app --reload
```

Abre `http://localhost:8000/docs` en el navegador o ejecuta desde otra ventana de PowerShell:

```powershell
# 1. Valorar a un cliente frecuente que lleva 250 días sin comprar (en riesgo de alto valor)
$c = @{id_cliente="13093"; frecuencia=6; recencia_dias=240; edad_dias=490; ticket_medio=520.0} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8000/valorar -Body $c -ContentType "application/json"

# 2. Priorizar una cartera en modo rescate con capacidad máxima de 2 contactos
$cartera = @{
  clientes = @(
    @{id_cliente="VIP_ACTIVO"; frecuencia=12; recencia_dias=480; edad_dias=500; ticket_medio=600.0},
    @{id_cliente="EN_RIESGO_1"; frecuencia=6; recencia_dias=230; edad_dias=500; ticket_medio=650.0},
    @{id_cliente="EN_RIESGO_2"; frecuencia=4; recencia_dias=210; edad_dias=460; ticket_medio=480.0},
    @{id_cliente="PERDIDO"; frecuencia=9; recencia_dias=60; edad_dias=520; ticket_medio=400.0}
  )
} | ConvertTo-Json -Depth 4
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/priorizar?modo=rescate&capacidad=2" -Body $cartera -ContentType "application/json"
```

Para detener el servidor: **Ctrl + C**.

## 3b · Abrir la app Streamlit

```powershell
streamlit run app/app.py --server.port 8615
```

Se abre en `http://localhost:8615`. Escribe un `Customer ID` (por ejemplo `13093`, cliente en riesgo de alto valor, o `12346`, campeón activo) para ver su CLV neto, $P(\text{vivo})$, probabilidad de recompra, segmento, acción recomendada y lo que realmente compró en los 6 meses de prueba. La pestaña «Simular cliente» usa el mismo motor que `POST /valorar`.

## 4 · Descargar los datos (10 s)

```powershell
python descargar_datos.py
```

Si `retail-rfm-cohortes-sql` ya descargó `online_retail_ii.csv`, lo reutiliza al instante; de lo contrario, baja el ZIP oficial de UCI y convierte las dos hojas del Excel (`1.067.371` filas) a `data/raw/online_retail_ii.csv`.

## 5 · Re-entrenar todo el pipeline (~20 s)

```powershell
python -m clv.entrenar
```

Ejecuta los 8 pasos en ~20 segundos (incluye 1.000 remuestreos bootstrap) y regenera `models/modelo_clv.joblib`, `models/metadata.json`, `reports/*.csv` (incluida `cartera_clientes.csv`, que usa la app), `reports/resultados.json` y las 9 figuras en `reports/figures/`.

## 6 · Notebooks

```powershell
jupyter notebook notebooks
```

Para volver a ejecutar los 3 notebooks desde terminal:

```powershell
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 01_datos_limpieza_y_corte_temporal.ipynb 02_bgnbd_gamma_gamma_y_evaluacion.ipynb 03_segmentacion_retencion_y_api.ipynb
```

---

## Preguntas de entrevista (con respuesta)

**1. ¿Por qué usar BG/NBD + Gamma-Gamma en vez de entrenar directamente un regresor como XGBoost o LightGBM sobre el gasto futuro?**
Los comparé en el mismo corte (`4.944` clientes, `183` días de prueba). LightGBM necesita etiquetas, así que se entrena en una ventana anterior (corte en diciembre de 2010, objetivo diciembre 2010 – junio 2011) que no incluye la temporada navideña: por eso subestima el ingreso del semestre de prueba en **−40,9 %**. Probé también entrenarlo con la misma estación (objetivo junio – diciembre 2010), pero con solo 190 días de historia el sesgo se invierte (+100,5 %) y el AUC baja a 0,774. BG/NBD no necesita etiquetas y usa toda la historia hasta el corte. Con un **bootstrap pareado** de 1.000 remuestreos, BG/NBD es mejor en AUC de recompra (**0,810 vs 0,800**, diferencia +0,010 con IC 95 % [+0,004; +0,017]), Brier y Spearman, con diferencias pequeñas pero significativas; en la captura del Top 20 % (72,0 % vs 71,6 %) la diferencia **no** es significativa. Conclusión honesta: para ordenar clientes ambos sirven; BG/NBD gana en calibración, en pronóstico agregado (−10,0 % de ingreso, −1,4 % de compras) e interpretabilidad.

**2. ¿Cuál es la diferencia entre un entorno contractual y uno no contractual, y por qué importa aquí?**
En un negocio contractual (Netflix, telefonía, SaaS) el cliente comunica su baja y sabemos exactamente quién sigue activo. En el retail no contractual (como esta tienda británica de regalos), el abandono es **no observable (silencioso)**: cuando alguien lleva 5 meses sin comprar, no sabemos si ya se fue para siempre o si simplemente compra dos veces al año. BG/NBD resuelve ese problema separando la frecuencia histórica ($x$) de la recencia respecto al alta ($t_x, T$) para estimar la probabilidad latente $P(\text{vivo en } T \mid x, t_x, T)$.

**3. ¿En qué consiste la «paradoja del comprador frecuente en silencio» en BG/NBD?**
Si dos clientes se dieron de alta hace 18 meses ($T = 546$ días) y ninguno ha comprado en los últimos 9 meses (270 días), la intuición ingenua de una tabla RFM dice que quien compró 12 veces es «mejor cliente» que quien compró 2 veces. BG/NBD dice lo contrario: quien compraba casi cada mes ($x = 12$) y de golpe encadena 9 meses de silencio tiene una probabilidad de seguir vivo **mucho menor** (~6 %) que quien compró pocas veces ($x = 2$, ~86 %). Los datos de la prueba lo confirman: con más de 300 días sin comprar, ninguno de los 7 clientes con $x \ge 7$ volvió, frente al 24 % de los de $x$ = 1–6 (muestra pequeña, pero en la dirección que predice el modelo). Para el comprador frecuente, ese silencio prolongado es una anomalía incompatible con su ritmo habitual $\lambda$.

**4. En BG/NBD, ¿por qué $P(\text{vivo en } T)$ vale 100 % para todos los clientes con $x = 0$ (sin compras repetidas), y cómo lo resolviste para calibrar la recompra?**
Porque en el modelo de Fader, Hardie & Lee (2005) el sorteo de abandono $\text{Bernoulli}(p)$ ocurre inmediatamente *después* de una compra repetida (o el cliente con $x=0$ no tiene compras intermedias que acoten el momento del abandono). Sin embargo, a medida que crece la antigüedad $T$ sin repetir compra ($x=0$), la distribución posterior de su tasa de compra $\lambda \mid x=0, T \sim \text{Gamma}(r, \alpha + T)$ se desplaza hacia cero. Por eso calculé y expuse explícitamente la probabilidad exacta de realizar al menos una compra en los próximos 6 meses:
$$P(Y(t) \ge 1 \mid x, t_x, T) = P(\text{vivo} \mid x, t_x, T) \cdot \left[1 - \left(\frac{\alpha + T}{\alpha + T + t}\right)^{r + x}\right]$$
Esa probabilidad sí decae suavemente con $T$ incluso para $x = 0$ y logra un Brier de **0,1774** y una curva de calibración por deciles casi diagonal.

**5. ¿Por qué consolidaste las transacciones por día de compra (`customer_id`, `fecha_dia`) en vez de contar facturas individuales?**
En UCI Online Retail II es frecuente que un cliente emita dos o tres facturas el mismo día con pocos minutos de diferencia (por ejemplo, cuando añade artículos olvidados a su pedido). Si contáramos facturas, un cliente con dos facturas el día de su alta parecería un comprador repetidor ($x = 1, t_x = 0$) que volvió a comprar y luego desapareció, sesgando hacia arriba tanto la tasa $\lambda$ como la probabilidad de baja $p$. En los modelos BTYD la unidad natural es el día con actividad de compra.

**6. ¿Qué supuesto clave exige el modelo Gamma-Gamma y cómo verificaste que se cumpliera?**
Gamma-Gamma asume que el ticket medio por compra $m_x$ es estocásticamente independiente del número de compras $x$ para poder multiplicar $\mathbb{E}[Y(t)] \times \mathbb{E}[M \mid x, m_x]$. En los datos crudos existen distribuidores mayoristas con pedidos de hasta `38.663 GBP` que inflan la media en alta frecuencia. Acoté el ticket diario en el percentil 99,5 (`3.000 GBP`) para que un puñado de cuentas mayoristas no distorsionara la escala $\gamma$, y medí la correlación: la lineal es débil (Pearson $r = 0,159$), pero la de rangos es $\rho = 0,249$ (Spearman, $p \approx 3 \times 10^{-48}$), así que el supuesto **se cumple solo de forma aproximada**: los clientes frecuentes tienen tickets algo mayores y el modelo los contrae de más hacia la media, lo que contribuye a subestimar su ingreso.

**7. ¿Qué hace la contracción empírica de Bayes (*shrinkage*) en Gamma-Gamma?**
El ticket esperado de un cliente es un promedio ponderado entre la media de toda la cartera ($\mu_0 = 392,81\text{ GBP}$) y su propio promedio histórico $m_x$:
$$\mathbb{E}[M \mid x, m_x] = (1 - w_x)\,\mu_0 + w_x\,m_x, \qquad w_x = \frac{p x}{p x + q - 1}$$
Con nuestros parámetros ($p = 2,30, q = 3,84$), si un cliente hizo solo $x = 1$ compra repetida de 1.200 GBP, el modelo le da un peso $w_1 = 44,8\%$ a ese dato y $55,2\%$ a la media poblacional (predice ~754 GBP, protegiéndose del ruido). En cambio, si mantuvo ese ticket durante $x = 10$ compras, el peso sube a $w_{10} = 89,0\%$ (predice ~1.111 GBP).

**8. ¿Por qué usaste la función hipergeométrica de Gauss ${}_2F_1$ y qué cuidado numérico tuviste cuando $a < 1$?**
La esperanza condicional $\mathbb{E}[Y(t) \mid x, t_x, T]$ integra la tasa de Poisson y la probabilidad de abandono sobre sus distribuciones posteriores Gamma y Beta, lo que conduce a la ecuación cerrada de Fader, Hardie & Lee (2005, Ec. 10) en términos de ${}_2F_1(r+x, b+x; a+b+x-1; \frac{t}{\alpha+T+t})$. En datos reales de retail suele ocurrir que $a < 1$ (aquí $a = 0,1518$). En ese régimen la función Beta tiene una singularidad integrable en $p = 0$; usar una cuadratura ingenua en $p \in (0, 1)$ falla por concentración de masa cerca de cero. La continuación analítica de `scipy.special.hyp2f1` (verificada en nuestros tests contra una cuadratura de 64 nodos con el cambio de variable $p = u^{1/a}$ que elimina la singularidad) es exacta en milisegundos.

**9. ¿A quién conviene enviar cupones de reactivación? ¿Cómo lo validaste sin un experimento?**
En los datos no hubo campañas, así que no se puede medir el efecto causal. Lo que sí validé con los 6 meses de prueba es a quién elige cada política y cuánto tendría que funcionar para pagarse: de los contactados que **no** volvieron, qué % habría que rescatar (a su ritmo de gasto histórico) para cubrir 6 GBP por contacto. Contactar a los **3.362** clientes con más de 60 días sin comprar exige rescatar el **8,6 %**; la regla RFM, el 6,8 %; el top 20 % de gasto histórico inactivo, el **3,0 %** [2,3; 3,9]; y la zona de riesgo de BG/NBD (190 clientes), el **3,6 %** [2,5; 5,0]. Las dos últimas no se distinguen estadísticamente, pero eligen gente distinta: del top 20 % de gasto volvió solo el 69,8 % (riesgo de regalar el incentivo), y de la zona de riesgo de BG/NBD, el 40,0 %. El siguiente paso sería un experimento A/B entre ambas.

**10. ¿Por qué no reportas un ROI único de la campaña de rescate?**
Porque los datos no traen ninguna campaña y cualquier ROI en dinero sería circular: el margen «recuperado» se calcularía con las predicciones del propio modelo y con una efectividad supuesta, y la política se elige con esa misma fórmula, así que ganaría por construcción. Lo presento solo como escenario, con su sensibilidad: el ROI de la política 4 va de +6,1 % (efectividad 5 %) a +748,5 % (efectividad 40 %), es decir, la cifra depende casi por completo del supuesto. Lo robusto es que contactar a todos los inactivos pierde dinero con cualquier supuesto, y lo que sí se valida con datos es a quién elige cada política y cuánto rescate necesitaría para pagarse.

**11. ¿Qué concentración de valor encontraste en la cartera y qué implica para el negocio?**
El **20 % superior** de los clientes priorizados por BG/NBD + Gamma-Gamma genera el **72,0 % de todo el ingreso real** en los 6 meses siguientes (y el Top 10 % genera el **57,1 %**, un lift de **5,7×**). El segmento de **Campeones activos** (32,3 % de los clientes) aporta el **80,4 %** de la facturación futura (`2,96 millones de GBP`). Perder una fracción pequeña de ese grupo o del segmento **En riesgo de alto valor** (197 clientes) impacta más que toda la cola de compradores ocasionales.

**12. ¿Dónde se equivoca el modelo?**
En los «Inactivos / perdidos»: les da $P(\text{vivo})$ media de 7,5 %, pero el 20,5 % volvió y gastaron 4,0 veces lo esperado. Son 39 clientes muy frecuentes con un silencio largo, probablemente estacionales, algo que la tasa constante de BG/NBD no captura. También subestima a los Campeones (gastaron 1,18 veces lo esperado) por el pico navideño y por el supuesto de Gamma-Gamma.

**13. ¿Cómo llevarías este sistema a producción?**
Ejecutando un proceso semanal o mensual que recalibre $(r, \alpha, a, b)$ y $(p, q, \gamma)$ sobre la ventana móvil de los últimos 18–24 meses, versione el artefacto junto con sus métricas de control en `metadata.json`, y publique las puntuaciones en el CRM o a través de la API FastAPI (`/valorar` y `/priorizar`) para disparar automáticamente la campaña de rescate cuando un cliente valioso cruce el umbral $P(\text{vivo}) < 0,70$.
