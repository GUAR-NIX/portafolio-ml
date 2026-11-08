# Guía paso a paso (Windows)

El proyecto **ya viene entrenado y ejecutado**: modelo en `models/`, notebooks con resultados, figuras y tablas en `reports/`. Esta guía es para volver a correrlo, por ejemplo para demostrarlo en una entrevista.

## 1 · Crear el entorno e instalar dependencias (3 min)

```powershell
cd bank-marketing-umbral-costo   # la carpeta del repositorio
py -3.14 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

> Si PowerShell bloquea la activación, ejecuta una vez: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

## 2 · Pruebas (10 s)

```powershell
pytest
```

Resultado esperado: `30 passed`. La prueba con datos reales se salta si no descargaste el CSV (paso 4); el resto usa datos sintéticos de resultado conocido y el modelo incluido.

## 3 · La API

```powershell
uvicorn api.main:app --reload
```

Abre `http://localhost:8000/docs` → `POST /puntuar` → **Try it out** → **Execute**. Prueba esto:

1. Con el ejemplo por defecto (cliente nuevo, mayo, euríbor 1,3) → probabilidad ≈ 7,8 %, `llamar: false`.
2. Cambia `resultado_previo` a `"success"`, `dias_desde_contacto` a `6` y `contactos_previos` a `2` → ≈ 33 %, `llamar: true`, y la primera razón es «Días desde el último contacto».
3. En `POST /priorizar` manda varios clientes con `capacidad=1`: solo el primero queda con `llamar: true`.
4. En `/priorizar` pon `costo_llamada=20` y `beneficio_exito=80`: el umbral pasa a 25 % sin re-entrenar.

Desde otra ventana de PowerShell:

```powershell
$c = @{edad=35; ocupacion="admin."; estado_civil="single"; educacion="university.degree"; mes="may"; dia_semana="thu"; dias_desde_contacto=6; contactos_previos=2; resultado_previo="success"; var_empleo=-1.8; ipc=92.893; confianza_consumidor=-46.2; euribor_3m=1.299; num_empleados=5099.1} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8000/puntuar -Body $c -ContentType "application/json"
```

Para cerrarla: **Ctrl + C**.

## 4 · Descargar los datos (10 s)

```powershell
python descargar_datos.py
```

Baja el zip oficial de UCI (1 MB) y deja `data/raw/bank-additional-full.csv` (41.188 filas). Verifica el número de filas.

## 5 · Re-entrenar todo (≈ 4 min)

```powershell
python -m campana.entrenar
```

Muestra 8 pasos (datos, fuga, desbalance, calibración y umbral, prueba, SHAP, estrés temporal, guardar) y termina con `Listo en … s`. Usa < 2 GB de RAM. Regenera `models/`, `reports/` y las 8 figuras; los resultados deben coincidir con el README (semilla fija). El paso más lento es SMOTE (≈ 1–2 min).

## 6 · Notebooks

```powershell
jupyter notebook notebooks
```

Necesitan los datos (paso 4). Para volver a ejecutarlos todos sin abrirlos:

```powershell
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 01_datos_y_fuga.ipynb 02_modelos_desbalance_calibracion.ipynb 03_umbral_ganancia_shap_api.ipynb
```

## 7 · Cambiar los supuestos de negocio

En `src/campana/config.py`: `COSTO_LLAMADA` y `BENEFICIO_EXITO`. Re-entrena (paso 5) para que el umbral por defecto y todas las tablas se actualicen. Para una prueba rápida no hace falta: la API acepta los costos por petición.

---

## Preguntas de entrevista (con respuesta)

**1. ¿Por qué no usaste `duration` si es la variable más predictiva?**
Porque es una fuga de información: es la duración de la llamada, que solo se conoce cuando la llamada terminó (0 s ⇒ «no» seguro). El modelo se usa para decidir *a quién llamar*, antes de llamar. Con ella el AUC sería 0,952, pero ese número no existe en producción. Sin ella, 0,815.

**2. ¿Hay otra fuga menos obvia?**
Sí, `campaign`: cuenta las llamadas de esta campaña incluida la última. El banco deja de llamar cuando el cliente acepta, así que depende del resultado, y antes de empezar no se conoce. También la excluí, aunque aportaba poco (AUC +0,001). Hay una prueba automática que impide que una variable con fuga entre al modelo.

**3. ¿Por qué validación estratificada y no temporal?**
El 70 % de los datos es de 2008 y lo que viene después es otro régimen (tasa de éxito de 6 % a 25–57 %, euríbor de 5 % a 0,7 %). Una partición temporal mediría sobre todo el cambio de régimen con pocos datos por mes. Por eso la evaluación principal es estratificada 80/20 con validación cruzada para todas las decisiones, y aparte hice una **prueba de estrés temporal** que muestra la degradación (AUC 0,70) y cómo mitigarla (re-entrenar cada mes: AUC 0,79).

**4. La clase está desbalanceada (11 %). ¿Usaste SMOTE o pesos?**
Los comparé con validación cruzada. Ni pesos ni submuestreo mejoran el orden (average precision igual), SMOTE lo empeora (0,464 → 0,416), y todos rompen la probabilidad: el modelo cree que acepta el 38 % cuando acepta el 11 %. El desbalance se resuelve mejor con probabilidades calibradas y un umbral económico. El modelo final no usa ninguno.

**5. ¿Por qué no usaste el umbral 0,5?**
Porque el 0,5 supone que un falso positivo cuesta lo mismo que un falso negativo. Aquí una llamada inútil cuesta 8 USD y un depósito perdido 80 USD. Con probabilidad calibrada conviene llamar si `p × 80 > 8`, o sea p > 10 %. Busqué el umbral que maximiza la ganancia fuera de pliegue y salió 11 %, casi el teórico. Con 0,5 se llamaría a muy pocos.

**6. ¿Qué significa que el modelo esté calibrado y cómo lo mediste?**
Que de los clientes a los que el modelo da 20 %, aceptan ~20 %. Lo medí con la curva de calibración por deciles y con el Brier score (0,0751 en prueba; Brier skill 0,25 frente a predecir siempre la tasa base). Calibré con Platt usando predicciones fuera de pliegue para que ningún cliente calibrara su propia predicción.

**7. ¿Cuánto dinero gana el banco con el modelo?**
Con los supuestos (8 USD por cliente, 80 USD por depósito), en los 8.236 clientes de prueba: llamar a todos deja 8.352 USD; el modelo llama al 18,8 %, captura el 65,5 % de los depósitos y deja 36.256 USD (×4,3). Si cambian los costos, el umbral se recalcula como costo/beneficio sin re-entrenar; en 8 de 9 escenarios de la tabla de sensibilidad el modelo supera a llamar a todos.

**8. ¿Y si el call center solo puede hacer 1.000 llamadas?**
Entonces no se usa el umbral sino el ranking: se llama a los 1.000 con mayor probabilidad. La curva de ganancia acumulada dice cuánto se captura: el top 10 % trae el 47 % de los depósitos (lift 4,7×), el top 20 % el 66 %. La API tiene el parámetro `capacidad` para esto.

**9. ¿Qué variables pesan más? ¿Tiene sentido?**
Según SHAP: nº de empleados de la economía, euríbor, tipo de teléfono, mes y variación del empleo. Tiene sentido (en 2009–2010, con tasas bajas, el depósito garantizado atraía más y el banco seleccionaba mejor), pero hay que leerlo con cuidado: el mes mezcla estacionalidad con el régimen, y SHAP explica el modelo, no causalidad.

**10. ¿Por qué LightGBM y no solo regresión logística?**
LightGBM gana en todo pero por poco (AUC 0,815 vs 0,801; AP 0,49 vs 0,44; 1.800 USD más en prueba). La logística sería una alternativa válida si se necesitara un modelo más simple de auditar. Elegí por ganancia calibrada fuera de pliegue, no por AUC.

**11. ¿Qué harías para ponerlo en producción?**
Re-entrenar mensualmente con los resultados de la campaña anterior (la prueba temporal mostró que es necesario), monitorear tasa real vs. predicha y el AUC por mes, versionar el modelo con su metadata (ya se guarda en `metadata.json`), y servirlo con la API (FastAPI con validación pydantic). Para cambios de costos no hace falta re-entrenar.
