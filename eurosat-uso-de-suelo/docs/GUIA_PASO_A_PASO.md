# Guía Paso a Paso y Manual de Entrevista Técnica: Clasificación Satelital de Uso de Suelo en CPU (`eurosat-uso-de-suelo`)

Esta guía explica la física de teledetección óptica de la misión **Copernicus Sentinel-2**, la arquitectura de **Transfer Learning en CPU** implementada en este repositorio sobre el benchmark oficial **EuroSAT (`27.000` escenas)**, y responde **12 preguntas de entrevista** con las métricas medidas en la partición de prueba (5.400 imágenes).

---

## 1. Fundamentos de Teledetección Satelital (Sentinel-2 MSI) y el Problema Espectral vs. Espacial

### 1.1. Geometría y Radiometría de EuroSAT
* Cada imagen de **EuroSAT RGB** es un recorte ortorrectificado del sensor multiespectral **MSI (*MultiSpectral Instrument*)** a bordo de los satélites **Sentinel-2A/2B** de la Agencia Espacial Europea (ESA).
* **Resolución espacial:** `10 metros por píxel` en las bandas visibles:
  * **Rojo ($R$, Banda `B04`):** centrada en $\lambda = 665\text{ nm}$ (fuerte absorción por clorofila-$a$ en vegetación sana).
  * **Verde ($G$, Banda `B03`):** centrada en $\lambda = 560\text{ nm}$ (pico de reflectancia visible de las hojas).
  * **Azul ($B$, Banda `B02`):** centrada en $\lambda = 490\text{ nm}$ (sensible a cuerpos de agua y dispersión atmosférica de Rayleigh).
* **Cobertura física de cada parche:** Un recorte de $64\times 64\text{ píxeles}$ equivale a un cuadrado de $640\text{ m}\times 640\text{ m} = 409.600\text{ m}^2 \approx \mathbf{40,96\text{ hectáreas}}$.

### 1.2. Los 54 Descriptores Físicos del Baseline de Teledetección (`M1`)
En `src/eurosat_suelo/datos.py` (`extraer_vector_handcrafted_imagen`) construimos un vector de **54 variables físicas** para evaluar hasta dónde llega el enfoque clásico sin redes neuronales profundas:
1. **Estadísticos de reflectancia por banda (`21` variables):** Media, desviación estándar y cuantiles ($P_{10}, P_{25}, P_{50}, P_{75}, P_{90}$) de $R, G, B$.
2. **Índices espectrales visibles (`8` variables):** Media y desviación estándar por píxel de:
   $$\text{ExG} = \frac{2G - R - B}{R + G + B + \varepsilon}, \quad \text{NGRDI} = \frac{G - R}{G + R + \varepsilon}, \quad \text{NGBDI} = \frac{G - B}{G + B + \varepsilon}, \quad \text{IBW} = \frac{B - R}{B + R + \varepsilon}$$
3. **Correlaciones inter-banda (`3` variables):** $\rho(R,G), \rho(R,B), \rho(G,B)$.
4. **Morfología y gradientes espaciales (`10` variables):** Magnitud media, desviación estándar y cuantiles ($P_{50}, P_{90}, P_{99}$) del gradiente de luminancia $|\nabla I|$, energías direccionales horizontal/vertical, anisotropía direccional y contraste centro-periferia (bloque central $32\times 32$ frente al borde).
5. **Distribución de luminancia y saturación (`12` variables):** Cuantiles ($P_5, P_{20}, P_{40}, P_{60}, P_{80}, P_{95}$) de brillo y saturación cromática.

---

## 2. Por qué falla la estadística espectral en clases geométricas

Particiones oficiales: **16.200** imágenes de entrenamiento, **5.400** de validación y **5.400** de prueba. Todas las cifras de esta guía son de la prueba y salen de `reports/`.

**El punto ciego del enfoque clásico:**
- El F1 de M1 es de solo **73,24 %** en `Highway` y **84,47 %** en `River`.
- La calzada de una autopista ocupa una franja estrecha del parche de 41 ha, y el resto es paisaje agrícola. Los promedios y cuantiles del parche describen el fondo, no la vía.
- Una CNN conserva la topología espacial y reconoce la línea continua que cruza la escena. Con el ensamble M5, `Highway` sube a **94,57 %** (+21,33 pts) y `River` a **96,74 %** (+12,27 pts).

**Por qué la fusión ayuda a una CNN congelada:**
- La CNN de ImageNet normaliza los canales y atenúa la radiometría absoluta. Los índices espectrales la conservan.
- Con la CNN congelada, M2 (90,67 %) no supera de forma significativa a M1 (90,30 %): el IC bootstrap de la diferencia es [−0,61; +1,33] puntos.
- Al concatenar embeddings e índices (M3) la exactitud sube a **94,33 %**, una mejora significativa de +3,67 pts.
- Al ajustar la red (M4, **96,52 %**), los índices aportan poco: el ensamble M5 (**96,98 %**) gana solo +0,46 pts, IC [+0,20; +0,72], y cuesta más del doble de latencia.

---

## 3. Decisiones clave para entrenar en CPU (sin GPU)

1. **Backbone ligero.** `MobileNetV3-Small` usa convoluciones separables en profundidad y bloques *Squeeze-and-Excitation*. El fine-tuning de 10 épocas sobre 16.200 imágenes tarda **unos 5 minutos** con 4 hilos (≈30 s por época), y el pipeline completo, 11,3 min.
2. **Descongelamiento parcial con tasas diferenciales.** Solo se entrenan los últimos 4 bloques (`features[-4:]`) y el cabezal: 886.770 parámetros. Se usa `AdamW` con `lr = 4e-4` en el backbone y `1.8e-3` en el cabezal, `CosineAnnealingLR` y `label_smoothing = 0.05`. Se guarda la época con mejor exactitud en **validación**.
3. **Simetría $D_4$.** Las imágenes nadir no tienen «arriba»: rotarlas 90° o reflejarlas da la misma clase y no requiere interpolación. Se usa en el entrenamiento y en el TTA horizontal.
4. **Calibración y decisiones en validación.** Temperaturas, peso del ensamble (0,75 / 0,25) y umbral τ se eligen con la partición de validación. La prueba no se toca hasta reportar.

---

## 4. Cuestionario de entrevista técnica (13 preguntas)

### P1. ¿Por qué funcionan para Sentinel-2 unos pesos de ImageNet?
**Respuesta:** Porque las primeras capas aprenden filtros genéricos de bordes, frecuencias y texturas que sirven para cualquier imagen óptica. Con la red **congelada** y solo una regresión logística (5.770 parámetros), M2 logra **90,67 %**. Pero con 16.200 etiquetas eso no supera a los 54 índices (90,30 %). La ventaja del preentrenamiento se nota cuando hay **pocas** etiquetas (P10) o al **ajustar** la red (P2).

### P2. ¿*Linear probe* o *fine-tuning* parcial?
**Respuesta:**
- La **sonda lineal** se entrena en segundos, casi no sobreajusta con pocos datos y solo guarda vectores de 576-d.
- El **fine-tuning parcial** adapta los filtros de alto nivel a surcos, rotondas y meandros, y llega a **96,52 %** (+6,22 pts sobre M1, significativo).
- **Regla práctica:** con decenas de imágenes por clase, sonda lineal o híbrido; con cientos o miles, fine-tuning con una tasa de aprendizaje baja en el backbone.

### P3. ¿Por qué tasas distintas en el backbone (`4e-4`) y en el cabezal (`1.8e-3`)?
**Respuesta:** El cabezal empieza aleatorio. Con una tasa alta en toda la red, sus gradientes grandes destruirían los filtros preentrenados (*catastrophic forgetting*). Una tasa unas 4,5 veces menor en `features[-4:]` protege la representación mientras se adapta.

### P4. ¿Por qué M1 se queda en 73,24 % de F1 en `Highway` y M5 llega a 94,57 %?
**Respuesta:** Una autopista de 20–30 m son 2 o 3 píxeles de ancho. En los estadísticos globales del parche esa señal se diluye. La CNN conserva su posición y continuidad. Grad-CAM (figura 09) y los mapas de oclusión (figura 08) muestran que la red mira la vía. Las confusiones que quedan son justamente las geométricas: 18 ríos predichos como autopista y 8 autopistas como río.

### P5. ¿Qué aporta combinar índices espectrales y embeddings?
**Respuesta:**
- **Sobre la CNN congelada aporta mucho:** M3 − M2 = +3,67 pts, IC [+3,00; +4,39]. Los índices conservan la radiometría absoluta que la normalización de ImageNet atenúa.
- **Sobre la red ajustada aporta poco:** M5 − M4 = +0,46 pts. Es significativo, pero en producción podría bastar M4 si importa la latencia.

### P6. ¿Qué es la simetría $D_4$ y por qué no sirve en fotos normales?
**Respuesta:** Son las 8 simetrías del cuadrado: 4 rotaciones con o sin reflejo. En una foto terrestre la gravedad fija el «arriba», pero en una toma nadir la clase es invariante bajo $D_4$. Además, rotar 90° una matriz es una permutación exacta de índices, sin el desenfoque de interpolar.

### P7. ¿Qué tan robustos son los modelos al velo, al ruido y al desenfoque?
**Respuesta (exactitud en prueba, `reports/robustez_perturbaciones.csv`):**
- **Velo simulado al 20 %:** M1 cae de 90,30 % a 57,61 %; M5, de 96,98 % a 76,09 %. El velo es una capa sintética, no nubosidad real.
- **Ruido gaussiano σ = 10:** M1 cae a 24,48 % y M5 a 56,98 %. Con σ = 20, la CNN **congelada** (M2, 38,76 %) aguanta más que M5 (30,74 %): ajustar la red a imágenes limpias la volvió más sensible al ruido.
- **Desenfoque σ = 1 px:** M1 cae a 49,59 % y M5 a 73,57 %.
- **Conclusión honesta:** ningún modelo es robusto porque se entrenaron sin estas perturbaciones. El siguiente paso es usarlas como aumentos de datos.

### P8. ¿Cómo diseñaste la política de abstención?
**Respuesta:** Con una matriz de costos **supuesta**: 18 USD por revisión humana, 35 USD por error leve y 140 USD por error crítico entre macro-categorías.
- **Elección de τ:** el umbral se elige **en validación**, minimizando el costo. Resultó τ = 0,85 y se reporta en prueba.
- **Resultado en prueba:** se automatiza el **92,22 %** con **99,14 %** de exactitud y los errores críticos bajan de 118 a 34. El costo por 1.000 parcelas pasa de 3.350,93 a **2.339,81 USD**, una diferencia de −1.011,11 USD con IC [−1.445; −600].
- **Comparación justa:** M1 con su propio τ de validación cuesta 6.087,59 USD.
- **Sensibilidad:** con la revisión al doble de precio, el ahorro de abstenerse cae al 5,2 %. En un escenario extremo es −0,5 %, es decir, deja de compensar.
- **Aclaración:** τ se elige solo en validación; en prueba el costo mínimo habría estado en τ = 0,80, así que el 0,85 elegido en validación tiene un pequeño optimismo, pero no contamina la prueba.

### P9. ¿Grad-CAM u oclusión?
**Respuesta:**
- **Grad-CAM** pondera los mapas de activación de una capa por el gradiente medio de la clase. Se usa `features[8]`, de 6×6, porque el último bloque solo da 3×3 con entrada de 96 px. Cuesta una pasada hacia adelante y otra hacia atrás.
- **Oclusión** tapa ventanas de 14×14 y mide cuánto cae la probabilidad. Es más lenta, pero no depende de gradientes.

Ambas coinciden en que la red mira vías y cauces en los aciertos. En los errores con alta confianza (figura 09), mira texturas de borde y cultivos.

### P10. ¿Qué muestra la curva *Few-Shot*?
**Respuesta (`reports/curva_few_shot_eurosat.csv`):**
- Con **5 imágenes por clase**, el GBDT espectral logra 47,43 % y el híbrido lineal (embeddings + índices) **69,17 %**.
- Con 25 por clase: 75,56 % frente a **83,59 %**.
- Con 1.000 por clase la ventaja se reduce a 2,8 pts.

Para mapear coberturas en Ecuador, unas decenas de parcelas etiquetadas por clase ya dan un punto de partida útil, que luego se mejora con fine-tuning.

### P11. ¿Por qué reescalar de 64×64 a 96×96?
**Respuesta:** MobileNetV3-Small reduce la resolución 32 veces. Con 64 px el mapa final sería de 2×2; con 96 px es de 3×3, y el bloque `features[8]` usado en Grad-CAM queda en 6×6. El costo extra en CPU es pequeño.

### P12. ¿Cómo desplegaste el modelo sin inflar el repositorio?
**Respuesta:** Se versionan cuatro archivos que suman unos 17 MB:

| Archivo | Tamaño | Uso |
| :--- | ---: | :--- |
| `clasificador_eurosat.joblib` | 4,5 MB | GBDT, sonda, MLP, temperaturas, peso del ensamble y τ |
| `mobilenetv3_eurosat_finetuned.onnx` | 4,3 MB | M4 para la API |
| `mobilenetv3_imagenet_extractor.onnx` | 3,7 MB | embeddings de M3 para la API |
| `mobilenetv3_eurosat_finetuned.pt` | 4,4 MB | Grad-CAM en la app |

- **Inferencia:** la API usa solo ONNX Runtime. Una imagen tarda 0,94 ms frente a 5,84 ms en PyTorch, con salidas idénticas: diferencia máxima de logits de 8,5·10⁻⁵.
- **Si falta un archivo:** la API responde 503 con la instrucción para regenerarlo.
- **Verificación:** una prueba comprueba que la inferencia ONNX reproduce las probabilidades de M5 calculadas con PyTorch.

### P13. ¿Qué pasa con la confusión entre cultivo anual y permanente?
**Respuesta:** En la prueba, de las 1.134 parcelas de las dos clases, el modelo final (M5) las confunde entre sí en 10 casos (0,88 %: 2 anual → permanente y 8 permanente → anual), frente a 26 (2,29 %) de los índices espectrales (M1): la red reduce esa confusión un 62 %.
- **No es el error principal:** el destino principal de los errores de cultivo permanente es la vegetación herbácea (23 casos), y en cultivo anual los errores se reparten entre pastizal, herbácea, bosque y permanente.
- **Es asimétrica** (8 de 10 son permanente → anual) y esos 10 errores son «leves» (misma macro-categoría, 35 USD en la matriz de costos supuesta).
- **La abstención los filtra:** 7 de los 10 tienen confianza menor a τ = 0,85 y van a revisión; los otros 3 cuestan 105 USD. En total son unos 231 USD de los 12.635 USD de la prueba (1,8 %).
- **Hipótesis, no medida:** la diferencia real es el ciclo del cultivo, que una imagen RGB de una sola fecha no ve; ayudarían NIR/borde rojo o varias fechas.

---

## 5. Comandos de verificación

```powershell
.\.venv\Scripts\python.exe descargar_datos.py            # 27.000 imágenes oficiales
.\.venv\Scripts\python.exe -m eurosat_suelo.entrenar      # ≈11 min en CPU: modelos, ONNX, reports/
.\.venv\Scripts\python.exe -m eurosat_suelo.graficos      # 10 figuras
.\.venv\Scripts\python.exe scripts\generar_notebooks.py   # 3 notebooks ejecutados
.\.venv\Scripts\pytest.exe -q                             # pruebas
.\.venv\Scripts\streamlit.exe run app/app.py --server.port 8616
```
