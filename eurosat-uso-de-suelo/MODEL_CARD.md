# Model Card · Clasificador de uso de suelo Sentinel-2 (`eurosat-uso-de-suelo`)

## 1. Detalles del modelo
- **Modelo de producción (M5):** ensamble de dos partes, con pesos elegidos en validación por log-loss:
  - **0,75 ×** MobileNetV3-Small preentrenada en ImageNet-1K con *fine-tuning* parcial: últimos 4 bloques y cabezal, aumentos diédricos $D_4$ y TTA horizontal.
  - **0,25 ×** un MLP que fusiona embeddings de 576-d de la CNN congelada con 54 índices espectrales y de textura.
- **Calibración:** escalado por temperatura ajustado en **validación** para cada componente (T = 0,72 en el fine-tuning y 2,00 en la fusión) y otra vez para la mezcla (T = 0,85).
- **Inferencia:** ONNX Runtime en CPU con 4 hilos. Los dos grafos ONNX incluyen el reescalado a 96×96 y la normalización. La latencia del componente fine-tuned es de 0,94 ms por imagen individual, frente a 5,84 ms en PyTorch.
- **Versión:** 2.0.0 · **Licencia del código:** MIT (ver `LICENSE`); los datos siguen su propia licencia.
- **Artefactos versionados en `models/` (≈17 MB):**
  - `clasificador_eurosat.joblib`;
  - `mobilenetv3_eurosat_finetuned.onnx`;
  - `mobilenetv3_imagenet_extractor.onnx`;
  - `mobilenetv3_eurosat_finetuned.pt` (solo para Grad-CAM).

  Si falta alguno, la API responde 503 con la instrucción para regenerarlo.

## 2. Uso previsto
- **Uso:** clasificar parches RGB de Sentinel-2 (10 m/píxel, 64×64 px ≈ 41 ha) en 10 coberturas de suelo para apoyar el monitoreo catastral o ambiental.
- **Humano en el bucle:** si la confianza calibrada es menor que **τ = 0,85** o hay alerta radiométrica de velo o bajo contraste, la escena se deriva a un fotointérprete (`DERIVAR_FOTOINTERPRETE_SIG`). τ se eligió minimizando un costo **supuesto** en validación.
- **Fuera de alcance:**
  - fotos aéreas o de dron;
  - otras resoluciones;
  - escenas con nubes reales (EuroSAT las excluye);
  - decisiones legales sin revisión humana.

## 3. Datos
- **Fuente:** EuroSAT RGB (Helber et al., 2019), espejo `timm/eurosat-rgb`, particiones oficiales.
- **Entrenamiento:** **16.200** imágenes.
- **Validación:** **5.400**. Se usaron para la época del fine-tuning, las temperaturas, el peso del ensamble y τ.
- **Prueba:** **5.400**. Solo para reportar.
- **Clases:** desbalance leve, de 2.000 a 3.000 escenas por clase en el corpus. Por eso se reporta también el Macro-F1.

## 4. Métricas en prueba (5.400 escenas)

| Modelo | Exactitud [IC 95 % bootstrap] | Macro-F1 | ECE | Errores críticos | Costo supuesto / 1.000 parcelas |
| :--- | ---: | ---: | ---: | ---: | ---: |
| M1 · 54 índices + GBDT | 90,30 % [89,50; 91,09] | 0,8988 | 0,0116 | 374 | 10.668,52 USD |
| M2 · CNN congelada + sonda | 90,67 % [89,83; 91,46] | 0,9035 | 0,0077 | 347 | 10.013,89 USD |
| M3 · fusión MLP | 94,33 % [93,72; 94,94] | 0,9419 | 0,0062 | 226 | 6.377,78 USD |
| M4 · fine-tuning $D_4$ | 96,52 % [96,00; 96,98] | 0,9639 | 0,0061 | 140 | 3.940,74 USD |
| **M5 · ensamble calibrado** | **96,98 %** [96,50; 97,41] | **0,9688** | **0,0059** | **118** | **3.350,93 USD** |
| **M5 con abstención (τ = 0,85 de validación)** | **99,14 %** en el 92,22 % automatizado | — | — | **34** | **2.339,81 USD** |

**Bootstrap pareado (2.000 réplicas):**
- M2 frente a M1 **no** es significativo: +0,37 pts, IC [−0,61; +1,33].
- Los saltos M3 − M2, M4 − M3 y M5 − M4 sí lo son. El último es pequeño: +0,46 pts, IC [+0,20; +0,72].
- La abstención ahorra 1.011,11 USD por 1.000 parcelas frente a M5 sin abstención, IC [−1.445; −600].

**Calibración (ECE en prueba antes → después de la temperatura):**
- M1: 0,0273 → 0,0116.
- M4: 0,0440 → 0,0061.
- M5: 0,0445 → 0,0059.

## 5. Costos supuestos y sensibilidad
Los costos (18 USD por revisión humana, 35 USD por error dentro de la misma macro-categoría y 140 USD por error crítico entre macro-categorías) **no** provienen de tarifas reales: son supuestos ilustrativos.

En 27 escenarios que multiplican cada costo por 0,5, 1 o 2 (`reports/sensibilidad_costos.csv`):
- M5 con abstención siempre cuesta entre un 68,4 % y un 89,0 % menos que M1.
- El ahorro de la abstención sobre M5 va de −0,5 % a 65,0 %. Con revisión cara y errores baratos, abstenerse deja de compensar.

## 6. Robustez, explicabilidad y limitaciones
1. **Perturbaciones (exactitud de M5 en prueba):**
   - **velo simulado** al 20 %: 76,09 %;
   - ruido gaussiano σ = 10: 56,98 %;
   - desenfoque σ = 1 px: 73,57 %.

   Ningún modelo es robusto: todos se entrenaron con imágenes limpias. M1 es el más frágil (24,48 % con σ = 10) y, con ruido fuerte, la CNN congelada (M2) aguanta mejor que la fine-tuned. El velo es una capa sintética, no nubosidad real. Se recomienda entrenar con ruido y desenfoque como aumentos de datos.
2. **Explicabilidad:** Grad-CAM sobre `features[8]` (rejilla 6×6) y mapas de oclusión. En los aciertos se concentran en vías y cauces. Los errores con alta confianza muestran que la red se fija en texturas de borde, por ejemplo un río con riberas agrícolas predicho como autopista.
3. **Solo RGB:** sin NIR ni borde rojo. Las confusiones principales que quedan son Cultivo permanente → Vegetación herbácea (23 casos) y Río → Autopista (18 casos). Cultivo anual y permanente se confunden entre sí en 10 de 1.134 parcelas de prueba (0,88 %; 2 anual → permanente y 8 permanente → anual) frente a 26 (2,29 %) del baseline espectral; es un error leve (misma macro-categoría), 7 de los 10 casos quedan bajo τ = 0,85 y van a revisión, y el costo de los 10 es ~1,8 % del costo total de la prueba. Probable causa (no medida): el ciclo del cultivo no se ve en una imagen RGB de una fecha. Detalle en el README.
4. **Sesgo geográfico:** EuroSAT cubre 34 países europeos. Para usarlo en Ecuador hay que reajustar con escenas locales. La curva Few-Shot muestra que con 25 imágenes por clase el híbrido lineal alcanza 83,59 % y con 100, 89,41 %.
5. **Selección de τ:** en prueba el óptimo habría sido 0,80, no 0,85. La diferencia muestra el pequeño optimismo de ajustar en una partición y reportar en otra; aun así es la práctica correcta.
