# Portafolio de Machine Learning

Clasificación con costo, pronóstico, recomendación, valor de vida del cliente, visión por computadora y detección de fraude, con validación sin fuga, intervalos y APIs.

**Herramientas principales:** scikit-learn · LightGBM · PyTorch · SHAP · FastAPI

Cada carpeta es un proyecto independiente, con su propio README (resultados primero), guía paso a paso, pruebas automáticas y cómo reproducirlo. Los datos crudos no se versionan: cada proyecto trae un script que los descarga desde la fuente oficial.

## Proyectos

| Proyecto | De qué trata |
|---|---|
| [**📞 ¿A quién llamar? Campaña de depósitos a plazo con ML**](bank-marketing-umbral-costo/) | Modelo que decide a qué clientes debe llamar un banco para ofrecer un depósito a plazo, con la probabilidad calibrada de que acepten y un umbral elegido por costo/beneficio de cada llamada. Datos reales de telemercadeo de un… |
| [**💎 ¿Quién sigue vivo y cuánto vale? CLV y retención en retail (BG/NBD + Gamma-Gamma)**](clv-retail-bgnbd/) | Sistema probabilístico que estima qué clientes siguen activos, cuántas veces comprarán, cuál será su ticket medio y a quién conviene rescatar en un comercio minorista donde el abandono ocurre en silencio. Evaluado sobre UCI… |
| [**🚲 ¿Cuántas bicis saldrán de cada estación mañana? Pronóstico de demanda por estación y hora (Citi Bike + clima)**](demanda-citibike-pronostico/) | Citi Bike mueve decenas de miles de bicicletas al día y cada noche hay que rebalancear: llevar bicis a las estaciones que se vaciarán en las horas pico de mañana. Para decidirlo hace falta saber, la noche anterior, cuántas… |
| [**💳 Detección de fraude con tarjetas: anomalías no supervisadas frente a modelos supervisados**](deteccion-fraude-tarjetas/) | Datos: 284.807 transacciones reales de titulares europeos (ULB / Worldline, dos días de septiembre de 2013), con 492 fraudes (0,17 %). |
| [**🛰️ Uso de suelo con Sentinel-2 (EuroSAT): índices espectrales vs. Transfer Learning en CPU**](eurosat-uso-de-suelo/) | Datos: 27.000 escenas de EuroSAT RGB en sus particiones oficiales: 16.200 de entrenamiento · 5.400 de validación · 5.400 de prueba. |
| [**🎬 ¿Qué película ver ahora? Recomendador sobre MovieLens 25M**](recomendador-movielens/) | Sistema de recomendación de películas entrenado con 25 millones de calificaciones de 162 mil usuarios (MovieLens 25M, 1995 → 2019). Combina filtrado colaborativo implícito (ALS), un modelo de contenido con géneros y el Tag Genome… |

## Cómo usar este repositorio

```bash
git clone https://github.com/GUAR-NIX/portafolio-ml.git
cd portafolio-ml/<proyecto>
```

Dentro de cada proyecto, sigue la sección «Cómo reproducir» de su README y `docs/GUIA_PASO_A_PASO.md`.

## Autor

**Erick Guarnizo** · [LinkedIn](https://www.linkedin.com/in/erick-g-571a7b3b4)

## Licencia

El código está bajo licencia [MIT](LICENSE). Los datos de cada proyecto conservan la licencia de su fuente, indicada en su README.
