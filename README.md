# SolarGuard AI

> Prototipo académico de inteligencia artificial para clasificar visualmente el estado de paneles fotovoltaicos y apoyar la priorización del mantenimiento.

SolarGuard AI analiza imágenes de paneles solares mediante un modelo preentrenado de clasificación y presenta una condición visual, su nivel de confianza y una recomendación básica de atención. El sistema está diseñado como herramienta de apoyo: no reemplaza la inspección de un técnico ni emite un diagnóstico eléctrico definitivo.

## Estado del proyecto

El repositorio se encuentra en la etapa inicial de implementación. La configuración base de Python, `uv`, la ingesta, la validación, el preprocesamiento y el servicio de inferencia ya están preparados; la interfaz de Streamlit y las reglas de priorización forman parte del desarrollo planificado.

## Objetivos

- Integrar el modelo `solarscan-yolov8n-cls` para clasificar imágenes de paneles.
- Validar y preparar las imágenes antes de la inferencia.
- Convertir la condición predicha en una prioridad y una acción de mantenimiento.
- Ofrecer una interfaz clara para cargar imágenes y consultar resultados.
- Mantener un proceso reproducible mediante control de versiones, pruebas y documentación.

## Modelo y clases

El prototipo utilizará `solarscan-yolov8n-cls`, un clasificador basado en YOLOv8n-cls publicado en [Hugging Face](https://huggingface.co/SaifElgalaly/solarscan-yolov8n-cls). El modelo recibe fotografías RGB de paneles, redimensiona el lado menor a 224 píxeles, aplica un recorte central de 224 x 224 y escala los valores a `0..1`. Reconoce seis condiciones visibles:

| Clase | Interpretación | Acción propuesta |
| --- | --- | --- |
| `Clean` | Panel visualmente limpio | Sin intervención inmediata. |
| `Dusty` | Acumulación visible de polvo | Programar limpieza. |
| `Bird-drop` | Presencia de excremento de aves | Programar limpieza. |
| `Physical-Damage` | Daño físico visible | Priorizar inspección o reparación. |
| `Electrical-damage` | Posible daño eléctrico visible | Priorizar revisión técnica. |
| `Snow-Covered` | Panel cubierto por nieve | Registrar la condición; tiene baja relevancia para el contexto local. |

La confianza del modelo debe interpretarse junto con el contexto de la imagen y la revisión humana. Iluminación, sombras, reflejos, cámara, ángulo y condiciones reales de campo pueden afectar la predicción.

## Alcance

### Incluido en el prototipo

- Modelo preentrenado de Hugging Face.
- Carga y validación de imágenes.
- Clasificación y nivel de confianza.
- Priorización y recomendación de mantenimiento.
- Interfaz web con Streamlit.

### Evolución prevista

- Persistencia de resultados e historial de inspecciones.
- Panel de seguimiento y notificaciones externas.
- Optimización adicional del rendimiento.
- Automatización de despliegue.

### Fuera del alcance

- Control real de drones.
- Integración con sensores IoT o sistemas SCADA.
- Ejecución física del mantenimiento.
- Predicción de fallas futuras.
- Uso comercial del prototipo sin una validación adicional.

## Requisitos

- Python `3.13`.
- [uv](https://docs.astral.sh/uv/) para gestionar el entorno y las dependencias.
- Windows, macOS o Linux.

El runtime previsto para la inferencia es [ONNX Runtime](https://onnxruntime.ai/), porque SolarScan publica el artefacto `best.onnx` y un ejemplo de ejecución con ese runtime. La ingesta actual acepta JPEG, PNG y TIFF/GeoTIFF legible, pero entrega las imágenes en RGB para respetar el contrato del modelo; no calcula NDVI ni NDWI.

## Instalación

Clona el repositorio y sincroniza el entorno:

```bash
git clone https://github.com/karolldahian/solarguard-ai.git
cd solarguard-ai
uv sync
```

Para comprobar que las dependencias están disponibles:

```bash
uv run python -c "import onnxruntime, streamlit, pandas, plotly; print('Entorno listo')"
```

Para ejecutar todas las pruebas:

```bash
uv run pytest -q
```

## Ingesta de imágenes

El módulo `solarguard_ai.ingesta` expone dos operaciones:

- `load_image(source)`: valida un archivo o fuente binaria, comprueba su integridad, dimensiones y formato, y devuelve una imagen RGB lista para el preprocesamiento.
- `load_images(sources)`: procesa un lote y reporta el índice de la imagen que produzca un error.

Se aceptan extensiones `.jpg`, `.jpeg`, `.png`, `.tif` y `.tiff`. Los archivos corruptos, las extensiones desconocidas, las discrepancias entre extensión y contenido y las imágenes excesivamente grandes se rechazan con errores descriptivos.

## Preprocesamiento

El módulo `solarguard_ai.preprocesamiento` sigue el contrato publicado por SolarScan:

1. Convierte la imagen a RGB.
2. Redimensiona el lado menor a 224 píxeles con interpolación bilineal.
3. Aplica un recorte central de `224 x 224`.
4. Escala los valores RGB a `0..1`.
5. Reordena el tensor a `NCHW`, la forma esperada por el artefacto ONNX.

La función `preprocess_for_solarscan()` devuelve un tensor `float32` con forma `(1, 3, 224, 224)`. También se incluyen aumentos geométricos básicos, separación de canales RGB y utilidades opcionales para calcular NDVI y NDWI sobre arreglos multiespectrales. Estos índices no se incorporan al tensor SolarScan, porque el modelo fue entrenado para fotografías RGB.

## Servicio de inferencia

El módulo `solarguard_ai.inferencia` utiliza el archivo ONNX publicado por SolarScan. Los pesos no se incluyen en el repositorio; deben descargarse desde [Hugging Face](https://huggingface.co/saifElgalaly/solarscan-yolov8n-cls) y pasarse mediante su ruta local:

```python
from solarguard_ai.inferencia import SolarScanInference
from solarguard_ai.preprocesamiento import preprocess_for_solarscan

service = SolarScanInference("models/best.onnx")
tensor = preprocess_for_solarscan(image)
result = service.predict(tensor)

print(result.predicted_class)
print(result.confidence)
print(result.probabilities)
```

El servicio también expone `predict_batch()` para varias imágenes y almacena en caché las predicciones cuyo tensor sea idéntico. Una confianza inferior a `0.5` devuelve `Unknown`, siguiendo el comportamiento descrito en la ficha del modelo. Las entradas deben tener forma `(1, 3, 224, 224)` y valores `float32` entre `0` y `1`.

## Uso actual

El punto de entrada configurado actualmente es un comando de verificación del paquete:

```bash
uv run solarguard-ai
```

La aplicación de Streamlit y el flujo de inferencia se encuentran planificados, pero todavía no están incluidos en el repositorio. Cuando se incorpore `app.py`, el comando previsto será:

```bash
uv run streamlit run app.py
```

## Estructura del repositorio

```text
solarguard-ai/
├── src/
│   └── solarguard_ai/
│       ├── __init__.py
│       ├── ingesta.py
│       ├── preprocesamiento.py
│       └── inferencia.py
├── tests/
│   ├── test_ingesta.py
│   ├── test_preprocesamiento.py
│   └── test_inferencia.py
├── .python-version
├── LICENSE
├── Makefile
├── README.md
├── pyproject.toml
└── uv.lock
```

La estructura objetivo contempla separar la ingesta, el preprocesamiento, la inferencia, la priorización y la generación de tickets, además de incorporar pruebas automatizadas y una interfaz Streamlit.

## Plan de trabajo

1. Definir criterios de éxito y validar el modelo junto con su Model Card.
2. Construir el servicio de inferencia y las reglas de priorización.
3. Integrar la interfaz Streamlit y la generación de tickets de mantenimiento.
4. Añadir seguimiento de ejecuciones con MLflow y revisión con Ruff.
5. Preparar Docker y realizar la validación final.

## Consideraciones de uso responsable

Los resultados son una clasificación visual inicial y pueden contener errores. Toda recomendación relacionada con daño físico o eléctrico debe ser confirmada por personal técnico calificado antes de intervenir un panel o una instalación.

## Referencias

- [SolarScan: solar panel condition classifier](https://huggingface.co/SaifElgalaly/solarscan-yolov8n-cls), modelo base.
- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/), arquitectura y herramientas de visión por computador.
- Wirth, R. y Hipp, J. (2000). *CRISP-DM: Towards a standard process model for data mining*.

## Equipo

Proyecto desarrollado en la Especialización en Inteligencia Artificial de la Universidad Autónoma de Occidente por Jose Fernando Luque Cajiao, Julio Cesar Rosero Porras, Karoll Dahian Ramirez Marulanda y Jarvin David Alvarado Garces.

## Licencia

Este proyecto se distribuye bajo la [licencia MIT](LICENSE).
