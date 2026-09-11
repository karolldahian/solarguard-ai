# SolarGuard AI

> Prototipo académico de inteligencia artificial para clasificar visualmente el estado de paneles fotovoltaicos y apoyar la priorización del mantenimiento.

SolarGuard AI analiza imágenes de paneles solares mediante un modelo preentrenado de clasificación y presenta una condición visual, su nivel de confianza y una recomendación básica de atención. El sistema está diseñado como herramienta de apoyo: no reemplaza la inspección de un técnico ni emite un diagnóstico eléctrico definitivo.

## Estado del proyecto

El repositorio se encuentra en etapa de desarrollo. Ya se implementaron la configuración base con `uv`, la ingesta y validación de imágenes, el preprocesamiento compatible con el modelo, las reglas de priorización (tickets), la integración gRPC entre interfaz y backend, y una primera versión de la interfaz Streamlit. La integración del modelo real de inferencia (ONNX), el seguimiento con MLflow y la preparación para despliegue forman parte del trabajo planificado.

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

Para ejecutar las pruebas de ingesta:

```bash
uv run pytest tests/test_ingesta.py -q
```

## Ingesta de imágenes

El módulo `solarguard_ai.ingesta` expone dos operaciones:

- `load_image(source)`: valida un archivo o fuente binaria, comprueba su integridad, dimensiones y formato, y devuelve una imagen RGB lista para el preprocesamiento.
- `load_images(sources)`: procesa un lote y reporta el índice de la imagen que produzca un error.

Se aceptan extensiones `.jpg`, `.jpeg`, `.png`, `.tif` y `.tiff`. Los archivos corruptos, las extensiones desconocidas, las discrepancias entre extensión y contenido y las imágenes excesivamente grandes se rechazan con errores descriptivos.

## Integración gRPC (interfaz ↔ backend)

La interfaz (Streamlit) y el backend se comunican con gRPC. La interfaz solo sube la imagen y dibuja el ticket; todo el procesamiento lo hace el backend:

```text
app.py (Streamlit) ──gRPC──▶ solarguard_ai.servidor_grpc
   cliente_grpc.py               │
                                  ├─ ingesta.load_image           (validar → RGB)
                                  ├─ preprocesamiento.preprocesar (tensor 1×3×224×224)
                                  ├─ inferencia.predecir          (clasificación, simulada)
                                  └─ tickets.generar_ticket       (condición → ticket)
```

El contrato se define en `proto/solarguard.proto` y se regenera con `make grpc-gen`. Los archivos generados (`solarguard_pb2.py` y `solarguard_pb2_grpc.py`) se versionan en `solarguard_ai/grpc_interface/` para que nadie necesite regenerarlos.

> La inferencia es por ahora una **simulación determinista** (misma imagen, mismo resultado). Cuando el módulo de inferencia real con ONNX Runtime esté listo, solo cambia la función `inferencia.predecir()`.

Para levantar el sistema se necesitan dos terminales:

```bash
# Terminal 1: el backend gRPC
uv run python -m solarguard_ai.servidor_grpc

# Terminal 2: la interfaz web
uv run streamlit run app.py
```

Pruebas de la integración:

```bash
uv run pytest tests/test_grpc.py -q
```

## Uso actual

El punto de entrada del paquete es un comando de verificación:

```bash
uv run solarguard-ai
```

La interfaz web de Streamlit y el backend gRPC ya están incluidos en el repositorio. Para usarlos se deben abrir dos terminales:

```bash
uv run python -m solarguard_ai.servidor_grpc
uv run streamlit run app.py
```

## Estructura del repositorio

```text
solarguard-ai/
├── proto/
│   └── solarguard.proto
├── scripts/
│   └── grpc_gen.py
├── src/
│   └── solarguard_ai/
│       ├── grpc_interface/
│       │   ├── solarguard_pb2.py
│       │   └── solarguard_pb2_grpc.py
│       ├── __init__.py
│       ├── ingesta.py
│       ├── preprocesamiento.py
│       ├── inferencia.py
│       ├── tickets.py
│       ├── servidor_grpc.py
│       └── cliente_grpc.py
├── tests/
│   ├── test_ingesta.py
│   ├── test_preprocesamiento.py
│   ├── test_tickets.py
│   └── test_grpc.py
├── app.py
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
2. Implementar el preprocesamiento compatible con la entrada RGB de 224 x 224.
3. Construir el servicio de inferencia y las reglas de priorización.
4. Integrar la interfaz Streamlit y la generación de tickets de mantenimiento.
5. Añadir seguimiento de ejecuciones con MLflow y revisión con Ruff.
6. Preparar Docker y realizar la validación final.

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
