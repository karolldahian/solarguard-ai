# SolarGuard AI

> Prototipo académico de inteligencia artificial para clasificar visualmente el estado de paneles fotovoltaicos y apoyar la priorización del mantenimiento.

SolarGuard AI analiza imágenes de paneles solares mediante un modelo preentrenado de clasificación y presenta una condición visual, su nivel de confianza y una recomendación básica de atención. El sistema está diseñado como herramienta de apoyo: no reemplaza la inspección de un técnico ni emite un diagnóstico eléctrico definitivo.

## Estado del proyecto

El repositorio se encuentra en la etapa inicial de implementación. La configuración base de Python, `uv` y las dependencias ya está preparada; la integración del modelo, la interfaz de Streamlit, las reglas de priorización y las pruebas forman parte del desarrollo planificado.

## Objetivos

- Integrar el modelo `solarscan-yolov8n-cls` para clasificar imágenes de paneles.
- Validar y preparar las imágenes antes de la inferencia.
- Convertir la condición predicha en una prioridad y una acción de mantenimiento.
- Ofrecer una interfaz clara para cargar imágenes y consultar resultados.
- Mantener un proceso reproducible mediante control de versiones, pruebas y documentación.

## Modelo y clases

El prototipo utilizará `solarscan-yolov8n-cls`, un clasificador basado en YOLOv8n-cls publicado en [Hugging Face](https://huggingface.co/SaifElgalaly/solarscan-yolov8n-cls). El modelo trabaja con imágenes de entrada de 224 x 224 píxeles y reconoce seis condiciones visibles:

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

TensorFlow se mantiene en Python 3.13 porque sus ruedas disponibles no son compatibles con Python 3.14 en la configuración actual del proyecto.

## Instalación

Clona el repositorio y sincroniza el entorno:

```bash
git clone https://github.com/karolldahian/solarguard-ai.git
cd solarguard-ai
uv sync
```

Para comprobar que las dependencias están disponibles:

```bash
uv run python -c "import tensorflow, streamlit, pandas, plotly; print('Entorno listo')"
```

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
│       └── __init__.py
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
2. Implementar la carga, validación y el preprocesamiento de imágenes.
3. Construir el servicio de inferencia y las reglas de priorización.
4. Integrar la interfaz Streamlit y la generación de tickets de mantenimiento.
5. Añadir pruebas con Pytest, seguimiento de ejecuciones con MLflow y revisión con Ruff.
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
