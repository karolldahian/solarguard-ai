# Guía de Despliegue y Orquestación con Docker — SolarGuard AI

Este documento describe la arquitectura de contenerización y el procedimiento de despliegue local mediante Docker y Docker Compose para el proyecto **SolarGuard AI**.

---

## 1. Arquitectura de Contenedores

La solución utiliza una imagen base estandarizada sobre **Python 3.13-slim** gestionada al 100% con **Astral `uv`**, orquestando tres servicios interconectados mediante una red interna:

```
                  ┌──────────────────────────────┐
                  │      Docker Host Network     │
                  └──────────────┬───────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         ▼                       ▼                       ▼
┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
│ solarguard-      │   │ solarguard-      │   │ solarguard-      │
│ frontend         │   │ backend          │   │ mlflow           │
│ (Streamlit)      │   │ (Servidor gRPC)  │   │ (Tracking Server)│
│ Puerto: 8501     │   │ Puerto: 50051    │   │ Puerto: 5000     │
└────────┬─────────┘   └────────┬─────────┘   └────────┬─────────┘
         │                      │                      │
         │                      ▼                      ▼
         │             ┌──────────────────┐   ┌──────────────────┐
         └────────────►│ Red Interna      │──►│ Volumen Persist. │
                       │ DNS: backend     │   │ mlflow_data      │
                       │ DNS: mlflow      │   │ /data/mlflow.db  │
                       └──────────────────┘   └──────────────────┘
```

### Servicios Orquestados:

1. **`solarguard-backend` (gRPC):**
   - **Puerto expuesto:** `50051:50051`.
   - **Punto de entrada:** `uv run python -m solarguard_ai.servidor_grpc --puerto 50051`.
   - **Responsabilidad:** Inferencia ONNX (`best.onnx`), preprocesamiento de imágenes, cálculo de severidad y emisión de tickets.

2. **`solarguard-frontend` (Streamlit):**
   - **Puerto expuesto:** `8501:8501`.
   - **Punto de entrada:** `uv run streamlit run app.py --server.port=8501 --server.address=0.0.0.0`.
   - **Responsabilidad:** Dashboard visual interactivo, análisis individual, análisis por lote, historial de inspecciones y mapa de riesgo.

3. **`solarguard-mlflow` (Tracking Server):**
   - **Puerto expuesto:** `5000:5000`.
   - **Punto de entrada:** `uv run mlflow server --host 0.0.0.0 --port 5000`.
   - **Persistencia:** Volumen Docker nombrado `mlflow_data` montado en `/data`.

---

## 2. Requisitos Previos

- **Docker Engine** >= 24.0 o **Docker Desktop** (Linux, Windows con WSL2 o macOS).
- **Docker Compose** v2 o superior (`docker compose version`).
- Artefacto del modelo preentrenado descargado en `models/best.onnx`.

---

## 3. Comandos de Operación

### 3.1 Construir las imágenes y levantar el ecosistema completo
```bash
# Construcción y arranque en segundo plano
docker compose up -d --build
```

### 3.2 Verificar el estado de los contenedores
```bash
docker compose ps
```

### 3.3 Consultar registros (logs) en tiempo real
```bash
# Ver logs de todos los servicios
docker compose logs -f

# Ver logs exclusivos del backend gRPC
docker compose logs -f backend

# Ver logs de Streamlit
docker compose logs -f frontend
```

### 3.4 Detener y limpiar los contenedores
```bash
# Detener los servicios preservando volúmenes de datos
docker compose down

# Detener los servicios y eliminar volúmenes persistentes
docker compose down -v
```

---

## 4. Acceso a las Interfaces Web

Una vez levantado el entorno con `docker compose up`:

| Servicio | URL de Acceso Local | Descripción |
| :--- | :--- | :--- |
| **SolarGuard Dashboard** | [http://localhost:8501](http://localhost:8501) | Interfaz web de Streamlit para diagnóstico de paneles. |
| **MLflow Tracking UI** | [http://localhost:5000](http://localhost:5000) | Métricas de latencia, tasa de aciertos y parámetros de corrida. |
| **Servidor gRPC** | `localhost:50051` | Canal RPC binario para clientes headless o integraciones externas. |

---

## 5. Variables de Entorno de los Contenedores

| Variable | Servicio | Valor por Defecto | Propósito |
| :--- | :--- | :--- | :--- |
| `PYTHONUNBUFFERED` | Todos | `1` | Salida inmediata de logs a stdout/stderr sin búfer. |
| `MLFLOW_TRACKING_URI` | `frontend`, `backend` | `http://mlflow:5000` | Resolución de red interna DNS hacia el servidor de métricas. |
| `MLFLOW_ENABLED` | `frontend`, `backend` | `true` | Habilita o deshabilita el envío de telemetría a MLflow. |
| `UV_SYSTEM_PYTHON` | Base | `1` | Permite a `uv` operar de manera optimizada dentro del contenedor. |
