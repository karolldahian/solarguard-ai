# SolarGuard AI — Arquitectura del Sistema y Contratos de Integración

> **Documento de Arquitectura y Especificación Técnica (Issue #4)**  
> **Proyecto:** SolarGuard AI — Detección y Priorización de Mantenimiento en Paneles Fotovoltaicos con IA  
> **Especialización en Inteligencia Artificial — Universidad Autónoma de Occidente (UAO)**  
> **Autor / Arquitecto:** Jose Fernando Luque Cajiao  

---

## 1. Visión General y Principios de Diseño

SolarGuard AI es una plataforma modular y desacoplada concebida para la auditoría visual automatizada de módulos solares fotovoltaicos. Su propósito es clasificar el estado físico-superficial del panel, calcular la prioridad operativa de atención y generar de manera desatendida los tickets de mantenimiento para las cuadrillas técnicas.

### Principios Fundamentales (AGENTS.md):
1. **Alta Cohesión y Bajo Acoplamiento:** Cada módulo (`ingesta`, `preprocesamiento`, `inferencia`, `priorizacion`, `tickets`, `view`) posee una responsabilidad única y aislada.
2. **Independencia Tecnológica de la Lógica de Negocio:** El motor de reglas de priorización y el emisor de tickets no dependen de ONNX Runtime, TensorFlow ni librerías de visión por computador.
3. **No Diagnóstico Eléctrico Definitivo:** Toda inferencia visual es una alerta preliminar sujeta a verificación de personal técnico calificado.
4. **Resiliencia y Modo Degradado:** Los servicios permiten ejecución simulada (*dry-run*) y manejo controlado de fallos sin interrumpir la operación.

---

## 2. Diagrama de Arquitectura del Sistema

```mermaid
flowchart TD
    subgraph UI_Layer ["Capa de Presentación"]
        ST[Streamlit Web App]
        CLI[Interfaz CLI / Script Batch]
    end

    subgraph API_Layer ["Capa de Integración / API"]
        GW[API Gateway / Backend FastAPI]
        GRPC[Servidor gRPC - Streaming Lotes]
    end

    subgraph Core_Pipeline ["Pipeline de Procesamiento e IA"]
        direction TB
        ING[solarguard_ai.ingesta<br/><i>Validación y Carga RGB</i>]
        PRE[solarguard_ai.preprocesamiento<br/><i>Redimensión 224x224 NCHW</i>]
        INF[solarguard_ai.inferencia<br/><i>ONNX Runtime - SolarScan</i>]
        PRI[solarguard_ai.priorizacion<br/><i>Motor de Reglas y Umbrales</i>]
        TCK[solarguard_ai.tickets<br/><i>Generación y Despacho</i>]

        ING -->|LoadedImage| PRE
        PRE -->|FloatTensor 1,3,224,224| INF
        INF -->|PredictionResult| PRI
        PRI -->|PriorityResult| TCK
    end

    subgraph External_Services ["Servicios y Almacenamiento Externo"]
        HF[(Hugging Face Hub<br/>best.onnx)]
        GH[(GitHub API<br/>Issues / Tickets)]
        MLF[(MLflow Tracking Server)]
        CONF[(config/prioritization.toml)]
    end

    ST -->|HTTP / multipart| GW
    CLI -->|Invocación directa| ING
    GW --> ING
    GRPC --> ING
    HF -.->|Descarga artefacto| INF
    CONF -.->|Umbral operativo| PRI
    TCK -->|REST API Issue| GH
    INF -.->|Métricas y Latencia| MLF
```

---

## 3. Contratos de Datos Entre Módulos Internos

Cada capa del pipeline consume y entrega contratos inmutables basados en `dataclasses` tipadas con Type Hints estrictos (PEP 484):

### 3.1. Ingesta (`solarguard_ai.ingesta`)
- **Entrada:** `source: str | Path | BinaryIO` (JPEG, PNG, TIFF).
- **Salida:** `LoadedImage`
  ```python
  @dataclass(frozen=True)
  class LoadedImage:
      image: Image.Image       # Imagen en modo RGB
      format: str              # Formato de origen ('JPEG', 'PNG', etc.)
      size: tuple[int, int]    # Dimensiones originales (width, height)
  ```

### 3.2. Preprocesamiento (`solarguard_ai.preprocesamiento`)
- **Entrada:** `image: Image.Image | NDArray[np.generic]`
- **Salida:** `FloatTensor` de tipo `np.float32` normalizado `0..1` con forma exacta `(1, 3, 224, 224)` en orden de canales `NCHW`.

### 3.3. Inferencia (`solarguard_ai.inferencia`)
- **Entrada:** `tensor: FloatTensor`
- **Salida:** `PredictionResult`
  ```python
  @dataclass(frozen=True)
  class PredictionResult:
      predicted_class: str             # Clean, Dusty, Bird-drop, Electrical-damage, etc.
      confidence: float                # 0.0 a 1.0
      probabilities: dict[str, float]  # Distribución completa de probabilidades
  ```

### 3.4. Priorización Operativa (`solarguard_ai.priorizacion`)
- **Entrada:** `predicted_class: str`, `confidence: float`, `review_threshold: float`
- **Salida:** `PriorityResult`
  ```python
  @dataclass(frozen=True)
  class PriorityResult:
      priority: PriorityLevel          # "high" | "medium" | "low"
      recommended_action: str          # Acción operativa inmediata
      requires_human_review: bool      # True si confidence < review_threshold o Unknown
      reason: str                      # Justificación técnica de la prioridad
  ```

### 3.5. Generación de Tickets (`solarguard_ai.tickets`)
- **Entrada:** `priority_result: PriorityResult`, `panel_id: str`, `location: str`, `confidence: float`
- **Salida:** `MaintenanceTicket` y `TicketCreationResult`
  ```python
  @dataclass(frozen=True)
  class MaintenanceTicket:
      title: str                       # Título estandarizado para triaje
      body: str                        # Cuerpo estructurado en Markdown con alertas y descargos
      labels: list[str]                # ['maintenance', 'urgent', 'electrical', ...]
      assignees: list[str]             # ['ing-electrico', ...]
      severity: str                    # 'high' | 'medium' | 'low'
      panel_id: str
      location: str
      predicted_class: str
      confidence: float
      requires_human_review: bool
      created_at: str                  # ISO 8601 UTC

  @dataclass(frozen=True)
  class TicketCreationResult:
      status: str                      # 'created' | 'simulated' | 'skipped' | 'failed'
      message: str
      ticket: MaintenanceTicket | None
      issue_number: int | None
      issue_url: str | None
  ```

---

## 4. Especificación OpenAPI 3.1.0 de Endpoints REST

La API del backend expone los siguientes endpoints para el frontend Streamlit y clientes externos:

```yaml
openapi: 3.1.0
info:
  title: SolarGuard AI REST API
  description: Servicio de inferencia, priorización y generación de tickets para mantenimiento fotovoltaico.
  version: 1.0.0
paths:
  /health:
    get:
      summary: Verificación de estado y salud del servicio
      responses:
        '200':
          description: Servicio saludable
          content:
            application/json:
              schema:
                type: object
                properties:
                  status: { type: string, example: "ok" }
                  model_loaded: { type: boolean, example: true }
                  version: { type: string, example: "0.1.0" }

  /api/v1/predict:
    post:
      summary: Clasificación visual de imagen de panel solar
      requestBody:
        required: true
        content:
          multipart/form-data:
            schema:
              type: object
              required: [file]
              properties:
                file:
                  type: string
                  format: binary
      responses:
        '200':
          description: Inferencia exitosa
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/PredictionResponse'
        '400':
          $ref: '#/components/responses/400BadRequest'
        '415':
          $ref: '#/components/responses/415UnsupportedMedia'

  /api/v1/prioritize:
    post:
      summary: Evaluación de prioridad operativa de mantenimiento
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required: [predicted_class, confidence]
              properties:
                predicted_class: { type: string, example: "Electrical-damage" }
                confidence: { type: number, minimum: 0.0, maximum: 1.0, example: 0.94 }
                review_threshold: { type: number, example: 0.70 }
      responses:
        '200':
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/PriorityResponse'

  /api/v1/tickets:
    post:
      summary: Emisión de ticket de mantenimiento en GitHub Issues
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required: [panel_id, predicted_class, confidence, priority]
              properties:
                panel_id: { type: string, example: "PANEL-B04-12" }
                location: { type: string, example: "Inversor 2 - Bloque Sur" }
                predicted_class: { type: string, example: "Electrical-damage" }
                confidence: { type: number, example: 0.94 }
                priority: { type: string, enum: [high, medium, low], example: "high" }
                force: { type: boolean, default: false }
      responses:
        '201':
          description: Ticket creado o simulado exitosamente
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/TicketResponse'

  /api/v1/pipeline:
    post:
      summary: Pipeline integral de auditoría (Imagen -> Inferencia -> Prioridad -> Ticket)
      requestBody:
        required: true
        content:
          multipart/form-data:
            schema:
              type: object
              required: [file, panel_id]
              properties:
                file: { type: string, format: binary }
                panel_id: { type: string, example: "PANEL-B04-12" }
                location: { type: string, example: "Parque Solar Yumbo" }
                create_ticket: { type: boolean, default: true }
      responses:
        '200':
          content:
            application/json:
              schema:
                type: object
                properties:
                  prediction: { $ref: '#/components/schemas/PredictionResponse' }
                  priority: { $ref: '#/components/schemas/PriorityResponse' }
                  ticket: { $ref: '#/components/schemas/TicketResponse' }

components:
  schemas:
    PredictionResponse:
      type: object
      properties:
        predicted_class: { type: string, example: "Physical-Damage" }
        confidence: { type: number, example: 0.925 }
        probabilities:
          type: object
          additionalProperties: { type: number }

    PriorityResponse:
      type: object
      properties:
        priority: { type: string, enum: [high, medium, low] }
        recommended_action: { type: string }
        requires_human_review: { type: boolean }
        reason: { type: string }

    TicketResponse:
      type: object
      properties:
        status: { type: string, enum: [created, simulated, skipped, failed] }
        message: { type: string }
        issue_number: { type: integer, nullable: true }
        issue_url: { type: string, nullable: true }

  responses:
    400BadRequest:
      description: Solicitud inválida (parámetros faltantes o corruptos)
    415UnsupportedMedia:
      description: Formato de archivo no soportado (solo JPEG, PNG, TIFF)
    500InternalError:
      description: Error interno de inferencia o comunicación con GitHub
```

---

## 5. Códigos de Error HTTP y Excepciones del Dominio

| Código HTTP | Significado | Excepción Interna de Dominio | Causa Común |
| :--- | :--- | :--- | :--- |
| **`200 OK`** | Operación completada con éxito | Ninguna | Inferencia o priorización resuelta |
| **`201 Created`** | Recurso creado exitosamente | Ninguna | Issue de GitHub generado en el repositorio |
| **`400 Bad Request`** | Parámetros inválidos | `ValueError`, `PrioritizationError` | Confianza fuera de rango `[0, 1]`, clase no soportada |
| **`415 Unsupported Media`** | Tipo de archivo no permitido | `ImageIngestionError` | Archivo PDF, SVG, o extensión desconocida |
| **`422 Unprocessable`** | Archivo legible pero tensor inválido | `InferenceError` | Imagen vacía o dimensiones incompatibles |
| **`500 Internal Server Error`** | Falla interna del servicio | `TicketGenerationError` | Fallo de conexión con GitHub API o carga del modelo |
| **`503 Service Unavailable`** | Inferencia no disponible | `InferenceError` | Archivo `best.onnx` no encontrado en ruta local |

---

## 6. Mapeo Organizacional y Reglas de Triaje

Para dar cumplimiento a la operación en campo y despacho a cuadrillas:

| Clase Diagnosticada | Nivel de Severidad | Responsable Asignado | SLA Operativo | Etiquetas GitHub |
| :--- | :--- | :--- | :--- | :--- |
| **`Electrical-damage`** | **Alta (`high`)** | `@ing-electrico` | `< 4 horas` | `maintenance`, `urgent`, `severity:high`, `electrical`, `risk` |
| **`Physical-Damage`** | **Alta (`high`)** | `@tecnico-campo` | `< 8 horas` | `maintenance`, `urgent`, `severity:high`, `structural`, `hardware` |
| **`Dusty`** | **Media (`medium`)** | `@mantenimiento-limpieza` | `< 48 horas` | `maintenance`, `severity:medium`, `cleaning`, `preventive` |
| **`Bird-drop`** | **Media (`medium`)** | `@mantenimiento-limpieza` | `< 24 horas` | `maintenance`, `severity:medium`, `cleaning`, `biological` |
| **`Snow-Covered`** | **Media (`medium`)** | `@mantenimiento-limpieza` | `< 12 horas` | `maintenance`, `severity:medium`, `cleaning`, `weather` |
| **`Unknown`** | **Media (`medium`)** | `@supervisor-triaje` | `< 12 horas` | `maintenance`, `severity:medium`, `triaje`, `review-needed`, `requires-human-review` |
| **`Clean`** | **Baja (`low`)** | *(Sin asignación)* | N/A | `routine` (No genera ticket automático) |

---

## 7. Verificación y Calidad

- **Pruebas Unitarias:** Cada contrato de datos cuenta con cobertura en `tests/test_ingesta.py`, `tests/test_preprocesamiento.py`, `tests/test_inferencia.py`, `tests/test_priorizacion.py` y `tests/test_tickets.py`.
- **Estructura AAA:** Las pruebas se rigen por Arrange-Act-Assert sin efectos colaterales de red ni dependencias de tokens en CI.
- **Herramientas de Calidad:** Validado bajo `ruff check` y `ruff format` conforme a Python 3.13.
