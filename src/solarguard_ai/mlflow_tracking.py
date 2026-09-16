"""MLflow tracking para SolarGuard AI.

Configuración centralizada y funciones helper para registrar métricas
de inferencia, latencias y eventos en MLflow.
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

# Configuración por defecto (override via env vars)
DEFAULT_TRACKING_URI = "http://localhost:5000"
DEFAULT_EXPERIMENT = "solarguard-inference"
DEFAULT_ENABLED = True


@dataclass(frozen=True)
class MlflowConfig:
    """Configuración inmutable de MLflow."""

    tracking_uri: str
    experiment_name: str
    enabled: bool


# Singleton para cliente MLflow
_mlflow_client: MlflowClient | None = None
_config: MlflowConfig | None = None
_init_lock = threading.Lock()
_run_counter = 0
_run_counter_lock = threading.Lock()


def _load_config() -> MlflowConfig:
    """Carga configuración desde variables de entorno."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT", DEFAULT_EXPERIMENT)
    enabled = os.environ.get("MLFLOW_ENABLED", str(DEFAULT_ENABLED)).lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    return MlflowConfig(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        enabled=enabled,
    )


def _get_config() -> MlflowConfig:
    """Obtiene la configuración (cached)."""
    global _config
    if _config is None:
        _config = _load_config()
    return _config


def _get_client() -> MlflowClient:
    """Inicializa y retorna el cliente MLflow (lazy singleton)."""
    global _mlflow_client
    if _mlflow_client is None:
        with _init_lock:
            if _mlflow_client is None:
                config = _get_config()
                mlflow.set_tracking_uri(config.tracking_uri)
                _mlflow_client = MlflowClient(tracking_uri=config.tracking_uri)
                # Crear experimento si no existe
                exp = _mlflow_client.get_experiment_by_name(config.experiment_name)
                if exp is None:
                    _mlflow_client.create_experiment(config.experiment_name)
                mlflow.set_experiment(config.experiment_name)
    return _mlflow_client


def is_enabled() -> bool:
    """Verifica si MLflow está habilitado."""
    return _get_config().enabled


def _next_run_id() -> int:
    """Genera un ID de run incremental para naming."""
    global _run_counter
    with _run_counter_lock:
        _run_counter += 1
        return _run_counter


@contextmanager
def start_run(run_name: str | None = None, tags: dict[str, str] | None = None):
    """Context manager para un run de MLflow.

    Si MLflow está deshabilitado, hace no-op y yield None.

    Args:
        run_name: Nombre opcional del run (se autogenera si no se da).
        tags: Tags adicionales para el run.

    Yields:
        mlflow.ActiveRun | None: El run activo o None si deshabilitado.
    """
    if not is_enabled():
        yield None
        return

    _ = _get_client()
    _ = _get_config()

    if run_name is None:
        run_name = f"run-{_next_run_id():06d}-{int(time.time())}"

    run_tags = {"mlflow.runName": run_name}
    if tags:
        run_tags.update(tags)

    with mlflow.start_run(run_name=run_name, tags=run_tags) as run:
        yield run


def log_prediction(
    predicted_class: str,
    confidence: float,
    latency_ms: float,
    cache_hit: bool = False,
    probabilities: dict[str, float] | None = None,
    model_path: str | None = None,
    confidence_threshold: float | None = None,
    panel_id: str | None = None,
    run: Any | None = None,
) -> None:
    """Registra una predicción individual en MLflow.

    Usa nested runs para evitar colisiones de parámetros cuando se loguean
    múltiples predicciones en el mismo run padre.

    Args:
        predicted_class: Clase predicha.
        confidence: Confianza (0.0-1.0).
        latency_ms: Latencia de inferencia en ms.
        cache_hit: Si el resultado vino de caché.
        probabilities: Dict completo de probabilidades por clase.
        model_path: Ruta del modelo ONNX.
        confidence_threshold: Umbral de confianza usado.
        panel_id: Identificador del panel (opcional).
        run: Run activo de MLflow (opcional, usa contexto actual si None).
    """
    if not is_enabled():
        return

    # Inicializar cliente y URI antes de loguear
    _ = _get_client()
    _ = _get_config()

    metrics = {
        "confidence": confidence,
        "latency_ms": latency_ms,
        "cache_hit": 1.0 if cache_hit else 0.0,
        "is_unknown": 1.0 if predicted_class == "Unknown" else 0.0,
    }

    params = {}
    if model_path:
        params["model_path"] = str(model_path)
    if confidence_threshold is not None:
        params["confidence_threshold"] = confidence_threshold
    if panel_id:
        params["panel_id"] = panel_id

    tags = {
        "predicted_class": predicted_class,
        "event_type": "prediction",
    }

    # Log probabilidades como métricas individuales si están disponibles
    if probabilities:
        for cls, prob in probabilities.items():
            safe_cls = cls.replace("-", "_").replace(" ", "_")
            metrics[f"prob_{safe_cls}"] = prob

    # Usar nested run para cada predicción (evita colisiones de params en mismo run_id)
    if run is not None:
        # Ya estamos en un run, crear nested run
        with mlflow.start_run(nested=True, run_name=f"prediction-{predicted_class}"):
            for k, v in metrics.items():
                mlflow.log_metric(k, v)
            for k, v in params.items():
                mlflow.log_param(k, v)
            for k, v in tags.items():
                mlflow.set_tag(k, v)
    else:
        # No hay run activo, crear uno nuevo para esta predicción
        with mlflow.start_run(run_name=f"prediction-{predicted_class}"):
            for k, v in metrics.items():
                mlflow.log_metric(k, v)
            for k, v in params.items():
                mlflow.log_param(k, v)
            for k, v in tags.items():
                mlflow.set_tag(k, v)


def log_pipeline_request(
    panel_id: str,
    image_size: tuple[int, int],
    image_format: str,
    total_latency_ms: float,
    ingestion_latency_ms: float | None = None,
    preprocessing_latency_ms: float | None = None,
    inference_latency_ms: float | None = None,
    ticket_latency_ms: float | None = None,
    predicted_class: str | None = None,
    confidence: float | None = None,
    priority: str | None = None,
    requires_human_review: bool | None = None,
    ticket_status: str | None = None,
    error: str | None = None,
) -> None:
    """Registra métricas completas del pipeline gRPC.

    Usa nested run para aislar cada request.

    Args:
        panel_id: Identificador del panel.
        image_size: (width, height) de la imagen original.
        image_format: Formato de imagen (JPEG, PNG, etc.).
        total_latency_ms: Latencia total end-to-end.
        ingestion_latency_ms: Latencia de ingesta (opcional).
        preprocessing_latency_ms: Latencia de preprocesamiento (opcional).
        inference_latency_ms: Latencia de inferencia (opcional).
        ticket_latency_ms: Latencia de generación de ticket (opcional).
        predicted_class: Clase predicha (opcional).
        confidence: Confianza (opcional).
        priority: Prioridad operativa (opcional).
        requires_human_review: Si requiere revisión humana (opcional).
        ticket_status: Estado del ticket (created/simulated/skipped/failed).
        error: Mensaje de error si falló (opcional).
    """
    if not is_enabled():
        return

    _ = _get_client()
    _ = _get_config()

    with mlflow.start_run(run_name=f"pipeline-{panel_id}", nested=True):
        metrics = {
            "total_latency_ms": total_latency_ms,
            "image_width": float(image_size[0]),
            "image_height": float(image_size[1]),
        }

        if ingestion_latency_ms is not None:
            metrics["ingestion_latency_ms"] = ingestion_latency_ms
        if preprocessing_latency_ms is not None:
            metrics["preprocessing_latency_ms"] = preprocessing_latency_ms
        if inference_latency_ms is not None:
            metrics["inference_latency_ms"] = inference_latency_ms
        if ticket_latency_ms is not None:
            metrics["ticket_latency_ms"] = ticket_latency_ms
        if confidence is not None:
            metrics["confidence"] = confidence
        if requires_human_review is not None:
            metrics["requires_human_review"] = 1.0 if requires_human_review else 0.0

        params = {
            "panel_id": panel_id,
            "image_format": image_format,
        }

        tags = {
            "event_type": "pipeline_request",
        }

        if predicted_class:
            tags["predicted_class"] = predicted_class
        if priority:
            tags["priority"] = priority
        if ticket_status:
            tags["ticket_status"] = ticket_status
        if error:
            tags["error"] = "true"
            tags["error_message"] = error[:200]

        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        for k, v in params.items():
            mlflow.log_param(k, v)
        for k, v in tags.items():
            mlflow.set_tag(k, v)


def log_streamlit_request(
    panel_id: str,
    image_size: tuple[int, int],
    image_format: str,
    total_latency_ms: float,
    network_latency_ms: float | None = None,
    predicted_class: str | None = None,
    confidence: float | None = None,
    priority: str | None = None,
    error: str | None = None,
) -> None:
    """Registra métricas desde Streamlit (frontend).

    Usa nested run para aislar cada request.
    """
    if not is_enabled():
        return

    _ = _get_client()
    _ = _get_config()

    with mlflow.start_run(run_name=f"streamlit-{panel_id}", nested=True):
        metrics = {
            "streamlit_total_latency_ms": total_latency_ms,
            "image_width": float(image_size[0]),
            "image_height": float(image_size[1]),
        }

        if network_latency_ms is not None:
            metrics["streamlit_network_latency_ms"] = network_latency_ms
        if confidence is not None:
            metrics["confidence"] = confidence

        params = {
            "panel_id": panel_id,
            "image_format": image_format,
            "source": "streamlit",
        }

        tags = {
            "event_type": "streamlit_request",
        }

        if predicted_class:
            tags["predicted_class"] = predicted_class
        if priority:
            tags["priority"] = priority
        if error:
            tags["error"] = "true"
            tags["error_message"] = error[:200]

        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        for k, v in params.items():
            mlflow.log_param(k, v)
        for k, v in tags.items():
            mlflow.set_tag(k, v)


def log_error(
    error_type: str,
    error_message: str,
    context: dict[str, Any] | None = None,
) -> None:
    """Registra un error en MLflow.

    Usa nested run para aislar cada error.

    Args:
        error_type: Tipo de error (ej: 'InferenceError', 'IngestionError').
        error_message: Mensaje descriptivo.
        context: Contexto adicional (panel_id, etc.).
    """
    if not is_enabled():
        return

    _ = _get_client()
    _ = _get_config()

    with mlflow.start_run(run_name=f"error-{error_type}", nested=True):
        tags = {
            "event_type": "error",
            "error_type": error_type,
        }

        if context:
            for k, v in context.items():
                if isinstance(v, (str, int, float, bool)):
                    tags[f"ctx_{k}"] = str(v)

        mlflow.set_tags(tags)
        mlflow.log_metric("error_count", 1)


__all__ = [
    "MlflowConfig",
    "is_enabled",
    "log_error",
    "log_pipeline_request",
    "log_prediction",
    "log_streamlit_request",
    "start_run",
]
