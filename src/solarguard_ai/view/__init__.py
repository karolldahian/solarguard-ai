"""Capa de presentacion de SolarGuard AI (Streamlit)."""

from solarguard_ai.view.diagnostico import (
    render_configuration_error,
    render_diagnosis,
    render_model_unavailable,
    render_prediction,
    render_priority,
    render_probabilities,
)
from solarguard_ai.view.modelo import (
    get_inference_service,
    get_priority_config,
    resolve_model_path,
    resolve_priority_config_path,
)
from solarguard_ai.view.pagina_principal import (
    render_header,
    render_image_metadata,
    render_image_preview,
    upload_image,
)

__all__ = [
    "get_inference_service",
    "get_priority_config",
    "render_configuration_error",
    "render_diagnosis",
    "render_header",
    "render_image_metadata",
    "render_image_preview",
    "render_model_unavailable",
    "render_prediction",
    "render_priority",
    "render_probabilities",
    "resolve_model_path",
    "resolve_priority_config_path",
    "upload_image",
]
